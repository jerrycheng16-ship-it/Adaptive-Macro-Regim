import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. 頁面基本配置 (Dark Theme Layout)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Adaptive Macro Regimes Terminal",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #0F172A; color: #F8FAFC; }
    .stTable { background-color: #1E293B; border-radius: 0.5rem; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

REGIME_CONFIG = {
    'Expansion':  {'minScore': 0.30, 'name': '擴張期 (Expansion)',  'color': '#10B981', 'bgColor': 'rgba(16, 185, 129, 0.20)', 'eqW': 1.00, 'tltW': 0.00, 'cashW': 0.00},
    'Recovery':   {'minScore': 0.00, 'name': '復甦期 (Recovery)',   'color': '#3B82F6', 'bgColor': 'rgba(59, 130, 246, 0.20)',  'eqW': 0.80, 'tltW': 0.00, 'cashW': 0.20},
    'Slowdown':   {'minScore':-0.30, 'name': '放緩期 (Slowdown)',   'color': '#F59E0B', 'bgColor': 'rgba(245, 158, 11, 0.22)',  'eqW': 0.50, 'tltW': 0.30, 'cashW': 0.20},
    'Contraction':{'minScore':-3.00, 'name': '收縮期 (Contraction)', 'color': '#EF4444', 'bgColor': 'rgba(239, 68, 68, 0.25)',  'eqW': 0.10, 'tltW': 0.60, 'cashW': 0.30}
}

# -----------------------------------------------------------------------------
# 2. 原始市場數據快取 (從 1985 年下載預留足夠暖機視窗)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=86400)
def load_raw_market_data():
    tickers = {
        'Equity': 'VFINX',      # S&P 500 (SPY 代理)
        'MidTreasury': 'VFITX',  # 中天期國債 (IEF 代理)
        'LongTreasury': 'VUSTX', # 長天期國債 (TLT 代理)
        'HighYield': 'VWEHX',   # 高收益債 (HYG 代理)
        'Cash': 'VFISX'         # 短期國債 (BIL 代理)
    }
    
    raw_df = yf.download(list(tickers.values()), start="1985-01-01", auto_adjust=True)
    
    if isinstance(raw_df.columns, pd.MultiIndex):
        df = raw_df['Close'] if 'Close' in raw_df.columns.levels[0] else raw_df.iloc[:, :len(tickers)]
    else:
        df = raw_df

    df = df.rename(columns={v: k for k, v in tickers.items()}).dropna()
    monthly_data = df.resample('ME').last()
    returns = monthly_data.pct_change().dropna()
    
    raw_growth = monthly_data['Equity'].pct_change(6)
    raw_credit = (monthly_data['HighYield'] / monthly_data['MidTreasury']).pct_change(6)
    raw_rates = monthly_data['LongTreasury'].pct_change(6)
    
    raw_signals = pd.DataFrame({'Growth': raw_growth, 'Credit': raw_credit, 'Rates': raw_rates}).dropna()
    return raw_signals, returns

# -----------------------------------------------------------------------------
# 3. 可調參數核心運算引擎 (固定 1993-01 起點，解決 Benchmark 移動問題)
# -----------------------------------------------------------------------------
def run_macro_model(raw_signals, returns, z_window, ic_window):
    # 1. Z-Score 標準化
    z_signals = pd.DataFrame(index=raw_signals.index)
    for col in raw_signals.columns:
        mean = raw_signals[col].rolling(z_window).mean()
        std = raw_signals[col].rolling(z_window).std()
        z_signals[col] = (raw_signals[col] - mean) / std.replace(0, 1)
        
    z_signals = z_signals.dropna()
    
    # 2. 滾動 IC 適應性權重
    fwd_ret = returns['Equity'].reindex(z_signals.index).shift(-1)
    rolling_ic = pd.DataFrame(index=z_signals.index)
    for col in z_signals.columns:
        rolling_ic[col] = z_signals[col].rolling(ic_window).corr(fwd_ret)
        
    weights = rolling_ic.map(lambda x: max(x, 0) + 0.1 if pd.notnull(x) else 0.1)
    weight_sum = weights.sum(axis=1)
    weights = weights.div(weight_sum, axis=0)
    
    macro_score = (z_signals * weights).sum(axis=1).dropna()
    
    # 3. 體制劃分與部位對應
    def get_regime_info(s):
        if s > REGIME_CONFIG['Expansion']['minScore']: return 'Expansion'
        elif s > REGIME_CONFIG['Recovery']['minScore']: return 'Recovery'
        elif s > REGIME_CONFIG['Slowdown']['minScore']: return 'Slowdown'
        else: return 'Contraction'
        
    regimes = macro_score.map(get_regime_info)
    
    alloc_eq = regimes.map(lambda r: REGIME_CONFIG[r]['eqW']).shift(1)
    alloc_tlt = regimes.map(lambda r: REGIME_CONFIG[r]['tltW']).shift(1)
    alloc_cash = regimes.map(lambda r: REGIME_CONFIG[r]['cashW']).shift(1)
    
    # 統一將回測起始點固定為 1993-01-01
    fixed_start_date = pd.to_datetime('1993-01-01')
    valid_idx = alloc_eq.dropna().loc[fixed_start_date:].index
    
    ret_eq = returns['Equity'].loc[valid_idx]
    ret_tlt = returns['LongTreasury'].loc[valid_idx]
    ret_cash = returns['Cash'].loc[valid_idx]
    
    strat_ret = (alloc_eq.loc[valid_idx] * ret_eq) + \
                (alloc_tlt.loc[valid_idx] * ret_tlt) + \
                (alloc_cash.loc[valid_idx] * ret_cash)
                
    backtest_df = pd.DataFrame({
        'Regime': regimes.loc[valid_idx],
        'SPY_Ret': ret_eq,
        'Cash_Ret': ret_cash,
        'Strat_Ret': strat_ret
    }, index=valid_idx)
    
    monthly_details = pd.DataFrame({
        'Month_Str': z_signals.index.strftime('%Y-%m'),
        'Macro_Score': macro_score,
        'Growth_Z': z_signals['Growth'],
        'Credit_Z': z_signals['Credit'],
        'Rates_Z': z_signals['Rates'],
        'Regime': regimes
    }, index=z_signals.index)
    
    return backtest_df, monthly_details

# 載入資料
try:
    with st.spinner("正在下載行情數據..."):
        raw_signals, returns = load_raw_market_data()
except Exception as e:
    st.error(f"數據下載失敗: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 4. Header & Modal 按鈕
# -----------------------------------------------------------------------------
col_header, col_btn1, col_btn2 = st.columns([2.5, 1, 1])

with col_header:
    st.title("Adaptive Macro Regimes Terminal")
    st.caption("Inspired by Jim Masturzo (Syzygy Asset Management / Research Affiliates)")

with col_btn1:
    if st.button("📄 論文出處與數據說明"):
        @st.dialog("論文出處與金融代理數據計算說明")
        def show_paper_info():
            st.markdown("""
            **📄 參考學術論文:**
            * *Adaptive Macro Regimes for Dynamic Equity Allocation* (Jim Masturzo, Omid Shakernia, Alex Pickard)
            * Published in *The Journal of Portfolio Management*
            
            ---
            **⚙️ 靈活性參數設計:**
            1. **Z-Score 滾動視窗 (12M - 60M):** 控制計算標準分數時的歷史參考記憶長度。預設 36 個月。
            2. **IC 權重視窗 (3M - 24M):** 控制模型對近期因子失效/生效的適應靈敏度。預設 8 個月。
            """)
        show_paper_info()

with col_btn2:
    if st.button("⚡ Adaptive Strategy 策略說明"):
        @st.dialog("Adaptive Strategy 策略標的與調倉時序說明")
        def show_strat_info():
            st.markdown("""
            **⏱️ 當月訊號預測下月 (Month T Signal for Month T+1 Allocation):**
            * 模型於 **$T$ 月底**讀取當期與過去數據計算 Macro Score 並判定 Regime，用於決定 **$T+1$ 月**整個月的資產配置。
            
            ---
            **📊 體制判斷與資產配置矩陣:**
            * **Expansion (> +0.30):** 100% SPY
            * **Recovery (0.00 ~ +0.30):** 80% SPY / 20% Cash (BIL)
            * **Slowdown (-0.30 ~ 0.00):** 50% SPY / 30% TLT / 20% Cash (BIL)
            * **Contraction (< -0.30):** 10% SPY / 60% TLT / 30% Cash (BIL)
            """)
        show_strat_info()

st.divider()

# -----------------------------------------------------------------------------
# 5. 參數控制與特定月份動態查詢 (IC 預設改為 8M)
# -----------------------------------------------------------------------------
st.markdown("### 🎛️ 策略模型參數調校 (Model Parameter Control)")

col_z, col_ic = st.columns(2)
with col_z:
    z_win_sel = st.slider(
        "Z-Score 滾動視窗 (Z-Score Lookback Window - Months):",
        min_value=12, max_value=60, value=36, step=6,
        help="決定計算標準差與均值的歷史參考長度。預設 36 個月。"
    )

with col_ic:
    ic_win_sel = st.slider(
        "適應性 IC 權重視窗 (IC Weighting Lookback Window - Months):",
        min_value=3, max_value=24, value=8, step=1, # 已調整預設值為 8
        help="決定統計因子與未來股市相關性的視窗。視窗越短，權重調整越靈敏。預設 8 個月。"
    )

df_bt, df_details = run_macro_model(raw_signals, returns, z_win_sel, ic_win_sel)

st.divider()

# 月份查詢
all_months = df_details['Month_Str'].tolist()

col_sel1, col_sel2 = st.columns([2, 3])
with col_sel1:
    selected_month_str = st.selectbox(
        "🔍 選擇查詢月份 (Select Historical Month):",
        options=all_months[::-1],
        index=0
    )

m_row = df_details[df_details['Month_Str'] == selected_month_str].iloc[0]

m_score = m_row['Macro_Score']
m_g = m_row['Growth_Z']
m_c = m_row['Credit_Z']
m_r = m_row['Rates_Z']
m_regime = m_row['Regime']
m_cfg = REGIME_CONFIG[m_regime]

col_p1, col_p2, col_p3 = st.columns([1.2, 1.2, 1])

with col_p1:
    st.markdown(f"### 當月標準化代理數據 (`{selected_month_str}`)")
    st.metric(f"成長代理 Z-Score (前{z_win_sel}M基準)", f"{m_g:+.2f}")
    st.metric(f"信用代理 Z-Score (前{z_win_sel}M基準)", f"{m_c:+.2f}")
    st.metric(f"利率代理 Z-Score (前{z_win_sel}M基準)", f"{m_r:+.2f}")

with col_p2:
    st.markdown(f"### 當前體制判斷 (`{selected_month_str}`)")
    st.metric("當月適應性宏觀得分 (Macro Score)", f"{m_score:+.2f}")
    st.markdown(f"**當月判定體制：** <span style='color:{m_cfg['color']}; font-size: 1.25rem; font-weight: bold;'>{m_cfg['name']}</span>", unsafe_allow_html=True)
    st.markdown(f"**下月建議股票比重 (SPY)：** **{int(m_cfg['eqW']*100)}%**")

with col_p3:
    st.markdown("### 下月資產配置預測比重")
    fig_donut = go.Figure(data=[go.Pie(
        labels=['股票 (SPY)', '長債 (TLT)', '現金 (BIL)'],
        values=[m_cfg['eqW']*100, m_cfg['tltW']*100, m_cfg['cashW']*100],
        hole=.6,
        marker_colors=[m_cfg['color'], '#6366F1', '#334155']
    )])
    fig_donut.update_layout(
        showlegend=True,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94A3B8')
    )
    st.plotly_chart(fig_donut, use_container_width=True)

st.divider()

# -----------------------------------------------------------------------------
# 6. 可選擇時間區間之動態回測 (Interactive Date Range Selector)
# -----------------------------------------------------------------------------
st.markdown("## 歷史體制動態回測 (Interactive Backtest Engine)")

min_bdate = df_bt.index.min().date()
max_bdate = df_bt.index.max().date()

start_date_sel, end_date_sel = st.slider(
    "📅 調整歷史回測時間區間 (Select Backtest Range):",
    min_value=min_bdate,
    max_value=max_bdate,
    value=(min_bdate, max_bdate),
    format="YYYY-MM"
)

sub_bt = df_bt.loc[pd.to_datetime(start_date_sel):pd.to_datetime(end_date_sel)].copy()

if len(sub_bt) < 2:
    st.warning("請選擇包含至少兩個月以上的時間區間。")
    st.stop()

sub_bt['SPY_Cum'] = (1 + sub_bt['SPY_Ret']).cumprod() * 100
sub_bt['Strat_Cum'] = (1 + sub_bt['Strat_Ret']).cumprod() * 100

# -----------------------------------------------------------------------------
# 7. 回測表格與動態圖表
# -----------------------------------------------------------------------------
st.markdown(f"### `{start_date_sel.strftime('%Y-%m')}` 至 `{end_date_sel.strftime('%Y-%m')}` 體制統計與月報酬率 (Z-Score: {z_win_sel}M, IC: {ic_win_sel}M)")

stats_list = []
sub_total_m = len(sub_bt)

for key, c in REGIME_CONFIG.items():
    s_df = sub_bt[sub_bt['Regime'] == key]
    cnt = len(s_df)
    pct = (cnt / sub_total_m) * 100 if sub_total_m > 0 else 0
    avg_spy = s_df['SPY_Ret'].mean() * 100 if cnt > 0 else 0
    avg_cash = s_df['Cash_Ret'].mean() * 100 if cnt > 0 else 0
    avg_strat = s_df['Strat_Ret'].mean() * 100 if cnt > 0 else 0
    
    stats_list.append({
        '宏觀體制 (Regime)': c['name'],
        '區間內月份數 (佔比)': f"{cnt} 個月 ({pct:.1f}%)",
        '股票 (SPY) 月報酬': f"{avg_spy:+.2f}%",
        '現金/短債 (BIL) 月報酬': f"{avg_cash:+.2f}%",
        '適應性策略 (Strategy) 月報酬': f"{avg_strat:+.2f}%"
    })

st.table(pd.DataFrame(stats_list))

# --- 走勢圖與色帶標示 ---
st.markdown("### 累積報酬率曲線與背景體制色帶 (Equity Curves & Regime Bands)")

st.markdown("""
<div style="display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 0.8rem; font-size: 0.85rem;">
    <span style="display: flex; align-items: center; gap: 0.3rem;"><span style="width:12px; height:12px; background-color:#10B981; display:inline-block; border-radius:2px;"></span> 🟢 擴張 (Expansion)</span>
    <span style="display: flex; align-items: center; gap: 0.3rem;"><span style="width:12px; height:12px; background-color:#3B82F6; display:inline-block; border-radius:2px;"></span> 🔵 復甦 (Recovery)</span>
    <span style="display: flex; align-items: center; gap: 0.3rem;"><span style="width:12px; height:12px; background-color:#F59E0B; display:inline-block; border-radius:2px;"></span> 🟡 放緩 (Slowdown)</span>
    <span style="display: flex; align-items: center; gap: 0.3rem;"><span style="width:12px; height:12px; background-color:#EF4444; display:inline-block; border-radius:2px;"></span> 🔴 收縮 (Contraction)</span>
</div>
""", unsafe_allow_html=True)

fig_line = go.Figure()

fig_line.add_trace(go.Scatter(
    x=sub_bt.index, y=sub_bt['Strat_Cum'],
    mode='lines', name='適應性宏觀體制策略 (Adaptive Strategy)',
    line=dict(color='#10B981', width=2.5)
))
fig_line.add_trace(go.Scatter(
    x=sub_bt.index, y=sub_bt['SPY_Cum'],
    mode='lines', name='買入持有 S&P 500 (Buy & Hold SPY)',
    line=dict(color='#94A3B8', width=1.5, dash='dash')
))

current_reg = None
start_d = None

for i in range(len(sub_bt)):
    date = sub_bt.index[i]
    reg = sub_bt['Regime'].iloc[i]
    
    if reg != current_reg:
        if current_reg is not None:
            fig_line.add_vrect(
                x0=start_d.strftime('%Y-%m-%d'), x1=date.strftime('%Y-%m-%d'),
                fillcolor=REGIME_CONFIG[current_reg]['bgColor'],
                opacity=1.0, layer="below", line_width=0
            )
        current_reg = reg
        start_d = date

if current_reg is not None:
    fig_line.add_vrect(
        x0=start_d.strftime('%Y-%m-%d'), x1=sub_bt.index[-1].strftime('%Y-%m-%d'),
        fillcolor=REGIME_CONFIG[current_reg]['bgColor'],
        opacity=1.0, layer="below", line_width=0
    )

fig_line.update_layout(
    yaxis_type="log",
    yaxis=dict(
        tickformat="$~s",
        gridcolor='#334155'
    ),
    xaxis=dict(
        type="date",
        showgrid=False
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#94A3B8'),
    margin=dict(t=10, b=10, l=10, r=10),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=11)
    )
)

st.plotly_chart(fig_line, use_container_width=True)

# -----------------------------------------------------------------------------
# 8. 選定區間之核心 KPI 指標計算
# -----------------------------------------------------------------------------
st.markdown("### 選定時間區間核心績效指標 (Selected Range KPI Metrics)")

def calc_kpis(ret_series):
    cum = (1 + ret_series).cumprod()
    n_years = len(ret_series) / 12.0
    cagr = (cum.iloc[-1]) ** (1 / n_years) - 1 if n_years > 0 else 0
    vol = ret_series.std() * np.sqrt(12)
    sharpe = cagr / vol if vol != 0 else 0
    dd = cum / cum.cummax() - 1
    mdd = dd.min()
    return cagr*100, vol*100, mdd*100, sharpe

s_cagr, s_vol, s_mdd, s_sharpe = calc_kpis(sub_bt['Strat_Ret'])
b_cagr, b_vol, b_mdd, b_sharpe = calc_kpis(sub_bt['SPY_Ret'])

kpi_data = [
    {
        '投資策略名稱 (Strategy)': f'適應性宏觀體制策略 (Z:{z_win_sel}M, IC:{ic_win_sel}M)',
        '區間年化報酬率 (CAGR)': f"+{s_cagr:.2f}%",
        '年化波動度 (Volatility)': f"{s_vol:.2f}%",
        '區間最大回撤 (Max Drawdown)': f"{s_mdd:.2f}%",
        '夏普比率 (Sharpe Ratio)': f"{s_sharpe:.2f}"
    },
    {
        '投資策略名稱 (Strategy)': '買入持有 S&P 500 (Buy & Hold SPY)',
        '區間年化報酬率 (CAGR)': f"+{b_cagr:.2f}%",
        '年化波動度 (Volatility)': f"{b_vol:.2f}%",
        '區間最大回撤 (Max Drawdown)': f"{b_mdd:.2f}%",
        '夏普比率 (Sharpe Ratio)': f"{b_sharpe:.2f}"
    }
]

st.table(pd.DataFrame(kpi_data))
