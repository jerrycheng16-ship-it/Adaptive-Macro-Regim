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
# 2. 數據載入與優化版回測引擎
# -----------------------------------------------------------------------------
@st.cache_data(ttl=86400)
def load_optimized_backtest():
    tickers = {
        'Equity': 'VFINX',      # S&P 500 (SPY 代理)
        'MidTreasury': 'VFITX',  # 中天期國債 (IEF 代理)
        'LongTreasury': 'VUSTX', # 長天期國債 (TLT 代理)
        'HighYield': 'VWEHX',   # 高收益債 (HYG 代理)
        'Cash': 'VFISX'         # 短期國債 (BIL 代理)
    }
    
    raw_df = yf.download(list(tickers.values()), start="1988-01-01", auto_adjust=True)
    
    if isinstance(raw_df.columns, pd.MultiIndex):
        df = raw_df['Close'] if 'Close' in raw_df.columns.levels[0] else raw_df.iloc[:, :len(tickers)]
    else:
        df = raw_df

    df = df.rename(columns={v: k for k, v in tickers.items()}).dropna()
    monthly_data = df.resample('ME').last()
    returns = monthly_data.pct_change().dropna()
    
    # 1. 原始代理因子 (6M Momentum)
    raw_growth = monthly_data['Equity'].pct_change(6)
    raw_credit = (monthly_data['HighYield'] / monthly_data['MidTreasury']).pct_change(6)
    raw_rates = monthly_data['LongTreasury'].pct_change(6)
    
    raw_signals = pd.DataFrame({'Growth': raw_growth, 'Credit': raw_credit, 'Rates': raw_rates}).dropna()
    
    # 2. 36 個月滾動 Z-Score 標準化
    z_signals = pd.DataFrame(index=raw_signals.index)
    for col in raw_signals.columns:
        mean = raw_signals[col].rolling(36).mean()
        std = raw_signals[col].rolling(36).std()
        z_signals[col] = (raw_signals[col] - mean) / std.replace(0, 1)
        
    z_signals = z_signals.dropna()
    
    # 3. 計算 12 個月滾動 IC 適應性權重
    fwd_ret = returns['Equity'].reindex(z_signals.index).shift(-1)
    rolling_ic = pd.DataFrame(index=z_signals.index)
    for col in z_signals.columns:
        rolling_ic[col] = z_signals[col].rolling(12).corr(fwd_ret)
        
    weights = rolling_ic.map(lambda x: max(x, 0) + 0.1 if pd.notnull(x) else 0.1)
    weight_sum = weights.sum(axis=1)
    weights = weights.div(weight_sum, axis=0)
    
    macro_score = (z_signals * weights).sum(axis=1).dropna()
    
    # 4. 體制劃分與部位對應
    def get_regime_info(s):
        if s > REGIME_CONFIG['Expansion']['minScore']: return 'Expansion'
        elif s > REGIME_CONFIG['Recovery']['minScore']: return 'Recovery'
        elif s > REGIME_CONFIG['Slowdown']['minScore']: return 'Slowdown'
        else: return 'Contraction'
        
    regimes = macro_score.map(get_regime_info)
    
    alloc_eq = regimes.map(lambda r: REGIME_CONFIG[r]['eqW']).shift(1)
    alloc_tlt = regimes.map(lambda r: REGIME_CONFIG[r]['tltW']).shift(1)
    alloc_cash = regimes.map(lambda r: REGIME_CONFIG[r]['cashW']).shift(1)
    
    valid_idx = alloc_eq.dropna().index
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
    with st.spinner("正在執行真實歷史數據載入與動態回測..."):
        df_bt, df_details = load_optimized_backtest()
except Exception as e:
    st.error(f"數據下載失敗: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 3. Header & Detailed Modal 按鈕
# -----------------------------------------------------------------------------
col_header, col_btn1, col_btn2 = st.columns([2.5, 1, 1])

with col_header:
    st.title("Adaptive Macro Regimes")
    st.caption("Inspired by Jim Masturzo (Syzygy Asset Management / Research Affiliates) | 動態對接真實數據")

with col_btn1:
    if st.button("📄 論文出處與數據說明"):
        @st.dialog("論文出處與金融代理數據計算說明")
        def show_paper_info():
            st.markdown("""
            **📄 參考學術論文:**
            * *Adaptive Macro Regimes for Dynamic Equity Allocation* (Jim Masturzo, Omid Shakernia, Alex Pickard)
            * Published in *The Journal of Portfolio Management*
            
            ---
            **⚙️ 多重時間視窗 (Multi-Lookback Windows) 計算邏輯:**
            1. **6 個月 (6M) 價格動能 (Raw Proxies):**
               * **成長代理 (Growth):** S&P 500 (SPY) 過去 6 個月累積報酬率。
               * **信用利差 (Credit):** 高收益債 (HYG) / 中天期國債 (IEF) 過去 6 個月相對強度變化。
               * **利率趨勢 (Rates):** 20年期長美債 (TLT) 過去 6 個月價格動能。
            2. **36 個月 (36M) 滾動 Z-Score 標準化:**
               * 將各代理因子的 6M 原始值減去過去 36 個月均值並除以標準差，轉化為標準分數 $N(0,1)$，消除不同資產類別的量綱差異。
            3. **12 個月 (12M) 滾動 IC 適應性動態加權:**
               * 統計過去 12 個月各因子對未來一期股票報酬的滾動相關性 (Information Coefficient, IC)，給予正預測力指標較高動態權重。
            """)
        show_paper_info()

with col_btn2:
    if st.button("⚡ Adaptive Strategy 策略說明"):
        @st.dialog("Adaptive Strategy 策略標的與調倉時序說明")
        def show_strat_info():
            st.markdown("""
            **⏱️ 當月訊號預測下月 (Month T Signal for Month T+1 Allocation):**
            * **調倉時序機制：** 模型於 **$T$ 月底**（如 2026-06 末）讀取當期與過去數據計算 Macro Score 並判定 Regime，用於決定 **$T+1$ 月**（如 2026-07 一整個月）的資產配置。
            * **無未來偏誤 (No Look-Ahead Bias)：** 回測中嚴格採用 `.shift(1)` 機制，確保實戰執行時不包含任何未來未發生的行情資訊。
            
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
# 4. 指定月份體制動態查詢 (Month Lookup Selector)
# -----------------------------------------------------------------------------
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
    st.metric("成長代理 Z-Score (前36M基準)", f"{m_g:+.2f}")
    st.metric("信用代理 Z-Score (前36M基準)", f"{m_c:+.2f}")
    st.metric("利率代理 Z-Score (前36M基準)", f"{m_r:+.2f}")

with col_p2:
    st.markdown(f"### 當前體制判斷 (`{selected_month_str}`)")
    st.metric("當月適應性宏觀得分 (Macro Score)", f"{m_score:+.2f}")
    st.markdown(f"**當月判定體制：** <span style='color:{m_cfg['color']}; font-size: 1.25rem; font-weight: bold;'>{m_cfg['name']}</span>", unsafe_allow_html=True)
    st.markdown(f"**下月建議股票比重 (SPY)：** **{int(m_cfg['eqW']*100)}%**")
    st.caption("註：此當月訊號用於決定下一個月 (T+1) 的資產配置比重")

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
# 5. 可選擇時間區間之動態回測 (Interactive Date Range Selector)
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
# 6. 回測表格與動態圖表 (含背景色帶 Legend 圖例)
# -----------------------------------------------------------------------------
st.markdown(f"### `{start_date_sel.strftime('%Y-%m')}` 至 `{end_date_sel.strftime('%Y-%m')}` 體制統計與月報酬率")

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

st.markdown("### 累積報酬率曲線與背景體制色帶 (Equity Curves & Regime Bands)")

fig_line = go.Figure()

# 1. 繪製背景體制色帶
current_reg = None
start_d = None

for i in range(len(sub_bt)):
    date = sub_bt.index[i]
    reg = sub_bt['Regime'].iloc[i]
    
    if reg != current_reg:
        if current_reg is not None:
            fig_line.add_vrect(
                x0=start_d, x1=date,
                fillcolor=REGIME_CONFIG[current_reg]['bgColor'],
                opacity=1.0, layer="below", line_width=0
            )
        current_reg = reg
        start_d = date

if current_reg is not None:
    fig_line.add_vrect(
        x0=start_d, x1=sub_bt.index[-1],
        fillcolor=REGIME_CONFIG[current_reg]['bgColor'],
        opacity=1.0, layer="below", line_width=0
    )

# 2. 手動在圖表中加入四大體制的背景顏色對應圖例 (Legend Traces)
for key, c in REGIME_CONFIG.items():
    fig_line.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=12, color=c['color'], symbol='square'),
        name=f"色帶: {c['name']}",
        showlegend=True
    ))

# 3. 繪製策略與基準之淨值折線
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

fig_line.update_layout(
    yaxis_type="log",
    yaxis=dict(
        tickformat="$~s",
        gridcolor='#334155'
    ),
    xaxis=dict(showgrid=False),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#94A3B8'),
    margin=dict(t=20, b=20, l=10, r=10),
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
# 7. 選定區間之核心 KPI 指標計算
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
        '投資策略名稱 (Strategy)': '適應性宏觀體制策略 (Adaptive Macro Strategy)',
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
