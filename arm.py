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
    div[data-testid="stMetricValue"] { font-size: 2rem; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

REGIME_CONFIG = {
    'Expansion':  {'minScore': 0.40, 'name': '擴張期 (Expansion)',  'color': '#10B981', 'eqW': 1.00, 'tltW': 0.00, 'cashW': 0.00},
    'Recovery':   {'minScore': 0.00, 'name': '復甦期 (Recovery)',   'color': '#3B82F6', 'eqW': 0.80, 'tltW': 0.00, 'cashW': 0.20},
    'Slowdown':   {'minScore':-0.40, 'name': '放緩期 (Slowdown)',   'color': '#F59E0B', 'eqW': 0.50, 'tltW': 0.30, 'cashW': 0.20},
    'Contraction':{'minScore':-3.00, 'name': '收縮期 (Contraction)', 'color': '#EF4444', 'eqW': 0.10, 'tltW': 0.60, 'cashW': 0.30}
}

# -----------------------------------------------------------------------------
# 2. 論文優化版：Z-Score 標準化 + 滾動 IC + 多資產避險引擎
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
    
    # 2. 關鍵優化：對因子進行 36 個月滾動 Z-Score 標準化 (Normalize to Z-scores)
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
        
    weights = rolling_ic.map(lambda x: max(x, 0) if pd.notnull(x) else 0)
    weight_sum = weights.sum(axis=1).replace(0, 1)
    weights = weights.div(weight_sum, axis=0)
    
    # 加權合成 Adaptive Macro Score
    macro_score = (z_signals * weights).sum(axis=1).dropna()
    
    # 4. 體制劃分與部位對應
    def get_regime_info(s):
        if s > REGIME_CONFIG['Expansion']['minScore']: return 'Expansion'
        elif s > REGIME_CONFIG['Recovery']['minScore']: return 'Recovery'
        elif s > REGIME_CONFIG['Slowdown']['minScore']: return 'Slowdown'
        else: return 'Contraction'
        
    regimes = macro_score.map(get_regime_info)
    
    # 位移一期 (Shift 1) 避免未來資訊偏誤
    alloc_eq = regimes.map(lambda r: REGIME_CONFIG[r]['eqW']).shift(1)
    alloc_tlt = regimes.map(lambda r: REGIME_CONFIG[r]['tltW']).shift(1)
    alloc_cash = regimes.map(lambda r: REGIME_CONFIG[r]['cashW']).shift(1)
    
    valid_idx = alloc_eq.dropna().index
    ret_eq = returns['Equity'].loc[valid_idx]
    ret_tlt = returns['LongTreasury'].loc[valid_idx]
    ret_cash = returns['Cash'].loc[valid_idx]
    
    # 計算策略複合月報酬率
    strat_ret = (alloc_eq.loc[valid_idx] * ret_eq) + \
                (alloc_tlt.loc[valid_idx] * ret_tlt) + \
                (alloc_cash.loc[valid_idx] * ret_cash)
                
    backtest_df = pd.DataFrame({
        'Regime': regimes.loc[valid_idx],
        'SPY_Ret': ret_eq,
        'Cash_Ret': ret_cash,
        'Strat_Ret': strat_ret,
        'SPY_Cum': (1 + ret_eq).cumprod() * 100,
        'Strat_Cum': (1 + strat_ret).cumprod() * 100
    }, index=valid_idx)
    
    latest_date = macro_score.index[-1].strftime('%Y-%m')
    latest_score = macro_score.iloc[-1]
    latest_growth = z_signals['Growth'].iloc[-1]
    latest_credit = z_signals['Credit'].iloc[-1]
    latest_rates = z_signals['Rates'].iloc[-1]
    
    return backtest_df, latest_date, latest_score, latest_growth, latest_credit, latest_rates

# 執行載入
try:
    with st.spinner("正在執行優化版真實歷史數據回測..."):
        df_bt, latest_date, latest_score, latest_g, latest_c, latest_r = load_optimized_backtest()
except Exception as e:
    st.error(f"數據下載或回測失敗，細節: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 3. Header & Modal 按鈕
# -----------------------------------------------------------------------------
col_header, col_btn1, col_btn2 = st.columns([2.5, 1, 1])

with col_header:
    st.title("Adaptive Macro Regimes")
    st.caption("Inspired by Jim Masturzo (Syzygy Asset Management / Research Affiliates) | Z-Score 優化版")

with col_btn1:
    if st.button("📄 論文出處與數據說明"):
        @st.dialog("論文出處與金融代理數據計算說明")
        def show_paper_info():
            st.markdown("""
            **📄 參考學術論文:**
            * *Adaptive Macro Regimes for Dynamic Equity Allocation* (Jim Masturzo, Omid Shakernia, Alex Pickard)
            * Published in *The Journal of Portfolio Management*
            
            ---
            **⚙️ 本版優化核心機制:**
            1. **36M 滾動 Z-Score 標準化：** 將代理因子進行標準化，避免指標尺度差異造成體制判定偏誤。
            2. **TLT 長債避險：** 在放緩與收縮期配置長債，捕捉降息週期的資本利得。
            """)
        show_paper_info()

with col_btn2:
    if st.button("⚡ Adaptive Strategy 策略說明"):
        @st.dialog("Adaptive Strategy 策略標的與買賣/調倉機制說明")
        def show_strat_info():
            st.markdown("""
            **📊 優化版資產配置矩陣:**
            * **Expansion (> +0.40):** 100% SPY
            * **Recovery (0.00 ~ +0.40):** 80% SPY / 20% Cash
            * **Slowdown (-0.40 ~ 0.00):** 50% SPY / 30% TLT / 20% Cash
            * **Contraction (< -0.40):** 10% SPY / 60% TLT / 30% Cash
            """)
        show_strat_info()

st.divider()

# -----------------------------------------------------------------------------
# 4. 當前體制面板
# -----------------------------------------------------------------------------
col_p1, col_p2, col_p3 = st.columns([1.2, 1.2, 1])

with col_p1:
    st.markdown(f"### 當月標準化代理數據 (`{latest_date}`)")
    st.metric("成長代理 Z-Score", f"{latest_g:+.2f}")
    st.metric("信用代理 Z-Score", f"{latest_c:+.2f}")
    st.metric("利率代理 Z-Score", f"{latest_r:+.2f}")

with col_p2:
    st.markdown("### 當前體制判斷 (Real-time)")
    if latest_score > REGIME_CONFIG['Expansion']['minScore']: current_rKey = 'Expansion'
    elif latest_score > REGIME_CONFIG['Recovery']['minScore']: current_rKey = 'Recovery'
    elif latest_score > REGIME_CONFIG['Slowdown']['minScore']: current_rKey = 'Slowdown'
    else: current_rKey = 'Contraction'
    
    cfg = REGIME_CONFIG[current_rKey]
    
    st.metric("當前適應性宏觀得分 (Macro Score)", f"{latest_score:+.2f}")
    st.markdown(f"**當前體制：** <span style='color:{cfg['color']}; font-size: 1.25rem; font-weight: bold;'>{cfg['name']}</span>", unsafe_allow_html=True)
    st.markdown(f"**建議股票比重 (SPY)：** **{int(cfg['eqW']*100)}%**")

with col_p3:
    st.markdown("### 資產配置比重")
    fig_donut = go.Figure(data=[go.Pie(
        labels=['股票 (SPY)', '長債 (TLT)', '現金 (BIL)'],
        values=[cfg['eqW']*100, cfg['tltW']*100, cfg['cashW']*100],
        hole=.6,
        marker_colors=[cfg['color'], '#6366F1', '#334155']
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
# 5. 回測表格與圖表
# -----------------------------------------------------------------------------
st.markdown("## 歷史體制回測與資產月報酬率 (優化版真實歷史數據)")

stats_list = []
total_m = len(df_bt)

for key, c in REGIME_CONFIG.items():
    sub = df_bt[df_bt['Regime'] == key]
    cnt = len(sub)
    pct = (cnt / total_m) * 100 if total_m > 0 else 0
    avg_spy = sub['SPY_Ret'].mean() * 100 if cnt > 0 else 0
    avg_cash = sub['Cash_Ret'].mean() * 100 if cnt > 0 else 0
    avg_strat = sub['Strat_Ret'].mean() * 100 if cnt > 0 else 0
    
    stats_list.append({
        '宏觀體制 (Regime)': c['name'],
        '歷史月份數 (佔比)': f"{cnt} 個月 ({pct:.1f}%)",
        '股票 (SPY) 月報酬': f"{avg_spy:+.2f}%",
        '現金/短債 (BIL) 月報酬': f"{avg_cash:+.2f}%",
        '適應性策略 (Strategy) 月報酬': f"{avg_strat:+.2f}%"
    })

st.table(pd.DataFrame(stats_list))

st.markdown("### 歷史資產累積報酬率曲線 (Cumulative Equity Curves)")

fig_line = go.Figure()
fig_line.add_trace(go.Scatter(
    x=df_bt.index, y=df_bt['Strat_Cum'],
    mode='lines', name='適應性宏觀體制策略 (Adaptive Strategy)',
    line=dict(color='#10B981', width=2)
))
fig_line.add_trace(go.Scatter(
    x=df_bt.index, y=df_bt['SPY_Cum'],
    mode='lines', name='買入持有 S&P 500 (Buy & Hold SPY)',
    line=dict(color='#94A3B8', width=1.5, dash='dash')
))
fig_line.update_layout(
    yaxis_type="log",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#94A3B8'),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor='#334155'),
    margin=dict(t=20, b=20, l=10, r=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig_line, use_container_width=True)

st.markdown("### 策略核心績效指標對比表 (KPI Metrics)")

def calc_kpis(ret_series):
    cum = (1 + ret_series).cumprod()
    n_years = len(ret_series) / 12.0
    cagr = (cum.iloc[-1]) ** (1 / n_years) - 1
    vol = ret_series.std() * np.sqrt(12)
    sharpe = cagr / vol if vol != 0 else 0
    dd = cum / cum.cummax() - 1
    mdd = dd.min()
    return cagr*100, vol*100, mdd*100, sharpe

s_cagr, s_vol, s_mdd, s_sharpe = calc_kpis(df_bt['Strat_Ret'])
b_cagr, b_vol, b_mdd, b_sharpe = calc_kpis(df_bt['SPY_Ret'])

kpi_data = [
    {
        '投資策略名稱 (Strategy)': '適應性宏觀體制策略 (Adaptive Macro Strategy)',
        '年化報酬率 (CAGR)': f"+{s_cagr:.2f}%",
        '年化波動度 (Volatility)': f"{s_vol:.2f}%",
        '最大回撤 (Max Drawdown)': f"{s_mdd:.2f}%",
        '夏普比率 (Sharpe Ratio)': f"{s_sharpe:.2f}"
    },
    {
        '投資策略名稱 (Strategy)': '買入持有 S&P 500 (Buy & Hold SPY)',
        '年化報酬率 (CAGR)': f"+{b_cagr:.2f}%",
        '年化波動度 (Volatility)': f"{b_vol:.2f}%",
        '最大回撤 (Max Drawdown)': f"{b_mdd:.2f}%",
        '夏普比率 (Sharpe Ratio)': f"{b_sharpe:.2f}"
    }
]

st.table(pd.DataFrame(kpi_data))
