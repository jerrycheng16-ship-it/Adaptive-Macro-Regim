import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

# -----------------------------------------------------------------------------
# 1. 頁面基本配置 (Dark Theme Layout)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Adaptive Macro Regimes Terminal",
    page_icon="📈",
    layout="wide"
)

# 套用 CSS 樣式以貼近原先 HTML 的深色精緻質感
st.markdown("""
<style>
    .stApp { background-color: #0F172A; color: #F8FAFC; }
    .css-1r6slb0, .stCard { background-color: #1E293B; border: 1px solid #334155; border-radius: 0.75rem; padding: 1.25rem; }
    .metric-box { background-color: #0F172A; border: 1px solid #1E293B; border-radius: 0.5rem; text-align: center; padding: 1rem; }
    div[data-testid="stMetricValue"] { font-size: 2.25rem; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

# 體制設定矩陣
REGIME_CONFIG = {
    'Expansion': {'minScore': 0.20, 'name': '擴張期 (Expansion)', 'color': '#10B981', 'eqW': 1.00, 'cashW': 0.00},
    'Recovery':  {'minScore': 0.00, 'name': '復甦期 (Recovery)',  'color': '#3B82F6', 'eqW': 0.70, 'cashW': 0.30},
    'Slowdown':  {'minScore': -0.20,'name': '放緩期 (Slowdown)',  'color': '#F59E0B', 'eqW': 0.40, 'cashW': 0.60},
    'Contraction':{'minScore':-1.00,'name': '收縮期 (Contraction)', 'color': '#EF4444', 'eqW': 0.10, 'cashW': 0.90}
}

# -----------------------------------------------------------------------------
# 2. 自動抓取真實行情與回測計算 (yfinance + 24H Cache)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=86400)
def load_real_data_and_backtest():
    tickers = {
        'Equity': 'VFINX',      # S&P 500 (SPY 代理)
        'MidTreasury': 'VFITX',  # 中天期國債 (IEF 代理)
        'LongTreasury': 'VUSTX', # 長天期國債 (TLT 代理)
        'HighYield': 'VWEHX',   # 高收益債 (HYG 代理)
        'Cash': 'VFISX'         # 短期國債 (BIL 代理)
    }
    
    # 抓取自 1990 年迄今真實歷史數據
    df = yf.download(list(tickers.values()), start="1990-01-01")['Adj Close']
    df = df.rename(columns={v: k for k, v in tickers.items()}).dropna()
    
    monthly_data = df.resample('ME').last()
    returns = monthly_data.pct_change().dropna()
    
    # 代理因子計算 (6M Momentum)
    sig_growth = monthly_data['Equity'].pct_change(6)
    sig_credit = (monthly_data['HighYield'] / monthly_data['MidTreasury']).pct_change(6) # HYG/IEF
    sig_rates = monthly_data['LongTreasury'].pct_change(6)
    
    signals = pd.DataFrame({'Growth': sig_growth, 'Credit': sig_credit, 'Rates': sig_rates}).dropna()
    
    # 12 個月滾動 IC 算算動態適應性權重
    fwd_ret = returns['Equity'].shift(-1)
    rolling_ic = pd.DataFrame(index=signals.index)
    for col in signals.columns:
        rolling_ic[col] = signals[col].rolling(12).corr(fwd_ret)
        
    weights = rolling_ic.applymap(lambda x: max(x, 0) if pd.notnull(x) else 0)
    weight_sum = weights.sum(axis=1).replace(0, 1)
    weights = weights.div(weight_sum, axis=0)
    
    macro_score = (signals * weights).sum(axis=1).dropna()
    
    def get_regime_info(s):
        if s > 0.20: return 'Expansion', 1.00
        elif s > 0.00: return 'Recovery', 0.70
        elif s > -0.20: return 'Slowdown', 0.40
        else: return 'Contraction', 0.10
        
    regimes = macro_score.map(lambda s: get_regime_info(s)[0])
    eq_weights = macro_score.map(lambda s: get_regime_info(s)[1]).shift(1) # 無未來資訊偏誤
    
    valid_idx = eq_weights.dropna().index
    ret_eq = returns['Equity'].loc[valid_idx]
    ret_cash = returns['Cash'].loc[valid_idx]
    w = eq_weights.loc[valid_idx]
    
    strat_ret = w * ret_eq + (1 - w) * ret_cash
    
    backtest_df = pd.DataFrame({
        'Regime': regimes.loc[valid_idx],
        'SPY_Ret': ret_eq,
        'Cash_Ret': ret_cash,
        'Strat_Ret': strat_ret,
        'SPY_Cum': (1 + ret_eq).cumprod() * 100,
        'Strat_Cum': (1 + strat_ret).cumprod() * 100
    }, index=valid_idx)
    
    # 取得最新一期代理因子與得分
    latest_date = macro_score.index[-1].strftime('%Y-%m')
    latest_score = macro_score.iloc[-1]
    latest_growth = signals['Growth'].iloc[-1]
    latest_credit = signals['Credit'].iloc[-1]
    latest_rates = signals['Rates'].iloc[-1]
    
    return backtest_df, latest_date, latest_score, latest_growth, latest_credit, latest_rates

# 載入數據
with st.spinner("正在連線下載真實市場歷史數據並執行適應性回測..."):
    df_bt, latest_date, latest_score, latest_g, latest_c, latest_r = load_real_data_and_backtest()

# -----------------------------------------------------------------------------
# 3. 頁面 Header & Modal 彈出視窗對齊 HTML 版本
# -----------------------------------------------------------------------------
col_header, col_btn1, col_btn2 = st.columns([2.5, 1, 1])

with col_header:
    st.title("Adaptive Macro Regimes")
    st.caption("Inspired by Jim Masturzo (Syzygy Asset Management / Research Affiliates) | 連線真實行情自動更新")

with col_btn1:
    if st.button("📄 論文出處與數據說明"):
        @st.dialog("論文出處與金融代理數據計算說明")
        def show_paper_info():
            st.markdown("""
            **📄 參考學術論文 (Reference Paper):**
            * *Adaptive Macro Regimes for Dynamic Equity Allocation* (Jim Masturzo, Omid Shakernia, Alex Pickard)
            * Published in *The Journal of Portfolio Management*
            
            ---
            **💡 為什麼採用「金融市場價格」取代「經濟數據」？**
            * **發布延遲 (Publication Lag)：** 官方數據 (GDP, CPI) 通常延遲 1~2 個月，發布時市場早已計價。
            * **經常性修正 (Data Revisions)：** 初值常大幅修改，不適合實務即時交易。
            * 本系統採用無延遲的高頻市場價格作為代理指標。
            
            ---
            **⚙️ 三個代理數據計算邏輯:**
            1. **經濟成長 (Growth):** S&P 500 (VFINX/SPY) 過去 6 個月報酬率。
            2. **信用利差 (Credit Spread):** 高收益債 (VWEHX/HYG) 對中天期國債 (VFITX/IEF) 之相對強度 6M 變化。
            3. **利率趨勢 (Rates):** 20年期美債 (VUSTX/TLT) 過去 6 個月價格動能。
            """)
        show_paper_info()

with col_btn2:
    if st.button("⚡ Adaptive Strategy 策略說明"):
        @st.dialog("Adaptive Strategy 策略標的與買賣/調倉機制說明")
        def show_strat_info():
            st.markdown("""
            **🎯 交易標的 (Trading Universe):**
            * **Risk-On (股票):** S&P 500 ETF (SPY)
            * **Risk-Off (現金/短債):** 1-3M 短債/現金 (BIL)
            
            **⏱️ 調倉頻率 (Rebalancing):**
            * 每月最後一個交易日計算最新 Macro Score，於下個月首個交易日調整部位。
            
            ---
            **📊 買賣與部位劃分矩陣:**
            * **Expansion (> +0.20):** 100% SPY / 0% Cash
            * **Recovery (0.00 ~ +0.20):** 70% SPY / 30% Cash
            * **Slowdown (-0.20 ~ 0.00):** 40% SPY / 60% Cash
            * **Contraction (< -0.20):** 10% SPY / 90% Cash
            """)
        show_strat_info()

st.divider()

# -----------------------------------------------------------------------------
# 4. 當前最新一期數據與體制判斷面板 (Top Section)
# -----------------------------------------------------------------------------
col_p1, col_p2, col_p3 = st.columns([1.2, 1.2, 1])

with col_p1:
    st.markdown(f"### 當月宏觀代理數據 (`{latest_date}`)")
    st.metric("經濟成長代理 (SPY 6M)", f"{latest_g*100:+.2f}%")
    st.metric("信用利差代理 (HYG/IEF 6M)", f"{latest_c*100:+.2f}%")
    st.metric("利率趨勢代理 (TLT 6M)", f"{latest_r*100:+.2f}%")

with col_p2:
    st.markdown("### 當前體制判斷 (Real-time)")
    
    # 計算當前 Regime
    if latest_score > 0.20: current_rKey = 'Expansion'
    elif latest_score > 0.00: current_rKey = 'Recovery'
    elif latest_score > -0.20: current_rKey = 'Slowdown'
    else: current_rKey = 'Contraction'
    
    cfg = REGIME_CONFIG[current_rKey]
    
    st.metric("當前適應性宏觀得分 (Macro Score)", f"{latest_score:+.2f}")
    st.markdown(f"**當前體制：** <span style='color:{cfg['color']}; font-size: 1.25rem; font-weight: bold;'>{cfg['name']}</span>", unsafe_allow_html=True)
    st.markdown(f"**建議股票比重 (SPY)：** **{int(cfg['eqW']*100)}%**")

with col_p3:
    st.markdown("### 資產配置比重")
    fig_donut = go.Figure(data=[go.Pie(
        labels=['股票 (SPY)', '現金 (BIL)'],
        values=[cfg['eqW']*100, cfg['cashW']*100],
        hole=.6,
        marker_colors=[cfg['color'], '#334155']
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
# 5. 歷史真實數據回測與月報酬統計 (Backtest & Performance Table)
# -----------------------------------------------------------------------------
st.markdown("## 歷史體制回測與資產月報酬率 (1991 - 2026 真實歷史數據)")

# 四大體制統計
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

# -----------------------------------------------------------------------------
# 6. 累積淨值曲線圖 (Plotly Line Chart)
# -----------------------------------------------------------------------------
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
    yaxis_type="log", # 使用對數坐標軸展示跨三十年複利
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#94A3B8'),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor='#334155'),
    margin=dict(t=20, b=20, l=10, r=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig_line, use_container_width=True)

# -----------------------------------------------------------------------------
# 7. 策略核心 KPI 對比表格 (CAGR, Vol, MDD, Sharpe)
# -----------------------------------------------------------------------------
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
