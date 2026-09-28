import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 頁面配置
st.set_page_config(page_title="Adaptive Macro Regimes & SP500", layout="wide")

st.title("📈 Adaptive Macro Regimes & S&P 500 自動回測與最新投資建議")
st.caption("自動抓取最新歷史數據 | 計算 Macro Regimes | 每月動態資產配置建議")

# 1. 自動抓取最新歷史數據 (修正後相容新版 yfinance)
@st.cache_data(ttl=86400) # 快取 24 小時
def load_data():
    tickers = ["^GSPC", "^TNX"]
    raw_data = yf.download(tickers, start="2010-01-01")['Close']
    
    # 重新命名欄位
    raw_data = raw_data.rename(columns={'^GSPC': 'SP500', '^TNX': 'TNX'})
    df = raw_data.dropna()
    
    # 計算月度資料 (ME 代表 Month End)
    monthly = df.resample('ME').last()
    monthly['SP500_Return'] = monthly['SP500'].pct_change()
    monthly['TNX_Change'] = monthly['TNX'].diff()
    
    # 劃分 4 個 Regime
    monthly['Growth_Signal'] = monthly['SP500_Return'] > 0
    monthly['Rate_Signal'] = monthly['TNX_Change'] > 0
    
    def assign_regime(row):
        if row['Growth_Signal'] and not row['Rate_Signal']:
            return 'Goldilocks (金髮女孩)'
        elif row['Growth_Signal'] and row['Rate_Signal']:
            return 'Reflation (通膨過熱)'
        elif not row['Growth_Signal'] and row['Rate_Signal']:
            return 'Stagflation (滯脹)'
        else:
            return 'Deflation/Recession (衰退)'

    monthly['Regime'] = monthly.apply(assign_regime, axis=1)
    monthly = monthly.dropna()
    return monthly

data = load_data()

# 2. 取得最新月底的 Regime 與投資建議
latest_month = data.index[-1].strftime('%Y-%m')
latest_regime = data['Regime'].iloc[-1]

st.subheader(f"🗓️ 最新總經狀態評估 ({latest_month})")

col1, col2 = st.columns(2)
with col1:
    st.metric(label="當前市場 Regime", value=latest_regime)

with col2:
    advice_map = {
        'Goldilocks (金髮女孩)': "💡 **投資建議**：加碼股票 (特別是科技股與成長股)、加碼信用債。適度減碼現金。",
        'Reflation (通膨過熱)': "💡 **投資建議**：加碼原物料、能源股、價值股與抗通膨債 (TIPS)；適度減碼長天期公債。",
        'Stagflation (滯脹)': "💡 **投資建議**：提高現金比例、加碼黃金與防禦型板塊；大幅降低高風險股票部位。",
        'Deflation/Recession (衰退)': "💡 **投資建議**：加碼長期美國公債 (TLT)、高品質公債與防禦型股票 (如必選消費、醫療)。"
    }
    st.info(advice_map.get(latest_regime, "保持觀望"))

# 3. 統計各個 Regime 的平均月報酬率
st.subheader("📊 歷史數據實測：各 Regime 下 S&P 500 平均月報酬率")
regime_stats = data.groupby('Regime')['SP500_Return'].agg(
    平均月報酬率=lambda x: f"{x.mean()*100:.2f}%",
    勝率=lambda x: f"{(x > 0).mean()*100:.1f}%",
    樣本月份數='count'
).reset_index()

st.dataframe(regime_stats, use_container_width=True)

# 4. 繪製累計資產曲線
st.subheader("📈 S&P 500 歷史累計報酬與走勢")
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=(1 + data['SP500_Return'].fillna(0)).cumprod(), name='S&P 500 累積淨值'))
fig.update_layout(template="plotly_dark", height=400, margin=dict(l=20, r=20, t=30, b=20))
st.plotly_chart(fig, use_container_width=True)
