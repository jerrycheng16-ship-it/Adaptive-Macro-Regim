@st.cache_data(ttl=86400) # 快取 24 小時
def load_data():
    # 一次下載 S&P 500 與 10年期美債殖利率 的 Close 欄位
    tickers = ["^GSPC", "^TNX"]
    raw_data = yf.download(tickers, start="2010-01-01")['Close']
    
    # 重新命名欄位
    raw_data = raw_data.rename(columns={'^GSPC': 'SP500', '^TNX': 'TNX'})
    
    # 刪除缺失值
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
    
    # 移除第一個月因為 pct_change / diff 產生的 NaN
    monthly = monthly.dropna()
    return monthly
