from flask import Flask, render_template, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__, template_folder='.')

@app.route('/api/backtest')
def get_backtest_data():
    # 抓取真實金融市場數據
    tickers = ["SPY", "HYG", "IEF", "TLT", "BIL"]
    raw = yf.download(tickers, start="1990-01-01")['Close']
    
    # 處理 missing values 並轉為月度資料
    df = raw.ffill().bfill()
    monthly = df.resample('ME').last()
    
    # 1. 計算真實 6 個月代理數據 (6M Rolling Return)
    monthly['growth_proxy'] = monthly['SPY'].pct_change(6) # SPY 6M 報酬
    monthly['credit_proxy'] = (monthly['HYG'] / monthly['IEF']).pct_change(6) # HYG/IEF 相對強度
    monthly['rates_proxy'] = monthly['TLT'].pct_change(6) # TLT 6M 報酬
    
    # 月報酬率 (計算策略資產變化)
    monthly['spy_ret'] = monthly['SPY'].pct_change()
    monthly['bil_ret'] = monthly['BIL'].pct_change().fillna(0.001) # 無 BIL 數據時期以微小正報酬充當短債
    
    monthly = monthly.dropna()
    
    # 2. 計算 Adaptive Score 與 Regime 判定
    # 預設 Lookback 12 個月權重
    wG, wC, wR = 0.50, 0.30, 0.20
    monthly['macro_score'] = (monthly['growth_proxy']*wG + monthly['credit_proxy']*wC + monthly['rates_proxy']*wR) / (wG+wC+wR)
    monthly['macro_score'] = monthly['macro_score'].clip(-1.0, 1.0)
    
    def get_regime(score):
        if score > 0.20: return 'Expansion'
        elif score > 0.00: return 'Recovery'
        elif score > -0.20: return 'Slowdown'
        else: return 'Contraction'
        
    monthly['regime'] = monthly['macro_score'].apply(get_regime)
    
    # 3. 計算策略配置與資產淨值 (Strategy Equity Curve)
    weights = {'Expansion': 1.0, 'Recovery': 0.7, 'Slowdown': 0.4, 'Contraction': 0.1}
    monthly['eq_weight'] = monthly['regime'].map(weights)
    monthly['cash_weight'] = 1 - monthly['eq_weight']
    
    monthly['strat_ret'] = (monthly['spy_ret'] * monthly['eq_weight']) + (monthly['bil_ret'] * monthly['cash_weight'])
    
    monthly['spy_cum'] = (1 + monthly['spy_ret']).cumprod() * 100
    monthly['strat_cum'] = (1 + monthly['strat_ret']).cumprod() * 100
    
    # 4. 整理歷史資料回傳 JSON
    history = []
    for idx, row in monthly.iterrows():
        history.append({
            'date': idx.strftime('%Y-%m'),
            'regime': row['regime'],
            'spyRet': float(row['spy_ret']),
            'bilRet': float(row['bil_ret']),
            'stratRet': float(row['strat_ret']),
            'spyCum': float(row['spy_cum']),
            'stratCum': float(row['strat_cum']),
            'macroScore': float(row['macro_score']),
            'growthProxy': float(row['growth_proxy']),
            'creditProxy': float(row['credit_proxy']),
            'ratesProxy': float(row['rates_proxy'])
        })
        
    # 計算真實 KPI
    def calc_kpi(returns):
        ann_ret = (1 + returns.mean())**12 - 1
        ann_vol = returns.std() * np.sqrt(12)
        cum = (1 + returns).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        mdd = dd.min()
        sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
        return ann_ret, ann_vol, mdd, sharpe

    strat_cagr, strat_vol, strat_mdd, strat_sharpe = calc_kpi(monthly['strat_ret'])
    spy_cagr, spy_vol, spy_mdd, spy_sharpe = calc_kpi(monthly['spy_ret'])

    kpis = {
        'strat': {
            'cagr': f"{strat_cagr*100:+.2f}%",
            'vol': f"{strat_vol*100:.2f}%",
            'mdd': f"{strat_mdd*100:.2f}%",
            'sharpe': f"{strat_sharpe:.2f}"
        },
        'spy': {
            'cagr': f"{spy_cagr*100:+.2f}%",
            'vol': f"{spy_vol*100:.2f}%",
            'mdd': f"{spy_mdd*100:.2f}%",
            'sharpe': f"{spy_sharpe:.2f}"
        }
    }

    return jsonify({'history': history, 'kpis': kpis})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
