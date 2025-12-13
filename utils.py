import streamlit as st
import numpy as np
import yfinance as yf
from scipy.stats import norm, skewnorm
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd


# -------------------------
# --- データ取得・統計計算関数 ---
# -------------------------
# 月次データ取得と対数リターン・対数株価追加
def load_monthly_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, interval='1mo')#['Close', 'High', 'Low', 'Open', 'Volume']
    if df.empty:
        return df
    df['Log_Return'] = np.log(df['Close']).diff()
    df['Log_Close'] = np.log(df['Close'])
    df = df.dropna()
    return df

# 月次リターンの平均・標準偏差計算
def calculate_statistics(monthly_df):    
    monthly_mean_log = monthly_df['Log_Return'].mean()
    monthly_std_log = monthly_df['Log_Return'].std()
    return monthly_mean_log, monthly_std_log

# 月次ログリターン → 年率換算（ログ・通常リターン）
def annualize(mean_monthly_log, std_monthly_log):
    # ログリターンベースの換算
    mean_annual_log = mean_monthly_log * 12
    std_annual_log = std_monthly_log * np.sqrt(12)
    # 通常リターンに変換
    mean_annual_exp = np.exp(mean_annual_log) - 1
    return mean_annual_log, std_annual_log, mean_annual_exp

# VaR/CVaR計算 ---
def calculate_var_cvar(returns, alpha=0.05):
    var = np.percentile(returns, 100*alpha)
    cvar = returns[returns <= var].mean()
    return var, cvar

# 対数リターンにスキュー付き正規分布を当てはめたシミュレーション（対数価格スケール）
def monte_carlo_simulation_log(monthly_df, skew_params, n_sims=10000):
    T = len(monthly_df)  # 期間（月数）
    a, loc, scale = skew_params
    # シミュレーション（log return）
    simulated_returns = skewnorm.rvs(a, loc=loc, scale=scale, size=(n_sims, T))
    # 累積対数リターン
    cum_log_returns = np.cumsum(simulated_returns, axis=1)
    # 初期対数株価
    logS0 = np.log(float(monthly_df['Close'].iloc[0]))
    log_price_return = logS0 + cum_log_returns
    return log_price_return



def withdrawal_strategy(
    withdrawal, monthly_need, savings, max_savings, min_savings,
    option1_1="1-1-1",
    option1_2="1-2-1",
    option2_1="2-1-1",
    option2_2="2-2-1"
):
    """
    #(case1)取り崩し額が生活費を上回っていたら
    #  - (case1-1)貯金が上限に達していたら
    #    - (option1-1-1)余剰資金はすべて消費する（積極的に消費、消費優先）
    #    - (option1-1-2)生活費までを取り崩す（必要最低限の資産取り崩し、資産確保優先）
    #  - (case1-2)貯金が上限に達していなかったら
    #    - (option1-2-1)余剰金はすべて消費する（積極的に消費する、消費優先）
    #    - (option1-2-2)貯金最低額以上あれば、余剰金は消費する　貯金最低額以下ならば、余剰金は貯金する（貯金と消費のバランスを取る）
    #    - (option1-2-3)生活費を差し引いた余剰分を貯金に回す（現金貯金を手厚くする、貯金確保優先）
    #(case2)取り崩し額が生活費を下回っていたら
    #  - (case2-1)貯金から不足分を補えるなら
    #    - (option2-1-1)取り崩したうえで、不足分は別の手段で確保する/取り崩し額の範囲で生活する（貯金確保優先）
    #    - (option2-1-2)取り崩したうえで、貯金で不足分を補う（生活費確保優先）
    #    - (option2-1-3)取り崩しはせず、可能な限り貯金から補う　不足分は別の手段で確保する（資産確保優先）
    #  - (case2-2)貯金では不足分を補えないなら
    #    - (option2-2-1)取り崩したうえで、不足分は別の手段で確保する/取り崩し額の範囲で生活する（貯金確保優先）
    #    - (option2-2-2)取り崩しはせず、不足分は別の手段で確保する（資産確保優先）
    """
    # 戦略分岐（例）
    if withdrawal >= monthly_need:  # case1
        excess = withdrawal - monthly_need
        if savings >= max_savings:  # case1-1
            if option1_1 == "1-1-1":
                used = withdrawal
            elif option1_1 == "1-1-2":
                used = monthly_need
        else:  # case1-2
            if option1_2 == "1-2-1":
                used = withdrawal
            elif option1_2 == "1-2-2":
                # バランス型(1-2-2)
                if savings < min_savings:
                    to_savings = min(excess, max_savings - savings)
                    savings += to_savings
                    used = monthly_need + (excess - to_savings)
                else:
                    used = withdrawal
            else:
                excess = withdrawal - monthly_need
                if excess > 0:
                    savings += excess
                used = monthly_need
                #to_savings = min(excess, max_savings - savings)
                #savings += to_savings
                #used = monthly_need + (excess - to_savings)
    else:  # case2
        if savings >= monthly_need: #case2-1
            if option2_1 == "2-1-1":
                used = withdrawal
            elif option2_1 == "2-1-2":
                savings -= (monthly_need - withdrawal)
                used = monthly_need
            else:  # 2-1-3
                used = min(monthly_need, savings)
                savings -= used
        else: #case2-2
            if option2_2 == "2-2-1":
                used = withdrawal
            else:
                used = 0
    return used, savings


#月次データに対する分布当てはめ
def fit_distribution(df_monthly, ticker):
    # -------------------------
    # --- 対数リターンヒストグラム ---
    # -------------------------
    x_values = df_monthly['Log_Return'].values
    n_bin = 30
    fig = go.Figure()
    # --- 対数リターンのヒストグラム ---
    fig.add_trace(
        go.Histogram(x=x_values, nbinsx=n_bin, name="月次データのヒストグラム", marker_color='orange', histnorm='probability density'),
    )

    # --- 対数リターンの正規分布 ---
    # 統計量算出(月次)
    monthly_mean_log, monthly_std_log = calculate_statistics(df_monthly) #月次対数リターン、　対数リスク
    exp_return_monthly = np.exp(monthly_mean_log) - 1 #月次通常リターン　※リスクは簡易的に対数ベースのままとする
    # 統計量算出(年次変換)
    annual_mean_log, annual_std_log, annual_mean_exp = annualize(monthly_mean_log, monthly_std_log)#年次対数リターン、　対数リスク、通常リターン
    # 月次→年次VaR/CVaR
    annual_samples = np.random.choice(df_monthly['Log_Return'].values, size=(100000,12)).sum(axis=1)
    var_95, cvar_95 = calculate_var_cvar(annual_samples, alpha=0.05)
    x = np.linspace(x_values.min(), x_values.max(), 200)
    pdf = norm.pdf(x, loc=monthly_mean_log, scale=monthly_std_log)
    fig.add_trace(
        go.Scatter(x=x, y=pdf, mode='lines', name='正規分布と仮定', line=dict(color='red', width=2)),
    )

    # スキュー付き正規分布のパラメータを推定
    skew_params = skewnorm.fit(x_values)
    a, loc, scale = skew_params
    pdf_skew = skewnorm.pdf(x, *skew_params)
    # moments='mvsk'で平均(Mean)、分散(Variance)、歪度(Skewness)、尖度(Kurtosis)を返す
    model_mean_log, model_var_log, model_skew, _ = skewnorm.stats(a, loc=loc, scale=scale, moments='mvsk')
    model_std_log = np.sqrt(model_var_log)
    mode_estimate = x[np.argmax(pdf_skew)] #PDF の最大値の位置を取得（理論的ピーク）
    mode_exp = np.exp(mode_estimate) - 1                  # 月次通常リターンに変換
    mode_annual_log, _, mode_annual_exp = annualize(mode_estimate, model_std_log)  # 年次換算


    # モデル統計量：月次ログリターン → 年次換算（ログ・通常リターン）
    model_mean_annual_log, model_std_annual_log, model_mean_annual_exp = annualize(model_mean_log, model_std_log)
    # モデル統計量：年次VaR/CVaRを計算
    # 注意: skewnorm.rvsを使用してシミュレーションした結果からVaR/CVaRを計算します。
    # 標本サイズと期間（12ヶ月）を設定
    N_MC = 100000
    T_ANNUAL = 12
    # スキュー付き正規分布から年次リターンサンプルを生成
    annual_model_samples = skewnorm.rvs(a, loc=loc, scale=scale, size=(N_MC, T_ANNUAL)).sum(axis=1)
    model_var_95, model_cvar_95 = calculate_var_cvar(annual_model_samples, alpha=0.05)

    fig.add_trace(
        go.Scatter(x=x, y=pdf_skew, mode='lines', name=f'スキュー付き正規分布と仮定', line=dict(color='green', width=2, dash='dash')),
    )

    # レイアウト調整
    fig.update_layout(
        #title_text= ticker + "の月次対数チャートの変化率ヒストグラムと、当てはまりのよい分布を観察",
        xaxis_title="変化率（月次の対数チャートにおいて）",
        yaxis_title="確率密度",
        template="plotly_white",
        showlegend=True,
        height=400,
    )
    fig.update_layout(
        title=dict(
            text=ticker + "の月次対数リターンヒストグラム",
            x=0.5,   # 中央揃え
            xanchor='center',
            y=0.91,   # 上から少し下げる（デフォルトは1.0）
            yanchor='top'
        ),
        legend=dict(
            orientation="h",  # 横並び
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=120)  # 上の余白をpxで指定
    )
    # x軸を%表記に
    fig.update_xaxes(tickformat=".1%")

    
    # 統計量表示
    summary_table = pd.DataFrame({
        "指標": ["期待リターン", "最頻値", "リスク(標準偏差)", "歪度(Skewness)", "下振れリスク(VaR95%)", "平均損失(CVaR95%)"],
        
        "月次(正規)": [
            f"{exp_return_monthly*100:.2f}%", 
            "-",
            f"{monthly_std_log*100:.2f}%", 
            "0(定義)",
            "-", 
            "-"
        ],
        "年次(正規)": [
            f"{annual_mean_exp*100:.2f}%", 
            "-",
            f"{annual_std_log*100:.2f}%", 
            "-", 
            f"{var_95*100:.2f}%", 
            f"{cvar_95*100:.2f}%"
        ],
        
        "月次(スキュー付き)": [
            f"{(np.exp(model_mean_log)-1)*100:.2f}%", # モデル平均を通常リターンに変換
            f"{mode_exp*100:.2f}%",                     # 最頻値
            f"{model_std_log*100:.2f}%", 
            f"{model_skew:.2f}",
            "-", 
            "-"
        ],
        "年次(スキュー付き)": [
            f"{model_mean_annual_exp*100:.2f}%", 
            f"{mode_annual_exp*100:.2f}%",             # 年次最頻値
            f"{model_std_annual_log*100:.2f}%", 
            "-", 
            f"{model_var_95*100:.2f}%", 
            f"{model_cvar_95*100:.2f}%"
        ]
    })
    

    return skew_params, fig, summary_table

