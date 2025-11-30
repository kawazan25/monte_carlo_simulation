import streamlit as st
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from datetime import datetime
import utils

# キャッシュをクリアして実行
st.cache_data.clear()

#######################################################################################################################
# -------------------------
# --- 月次データに対する分布当てはめ ---
# -------------------------
st.title("リターン分布とモンテカルロシミュレーション")
st.subheader("月次データへの分布当てはめ")
st.markdown("""
- 月次ヒストリカルデータを対数チャート化し、その変化率を算出。  
- 月次の対数変化率の分布に当てはまりのよい分布を観察。 
- 正規分布よりも、Fat-tailに対応した「スキュー付き正規分布」が過去の分布をよく表している。
""")

# ティッカー選択
ticker_choice = st.selectbox("ティッカーを選択してください。またはcustomにして希望の銘柄を入力してください。(Yahoo! Finance登録銘柄)", ["VOO", "QQQ", "VT", "QLD", "custom"])
if ticker_choice == "custom":
    ticker = st.text_input("カスタムティッカーを入力してください（例: AAPL, TSLAなど）", value="AAPL")
else:
    ticker = ticker_choice

# 日付選択
current_year = datetime.now().year
current_month = datetime.now().month
years = list(range(1999, current_year + 1))
months = list(range(1, 13))

col1, col2 = st.columns(2)
with col1:
    year = st.selectbox("開始年", years, index=years.index(2009) if 2009 in years else 0)
with col2:
    month = st.selectbox("開始月", months, index=8)
start_date = f"{year}-{month:02d}-01" # フォーマットを整える (YYYY-MM-01)

col3, col4 = st.columns(2)
with col3:
    end_year = st.selectbox("終了年", years, index=years.index(current_year))
with col4:
    end_month = st.selectbox("終了月", months, index=current_month - 1)  # デフォルト今月
end_date = f"{end_year}-{end_month:02d}-01" # フォーマットを整える (YYYY-MM-01)

st.write(f"選択されたティッカー: **{ticker}**")
st.write(f"期間: **{start_date} 〜 {end_date or '現在'}**")

# Streamlitに描画するスペースを確保
chart_placeholder = st.empty()

# -------------------------
# --- データ取得・統計量 ---
# -------------------------
# 月次データ取得
df_monthly = utils.load_monthly_data(ticker, start_date, end_date)
if df_monthly.empty:
    st.error(f"ティッカー `{ticker}` のデータが取得できませんでした。入力を確認してください。")
    st.stop()  # ここで処理を中断（以降は実行されない）
# -------------------------
# --- 対数リターンヒストグラム ---
# -------------------------
skew_params, fig, summary_table = utils.fit_distribution(df_monthly, ticker)

# Streamlit に描画（古いグラフは置き換え）
chart_placeholder.plotly_chart(fig, use_container_width=True, clear_figure=True)

st.markdown("**統計量サマリー(正規分布 vs スキュー付き正規分布)**")
st.table(summary_table)

# --- 補足説明 ---
st.markdown("""
**補足説明:**  
- 統計量の表示は、イメージしやすいように期待リターンのみ対数チャートから通常チャートへのリターン換算をしています。  
- 月次のVaR/CVaRは省略、年次のみ計算しています。
- VaRは20回に1回(5%)の確率でこの割合以上下落することがあることを示しています。
- CVaRはその時の平均下落率を示しています。                      
- 以降のモンテカルロシミュレーション等の計算はすべて対数リターンベースで行います。（計算の簡易さの都合であり、通常リターンに換算する結果と同じ）
""")


# -------------------------
# --- モンテカルロシミュレーション対数株価 ---
# -------------------------
log_price_paths = utils.monte_carlo_simulation_log(df_monthly, skew_params, n_sims=5000)
# パーセンタイル（対数価格）
percentiles_log = np.percentile(log_price_paths, [2.5, 50, 97.5], axis=0)
# 実際の対数株価
actual_log_prices = df_monthly['Log_Close'].values
dates = df_monthly.index

# --- グラフ描画 ---
fig2 = go.Figure()
# シミュレーション（2.5%・50%・97.5%ライン）
fig2.add_trace(go.Scatter(
    x=dates, y=percentiles_log[0], mode='lines',
    name="シミュレーション下限 (2.5%)", line=dict(color='red', dash='dot')
))
fig2.add_trace(go.Scatter(
    x=dates, y=percentiles_log[2], mode='lines',
    name="シミュレーション上限 (97.5%)",
    fill="tonexty", fillcolor="rgba(173,216,230,0.2)",
    line=dict(color='green', dash='dot')
))
# 実際の対数株価
fig2.add_trace(go.Scatter(
    x=dates, y=actual_log_prices, mode='lines+markers',
    name="実際の対数株価", line=dict(color='black', width=2)
)) 
fig2.add_trace(go.Scatter(
    x=dates, y=percentiles_log[1], mode='lines',
    name="シミュレーション中央値 (50%)", line=dict(color='blue', width=2)
))

fig2.update_layout(
    #title_text=f"{ticker} の対数チャート<br>&モンテカルロシミュレーション<br>（スキュー付き正規分布）",
    xaxis_title="日付",
    yaxis_title="対数チャート",
    template="plotly_white",
    height=500
)
fig2.update_layout(
    title=dict(
        text=f"{ticker} の対数チャート<br>&モンテカルロシミュレーション<br>（スキュー付き正規分布）",
        x=0.5,   # 中央揃え
        xanchor='center',
        y=0.90,   # 上から少し下げる（デフォルトは1.0）
        yanchor='top'
    ),
    legend=dict(
        orientation="h",  # 横並び
        yanchor="bottom",
        y=1.03,
        xanchor="center",
        x=0.5
    ),
    margin=dict(t=200)  # 上の余白をpxで指定
)

# ---- グラフ用コンテナ（表示位置を固定） ----
graph_container = st.container()

#１回分のシミュレーション結果を追加描画
if st.button("シミュレーション例描画"):
    one_path = utils.monte_carlo_simulation_log(df_monthly, skew_params, n_sims=1)
    one_path = one_path[0]
    fig2.add_trace(go.Scatter(
        x=dates, y=one_path, mode="lines",
        name="シミュレーション1例",
        line=dict(color="red", width=1)
    ))

# ---- グラフ描画（ここが1回だけ）----
with graph_container:
    st.plotly_chart(fig2, use_container_width=True)
