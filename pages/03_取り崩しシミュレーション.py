import streamlit as st
import numpy as np
from scipy.stats import skewnorm
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
from plotly.subplots import make_subplots
from streamlit_js_eval import streamlit_js_eval
import os
import utils

#デバイス確認
def is_mobile_device(ua):
    ua = ua.lower()
    keywords = ["iphone", "android", "ipad", "mobile"]
    return any(k in ua for k in keywords)

#選択肢の見せ方を変える
def build_options(options, is_premium):
    """無料ユーザーなら2番目以降に鍵マークを付ける"""
    if is_premium:
        return options  # 有料ユーザーはそのまま
    else:
        # 無料ユーザー → 最初以外に鍵マーク付ける
        return [
            options[0]
        ] + [
            f"🔒 {opt}" for opt in options[1:]
        ]

#有料無料の分岐
def selectbox_with_lock(title, options_key, original_options, is_premium):
    """鍵付き selectbox の管理（無料はロック＆自動戻し）"""
    
    # 表示用オプション（鍵マーク付き）
    display_options = build_options(original_options, is_premium)

    # セッション初期化（widget が生成される前に）
    if options_key not in st.session_state:
        st.session_state[options_key] = display_options[0]

    # 表示(widget生成)
    selected = st.selectbox(title, display_options, key=options_key)

    # 無料ユーザーが鍵を選んだ場合
    if not is_premium and selected.startswith("🔒"):
        st.warning("この選択肢は有料です。利用するにはパスコードで認証してください。")
        st.warning("Note記事をご購入ください🙏")
    
    # 表示用の鍵マークを除去した「内部値」を返す
    clean_value = selected.replace("🔒 ", "")

    return clean_value.split(":")[0]


# 認証 UI
def premium_auth():
    st.markdown("🔐 有料版パスコード入力")

    if PREMIUM_PASS is None:
        st.warning("パスコードが設定されていません（管理者設定が必要です）")
        return

    # パスワード入力欄（自動補完オフ）
    password = st.text_input(
        "パスコードを入力してください",
        type="password",
        key="premium_pass",
        autocomplete="off",
        placeholder="パスコードを入力"
    )

    if st.button("認証する"):
        if password == PREMIUM_PASS:
            st.session_state["is_premium"] = True
            st.success("認証に成功しました！")
            st.rerun()
        else:
            st.error("パスコードが違います。もう一度お試しください。")


#######################################################################################################################
# キャッシュをクリアして実行
st.cache_data.clear()


#######################################################################################################################
# -------------------------
# --- 月次データに対する分布当てはめ ---
# -------------------------
st.title("取り崩しシミュレーション")
st.subheader("月次データへの分布当てはめ STEP.1")
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
a, loc, scale = skew_params

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


#######################################################################################################################
# -------------------------
# --- 資産取り崩しシミュレーション ---
# -------------------------
st.subheader("取り崩しシミュレーション STEP.2")
st.markdown("""
- モンテカルロシミュレーションを用いた資産取り崩しシミュレーションです。
- 単なる定率売却を続けるのではなく、資産の状況に応じて臨機応変に対応するような選択肢を含めました。
- インデックスのリターン、リスクはSTEP.1で算出したものが用いられます。 
- 積み立ては毎月一定、年初一括、時期により積み立て額を変更するなどカスタム可能です。
- 戦略を切り替えながら、自分の心理面とも相談してご参考ください。
- 戦略の選択肢に対しては是非ご意見をお寄せください。（アプリ実装の参考にさせていただきます）
""")

st.markdown("**基本設定**")
col1, col2 = st.columns(2)
with col1:
    initial_assets = st.number_input("初期投資資産（万円）", value=5000, step=100)
    initial_monthly_need = st.number_input("初期生活費（月額, 万円）", value=40, step=1)
with col2:
    initial_savings = st.number_input("初期現金貯金（万円）", value=1500, step=50)
    simulation_years = st.number_input("シミュレーション年数", value=30, step=1)

col1, col2 = st.columns(2)
with col1:
    inflation_rate = st.number_input("インフレ率（年率, %）", value=2.0, step=0.1)
with col2:
    adjust_need_for_inflation = st.checkbox("生活費をインフレ率に応じて増加させる", value=True)

st.markdown("**貯金の変動幅設定**")
col1, col2 = st.columns(2)
with col1:
    min_savings_ratio = st.number_input("貯金下限比率（資産に対して）[%]", value=10, step=1, min_value=0, max_value=100)
with col2:
    max_savings_ratio = st.number_input("貯金上限比率（資産に対して）[%]", value=30, step=1, min_value=0, max_value=100)
#st.caption("例：資産が1億円なら、貯金下限1000万円、上限3000万円。")

st.markdown("**取り崩し率設定**")
col1, col2 = st.columns(2)
with col1:
    withdrawal_rate = st.number_input("取り崩し率（月次, %）", value=1.7, step=0.1)

n_trials = 500 #試行回数（モンテカルロシミュレーション）

#戦略の選択
#資産に対する定率取り崩し額を計算する
#(case1)取り崩し額が生活費を上回っていたら
#  - (case1-1)貯金が上限に達していたら
#    - (option1-1-1)生活費までを取り崩す（必要最低限の資産取り崩し、資産確保優先）
#    - (option1-1-2)余剰資金はすべて消費する（積極的に消費、消費優先）
#  - (case1-2)貯金が上限に達していなかったら
#    - (option1-2-1)生活費を差し引いた余剰分を貯金に回す（現金貯金を手厚くする、貯金確保優先）
#    - (option1-2-2)貯金最低額以上あれば、余剰金は消費する　貯金最低額以下ならば、余剰金は貯金する（貯金と消費のバランスを取る）
#    - (option1-2-3)余剰金はすべて消費する（積極的に消費する、消費優先）
#(case2)取り崩し額が生活費を下回っていたら
#  - (case2-1)貯金から不足分を補えるなら
#    - (option2-1-1)取り崩したうえで、貯金で不足分を補う（生活費確保優先）
#    - (option2-1-2)取り崩したうえで、不足分は別の手段で確保する/取り崩し額の範囲で生活する（貯金確保優先）
#    - (option2-1-3)取り崩しはせず、可能な限り貯金から補う　不足分は別の手段で確保する（資産確保優先）
#  - (case2-2)貯金では不足分を補えないなら
#    - (option2-2-1)取り崩したうえで、不足分は別の手段で確保する/取り崩し額の範囲で生活する（貯金確保優先）
#    - (option2-2-2)取り崩しはせず、不足分は別の手段で確保する（資産確保優先）
st.markdown("**取り崩し戦略設定**")
#有料無料分岐
#環境変数からパスコードを取得
PREMIUM_PASS = os.getenv("PREMIUM_PASS_CODE", None)
# Session 初期化
if "is_premium" not in st.session_state:
    st.session_state["is_premium"] = False
#認証（環境変数確認）
premium_auth()

#有料無料の状態取得
is_premium = st.session_state.get("is_premium", False)

#選択肢
option1_1_list = [
    "1-1-1: 生活費までを取り崩す（必要最低限の資産取り崩し、資産確保優先）",
    "1-1-2: 余剰資金はすべて消費する（積極的に消費、消費優先）"
]

option1_2_list = [
    "1-2-1: 生活費を差し引いた余剰分を貯金に回す（現金貯金を手厚くする、貯金確保優先）",
    "1-2-2: 貯金最低額以上あれば、余剰金は消費する　貯金最低額以下ならば、余剰金は貯金する（バランス型）",
    "1-2-3: 余剰金はすべて消費する（積極的な消費）"
]

option2_1_list = [
    "2-1-1: 取り崩したうえで、貯金で不足分を補う（生活費確保優先）",
    "2-1-2: 取り崩したうえで、不足分は別の手段で確保する（貯金確保優先）",
    "2-1-3: 取り崩しはせず、可能な限り貯金から補う（資産確保優先）"
]

option2_2_list = [
    "2-2-1: 取り崩したうえで、不足分は別の手段で確保する（貯金確保優先）",
    "2-2-2: 取り崩しはせず、不足分は別の手段で確保する（資産確保優先）"
]

# ---- 鍵付きセレクトボックス ----

selected_option1_1 = selectbox_with_lock(
    "① 定率取り崩し額が生活費を上回り かつ 貯金額が上限に達していたら？",
    "option1_1",
    option1_1_list,
    is_premium
)

selected_option1_2 = selectbox_with_lock(
    "② 定率取り崩し額が生活費を上回り かつ 貯金額が上限に届いていなければ？",
    "option1_2",
    option1_2_list,
    is_premium
)

selected_option2_1 = selectbox_with_lock(
    "③ 定率取り崩し額が生活費を下回り かつ 貯金で不足分を補えるなら？",
    "option2_1",
    option2_1_list,
    is_premium
)

selected_option2_2 = selectbox_with_lock(
    "④ 定率取り崩し額が生活費を下回り かつ 貯金で補えないなら？",
    "option2_2",
    option2_2_list,
    is_premium
)
print(selected_option1_1)
print(selected_option1_2)
print(selected_option2_1)
print(selected_option2_2)

st.markdown("**グラフ表示範囲設定**")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**① 総資産**")
    y_min_total = st.number_input("最小値（万円）", value=0, key="y_min_total")
    y_max_total = st.number_input("最大値（万円）", value=10000, key="y_max_total")

    st.markdown("**③ 貯金**")
    y_min_savings = st.number_input("最小値（万円）", value=0, key="y_min_savings")
    y_max_savings = st.number_input("最大値（万円）", value=5000, key="y_max_savings")

with col2:
    st.markdown("**② 株式資産**")
    y_min_assets = st.number_input("最小値（万円）", value=0, key="y_min_assets")
    y_max_assets = st.number_input("最大値（万円）", value=10000, key="y_max_assets")

    st.markdown("**④ 生活費と使用額**")
    y_min_used = st.number_input("最小値（万円）", value=0, key="y_min_used")
    y_max_used = st.number_input("最大値（万円）", value=100, key="y_max_used")


ua = streamlit_js_eval(js_expressions="navigator.userAgent", key="ua_step3")

if ua is None:
    st.info("User-Agent を取得中...（ページが自動で再描画されます）")
    st.stop()

# UA 取得成功
is_mobile = is_mobile_device(ua)
# -------------------------
# シミュレーション実行ボタン
# -------------------------
if st.button("▶ シミュレーション実行(STEP2)"):
    # チェック対象のキー一覧
    option_keys = ["option1_1", "option1_2", "option2_1", "option2_2"]
    for key in option_keys:
        raw_value = st.session_state.get(key, "")
        if "🔒" in raw_value:
            st.error("有料の選択肢が選択されています。認証しないと実行できません。")
            st.stop()

    n_months = simulation_years * 12
    monthly_need = initial_monthly_need

    results = []
    for sim in range(n_trials):
        assets = initial_assets
        savings = initial_savings
        need = monthly_need
        total = assets + savings
        
        r = skewnorm.rvs(a, loc=loc, scale=scale, size=n_months)
        for m in range(n_months):
            # ランダムリターン
            assets *= np.exp(r[m])

            withdrawal = assets * (withdrawal_rate / 100)

            min_s = total * (min_savings_ratio / 100)
            max_s = total * (max_savings_ratio / 100)

            used, savings = utils.withdrawal_strategy(
                withdrawal, need, savings, max_s, min_s,
                selected_option1_1,
                selected_option1_2,
                selected_option2_1,
                selected_option2_2
            )

            assets -= used
            total = assets + savings

            results.append([sim, m, assets, savings, total, need, used])

            # 翌月
            if adjust_need_for_inflation:
                need *= (1 + inflation_rate / 100 / 12)
            if total <= 0:
                break
    
    #"Sim": 試行インデックス, "Month": 月インデックス, "Assets": 株式資産, "Savings": 貯金, "Total": 資産総額, "Need": 必要生活費, "Used": 消費額
    df = pd.DataFrame(results, columns=["Sim", "Month", "Assets", "Savings", "Total", "Need", "Used"])
    
    if is_mobile:
        # スマホは縦4つ
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=["総資産", "株式資産", "貯金", "必要生活費と消費額"],
            vertical_spacing=0.09
        )
        layout_mode = "mobile"

    else:
        # PCは2×2
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["総資産", "株式資産", "貯金", "必要生活費と消費額"],
            vertical_spacing=0.15,
            horizontal_spacing=0.10,
        )
        layout_mode = "pc"

    # --- サブプロット位置（PC とスマホで変わる） ---
    pos_total    = (1, 1) if layout_mode == "pc" else (1, 1)  # 総資産
    pos_assets   = (1, 2) if layout_mode == "pc" else (2, 1)  # 株式資産
    pos_savings  = (2, 1) if layout_mode == "pc" else (3, 1)  # 貯金
    pos_usage    = (2, 2) if layout_mode == "pc" else (4, 1)  # 必要生活費＆消費額


    # --- 総資産 ---
    df_total = df.groupby(["Sim", "Month"])["Total"].sum().unstack().T
    median = df_total.median(axis=1)
    p5 = df_total.quantile(0.025, axis=1)
    p95 = df_total.quantile(0.975, axis=1)

    fig.add_trace(go.Scatter(x=df_total.index, y=median, name="総資産 中央値", line=dict(color="black")), row=pos_total[0], col=pos_total[1])
    fig.add_trace(go.Scatter(x=df_total.index, y=p5, name="2.5%tile", line=dict(color="gray", dash="dot")), row=pos_total[0], col=pos_total[1])
    fig.add_trace(go.Scatter(x=df_total.index, y=p95, name="97.5%tile", fill="tonexty", fillcolor="rgba(200,200,200,0.2)", line=dict(color="gray", dash="dot")), row=pos_total[0], col=pos_total[1])
    fig.update_yaxes(range=[y_min_total, y_max_total], row=pos_total[0], col=pos_total[1])

    

    # --- 株式資産 ---
    df_assets = df.groupby(["Sim", "Month"])["Assets"].sum().unstack().T
    median = df_assets.median(axis=1)
    p5 = df_assets.quantile(0.025, axis=1)
    p95 = df_assets.quantile(0.975, axis=1)
    fig.add_trace(go.Scatter(x=df_assets.index, y=median, name="株式資産 中央値", line=dict(color="blue")),row=pos_assets[0], col=pos_assets[1])
    fig.add_trace(go.Scatter(x=df_assets.index, y=p5, name="2.5%tile", line=dict(color="lightblue", dash="dot")),row=pos_assets[0], col=pos_assets[1])
    fig.add_trace(go.Scatter(x=df_assets.index, y=p95, name="97.5%tile", fill="tonexty", fillcolor="rgba(173,216,230,0.2)", line=dict(color="lightblue", dash="dot")),row=pos_assets[0], col=pos_assets[1])
    fig.update_yaxes(range=[y_min_assets, y_max_assets], row=pos_assets[0], col=pos_assets[1])

    # --- 貯金 ---
    df_savings = df.groupby(["Sim", "Month"])["Savings"].sum().unstack().T
    median = df_savings.median(axis=1)
    p5 = df_savings.quantile(0.025, axis=1)
    p95 = df_savings.quantile(0.975, axis=1)
    fig.add_trace(go.Scatter(x=df_savings.index, y=median, name="貯金 中央値", line=dict(color="orange")),row=pos_savings[0], col=pos_savings[1])
    fig.add_trace(go.Scatter(x=df_savings.index, y=p5, name="2.5%tile", line=dict(color="gold", dash="dot")),row=pos_savings[0], col=pos_savings[1])
    fig.add_trace(go.Scatter(x=df_savings.index, y=p95, name="97.5%tile", fill="tonexty", fillcolor="rgba(255,215,0,0.2)", line=dict(color="gold", dash="dot")),row=pos_savings[0], col=pos_savings[1])
    fig.update_yaxes(range=[y_min_savings, y_max_savings], row=pos_savings[0], col=pos_savings[1])

    # --- 必要生活費 & 消費額（同じグラフに描画） ---
    df_need = df.groupby(["Sim", "Month"])["Need"].sum().unstack().T
    df_used = df.groupby(["Sim", "Month"])["Used"].sum().unstack().T

    median_need = df_need.median(axis=1)
    median_used = df_used.median(axis=1)
    p5 = df_used.quantile(0.025, axis=1)
    p95 = df_used.quantile(0.975, axis=1)
    fig.add_trace(go.Scatter(x=median_need.index, y=median_need, name="必要生活費", line=dict(color="green")),row=pos_usage[0], col=pos_usage[1])
    fig.add_trace(go.Scatter(x=df_need.index, y=median_used, name="消費額 中央値", line=dict(color="red")),row=pos_usage[0], col=pos_usage[1])
    fig.add_trace(go.Scatter(x=df_used.index, y=p5, name="2.5%tile", line=dict(color="salmon", dash="dot")),row=pos_usage[0], col=pos_usage[1])
    fig.add_trace(go.Scatter(x=df_used.index, y=p95, name="97.5%tile", fill="tonexty", fillcolor="rgba(250,128,114,0.15)", line=dict(color="salmon", dash="dot")),row=pos_usage[0], col=pos_usage[1])
    fig.update_yaxes(range=[y_min_used, y_max_used], row=pos_usage[0], col=pos_usage[1])
    
    # --- レイアウト ---
    
    if is_mobile:
        fig.update_xaxes(title_text="経過月数", row=1, col=1)
        fig.update_yaxes(title_text="金額（万円）", row=1, col=1)

        fig.update_xaxes(title_text="経過月数", row=2, col=1)
        fig.update_yaxes(title_text="金額（万円）", row=2, col=1)

        fig.update_xaxes(title_text="経過月数", row=3, col=1)
        fig.update_yaxes(title_text="金額（万円）", row=3, col=1)

        fig.update_xaxes(title_text="経過月数", row=4, col=1)
        fig.update_yaxes(title_text="金額（万円）", row=4, col=1)
        fig.update_layout(
            xaxis_title="経過月数",
            yaxis_title="金額（万円）",
            height=1800,
            width=None,
            title=dict(
                text=f"モンテカルロシミュレーション結果",
                x=0.5,   # 中央揃え
                xanchor='center',
                y=0.95,   # 上から少し下げる（デフォルトは1.0）
                yanchor='top'
            ),
            legend=dict(
                orientation="h",  # 横並び
                yanchor="bottom",
                y=1.03,
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=300)  # 上の余白をpxで指定
        )
    else:
        fig.update_xaxes(title_text="経過月数", row=1, col=1)
        fig.update_yaxes(title_text="金額（万円）", row=1, col=1)

        fig.update_xaxes(title_text="経過月数", row=2, col=1)
        fig.update_yaxes(title_text="金額（万円）", row=2, col=1)

        fig.update_xaxes(title_text="経過月数", row=1, col=2)
        fig.update_yaxes(title_text="金額（万円）", row=1, col=2)

        fig.update_xaxes(title_text="経過月数", row=2, col=2)
        fig.update_yaxes(title_text="金額（万円）", row=2, col=2)
        fig.update_layout(
            xaxis_title="経過月数",
            yaxis_title="金額（万円）",
            height=900,
            width=1100,
            title=dict(
                text=f"モンテカルロシミュレーション結果",
                x=0.5,   # 中央揃え
                xanchor='center',
                y=0.95,   # 上から少し下げる（デフォルトは1.0）
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
    

    # グラフと実行時刻を保存
    st.session_state["fig_step3"] = fig
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.success("シミュレーションを実行しました。入力を変更したら再実行してください。")
    st.caption(f"実行時刻：{run_time}")

# -------------------------
# グラフ表示（過去の結果があれば再表示）
# -------------------------
if "fig_step3" in st.session_state:
    st.plotly_chart(st.session_state["fig_step3"], use_container_width=True)
