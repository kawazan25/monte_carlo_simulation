import streamlit as st
import numpy as np
from scipy.stats import skewnorm
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
import utils

# キャッシュをクリアして実行
st.cache_data.clear()

#######################################################################################################################
# -------------------------
# --- 月次データに対する分布当てはめ ---
# -------------------------
st.title("資産形成シミュレーション")
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
# --- モンテカルロシミュレーション資産形成シミュレーション ---
# -------------------------
st.subheader("資産形成シミュレーション STEP.2")
st.markdown("""
- ご自身の資産形成の可能性をシミュレーションで体験してみてください。  
- インデックスのリターン、リスクはSTEP.1で算出したものが用いられます。 
- 積み立ては毎月一定、年初一括、時期により積み立て額を変更するなどカスタム可能です。
- 資産形成戦略を考える材料にしてみてください。
""")
# -------------------------
# 投資期間設定
# -------------------------
investment_years = st.number_input("投資期間（年）", min_value=1, max_value=50, value=30, key="investment_years")
# -------------------------
# 開始年月
# -------------------------
years = list(range(1999, current_year + 50))  # 少し未来まで対応
months = list(range(1, 13))

col5, col6 = st.columns(2)
with col5:
    start_year = st.selectbox("開始年", years, index=years.index(current_year), key="start_year_step2")
with col6:
    start_month = st.selectbox("開始月", months, index=current_month-1, key="start_month_step2")

# -------------------------
# 終了年月（自動計算＋ユーザー変更可）
# -------------------------
end_year_auto = start_year + investment_years - 1
end_month_auto = start_month

col7, col8 = st.columns(2)
with col7:
    end_year = st.selectbox("終了年", years, index=years.index(end_year_auto), key="end_year_step2")
with col8:
    end_month = st.selectbox("終了月", months, index=end_month_auto-1, key="end_month_step2")

# 実際の投資期間を再計算
investment_years_actual = (end_year - start_year) + (1 if end_month >= start_month else 0)
st.write(f"実際の投資期間: **{investment_years_actual}年**")

# -------------------------
# 投資スケジュールテーブル
# -------------------------
st.markdown("**積立スケジュール設定**")
st.markdown("""
- 各行に対して期間・毎月積立額・年初一括額を設定できます  
- 「行を追加」で複数の積立パターンを入力可能
""")

if 'schedule' not in st.session_state:
    st.session_state.schedule = pd.DataFrame([{
        "開始年": start_year,
        "開始月": start_month,
        "終了年": start_year + 1,
        "終了月": start_month,
        "毎月積立額(万円)": 0,
        "年初一括額(1月)(万円)": 0
    }])

schedule_df_edited = st.data_editor(
    st.session_state.schedule,
    num_rows="dynamic",
    use_container_width=True,
)

# 空の場合は計算を止める
if st.session_state.schedule.empty:
    st.warning("積立スケジュールを入力してください。")
    st.stop()
st.markdown("※必要に応じて行を追加・削除して投資シナリオを自由に設定できます。")


# -------------------------
# シミュレーション設定入力
# -------------------------
# 総期間（月数）
n_months = investment_years_actual * 12

# 初期投資額
initial_investment = st.number_input(
    "初期投資額（万円）",
    min_value=0,
    value=100,
    key="initial_investment"
)

# 目標金額入力
target_amount = st.number_input(
    "目標資産額（万円）",
    min_value=0,
    value=1000  # デフォルト値
)
# -------------------------
# 表示範囲設定
# -------------------------
# 月ラベル
dates_sim = pd.date_range(start=f"{start_year}-{start_month:02d}-01", periods=n_months, freq='MS')
st.markdown("""グラフ表示範囲設定（任意）""")
col_x, col_y = st.columns(2)

with col_x:
    x_start = st.date_input("横軸開始年月", value=dates_sim[0], min_value=dates_sim[0], max_value=dates_sim[-1], key="start_time_step2")
    x_end   = st.date_input("横軸終了年月", value=dates_sim[-1], min_value=dates_sim[0], max_value=dates_sim[-1], key="end_time_step2")

with col_y:
    y_min = st.number_input("縦軸最小値（万円）", value=0, key="y_min_input_step2")
    y_max = st.number_input("縦軸最大値（万円）", value=10000, key="y_max_input_step2")


# -------------------------
# シミュレーションボタン
# -------------------------
if st.button("▶ シミュレーション実行(STEP2)"):
    df = schedule_df_edited.copy()

    #入力チェック
    required_cols = ["開始年", "開始月", "終了年", "終了月"]
    missing_rows = df[required_cols].isna().any(axis=1)

    if missing_rows.any():
        st.error("開始・終了の年/月が未入力の行があります。全て入力してください。")
        st.stop()

    # 数値列の欠損をデフォルト埋め
    df["毎月積立額(万円)"] = df["毎月積立額(万円)"].fillna(0)
    df["年初一括額(1月)(万円)"] = df["年初一括額(1月)(万円)"].fillna(0)

    # 型チェック（数値であるか）
    for col in ["開始年", "開始月", "終了年", "終了月", "毎月積立額(万円)", "年初一括額(1月)(万円)"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            st.error(f"列『{col}』に数値以外の入力があります。")
            st.stop()

    # ロジカルチェック（終了年月が開始年月より前になっていないか）
    invalid = (df["終了年"] * 12 + df["終了月"]) < (df["開始年"] * 12 + df["開始月"])
    if invalid.any():
        st.error("終了年月が開始年月より前の行があります。")
        st.stop()

    # ここまで通ればOK → デフォルト値補完後のクリーンデータを保存
    st.session_state.schedule = df
    # st.success("入力チェック完了。シミュレーションを開始します。")

    # 毎月積立額と年初一括額の配列作成
    monthly_contributions = np.zeros(n_months)
    for i, row in st.session_state.schedule.iterrows():
        # 月次積立
        for y in range(int(row["開始年"]), int(row["終了年"])+1):
            for m in range(1, 13):
                # 月インデックス計算
                month_idx = (y - start_year) * 12 + (m - start_month)
                if month_idx < 0 or month_idx >= n_months:
                    continue
                # 月次積立
                monthly_contributions[month_idx] += row["毎月積立額(万円)"]
        # 年初一括
        if row["年初一括額(1月)(万円)"] > 0:
            first_year = int(row["開始年"]) if int(row["開始月"]) == 1 else int(row["開始年"])+1
            for y in range(first_year, int(row["終了年"])+1):
                month_idx = (y - start_year) * 12  # 1月
                if 0 <= month_idx < n_months:
                    monthly_contributions[month_idx] += row["年初一括額(1月)(万円)"]

    # --- モンテカルロシミュレーション ---
    # シミュレーション回数
    n_sims = 5000

    # STEP.1で推定したスキュー付き正規分布パラメータ
    a, loc, scale = skew_params

    # 月次リターンシミュレーション
    simulated_log_returns = skewnorm.rvs(a, loc=loc, scale=scale, size=(n_sims, n_months))

    # 累積資産計算
    asset_paths = np.zeros_like(simulated_log_returns)
    asset_paths[:, 0] = initial_investment * 1e4 + monthly_contributions[0]*1e4  # 単位: 円
    for t in range(1, n_months):
        asset_paths[:, t] = asset_paths[:, t-1] * np.exp(simulated_log_returns[:, t]) + monthly_contributions[t]*1e4

    # パーセンタイル（2.5%,50%,97.5%）
    percentiles = np.percentile(asset_paths, [2.5,50,97.5], axis=0)

    # -------------------------
    # 目標資産額に到達するまでの期間分布
    # -------------------------
    time_to_target = []
    for sim in asset_paths:
        hit = np.where(sim >= target_amount*1e4)[0]  # 到達した月インデックス
        if len(hit) > 0:
            time_to_target.append(hit[0] + 1)  # 最初に到達した月（1始まり）
        else:
            time_to_target.append(np.nan)  # 未到達は NaN
    time_to_target = np.array(time_to_target)
    # NaN を除外
    valid_times = time_to_target[~np.isnan(time_to_target)]
    # 月 → 年換算
    years_to_target = valid_times / 12
    # パーセンタイル計算
    percentiles_time = np.percentile(years_to_target, [2.5, 50, 97.5])

    # --- 結果を session_state に保存 ---
    st.session_state["sim_result"] = {
        "dates_sim": dates_sim,
        "percentiles": percentiles,
        "target_amount": target_amount,
        "x_start": x_start,
        "x_end": x_end,
        "y_min": y_min,
        "y_max": y_max,
        "percentiles_time": percentiles_time,
        "years_to_target": years_to_target
    }
    st.session_state["sim_executed"] = True
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.success("シミュレーションを実行しました。入力を変更したら再実行してください。")
    st.caption(f"実行時刻：{run_time}")
# -------------------------
# 表示部：前回の結果を保持
# -------------------------
if "sim_executed" in st.session_state and st.session_state["sim_executed"]:
    result = st.session_state["sim_result"]

    # --- メイングラフ ---
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=result["dates_sim"], y=result["percentiles"][0]/1e4, mode='lines', name='下限(2.5%)', line=dict(color='red', dash='dot')))
    fig3.add_trace(go.Scatter(x=result["dates_sim"], y=result["percentiles"][2]/1e4, mode='lines', name='上限(97.5%)', fill="tonexty", fillcolor="rgba(173,216,230,0.2)", line=dict(color='green', dash='dot')))
    fig3.add_trace(go.Scatter(x=result["dates_sim"], y=result["percentiles"][1]/1e4, mode='lines', name='中央値(50%)', line=dict(color='blue', width=2)))
    # グラフに目標線を追加
    fig3.add_hline(
        y=result["target_amount"],
        line_dash="dash",
        line_color="purple",
        annotation_text="目標資産額",
        annotation_position="top right"
    )
    fig3.update_layout(
        #title="モンテカルロ資産形成シミュレーション",
        xaxis_title="年月",
        yaxis_title="資産額（万円）",
        xaxis=dict(range=[result["x_start"], result["x_end"]]),
        yaxis=dict(range=[result["y_min"], result["y_max"]]),
        template="plotly_white",
        height=500
    )
    fig3.update_layout(
        title=dict(
            text=f"モンテカルロ資産形成シミュレーション",
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
        margin=dict(t=120)  # 上の余白をpxで指定
    )
    st.plotly_chart(fig3, use_container_width=True)

    # --- 到達年数ヒストグラム ---
    fig4 = go.Figure()
    fig4.add_trace(go.Histogram(
        x=result["years_to_target"],
        nbinsx=60,
        name="到達までの年数分布",
        marker_color="skyblue"
    ))

    # 2.5%, 50%, 97.5%タイルに縦線を追加
    for p, val in zip([2.5, 50, 97.5], result["percentiles_time"]):
        fig4.add_vline(
            x=val,
            line_dash="dash",
            line_color="red",
            annotation_text=f"{p}%tile",
            annotation_position="top"
        )
    fig4.update_layout(
        title=dict(
            text=f"目標資産額に到達するまでの期間分布",
            x=0.5,   # 中央揃え
            xanchor='center',
            y=0.9,   # 上から少し下げる（デフォルトは1.0）
            yanchor='top'
        ),
        xaxis_title="到達年数",
        yaxis_title="シミュレーション回数",
        template="plotly_white",
        height=500
    )
    st.plotly_chart(fig4, use_container_width=True)

    # パーセンタイルの値をテキストで出力
    st.markdown(f"""
    **到達期間の統計値 (年):**
    - 2.5 %tile: {result["percentiles_time"][0]:.1f} 年
    - 50 %tile (中央値): {result["percentiles_time"][1]:.1f} 年
    - 97.5 %tile: {result["percentiles_time"][2]:.1f} 年
    """)

