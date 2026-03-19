"""
Streamlit Web App: 日米セクター リードラグ戦略ダッシュボード

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CFULL_END,
    CFULL_START,
    DATA_PROCESSED,
    DATA_RAW,
    JP_TICKERS,
    JP_TICKER_NAMES,
    NUM_FACTORS,
    QUANTILE_THRESHOLD,
    REGULARIZATION_LAMBDA,
    ROLLING_WINDOW,
    TEST_START,
)
from src.signal.prior_subspace import build_prior_subspace, compute_cfull
from src.signal.regularized_pca import regularized_pca, rolling_standardize
from src.portfolio.construction import construct_portfolio
from src.evaluation.metrics import compute_metrics

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="日米セクター リードラグ戦略",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

def _data_files_exist() -> bool:
    """Check whether the required processed data files exist."""
    us_path = DATA_PROCESSED / "us_returns.csv"
    jp_path = DATA_PROCESSED / "jp_returns.csv"
    map_path = DATA_PROCESSED / "us_jp_date_map.csv"
    return us_path.exists() and jp_path.exists() and map_path.exists()


@st.cache_data(show_spinner="データを読み込み中...")
def load_data():
    """Load pre-processed CSV data. Returns (us_ret, jp_ret_full, date_map)."""
    us_ret = pd.read_csv(
        DATA_PROCESSED / "us_returns.csv", index_col=0, parse_dates=True,
    )
    jp_ret = pd.read_csv(
        DATA_PROCESSED / "jp_returns.csv", header=[0, 1], index_col=0, parse_dates=True,
    )
    date_map = pd.read_csv(
        DATA_PROCESSED / "us_jp_date_map.csv", parse_dates=["us_date", "jp_next_date"],
    )
    return us_ret, jp_ret, date_map


@st.cache_data(show_spinner="JP始値データを読み込み中...")
def load_jp_open_prices():
    """Load JP ETF open prices from raw OHLC for share count calculation."""
    ohlc_path = DATA_RAW / "jp_etf_ohlc.csv"
    if not ohlc_path.exists():
        return None
    ohlc = pd.read_csv(ohlc_path, header=[0, 1], index_col=0, parse_dates=True)
    open_cols = {}
    for ticker in JP_TICKERS:
        if (ticker, "Open") in ohlc.columns:
            open_cols[ticker] = ohlc[(ticker, "Open")]
    if not open_cols:
        return None
    return pd.DataFrame(open_cols)


@st.cache_data(show_spinner="Cfull/V0/C0/z-scoreを計算中...")
def compute_all_artifacts():
    """Compute Cfull, V0, C0, combined returns, z-scores (all heavy work)."""
    us_ret = pd.read_csv(
        DATA_PROCESSED / "us_returns.csv", index_col=0, parse_dates=True,
    )
    jp_ret = pd.read_csv(
        DATA_PROCESSED / "jp_returns.csv", header=[0, 1], index_col=0, parse_dates=True,
    )
    jp_cc = jp_ret.xs("cc", axis=1, level="ReturnType")

    # Cfull (2010-2014, 9 US tickers automatically detected)
    cfull_corr, us_tickers_cfull = compute_cfull(
        us_ret, jp_cc,
        cfull_start=CFULL_START, cfull_end=CFULL_END,
        us_tickers_available=None,
    )
    jp_tickers = list(JP_TICKERS)
    V0, C0 = build_prior_subspace(us_tickers_cfull, jp_tickers, cfull_corr)

    # Combined returns matrix + rolling z-scores
    us = us_ret[us_tickers_cfull]
    jp = jp_cc[jp_tickers]
    combined = pd.concat([us, jp], axis=1).dropna()
    z_scores = rolling_standardize(combined.values, window=ROLLING_WINDOW)

    return us_tickers_cfull, C0, combined, z_scores


@st.cache_data(show_spinner="バックテストを実行中...")
def run_pca_sub_backtest():
    """Run PCA_SUB backtest and return daily returns Series."""
    us_ret = pd.read_csv(
        DATA_PROCESSED / "us_returns.csv", index_col=0, parse_dates=True,
    )
    jp_ret_full = pd.read_csv(
        DATA_PROCESSED / "jp_returns.csv", header=[0, 1], index_col=0, parse_dates=True,
    )
    date_map = pd.read_csv(
        DATA_PROCESSED / "us_jp_date_map.csv", parse_dates=["us_date", "jp_next_date"],
    )

    jp_cc = jp_ret_full.xs("cc", axis=1, level="ReturnType")
    jp_oc = jp_ret_full.xs("oc", axis=1, level="ReturnType")

    # Compute prior
    cfull_corr, us_tickers_cfull = compute_cfull(
        us_ret, jp_cc, cfull_start=CFULL_START, cfull_end=CFULL_END,
        us_tickers_available=None,
    )
    jp_tickers = list(JP_TICKERS)
    V0, C0 = build_prior_subspace(us_tickers_cfull, jp_tickers, cfull_corr)

    # Combined + z-scores
    us = us_ret[us_tickers_cfull]
    jp = jp_cc[jp_tickers]
    combined = pd.concat([us, jp], axis=1).dropna()
    z_scores = rolling_standardize(combined.values, window=ROLLING_WINDOW)

    # Date map lookup
    us_to_jp = {}
    for _, row in date_map.iterrows():
        us_to_jp[pd.Timestamp(row["us_date"])] = pd.Timestamp(row["jp_next_date"])

    test_start = pd.Timestamp(TEST_START)
    window = ROLLING_WINDOW
    K = NUM_FACTORS
    n_us = len(us_tickers_cfull)

    daily_rets = []
    for us_date in sorted(us_to_jp.keys()):
        jp_date = us_to_jp[us_date]
        if jp_date < test_start:
            continue
        if us_date not in combined.index:
            continue
        t = combined.index.get_loc(us_date)
        if t < window:
            continue

        z_win = z_scores[t - window : t]
        if np.isnan(z_win).any():
            continue

        V_K = regularized_pca(z_win, C0, lam=REGULARIZATION_LAMBDA, K=K)
        V_U = V_K[:n_us, :]
        V_J = V_K[n_us:, :]
        z_US_t = z_scores[t, :n_us]
        f_t = V_U.T @ z_US_t
        z_hat_J = V_J @ f_t

        signal = pd.Series(z_hat_J, index=jp_tickers)
        weights = construct_portfolio(signal, q=QUANTILE_THRESHOLD)

        if jp_date not in jp_oc.index:
            continue
        oc_ret = jp_oc.loc[jp_date, jp_tickers]
        port_ret = (weights * oc_ret).sum()
        daily_rets.append((jp_date, port_ret))

    if not daily_rets:
        return pd.Series(dtype=float)
    dates, rets = zip(*daily_rets)
    return pd.Series(rets, index=pd.DatetimeIndex(dates), name="PCA_SUB").sort_index()


# ---------------------------------------------------------------------------
# Signal generation helpers
# ---------------------------------------------------------------------------

def find_us_date_for_jp(jp_date: pd.Timestamp, date_map: pd.DataFrame):
    """Reverse-lookup: JP date -> US date."""
    match = date_map[date_map["jp_next_date"] == jp_date]
    if match.empty:
        return None
    return pd.Timestamp(match.iloc[-1]["us_date"])


def find_nearest_jp_date(target, date_map, direction="backward"):
    """Find the nearest JP business day in the date_map."""
    jp_dates = pd.DatetimeIndex(date_map["jp_next_date"].sort_values().unique())
    if direction == "backward":
        candidates = jp_dates[jp_dates <= target]
        return candidates[-1] if len(candidates) > 0 else None
    else:
        candidates = jp_dates[jp_dates >= target]
        return candidates[0] if len(candidates) > 0 else None


def generate_signal(us_date, combined, z_scores, us_tickers_cfull, C0):
    """Generate signal and weights for a single US date."""
    window = ROLLING_WINDOW
    K = NUM_FACTORS
    jp_tickers = list(JP_TICKERS)

    if us_date not in combined.index:
        return None, None, f"US日付 {us_date.date()} はデータに存在しません"

    t_loc = combined.index.get_loc(us_date)
    if t_loc < window * 2:
        return None, None, f"ローリングウィンドウに十分なデータがありません (位置={t_loc})"

    z_win = z_scores[t_loc - window : t_loc]
    if np.any(np.isnan(z_win)):
        return None, None, "PCAウィンドウにNaNが含まれています"

    V_K = regularized_pca(z_win, C0, lam=REGULARIZATION_LAMBDA, K=K)
    n_us = len(us_tickers_cfull)
    V_U = V_K[:n_us, :]
    V_J = V_K[n_us:, :]

    z_US_t = z_scores[t_loc, :n_us]
    f_t = V_U.T @ z_US_t
    z_hat_J = V_J @ f_t

    signal = pd.Series(z_hat_J, index=jp_tickers)
    weights = construct_portfolio(signal, q=QUANTILE_THRESHOLD)
    return signal, weights, None


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

st.sidebar.title("日米セクター リードラグ戦略")
page = st.sidebar.radio(
    "ページ選択",
    ["本日のシグナル", "バックテスト結果", "直近パフォーマンス"],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "**免責事項**\n\n"
    "本ツールは学術論文(中川ら, SIG-FIN-036)の"
    "再現実装であり、研究・教育目的で公開しています。\n\n"
    "特定の金融商品の売買を推奨するものではありません。"
    "投資判断はご自身の責任で行ってください。\n\n"
    "データソース: Yahoo Finance, Kenneth French Data Library"
)

# ---------------------------------------------------------------------------
# Data availability check
# ---------------------------------------------------------------------------

if not _data_files_exist():
    st.info("初回起動: データをダウンロード中...（1〜2分かかります）")
    with st.spinner("データ取得中..."):
        try:
            from src.data.fetch_jp_etf import fetch_jp_etf
            from src.data.fetch_us_etf import fetch_us_etf
            from src.data.fetch_ff_factors import fetch_ff_factors
            from src.data.preprocess import preprocess_returns
            from src.data.build_calendar import build_calendar

            fetch_jp_etf()
            fetch_us_etf()
            fetch_ff_factors()
            us_ret, jp_ret = preprocess_returns()
            jp_cc = jp_ret.xs("cc", axis=1, level="ReturnType")
            build_calendar(us_ret.index, jp_cc.index)
            st.success("データ取得完了! ページを再読み込みします...")
            st.rerun()
        except Exception as e:
            st.error(f"データ取得に失敗しました: {e}")
            st.stop()

# Load base data (cheap, cached)
us_ret, jp_ret_full, date_map = load_data()
jp_cc = jp_ret_full.xs("cc", axis=1, level="ReturnType")
jp_oc = jp_ret_full.xs("oc", axis=1, level="ReturnType")


# ===================================================================
# Page 1: Today's Signal
# ===================================================================
if page == "本日のシグナル":
    st.header("本日のシグナル (PCA SUB)")

    # Compute heavy artifacts (cached)
    us_tickers_cfull, C0, combined, z_scores = compute_all_artifacts()

    # Date picker
    jp_dates = pd.DatetimeIndex(date_map["jp_next_date"].sort_values().unique())
    default_jp = find_nearest_jp_date(pd.Timestamp.today(), date_map, direction="backward")
    if default_jp is None:
        default_jp = jp_dates[-1]

    col_date, col_capital = st.columns([1, 1])
    with col_date:
        selected_date = st.date_input(
            "対象日（日本市場）",
            value=default_jp.date(),
            min_value=jp_dates[0].date(),
            max_value=jp_dates[-1].date(),
        )
    with col_capital:
        capital = st.number_input(
            "投資金額（円）",
            value=5_000_000,
            min_value=100_000,
            step=1_000_000,
            format="%d",
        )

    target_ts = pd.Timestamp(selected_date)
    jp_date = find_nearest_jp_date(target_ts, date_map, direction="backward")
    if jp_date is None:
        jp_date = find_nearest_jp_date(target_ts, date_map, direction="forward")

    if jp_date is None:
        st.error("対応する営業日が見つかりません。")
        st.stop()

    if jp_date.date() != selected_date:
        st.info(f"{selected_date} は営業日ではありません。直近の営業日 {jp_date.date()} を使用します。")

    us_date = find_us_date_for_jp(jp_date, date_map)
    if us_date is None:
        st.error(f"{jp_date.date()} に対応するUS日付が見つかりません。")
        st.stop()

    st.markdown(
        f"**日本市場日付:** {jp_date.strftime('%Y-%m-%d')} |"
        f"**米国基準日:** {us_date.strftime('%Y-%m-%d')}"
    )

    # Generate signal
    signal, weights, err = generate_signal(us_date, combined, z_scores, us_tickers_cfull, C0)
    if err:
        st.error(err)
        st.stop()

    # Build display table
    rows = []
    jp_open_prices = load_jp_open_prices()

    for ticker in signal.sort_values(ascending=False).index:
        w = weights[ticker]
        if w > 0:
            side = "LONG"
        elif w < 0:
            side = "SHORT"
        else:
            side = "-"

        name = JP_TICKER_NAMES.get(ticker, "")
        sig_val = signal[ticker]
        target_jpy = abs(w) * capital  # 目標ポジション金額

        # ETFは1口単位で売買可能（個別株の100株単位とは異なる）
        shares_str = ""
        actual_jpy_str = ""
        if w != 0 and jp_open_prices is not None and ticker in jp_open_prices.columns:
            if jp_date in jp_open_prices.index:
                open_price = jp_open_prices.loc[jp_date, ticker]
                if pd.notna(open_price) and open_price > 0:
                    shares = max(1, int(target_jpy / open_price))
                    actual_jpy = shares * open_price
                    shares_str = f"{shares:,}"
                    actual_jpy_str = f"{actual_jpy:,.0f}"

        rows.append({
            "ティッカー": ticker,
            "セクター名": name,
            "サイド": side,
            "シグナル": round(sig_val, 4),
            "ウェイト": round(w, 4),
            "目標金額(円)": f"{target_jpy:,.0f}" if w != 0 else "",
            "株数": shares_str,
            "実金額(円)": actual_jpy_str,
        })

    df_display = pd.DataFrame(rows)

    # Style the side column (use .map for pandas >= 2.1)
    def style_side(val):
        if val == "LONG":
            return "color: #2ca02c; font-weight: bold"
        elif val == "SHORT":
            return "color: #d62728; font-weight: bold"
        return ""

    styled = df_display.style.map(style_side, subset=["サイド"])
    st.dataframe(
        styled,
        width="stretch",
        hide_index=True,
        height=35 * len(df_display) + 38,
    )
    st.caption("※ TOPIX-17業種ETFは1口単位で売買可能（個別株の100株単位とは異なります）")

    # Summary
    long_count = (weights > 0).sum()
    short_count = (weights < 0).sum()
    st.markdown(
        f"**ロング {long_count}銘柄** / **ショート {short_count}銘柄** / "
        f"計 {len(weights)}銘柄 |"
        f"グロスエクスポージャー: {weights.abs().sum():.1f} |"
        f"ネットエクスポージャー: {weights.sum():.4f}"
    )

    # Actual P&L if date is in the past
    if jp_date in jp_oc.index:
        oc_ret = jp_oc.loc[jp_date, list(JP_TICKERS)]
        port_ret = (weights * oc_ret).sum()

        if pd.Timestamp.today().normalize() > jp_date:
            st.markdown("---")
            st.subheader("実績（始値→終値）")

            pnl_jpy = port_ret * capital
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("ポートフォリオリターン", f"{port_ret * 100:+.4f}%")
            with col2:
                st.metric("損益（円）", f"{pnl_jpy:+,.0f}")
            with col3:
                individual_pnl = weights * oc_ret
                best_ticker = individual_pnl.idxmax()
                st.metric(
                    "最大寄与",
                    f"{JP_TICKER_NAMES.get(best_ticker, best_ticker)}",
                    f"{individual_pnl[best_ticker] * 100:+.4f}%",
                )


# ===================================================================
# Page 2: Backtest Summary
# ===================================================================
elif page == "バックテスト結果":
    st.header("バックテスト結果 (PCA SUB)")

    daily_rets = run_pca_sub_backtest()

    if daily_rets.empty:
        st.warning("バックテスト結果がありません。データを確認してください。")
        st.stop()

    # Metrics
    metrics = compute_metrics(daily_rets)

    st.subheader("パフォーマンス指標")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("年率リターン (AR)", f"{metrics['AR'] * 100:.2f}%")
    with col2:
        st.metric("年率リスク", f"{metrics['RISK'] * 100:.2f}%")
    with col3:
        st.metric("リスクリターン比 (RR)", f"{metrics['RR']:.2f}")
    with col4:
        st.metric("最大ドローダウン (MDD)", f"{metrics['MDD'] * 100:.2f}%")
    with col5:
        st.metric("累積リターン", f"{metrics['TOTAL_RETURN'] * 100:.2f}%")

    st.markdown(
        f"**テスト期間:** {daily_rets.index[0].strftime('%Y-%m-%d')} ~ "
        f"{daily_rets.index[-1].strftime('%Y-%m-%d')} |"
        f"**取引日数:** {len(daily_rets):,}"
    )

    # Cumulative returns chart
    st.subheader("累積リターン推移")
    cumulative = (1.0 + daily_rets).cumprod()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(cumulative.index, cumulative.values, color="#2ca02c", linewidth=1.2, label="PCA SUB")
    ax.axhline(y=1.0, color="grey", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_ylabel("累積リターン")
    ax.set_xlabel("日付")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Drawdown chart
    st.subheader("ドローダウン推移")
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color="#d62728")
    ax2.plot(drawdown.index, drawdown.values, color="#d62728", linewidth=0.8)
    ax2.set_ylabel("ドローダウン")
    ax2.set_xlabel("日付")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    # Annual breakdown
    st.subheader("年別リターン")
    annual = daily_rets.groupby(daily_rets.index.year).apply(
        lambda x: (1 + x).prod() - 1
    )
    annual_df = pd.DataFrame({
        "年": annual.index,
        "年間リターン": [f"{r * 100:+.2f}%" for r in annual.values],
        "取引日数": daily_rets.groupby(daily_rets.index.year).count().values,
    })
    st.dataframe(annual_df, width="stretch", hide_index=True)


# ===================================================================
# Page 3: Recent Performance
# ===================================================================
elif page == "直近パフォーマンス":
    st.header("直近パフォーマンス (PCA SUB)")

    daily_rets = run_pca_sub_backtest()

    if daily_rets.empty:
        st.warning("バックテスト結果がありません。データを確認してください。")
        st.stop()

    n_days = st.sidebar.slider("表示日数", min_value=5, max_value=60, value=10)
    recent = daily_rets.tail(n_days)

    # Recent daily P&L table
    st.subheader(f"直近{n_days}営業日の日次損益")

    cum_pnl = recent.cumsum()
    recent_df = pd.DataFrame({
        "日付": [d.strftime("%Y-%m-%d") for d in recent.index],
        "日次リターン": [f"{r * 100:+.4f}%" for r in recent.values],
        "累積リターン": [f"{c * 100:+.4f}%" for c in cum_pnl.values],
    })

    def highlight_return(val):
        if isinstance(val, str):
            if val.startswith("+"):
                return "color: #2ca02c"
            elif val.startswith("-"):
                return "color: #d62728"
        return ""

    styled = recent_df.style.map(
        highlight_return, subset=["日次リターン", "累積リターン"],
    )
    st.dataframe(styled, width="stretch", hide_index=True)

    # Summary metrics for the period
    col1, col2, col3 = st.columns(3)
    with col1:
        total = (1 + recent).prod() - 1
        st.metric(f"直近{n_days}日 累積リターン", f"{total * 100:+.4f}%")
    with col2:
        wins = (recent > 0).sum()
        st.metric("勝率", f"{wins}/{n_days} ({wins / n_days * 100:.0f}%)")
    with col3:
        avg = recent.mean()
        st.metric("平均日次リターン", f"{avg * 100:+.4f}%")

    # Cumulative P&L chart for recent period
    st.subheader("累積損益推移")
    cum_recent = (1 + recent).cumprod()

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#2ca02c" if r >= 0 else "#d62728" for r in recent.values]
    ax.bar(
        range(len(recent)), recent.values * 100,
        color=colors, alpha=0.7, label="日次リターン(%)",
    )
    ax2 = ax.twinx()
    ax2.plot(
        range(len(recent)), cum_recent.values,
        color="#1f77b4", linewidth=2, marker="o", markersize=4, label="累積リターン",
    )
    ax2.axhline(y=1.0, color="grey", linestyle="--", linewidth=0.7, alpha=0.5)

    ax.set_xticks(range(len(recent)))
    ax.set_xticklabels(
        [d.strftime("%m/%d") for d in recent.index], rotation=45, fontsize=9,
    )
    ax.set_ylabel("日次リターン (%)")
    ax2.set_ylabel("累積リターン")
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=9)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
