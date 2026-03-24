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
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
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
    page_title="Alpha Signal｜日米セクター クロスボーダー戦略",
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


@st.cache_data(show_spinner="データを読み込み中...", ttl=3600)
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


@st.cache_data(show_spinner="米国ETFデータを読み込み中...", ttl=3600)
def load_us_ohlc() -> pd.DataFrame:
    """Load US ETF OHLC from raw CSV."""
    return pd.read_csv(DATA_RAW / "us_etf_ohlc.csv", header=[0, 1], index_col=0, parse_dates=True)


@st.cache_data(show_spinner="JP始値データを読み込み中...", ttl=3600)
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


@st.cache_data(show_spinner="Cfull/V0/C0/z-scoreを計算中...", ttl=3600)
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


@st.cache_data(show_spinner="バックテストを実行中...", ttl=3600)
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

st.sidebar.title("Alpha Signal")
st.sidebar.caption("日米セクター クロスボーダー戦略")
page = st.sidebar.radio(
    "ページ選択",
    ["本日のシグナル", "バックテスト結果", "直近パフォーマンス", "📈 米国セクター動向", "🌙 夜間リスクチェック"],
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
    st.error(
        "Required data files are missing. Run the scheduled HF update job "
        "to regenerate data for this Space."
    )
    st.caption("Setup instructions: docs/hf-jobs-setup.md")
    st.stop()

# Load base data (cheap, cached)
us_ret, jp_ret_full, date_map = load_data()
jp_cc = jp_ret_full.xs("cc", axis=1, level="ReturnType")
jp_oc = jp_ret_full.xs("oc", axis=1, level="ReturnType")


# ===================================================================
# Page 1: Today's Signal
# ===================================================================
if page == "本日のシグナル":
    st.header("📊 本日のシグナル")

    # Compute heavy artifacts (cached)
    us_tickers_cfull, C0, combined, z_scores = compute_all_artifacts()
    jp_open_prices = load_jp_open_prices()

    # --- 設定行 ---
    jp_dates = pd.DatetimeIndex(date_map["jp_next_date"].sort_values().unique())
    # データに存在する最新の日付を常にデフォルトとして使用する
    # （JP市場未クローズ時でもUS終値ベースのシグナルを表示するため jp_dates[-1] を使用）
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
            step=500_000,
            format="%d",
        )

    # --- 日付解決 ---
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

    jp_has_closing = jp_date in jp_oc.index

    st.caption(
        f"🇯🇵 日本市場: {jp_date.strftime('%Y-%m-%d')}　|　"
        f"🇺🇸 米国基準日: {us_date.strftime('%Y-%m-%d')}"
    )

    if not jp_has_closing:
        st.info(
            f"📡 {jp_date.strftime('%Y-%m-%d')} の JP 終値データはまだ取得できていません（市場取引中または更新前）。"
            "シグナルは米国終値から計算済みです。終値実績は市場クローズ後のデータ更新後に表示されます。"
        )

    # --- シグナル生成 ---
    signal, weights, err = generate_signal(us_date, combined, z_scores, us_tickers_cfull, C0)
    if err:
        st.error(err)
        st.stop()

    # シグナルを [-1, +1] に正規化
    sig_max = signal.abs().max()
    norm_signal = signal / sig_max if sig_max > 0 else signal

    # ロング TOP2 / ショート TOP2
    long_series = signal[weights > 0].nlargest(2)
    short_series = signal[weights < 0].nsmallest(2)

    # ランクに応じた ★ 表示
    RANK_STARS = {1: "★★★", 2: "★★"}

    # 口数と実金額を計算
    def _shares_info(ticker: str, weight_val: float):
        target_jpy = abs(weight_val) * capital
        shares_label = "—"
        if (
            jp_open_prices is not None
            and ticker in jp_open_prices.columns
            and jp_date in jp_open_prices.index
        ):
            open_price = jp_open_prices.loc[jp_date, ticker]
            if pd.notna(open_price) and open_price > 0:
                shares = max(1, int(target_jpy / open_price))
                shares_label = f"{shares:,} 口"
        return target_jpy, shares_label

    # ランク1カード（大・グラデーション背景）
    def _card_rank1(ticker: str, rank: int, weight_val: float, grad: str, label: str):
        name = JP_TICKER_NAMES.get(ticker, ticker)
        nv = norm_signal[ticker]
        sign = "+" if nv >= 0 else ""
        stars = RANK_STARS.get(rank, "★")
        st.markdown(
            f"""
            <div style="
                background:{grad};
                border-radius:18px;
                padding:30px 20px 28px;
                color:white;
                text-align:center;
                box-shadow:0 6px 24px rgba(0,0,0,0.25);
                margin-bottom:14px;
            ">
                <div style="font-size:12px;opacity:.8;letter-spacing:2px;margin-bottom:6px;">{label}</div>
                <div style="font-size:40px;font-weight:900;line-height:1.2;margin-bottom:12px;">{name}</div>
                <div style="font-size:30px;letter-spacing:5px;margin-bottom:8px;">{stars}</div>
                <div style="font-size:26px;font-weight:bold;">{sign}{nv:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ランク2カード（小・枠線スタイル）
    def _card_rank2(ticker: str, rank: int, weight_val: float, border: str, fg: str, bg: str, label: str):
        name = JP_TICKER_NAMES.get(ticker, ticker)
        nv = norm_signal[ticker]
        sign = "+" if nv >= 0 else ""
        stars = RANK_STARS.get(rank, "★")
        st.markdown(
            f"""
            <div style="
                background:{bg};
                border:2px solid {border};
                border-radius:14px;
                padding:20px 16px 20px;
                text-align:center;
                margin-bottom:14px;
            ">
                <div style="font-size:11px;color:{fg};opacity:.7;letter-spacing:2px;margin-bottom:4px;">{label}</div>
                <div style="font-size:28px;font-weight:800;color:{fg};line-height:1.2;margin-bottom:10px;">{name}</div>
                <div style="font-size:22px;color:{border};letter-spacing:4px;margin-bottom:6px;">{stars}</div>
                <div style="font-size:20px;font-weight:bold;color:{fg};">{sign}{nv:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- 2列レイアウト: 左=ロング / 右=ショート ---
    st.markdown("---")
    col_long, col_short = st.columns(2)

    with col_long:
        st.markdown("### 🟢 買い（ロング）")
        tickers_l = list(long_series.index)
        if len(tickers_l) >= 1:
            _card_rank1(
                tickers_l[0], 1, weights[tickers_l[0]],
                "linear-gradient(135deg,#1a7a2e,#38c446)",
                "🏆  No.1  LONG",
            )
        if len(tickers_l) >= 2:
            _card_rank2(
                tickers_l[1], 2, weights[tickers_l[1]],
                "#2ca02c", "#155a20", "#e8f5e9",
                "No.2  LONG",
            )

    with col_short:
        st.markdown("### 🔴 売り（ショート）")
        tickers_s = list(short_series.index)
        if len(tickers_s) >= 1:
            _card_rank1(
                tickers_s[0], 1, weights[tickers_s[0]],
                "linear-gradient(135deg,#8b0000,#d62728)",
                "🏆  No.1  SHORT",
            )
        if len(tickers_s) >= 2:
            _card_rank2(
                tickers_s[1], 2, weights[tickers_s[1]],
                "#d62728", "#7a0000", "#fff0f0",
                "No.2  SHORT",
            )

    st.caption("※ TOPIX-17業種ETFは1口単位で売買可能（個別株の100株単位とは異なります）")

    # --- 全セクター シグナル一覧 ---
    st.markdown("---")
    st.markdown("#### 全セクター シグナル一覧")

    # ★ の決定（全銘柄スケールで相対判定）
    def _stars_full(norm_val: float) -> str:
        v = abs(norm_val)
        if v >= 0.7:
            return "★★★"
        elif v >= 0.4:
            return "★★"
        return "★"

    # 1行分のHTML生成
    def _row_html(rank: int, ticker: str, norm_val: float, is_top2: bool, is_long: bool) -> str:
        name = JP_TICKER_NAMES.get(ticker, ticker)
        sign = "+" if norm_val >= 0 else ""
        stars = _stars_full(norm_val)
        star_color = "#2ca02c" if is_long else "#d62728"
        text_color = "#155a20" if is_long else "#7a0000"
        bg = ("#d4edda" if is_long else "#f8d7da") if is_top2 else "transparent"
        weight = "700" if is_top2 else "400"
        return (
            f'<div style="display:flex;align-items:center;padding:6px 10px;'
            f'background:{bg};border-radius:6px;margin-bottom:3px;gap:6px;">'
            f'<span style="font-size:11px;color:#999;width:26px;text-align:right;">{rank}位</span>'
            f'<span style="font-size:14px;font-weight:{weight};color:{text_color};flex:1;">{name}</span>'
            f'<span style="font-size:14px;font-weight:bold;color:{text_color};width:50px;text-align:right;">'
            f'{sign}{norm_val:.2f}</span>'
            f'<span style="font-size:13px;color:{star_color};width:46px;text-align:center;">{stars}</span>'
            f'</div>'
        )

    # ロング候補（シグナル強い順）
    long_all = signal[weights > 0].sort_values(ascending=False)
    long_top2_set = set(list(long_series.index))
    long_html = "".join(
        _row_html(i + 1, tk, norm_signal[tk], tk in long_top2_set, is_long=True)
        for i, tk in enumerate(long_all.index)
    )

    # ショート候補（シグナル強い順 = 最も負の値が1位）
    short_all = signal[weights < 0].sort_values(ascending=True)
    short_top2_set = set(list(short_series.index))
    short_html = "".join(
        _row_html(i + 1, tk, norm_signal[tk], tk in short_top2_set, is_long=False)
        for i, tk in enumerate(short_all.index)
    )

    col_ll, col_ss = st.columns(2)
    with col_ll:
        st.markdown(
            f'<div style="font-size:14px;font-weight:bold;color:#1a7a2e;margin-bottom:8px;">'
            f'🟢 ロング候補（強い順）</div>{long_html}',
            unsafe_allow_html=True,
        )
    with col_ss:
        st.markdown(
            f'<div style="font-size:14px;font-weight:bold;color:#8b0000;margin-bottom:8px;">'
            f'🔴 ショート候補（強い順）</div>{short_html}',
            unsafe_allow_html=True,
        )

    # 中立（ウェイト=0）セクター
    neutral_tickers = [tk for tk in signal.index if weights[tk] == 0]
    if neutral_tickers:
        neutral_sorted = signal[neutral_tickers].abs().sort_values(ascending=False).index
        neutral_rows = "".join(
            f'<span style="display:inline-flex;align-items:center;gap:6px;'
            f'padding:4px 10px;border-radius:6px;background:#f5f5f5;margin:3px;">'
            f'<span style="font-size:13px;color:#555;">{JP_TICKER_NAMES.get(tk, tk)}</span>'
            f'<span style="font-size:12px;color:#888;font-weight:bold;">'
            f'{"+" if norm_signal[tk] >= 0 else ""}{norm_signal[tk]:.2f}</span>'
            f'</span>'
            for tk in neutral_sorted
        )
        st.markdown(
            f'<div style="margin-top:14px;">'
            f'<div style="font-size:14px;font-weight:bold;color:#666;margin-bottom:8px;">⚪ 中立（今日は見送り）</div>'
            f'<div style="display:flex;flex-wrap:wrap;gap:2px;">{neutral_rows}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # --- 過去日付の場合: 実績表示（JP終値データが存在する日のみ） ---
    if jp_has_closing and pd.Timestamp.today().normalize() > jp_date:
        oc_ret = jp_oc.loc[jp_date, list(JP_TICKERS)]
        port_ret = (weights * oc_ret).sum()
        pnl_jpy = port_ret * capital
        st.markdown("---")
        st.subheader("実績（始値→終値）")
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


# ===================================================================
# Page 4: US Sector Chart
# ===================================================================
elif page == "📈 米国セクター動向":
    st.header("📈 米国セクター動向")

    _US_NAMES = {
        "XLB": "素材", "XLC": "通信", "XLE": "エネルギー", "XLF": "金融",
        "XLI": "資本財", "XLK": "情報技術", "XLP": "生活必需品",
        "XLRE": "不動産", "XLU": "公益", "XLV": "ヘルスケア", "XLY": "一般消費財",
    }
    _PERIOD_DAYS = {"1週間": 7, "1ヶ月": 30, "3ヶ月": 90, "6ヶ月": 180}
    _COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    us_ohlc = load_us_ohlc()
    _available = [t for t in _US_NAMES if (t, "Close") in us_ohlc.columns]

    col_sel, col_period = st.columns([3, 1])
    with col_sel:
        selected_us = st.multiselect(
            "セクターを選択（最大5本）",
            options=_available,
            default=["XLK", "XLF", "XLE"],
            format_func=lambda t: f"{t}  {_US_NAMES[t]}",
            max_selections=5,
        )
    with col_period:
        period_label = st.selectbox("表示期間", list(_PERIOD_DAYS.keys()), index=1)

    if not selected_us:
        st.info("セクターを1本以上選択してください。")
        st.stop()

    # --- データ絞り込み ---
    end_dt = us_ohlc.index.max()
    start_dt = end_dt - pd.Timedelta(days=_PERIOD_DAYS[period_label])
    closes = pd.DataFrame(
        {t: us_ohlc[(t, "Close")] for t in selected_us},
    ).loc[lambda df: df.index >= start_dt]

    # --- チャート描画（期間開始=100に正規化して比較） ---
    fig, ax = plt.subplots(figsize=(12, 5))

    period_returns = {}
    for i, ticker in enumerate(selected_us):
        series = closes[ticker].dropna()
        if series.empty:
            continue
        pct_ret = (series.iloc[-1] / series.iloc[0] - 1) * 100
        period_returns[ticker] = pct_ret
        norm = series / series.iloc[0] * 100
        label = f"{ticker} {_US_NAMES[ticker]}  ({pct_ret:+.1f}%)"
        ax.plot(norm.index, norm.values, color=_COLORS[i % len(_COLORS)],
                linewidth=1.8, label=label)

    ax.axhline(y=100, color="grey", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_ylabel(f"基準値（{period_label}開始=100）")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # --- 期間リターン サマリーテーブル ---
    st.subheader("期間リターン まとめ")
    ret_rows = sorted(period_returns.items(), key=lambda x: x[1], reverse=True)
    ret_df = pd.DataFrame([
        {
            "ティッカー": t,
            "セクター": _US_NAMES[t],
            "期間リターン": f"{r:+.2f}%",
        }
        for t, r in ret_rows
    ])

    def _style_ret(val):
        if isinstance(val, str) and val.startswith("+"):
            return "color: #2ca02c; font-weight: bold"
        elif isinstance(val, str) and val.startswith("-"):
            return "color: #d62728; font-weight: bold"
        return ""

    st.dataframe(
        ret_df.style.map(_style_ret, subset=["期間リターン"]),
        hide_index=True,
        width="stretch",
    )

# ===================================================================
# Page 5: 夜間リスクチェック
# ===================================================================
elif page == "🌙 夜間リスクチェック":
    import feedparser
    import anthropic as _anthropic
    from datetime import datetime, timedelta, timezone

    st.header("🌙 夜間リスクチェック")
    st.caption("オーバーナイトトレード判断用ダッシュボード（15:15までに確認）")

    col_hd, col_btn = st.columns([5, 1])
    with col_btn:
        _refresh = st.button("🔄 今すぐ更新", use_container_width=True)

    JST = timezone(timedelta(hours=9))
    _now = datetime.now(JST)

    # ------------------------------------------------------------------
    # Helper: fetch RSS with fallback user-agent
    # ------------------------------------------------------------------
    def _fetch_rss(url: str) -> list:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }
        try:
            import requests as _req
            resp = _req.get(url, headers=headers, timeout=10)
            d = feedparser.parse(resp.text)
            return d.entries
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Section 1: マクロイベントカレンダー
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📅 1. マクロイベントカレンダー")
    st.caption("ソース: ZeroHedge / Reuters / Google News（今日〜7日以内）")

    _HIGH_KEYWORDS = ["FOMC", "Federal Reserve", "日銀", "BOJ", "CPI", "nonfarm", "payroll",
                      "雇用統計", "interest rate decision", "rate hike", "rate cut"]
    _NOTABLE_KEYWORDS = ["inflation", "GDP", "unemployment", "treasury", "yield", "Jackson Hole",
                         "monetary policy", "central bank", "economic outlook"]

    _GNEWS_SOURCES = [
        ("Google News(JP)", "https://news.google.com/rss/search?q=FOMC+OR+%E6%97%A5%E9%8A%80+OR+CPI+OR+%E9%9B%87%E7%94%A8%E7%B5%B1%E8%A8%88&hl=ja&gl=JP&ceid=JP:ja"),
        ("Google News(EN)", "https://news.google.com/rss/search?q=FOMC+OR+Fed+OR+BOJ+OR+CPI+OR+payroll&hl=en&gl=US&ceid=US:en"),
    ]

    @st.cache_data(ttl=1800, show_spinner=False)
    def _load_econ_calendar():
        now_utc  = datetime.now(timezone.utc)
        since    = now_utc - timedelta(days=7)   # 今日から7日前まで

        seen = set()
        rows = []
        for source_name, url in _GNEWS_SOURCES:
            entries = _fetch_rss(url)
            for e in entries:
                try:
                    pub = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
                except Exception:
                    continue
                # 7日以内の記事のみ
                if pub < since or pub > now_utc + timedelta(hours=1):
                    continue
                title = e.get("title", "").strip()
                link  = e.get("link", "")
                if not title or title in seen:
                    continue
                seen.add(title)
                is_high    = any(k.lower() in title.lower() for k in _HIGH_KEYWORDS)
                is_notable = any(k.lower() in title.lower() for k in _NOTABLE_KEYWORDS)
                if not (is_high or is_notable):
                    continue
                importance = "★★★" if is_high else "★★☆"
                pub_jst = pub.astimezone(JST)
                rows.append({
                    "日時(JST)": pub_jst.strftime("%m/%d %H:%M"),
                    "重要度": importance,
                    "イベント": title,
                    "ソース": source_name,
                    "_link": link,
                    "_high": is_high,
                    "_pub": pub,
                })

        rows.sort(key=lambda r: r["_pub"], reverse=True)
        return rows[:25]

    with st.spinner("経済カレンダー取得中…"):
        _cal_rows = _load_econ_calendar()

    if _refresh:
        st.cache_data.clear()
        _cal_rows = _load_econ_calendar()

    if not _cal_rows:
        st.warning("該当するマクロイベントが見つかりませんでした。キーワード（FOMC / CPI / 雇用統計 など）に関連するニュースがない可能性があります。")
    else:
        for row in _cal_rows:
            color = "#cc0000" if row["_high"] else "#cc7700"
            bg    = "rgba(204,0,0,0.06)" if row["_high"] else "transparent"
            st.markdown(
                f"""<div style="padding:5px 10px;margin:3px 0;background:{bg};border-radius:4px;border-left:3px solid {color}">
                <span style="color:{color};font-weight:bold">{row['重要度']}</span>
                &nbsp;<span style="color:#333;font-size:0.85em">{row['日時(JST)']}</span>
                &nbsp;<span style="color:#666;font-size:0.8em">[{row['ソース']}]</span>
                &nbsp;<a href="{row['_link']}" target="_blank" style="color:#111111;text-decoration:none">{row['イベント']}</a>
                </div>""",
                unsafe_allow_html=True,
            )

    # ------------------------------------------------------------------
    # Section 2: 市場環境
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 2. 市場環境")

    _MARKET_TICKERS = {
        "VIX":          ("^VIX",     "恐怖指数"),
        "ドル円":        ("USDJPY=X", "USD/JPY"),
        "金":            ("GC=F",     "Gold先物"),
        "原油WTI":       ("CL=F",     "WTI先物"),
        "日経先物(CME)": ("NKD=F",    "CME Nikkei"),
    }

    @st.cache_data(ttl=300, show_spinner=False)
    def _load_market_data():
        import yfinance as _yf
        results = {}
        for label, (ticker, desc) in _MARKET_TICKERS.items():
            try:
                t  = _yf.Ticker(ticker)
                fi = t.fast_info
                last = float(fi.last_price)

                # 前日終値: 直近5日の日足から2番目の終値
                hist = t.history(period="5d", interval="1d", auto_adjust=True)
                if hist is None or len(hist) < 2:
                    results[label] = None
                    continue
                prev = float(hist["Close"].iloc[-2])

                chg   = last - prev
                pct   = chg / prev * 100
                arrow = "↑" if chg > 0 else ("↓" if chg < 0 else "→")
                results[label] = {
                    "desc": desc,
                    "last": last,
                    "chg":  chg,
                    "pct":  pct,
                    "arrow": arrow,
                }
            except Exception:
                results[label] = None
        return results

    with st.spinner("市場データ取得中…"):
        _mkt = _load_market_data()

    _mkt_cols = st.columns(5)
    for i, (label, data) in enumerate(_mkt.items()):
        with _mkt_cols[i]:
            if data is None:
                st.metric(label, "取得失敗")
                continue
            chg_color = "#ff4444" if data["chg"] < 0 else "#44bb44"
            arrow = data["arrow"]
            st.markdown(
                f"""<div style="border:1px solid #333;border-radius:8px;padding:10px;text-align:center">
                <div style="font-size:0.8em;color:#aaa">{label}</div>
                <div style="font-size:1.4em;font-weight:bold">{data['last']:.2f}</div>
                <div style="color:{chg_color};font-size:0.9em">{arrow} {data['chg']:+.2f} ({data['pct']:+.1f}%)</div>
                <div style="font-size:0.75em;color:#666">{data['desc']}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    # ------------------------------------------------------------------
    # Section 3: 日本株関連ニュース
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🗾 3. 日本株関連ニュース")

    @st.cache_data(ttl=900, show_spinner=False)
    def _load_jp_news():
        import urllib.parse
        query = urllib.parse.quote("日本株 OR 日経平均 OR 東証")
        url = f"https://news.google.com/rss/search?q={query}&hl=ja&gl=JP&ceid=JP:ja"
        entries = _fetch_rss(url)
        rows = []
        for e in entries[:5]:
            try:
                pub = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
                pub_jst = pub.astimezone(JST).strftime("%m/%d %H:%M")
            except Exception:
                pub_jst = "不明"
            rows.append({
                "time": pub_jst,
                "title": e.get("title", ""),
                "link": e.get("link", ""),
            })
        return rows

    with st.spinner("ニュース取得中…"):
        _jp_news = _load_jp_news()

    if not _jp_news:
        st.warning("ニュースを取得できませんでした。")
    else:
        for item in _jp_news:
            st.markdown(
                f'<div style="padding:5px 0;border-bottom:1px solid #222">'
                f'<span style="color:#888;font-size:0.8em">{item["time"]}</span> '
                f'<a href="{item["link"]}" target="_blank" style="color:#7cb8ff">{item["title"]}</a>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ------------------------------------------------------------------
    # Section 4: AIコメント（Claude API）
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🤖 4. AIコメント（Claude）")

    def _build_summary_prompt(cal, mkt, news) -> str:
        lines = ["## マクロイベント（直近）"]
        for r in cal[:5]:
            lines.append(f"- [{r['重要度']}] {r['日時(JST)']} {r['イベント']}")

        lines.append("\n## 市場環境")
        for label, data in mkt.items():
            if data:
                lines.append(f"- {label}: {data['last']:.2f} ({data['arrow']}{data['pct']:+.1f}%)")

        lines.append("\n## 日本株ニュース（直近）")
        for item in news[:3]:
            lines.append(f"- {item['time']} {item['title']}")

        lines.append(
            "\n以上の情報を踏まえて、今夜オーバーナイトポジションを持つ際の"
            "注意点を3点、日本語で簡潔に述べてください。"
        )
        return "\n".join(lines)

    try:
        try:
            _api_key = st.secrets["ANTHROPIC_API_KEY"]
        except Exception:
            import os
            _api_key = os.environ.get("ANTHROPIC_API_KEY", "")

        if not _api_key:
            st.info("ANTHROPIC_API_KEY が設定されていません。Streamlit secrets または環境変数に設定してください。")
        else:
            _ai_btn_col, _ = st.columns([1, 3])
            with _ai_btn_col:
                _run_ai = st.button("💬 AIコメントを生成", use_container_width=True)

            if _run_ai:
                _prompt = _build_summary_prompt(_cal_rows, _mkt, _jp_news)
                with st.spinner("Claude が分析中…"):
                    _client = _anthropic.Anthropic(api_key=_api_key)
                    _msg = _client.messages.create(
                        model="claude-opus-4-6",
                        max_tokens=512,
                        messages=[
                            {
                                "role": "user",
                                "content": _prompt.encode("utf-8").decode("utf-8"),
                            }
                        ],
                    )
                    _ai_text = _msg.content[0].text

                st.markdown(
                    f'<div style="background:#1a1f2e;border-left:4px solid #7cb8ff;'
                    f'border-radius:4px;padding:16px;margin-top:8px;line-height:1.7">'
                    f'{_ai_text.replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True,
                )

    except Exception as _e:
        st.error(f"AI コメント生成エラー: {_e}")
