"""
日本 TOPIX-17 業種ETF の OHLCV データ取得
yfinance 経由で取得し data/raw/jp_etf_ohlc.csv に保存
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import DATA_RAW, DATA_START, DATA_END, JP_TICKERS

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_WAIT_SEC = 5


def fetch_jp_etf(
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """TOPIX-17 業種ETF の OHLCV を yfinance で取得する.

    Parameters
    ----------
    tickers : list[str] | None
        取得対象ティッカー。None の場合 config.JP_TICKERS を使用。
    start : str | None
        取得開始日 (YYYY-MM-DD)。None の場合 config.DATA_START。
    end : str | None
        取得終了日 (YYYY-MM-DD)。None の場合 config.DATA_END (= 直近)。

    Returns
    -------
    pd.DataFrame
        MultiIndex columns: (ticker, field)
        field ∈ {Open, High, Low, Close, Adj Close, Volume}
    """
    tickers = tickers or JP_TICKERS
    start = start or DATA_START
    end = end or DATA_END

    frames: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        df = _download_with_retry(ticker, start, end)
        if df is not None and not df.empty:
            frames[ticker] = df
            logger.info("OK  %s  rows=%d", ticker, len(df))
        else:
            logger.warning("FAILED  %s — skipped", ticker)

    if not frames:
        raise RuntimeError("全ティッカーの取得に失敗しました")

    # MultiIndex columns: (ticker, field)
    combined = pd.concat(frames, axis=1)  # keys = ticker names by default
    combined.index.name = "Date"
    combined.columns.names = ["Ticker", "Field"]

    # 保存
    out_path = _ensure_dir(DATA_RAW) / "jp_etf_ohlc.csv"
    combined.to_csv(out_path)
    logger.info("Saved → %s  shape=%s", out_path, combined.shape)

    return combined


def _download_with_retry(
    ticker: str,
    start: str | None,
    end: str | None,
) -> pd.DataFrame | None:
    """リトライ付きで単一ティッカーをダウンロードする."""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            df: pd.DataFrame = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
            )
            if df is not None and not df.empty:
                # yfinance may return MultiIndex columns for single ticker;
                # flatten to plain columns if needed.
                if isinstance(df.columns, pd.MultiIndex):
                    # yfinance returns (Price_field, Ticker) — drop the Ticker level
                    df.columns = df.columns.droplevel("Ticker")
                return df
        except Exception:
            logger.warning(
                "Attempt %d/%d failed for %s",
                attempt,
                _MAX_RETRIES,
                ticker,
                exc_info=True,
            )
        if attempt < _MAX_RETRIES:
            time.sleep(_RETRY_WAIT_SEC)

    return None


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
    )
    df = fetch_jp_etf()
    print(f"\nShape: {df.shape}")
    print(df.head())
