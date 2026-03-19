"""
リターン計算の前処理

US:  Close-to-Close リターン (Adj Close ベース)
JP:  Close-to-Close リターン + Open-to-Close リターン (分割調整済み)

出力:
  data/processed/us_returns.csv   — columns = tickers, index = Date
  data/processed/jp_returns.csv   — MultiIndex columns: (ticker, return_type)
                                    return_type ∈ {cc, oc}
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import DATA_RAW, DATA_PROCESSED

logger = logging.getLogger(__name__)


def preprocess_returns() -> tuple[pd.DataFrame, pd.DataFrame]:
    """保存済み OHLC から日次リターンを計算する.

    Returns
    -------
    (us_returns, jp_returns)
        us_returns: index=Date, columns=ticker, values=cc_return
        jp_returns: index=Date, MultiIndex columns=(ticker, return_type)
                    return_type ∈ {cc, oc}
    """
    # ---- US ----
    us_ohlc = _load_ohlc(DATA_RAW / "us_etf_ohlc.csv")
    us_returns = _compute_us_returns(us_ohlc)
    logger.info("US returns  shape=%s", us_returns.shape)

    # ---- JP ----
    jp_ohlc = _load_ohlc(DATA_RAW / "jp_etf_ohlc.csv")
    jp_returns = _compute_jp_returns(jp_ohlc)
    logger.info("JP returns  shape=%s", jp_returns.shape)

    # ---- 保存 ----
    out_dir = _ensure_dir(DATA_PROCESSED)

    us_out = out_dir / "us_returns.csv"
    us_returns.to_csv(us_out)
    logger.info("Saved → %s", us_out)

    jp_out = out_dir / "jp_returns.csv"
    jp_returns.to_csv(jp_out)
    logger.info("Saved → %s", jp_out)

    return us_returns, jp_returns


def _load_ohlc(path: Path) -> pd.DataFrame:
    """MultiIndex columns の OHLC CSV を読み込む."""
    if not path.exists():
        raise FileNotFoundError(f"{path} が見つかりません。先にデータ取得を実行してください。")
    df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    df.index.name = "Date"
    return df


def _compute_us_returns(ohlc: pd.DataFrame) -> pd.DataFrame:
    """US: Adj Close ベースの Close-to-Close リターン.

    ret_t = adj_close_t / adj_close_{t-1} - 1
    """
    tickers = ohlc.columns.get_level_values(0).unique()
    frames: dict[str, pd.Series] = {}

    for ticker in tickers:
        adj_close = ohlc[(ticker, "Adj Close")].astype(float)
        ret = adj_close / adj_close.shift(1) - 1.0
        frames[ticker] = ret

    returns = pd.DataFrame(frames)
    returns.index.name = "Date"

    # 先頭行は NaN (前日がないため)
    return returns


def _compute_jp_returns(ohlc: pd.DataFrame) -> pd.DataFrame:
    """JP: Close-to-Close リターン + Open-to-Close リターン.

    adj_ratio = adj_close / close
    adj_open  = open * adj_ratio

    cc_return = adj_close_t / adj_close_{t-1} - 1
    oc_return = adj_close_t / adj_open_t - 1
    """
    tickers = ohlc.columns.get_level_values(0).unique()
    pieces: dict[tuple[str, str], pd.Series] = {}

    for ticker in tickers:
        close = ohlc[(ticker, "Close")].astype(float)
        adj_close = ohlc[(ticker, "Adj Close")].astype(float)
        open_price = ohlc[(ticker, "Open")].astype(float)

        adj_ratio = adj_close / close
        adj_open = open_price * adj_ratio

        cc_ret = adj_close / adj_close.shift(1) - 1.0
        oc_ret = adj_close / adj_open - 1.0

        pieces[(ticker, "cc")] = cc_ret
        pieces[(ticker, "oc")] = oc_ret

    returns = pd.DataFrame(pieces)
    returns.columns = pd.MultiIndex.from_tuples(
        returns.columns, names=["Ticker", "ReturnType"]
    )
    returns.index.name = "Date"

    return returns


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
    )
    us_ret, jp_ret = preprocess_returns()

    print("\n=== US Returns ===")
    print(f"Shape: {us_ret.shape}")
    print(us_ret.head())

    print("\n=== JP Returns ===")
    print(f"Shape: {jp_ret.shape}")
    print(jp_ret.head())
