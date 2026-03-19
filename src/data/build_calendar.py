"""
US-JP 日付マッピング構築

CRITICAL LOGIC:
  各 US 営業日 t に対して、t の **翌日以降** で最も近い JP 営業日を紐付ける。
  （US 金曜 → 通常 JP 月曜、JP 月曜が祝日なら JP 火曜 …）
  複数の US 日付が同一 JP 日付にマッピングされる場合、最も遅い US 日付のみ保持。

出力: data/processed/us_jp_date_map.csv
  columns = [us_date, jp_next_date]
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_PROCESSED

logger = logging.getLogger(__name__)


def build_calendar(
    us_dates: pd.DatetimeIndex | np.ndarray,
    jp_dates: pd.DatetimeIndex | np.ndarray,
) -> pd.DataFrame:
    """US 営業日→次の JP 営業日のマッピングを構築する.

    Parameters
    ----------
    us_dates : array-like of datetime
        米国市場の取引日（ソート済み）
    jp_dates : array-like of datetime
        日本市場の取引日（ソート済み）

    Returns
    -------
    pd.DataFrame
        columns = [us_date, jp_next_date]
        複数 US 日→同一 JP 日の重複は最新 US 日のみ保持
    """
    us_dates = pd.DatetimeIndex(us_dates).sort_values()
    jp_dates = pd.DatetimeIndex(jp_dates).sort_values()

    jp_arr = jp_dates.values  # numpy datetime64 array for searchsorted

    mappings: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    for us_d in us_dates:
        # us_d より厳密に後ろの最初の JP 営業日を探す
        # searchsorted with side='right' は us_d 以下の要素の次の位置を返す
        idx = jp_arr.searchsorted(us_d.to_datetime64(), side="right")
        if idx < len(jp_arr):
            jp_next = pd.Timestamp(jp_arr[idx])
            mappings.append((us_d, jp_next))
        else:
            # US 日付以降に JP 営業日が存在しない → スキップ
            logger.debug("No JP date after %s — skipped", us_d.date())

    df = pd.DataFrame(mappings, columns=["us_date", "jp_next_date"])

    # 同一 jp_next_date に複数 US 日付がマッピングされる場合、
    # 最も遅い（=直近の）US 日付のみ保持
    df = df.sort_values("us_date")
    df = df.drop_duplicates(subset="jp_next_date", keep="last").reset_index(drop=True)

    logger.info(
        "Calendar built: %d US dates → %d unique mappings", len(us_dates), len(df)
    )

    # 保存
    out_path = _ensure_dir(DATA_PROCESSED) / "us_jp_date_map.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved → %s", out_path)

    return df


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# ------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
    )

    # 保存済み OHLC データからインデックスを読み込んで構築
    from src.config import DATA_RAW

    us_path = DATA_RAW / "us_etf_ohlc.csv"
    jp_path = DATA_RAW / "jp_etf_ohlc.csv"

    if not us_path.exists() or not jp_path.exists():
        print("先に fetch_us_etf / fetch_jp_etf を実行してください")
        raise SystemExit(1)

    # MultiIndex CSV の先頭2行がヘッダーなので header=[0,1]
    us_df = pd.read_csv(us_path, header=[0, 1], index_col=0, parse_dates=True)
    jp_df = pd.read_csv(jp_path, header=[0, 1], index_col=0, parse_dates=True)

    cal = build_calendar(us_df.index, jp_df.index)
    print(f"\nMappings: {len(cal)}")
    print(cal.head(10))
    print("…")
    print(cal.tail(5))
