"""
US-JP 日付マッピング構築

CRITICAL LOGIC:
  各 US 営業日 t に対して、t の **翌日以降** で最も近い JP 営業日を紐付ける。
  （US 金曜 → 通常 JP 月曜、JP 月曜が祝日なら JP 火曜 …）
  複数の US 日付が同一 JP 日付にマッピングされる場合、最も遅い US 日付のみ保持。

  JP データがまだ存在しない最新 US 日付（例: 当日の US 終値取得後・JP 市場未クローズ）
  については、土日・日本祝日を除いた翌 JP 営業日を計算して補完する。

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


def _next_jp_bday(us_date: pd.Timestamp) -> pd.Timestamp:
    """JP データがない US 日付に対して、翌 JP 営業日を計算する.

    土曜・日曜をスキップし、jpholiday が利用可能であれば日本の祝日もスキップする。
    jpholiday 未インストールの場合は土日スキップのみで動作する（フォールバック）。
    """
    try:
        import jpholiday  # type: ignore

        def _is_holiday(d: pd.Timestamp) -> bool:
            return bool(jpholiday.is_holiday(d.date()))

    except ImportError:
        logger.warning("jpholiday not installed — JP holiday check skipped (weekends only)")

        def _is_holiday(d: pd.Timestamp) -> bool:
            return False

    candidate = us_date + pd.Timedelta(days=1)
    while candidate.weekday() >= 5 or _is_holiday(candidate):
        candidate += pd.Timedelta(days=1)
    return candidate


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
        日本市場の取引日（ソート済み）。既知の JP 取引日リスト。
        JP データがない最新 US 日付は _next_jp_bday() で補完される。

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
        # us_d より厳密に後ろの最初の既知 JP 営業日を探す
        idx = jp_arr.searchsorted(us_d.to_datetime64(), side="right")
        if idx < len(jp_arr):
            jp_next = pd.Timestamp(jp_arr[idx])
        else:
            # JP データがまだ存在しない（当日 US 終値取得済み・JP 未クローズ等）
            # 土日・祝日をスキップして翌 JP 営業日を計算する
            jp_next = _next_jp_bday(us_d)
            logger.info(
                "JP data not yet available after %s — estimated next JP bday: %s",
                us_d.date(),
                jp_next.date(),
            )
        mappings.append((us_d, jp_next))

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
