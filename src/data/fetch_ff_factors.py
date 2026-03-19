"""
Fama-French ファクターデータ取得 (日本市場)
Kenneth French Data Library から日次ファクターを取得し
data/raw/ff_factors_jp.csv に保存

取得ファクター: MKT (= Mkt-RF), SMB, HML, RF, WML (momentum)
値は French Library の百分率表記を 100 で割り実数に変換。
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests

from src.config import DATA_RAW, FF_JAPAN_URL, FF_JAPAN_MOM_URL

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = 60


def fetch_ff_factors() -> pd.DataFrame:
    """日本市場 Fama-French 3 ファクター + Carhart モメンタムを取得する.

    Returns
    -------
    pd.DataFrame
        columns = [MKT, SMB, HML, RF, WML]
        index = DatetimeIndex (daily)
        値は実数 (e.g. 0.01 = 1%)
    """
    logger.info("Downloading Japan 3 Factors …")
    ff3 = _download_and_parse_ff(FF_JAPAN_URL, is_momentum=False)
    logger.info("  rows=%d", len(ff3))

    logger.info("Downloading Japan Momentum Factor …")
    mom = _download_and_parse_ff(FF_JAPAN_MOM_URL, is_momentum=True)
    logger.info("  rows=%d", len(mom))

    # 結合 (inner join — 両方にある日付のみ)
    merged = ff3.join(mom, how="inner")

    # 列名を正規化
    merged.columns = ["MKT", "SMB", "HML", "RF", "WML"]

    # 百分率 → 実数
    merged = merged / 100.0

    merged.index.name = "Date"

    # 保存
    out_path = _ensure_dir(DATA_RAW) / "ff_factors_jp.csv"
    merged.to_csv(out_path)
    logger.info("Saved → %s  shape=%s", out_path, merged.shape)

    return merged


def _download_and_parse_ff(url: str, *, is_momentum: bool) -> pd.DataFrame:
    """French Data Library の zip をダウンロードし日次データを抽出する."""
    resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        # zip 内のファイルは通常 1 つ
        csv_name = zf.namelist()[0]
        raw_text = zf.read(csv_name).decode("utf-8", errors="replace")

    return _parse_ff_csv(raw_text, is_momentum=is_momentum)


def _parse_ff_csv(raw_text: str, *, is_momentum: bool) -> pd.DataFrame:
    """French Data Library CSV のヘッダー部分をスキップし日次データを抽出する.

    French CSV の構造:
      - 先頭に説明行がある
      - 日次データセクションは YYYYMMDD 形式の日付で始まる行
      - 月次データセクションが続く場合がある（YYYYMM 形式 = 6桁）
      - 空行やテキスト行でセクション区切り
    """
    lines = raw_text.splitlines()

    data_rows: list[list[str]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            # 日次データ収集中に空行が来たらセクション終了
            if data_rows:
                break
            continue

        parts = stripped.split(",")
        date_candidate = parts[0].strip()

        # 日次データは YYYYMMDD (8桁の数字)
        if date_candidate.isdigit() and len(date_candidate) == 8:
            data_rows.append([p.strip() for p in parts])
        elif data_rows:
            # 日次データ収集中に非データ行が来たら終了
            break

    if not data_rows:
        raise ValueError("日次データ行が見つかりませんでした")

    if is_momentum:
        columns = ["date", "WML"]
    else:
        columns = ["date", "Mkt-RF", "SMB", "HML", "RF"]

    # 列数がデータと合わない場合、データ列数に合わせる
    n_cols = len(data_rows[0])
    if len(columns) != n_cols:
        # ファイルによって列数が異なる場合がある
        # 先頭を date とし、残りは自動命名
        columns = ["date"] + [f"col{i}" for i in range(1, n_cols)]
        logger.warning(
            "列数不一致: expected=%d, actual=%d → 自動命名",
            len(columns),
            n_cols,
        )

    df = pd.DataFrame(data_rows, columns=columns)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date")

    # 数値変換
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

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
    df = fetch_ff_factors()
    print(f"\nShape: {df.shape}")
    print(df.head(10))
    print(df.tail(5))
