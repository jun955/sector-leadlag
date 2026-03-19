from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger("update_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh market data and derived CSVs for the HF Space.",
    )
    parser.add_argument(
        "--mode",
        choices=("jp", "us", "all"),
        required=True,
        help="Which market data source to refresh before rebuilding derived files.",
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the cloned Space repository.",
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Create a git commit when data files changed.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push the commit to origin. Requires --commit.",
    )
    parser.add_argument(
        "--message",
        help="Override the default git commit message.",
    )
    return parser.parse_args()


def ensure_repo_root(repo_dir: Path) -> Path:
    repo_root = repo_dir.resolve()
    expected = (
        repo_root / "app.py",
        repo_root / "src",
        repo_root / "data",
    )
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Repository root looks incomplete: missing {', '.join(missing)}"
        )
    return repo_root


def configure_import_path(repo_root: Path) -> None:
    root_str = str(repo_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def refresh_market_data(mode: str) -> None:
    from src.data.build_calendar import build_calendar
    from src.data.fetch_jp_etf import fetch_jp_etf
    from src.data.fetch_us_etf import fetch_us_etf
    from src.data.preprocess import preprocess_returns

    if mode in {"jp", "all"}:
        LOGGER.info("Refreshing JP ETF data")
        fetch_jp_etf()

    if mode in {"us", "all"}:
        LOGGER.info("Refreshing US ETF data")
        fetch_us_etf()

    LOGGER.info("Rebuilding processed returns")
    us_returns, jp_returns = preprocess_returns()
    jp_cc = jp_returns.xs("cc", axis=1, level="ReturnType")

    LOGGER.info("Rebuilding US-JP trading calendar map")
    build_calendar(us_returns.index, jp_cc.index)


def validate_outputs(repo_root: Path) -> None:
    # The Streamlit app expects these files and schemas to exist before startup.
    raw_dir = repo_root / "data" / "raw"
    processed_dir = repo_root / "data" / "processed"

    required_raw = [
        raw_dir / "jp_etf_ohlc.csv",
        raw_dir / "us_etf_ohlc.csv",
    ]
    required_processed = [
        processed_dir / "jp_returns.csv",
        processed_dir / "us_returns.csv",
        processed_dir / "us_jp_date_map.csv",
    ]

    for path in required_raw + required_processed:
        if not path.exists():
            raise FileNotFoundError(f"Required output file is missing: {path}")

    us_returns = pd.read_csv(
        processed_dir / "us_returns.csv",
        index_col=0,
        parse_dates=True,
    )
    jp_returns = pd.read_csv(
        processed_dir / "jp_returns.csv",
        header=[0, 1],
        index_col=0,
        parse_dates=True,
    )
    date_map = pd.read_csv(
        processed_dir / "us_jp_date_map.csv",
        parse_dates=["us_date", "jp_next_date"],
    )

    if us_returns.empty:
        raise ValueError("us_returns.csv is empty")
    if jp_returns.empty:
        raise ValueError("jp_returns.csv is empty")
    if date_map.empty:
        raise ValueError("us_jp_date_map.csv is empty")

    return_types = set(jp_returns.columns.get_level_values(1))
    if {"cc", "oc"} - return_types:
        raise ValueError("jp_returns.csv is missing expected return types")

    if us_returns.index.isna().any():
        raise ValueError("us_returns.csv contains null dates")
    if jp_returns.index.isna().any():
        raise ValueError("jp_returns.csv contains null dates")
    if date_map[["us_date", "jp_next_date"]].isna().any().any():
        raise ValueError("us_jp_date_map.csv contains null dates")

    LOGGER.info(
        "Validated outputs: latest US=%s latest JP=%s latest map=(%s -> %s)",
        us_returns.index.max().date(),
        jp_returns.index.max().date(),
        date_map["us_date"].max().date(),
        date_map["jp_next_date"].max().date(),
    )


def run_git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )


def read_git_config(repo_root: Path, key: str) -> str:
    try:
        result = run_git(repo_root, "config", "--get", key)
    except subprocess.CalledProcessError:
        return ""
    return result.stdout.strip()


def ensure_git_identity(repo_root: Path) -> None:
    if not read_git_config(repo_root, "user.name"):
        run_git(repo_root, "config", "user.name", "hf-jobs-bot")
    if not read_git_config(repo_root, "user.email"):
        run_git(
            repo_root,
            "config",
            "user.email",
            "hf-jobs-bot@users.noreply.huggingface.co",
        )


def data_diff_status(repo_root: Path) -> str:
    result = run_git(repo_root, "status", "--short", "--", "data/raw", "data/processed")
    return result.stdout.strip()


def default_commit_message(mode: str) -> str:
    if mode == "jp":
        return "chore: refresh JP market data"
    if mode == "us":
        return "chore: refresh US market data"
    return "chore: refresh market data"


def commit_and_push(repo_root: Path, mode: str, message: str | None, push: bool) -> None:
    ensure_git_identity(repo_root)
    run_git(repo_root, "add", "data/raw", "data/processed")
    run_git(repo_root, "commit", "-m", message or default_commit_message(mode))

    if push:
        LOGGER.info("Pushing updated data to origin")
        run_git(repo_root, "push")


def main() -> int:
    args = parse_args()
    if args.push and not args.commit:
        raise SystemExit("--push requires --commit")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    repo_root = ensure_repo_root(args.repo_dir)
    configure_import_path(repo_root)

    LOGGER.info("Repository root: %s", repo_root)
    LOGGER.info("Refresh mode: %s", args.mode)

    refresh_market_data(args.mode)
    validate_outputs(repo_root)

    diff = data_diff_status(repo_root)
    if not diff:
        LOGGER.info("No changes detected in data/raw or data/processed")
        return 0

    LOGGER.info("Changed files:\n%s", diff)
    if args.commit:
        commit_and_push(repo_root, args.mode, args.message, args.push)
    else:
        LOGGER.info("Changes detected, but --commit was not requested")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        LOGGER.exception("Data refresh failed: %s", exc)
        raise SystemExit(1) from exc
