"""
Prior subspace construction for regularized PCA.

Builds the 3-factor prior basis V0 (global, country-spread, cyclical/defensive)
and the target correlation matrix C0 from the long-term Cfull estimate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    CFULL_END,
    CFULL_START,
    JP_CYCLICAL,
    JP_DEFENSIVE,
    JP_TICKERS,
    US_CYCLICAL,
    US_DEFENSIVE,
    US_LATE_LISTING,
    US_TICKERS,
)


def _gram_schmidt_step(v: np.ndarray, basis: list[np.ndarray]) -> np.ndarray:
    """Orthogonalize v against each vector in basis, then normalize."""
    u = v.copy()
    for b in basis:
        u = u - np.dot(u, b) * b
    norm = np.linalg.norm(u)
    if norm < 1e-12:
        raise ValueError("Gram-Schmidt produced zero vector — linearly dependent input")
    return u / norm


def get_available_us_tickers(date: pd.Timestamp | str | None = None) -> list[str]:
    """Return US tickers available at the given date.

    Parameters
    ----------
    date : pd.Timestamp | str | None
        If None, return all US tickers.  Otherwise exclude late-listed
        tickers whose listing date is strictly after *date*.
    """
    if date is None:
        return list(US_TICKERS)
    if isinstance(date, str):
        date = pd.Timestamp(date)
    available = []
    for t in US_TICKERS:
        if t in US_LATE_LISTING:
            listing_date = pd.Timestamp(US_LATE_LISTING[t])
            if date < listing_date:
                continue
        available.append(t)
    return available


def build_prior_subspace(
    us_tickers_available: list[str],
    jp_tickers: list[str],
    cfull_corr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the 3-factor prior subspace basis and target correlation matrix.

    Parameters
    ----------
    us_tickers_available : list[str]
        US sector tickers available at the current date (ordered).
    jp_tickers : list[str]
        JP sector tickers (ordered, always all 17).
    cfull_corr : np.ndarray
        Long-term correlation matrix of shape (N, N) where N =
        len(us_tickers_available) + len(jp_tickers).  Rows/columns
        ordered US-first, then JP.

    Returns
    -------
    V0 : np.ndarray, shape (N, 3)
        Prior subspace basis (global, country-spread, cyclical/defensive).
    C0 : np.ndarray, shape (N, N)
        Prior target correlation matrix.
    """
    n_us = len(us_tickers_available)
    n_jp = len(jp_tickers)
    n = n_us + n_jp

    if cfull_corr.shape != (n, n):
        raise ValueError(
            f"cfull_corr shape {cfull_corr.shape} does not match "
            f"expected ({n}, {n}) for {n_us} US + {n_jp} JP tickers"
        )

    # --- v1: global factor (equal weight, normalized) ---
    v1 = np.ones(n) / np.sqrt(n)

    # --- v2: country spread factor ---
    v2_raw = np.empty(n)
    v2_raw[:n_us] = +1.0
    v2_raw[n_us:] = -1.0
    v2 = _gram_schmidt_step(v2_raw, [v1])

    # --- v3: cyclical / defensive factor ---
    v3_raw = np.zeros(n)

    # US block
    for i, ticker in enumerate(us_tickers_available):
        if ticker in US_CYCLICAL:
            v3_raw[i] = +1.0
        elif ticker in US_DEFENSIVE:
            v3_raw[i] = -1.0
        # else neutral = 0

    # JP block
    for j, ticker in enumerate(jp_tickers):
        idx = n_us + j
        if ticker in JP_CYCLICAL:
            v3_raw[idx] = +1.0
        elif ticker in JP_DEFENSIVE:
            v3_raw[idx] = -1.0

    v3 = _gram_schmidt_step(v3_raw, [v1, v2])

    # --- V0 basis matrix ---
    V0 = np.column_stack([v1, v2, v3])  # (N, 3)

    # --- Project Cfull onto V0 to get factor variances ---
    # D0 = diag(V0^T @ Cfull @ V0)
    proj = V0.T @ cfull_corr @ V0  # (3, 3)
    D0 = np.diag(np.diag(proj))     # keep only diagonal

    # --- Reconstruct low-rank covariance and normalize to correlation ---
    C0_raw = V0 @ D0 @ V0.T  # (N, N)
    delta = np.diag(C0_raw)
    inv_sqrt_delta = np.diag(1.0 / np.sqrt(np.maximum(delta, 1e-12)))
    C0 = inv_sqrt_delta @ C0_raw @ inv_sqrt_delta

    return V0, C0


def compute_cfull(
    us_returns: pd.DataFrame,
    jp_returns: pd.DataFrame,
    cfull_start: str = CFULL_START,
    cfull_end: str = CFULL_END,
    us_tickers_available: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Compute the long-term correlation matrix from the estimation period.

    Parameters
    ----------
    us_returns : pd.DataFrame
        Daily returns for US tickers, columns = ticker symbols.
    jp_returns : pd.DataFrame
        Daily returns for JP tickers, columns = ticker symbols.
    cfull_start, cfull_end : str
        Start/end dates for the estimation window (inclusive).
    us_tickers_available : list[str] | None
        If None, determine automatically from data availability in
        the estimation period.  Late-listed tickers (XLC, XLRE) are
        excluded if they have no data in the window.

    Returns
    -------
    cfull_corr : np.ndarray, shape (N, N)
        Long-term correlation matrix (US block first, then JP).
    us_tickers_used : list[str]
        US tickers actually used (excludes late-listed ones missing
        from the estimation period).
    """
    # Slice to estimation period
    us_slice = us_returns.loc[cfull_start:cfull_end]
    jp_slice = jp_returns.loc[cfull_start:cfull_end]

    # Determine which US tickers are available
    if us_tickers_available is None:
        # Exclude tickers that are entirely NaN in the estimation window
        us_tickers_available = [
            t for t in US_TICKERS
            if t in us_slice.columns and us_slice[t].dropna().shape[0] > 0
        ]

    # Select columns in order
    us_data = us_slice[us_tickers_available].dropna(how="all")
    jp_data = jp_slice[list(JP_TICKERS)].dropna(how="all")

    # Align on common dates
    common_dates = us_data.index.intersection(jp_data.index)
    us_data = us_data.loc[common_dates]
    jp_data = jp_data.loc[common_dates]

    # Concatenate and drop any remaining NaN rows
    combined = pd.concat([us_data, jp_data], axis=1).dropna()

    # Compute correlation matrix
    cfull_corr = combined.corr().values

    return cfull_corr, us_tickers_available


if __name__ == "__main__":
    # Quick sanity check with synthetic data
    np.random.seed(42)

    us_avail = get_available_us_tickers("2016-01-01")
    jp = JP_TICKERS
    n = len(us_avail) + len(jp)
    print(f"US tickers at 2016-01-01: {us_avail}")
    print(f"N = {n} ({len(us_avail)} US + {len(jp)} JP)")

    # Synthetic correlation matrix (identity as placeholder)
    cfull_fake = np.eye(n)

    V0, C0 = build_prior_subspace(us_avail, jp, cfull_fake)
    print(f"\nV0 shape: {V0.shape}")
    print(f"C0 shape: {C0.shape}")
    print(f"V0 columns orthogonal? (V0^T V0 ~ I):\n{np.round(V0.T @ V0, 6)}")
    print(f"C0 diagonal (should be 1s): {np.diag(C0)[:5]}...")
