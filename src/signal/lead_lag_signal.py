"""
Lead-lag signal generation pipeline.

For each US trading day, uses regularized PCA on joint US-JP sector returns
to extract common factors, then projects US intraday information forward
onto next-day JP sector expected returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import (
    JP_TICKERS,
    NUM_FACTORS,
    REGULARIZATION_LAMBDA,
    ROLLING_WINDOW,
    US_LATE_LISTING,
    US_TICKERS,
)
from src.signal.prior_subspace import (
    build_prior_subspace,
    compute_cfull,
    get_available_us_tickers,
)
from src.signal.regularized_pca import regularized_pca, rolling_standardize


@dataclass
class SignalResult:
    """Container for the signal generation output."""

    signals: pd.DataFrame
    """DataFrame indexed by JP business date, columns = JP tickers."""

    us_date_map: dict[pd.Timestamp, pd.Timestamp]
    """Mapping from US date to corresponding JP date used."""

    ticker_availability: dict[pd.Timestamp, list[str]]
    """Which US tickers were available at each US date."""

    diagnostics: dict[str, object] = field(default_factory=dict)
    """Optional diagnostics (eigenvalues, factor scores, etc.)."""


def _build_combined_returns(
    us_returns: pd.DataFrame,
    jp_returns: pd.DataFrame,
    us_tickers: list[str],
    jp_tickers: list[str],
) -> pd.DataFrame:
    """Concatenate US and JP returns, aligned on date index.

    Missing values for late-listed US tickers are preserved as NaN.
    """
    us_sub = us_returns[us_tickers] if us_tickers else pd.DataFrame(index=us_returns.index)
    jp_sub = jp_returns[jp_tickers]
    combined = pd.concat([us_sub, jp_sub], axis=1)
    return combined


def _get_rolling_window_data(
    combined_returns: pd.DataFrame,
    end_loc: int,
    window: int,
) -> np.ndarray | None:
    """Extract a (window, N) array of returns ending at end_loc - 1.

    Returns None if insufficient non-NaN data is available.
    """
    start_loc = end_loc - window
    if start_loc < 0:
        return None
    chunk = combined_returns.iloc[start_loc:end_loc].values
    # Check for NaN — if any column is entirely NaN, skip
    if np.any(np.all(np.isnan(chunk), axis=0)):
        return None
    # Forward-fill isolated NaNs within the window (rare, from holidays)
    # Use column-wise linear interpolation
    if np.any(np.isnan(chunk)):
        df_tmp = pd.DataFrame(chunk).interpolate(method="linear", axis=0).bfill().ffill()
        chunk = df_tmp.values
    return chunk


def generate_signals(
    us_returns: pd.DataFrame,
    jp_returns: pd.DataFrame,
    date_map: dict[pd.Timestamp, pd.Timestamp],
    V0_func=build_prior_subspace,
    C0_func=compute_cfull,
    cfull_corr: np.ndarray | None = None,
    params: dict | None = None,
) -> SignalResult:
    """Main signal generation pipeline.

    Parameters
    ----------
    us_returns : pd.DataFrame
        Daily returns for US sector ETFs.  Index = dates, columns = tickers.
    jp_returns : pd.DataFrame
        Daily returns for JP sector ETFs.  Index = dates, columns = tickers.
    date_map : dict[pd.Timestamp, pd.Timestamp]
        Maps each US trading date to the next JP business day.
    V0_func : callable
        Function to build the prior subspace (for dependency injection in tests).
    C0_func : callable
        Function to compute the long-term correlation matrix.
    cfull_corr : np.ndarray | None
        Pre-computed long-term correlation matrix.  If None, it will be
        computed from the data using C0_func.
    params : dict | None
        Override default hyperparameters.  Keys:
        ``window`` (int), ``lam`` (float), ``K`` (int).

    Returns
    -------
    SignalResult
        Contains the signal DataFrame and metadata.
    """
    # --- Hyperparameters ---
    p = params or {}
    window: int = p.get("window", ROLLING_WINDOW)
    lam: float = p.get("lam", REGULARIZATION_LAMBDA)
    K: int = p.get("K", NUM_FACTORS)

    jp_tickers = list(JP_TICKERS)

    # --- Pre-compute rolling z-scores for all US and JP tickers ---
    # We will need per-ticker-set z-scores later, but pre-compute full
    # rolling statistics here for efficiency.
    # The actual z-score extraction per window is done inline because
    # the available ticker set changes over time.

    # --- Determine test dates (US dates present in date_map) ---
    us_dates_sorted = sorted(date_map.keys())

    # --- Cache: Cfull and C0/V0 per ticker configuration ---
    # Since the ticker set only changes at XLC and XLRE listing dates,
    # we cache the prior subspace per configuration.
    _prior_cache: dict[tuple[str, ...], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _get_prior(us_avail: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (V0, C0, cfull) for the given US ticker configuration."""
        cache_key = tuple(us_avail)
        if cache_key in _prior_cache:
            return _prior_cache[cache_key]

        if cfull_corr is not None and tuple(us_avail) == tuple(get_available_us_tickers(None)):
            # Use the pre-supplied full cfull
            cfull = cfull_corr
        else:
            cfull, _ = C0_func(us_returns, jp_returns, us_tickers_available=us_avail)

        V0, C0 = V0_func(us_avail, jp_tickers, cfull)
        _prior_cache[cache_key] = (V0, C0, cfull)
        return V0, C0, cfull

    # --- Pre-compute combined returns for each ticker configuration ---
    # Build combined returns DataFrames per configuration and their
    # rolling z-scores.
    _combined_cache: dict[tuple[str, ...], tuple[pd.DataFrame, np.ndarray]] = {}

    def _get_combined_and_zscores(us_avail: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
        cache_key = tuple(us_avail)
        if cache_key in _combined_cache:
            return _combined_cache[cache_key]

        combined = _build_combined_returns(us_returns, jp_returns, us_avail, jp_tickers)
        # Align to only dates where all columns have data
        combined = combined.dropna()

        # Rolling z-scores for the entire combined series
        zscores = rolling_standardize(combined.values, window)

        _combined_cache[cache_key] = (combined, zscores)
        return combined, zscores

    # --- Main loop ---
    signal_rows: list[dict[str, float]] = []
    signal_dates: list[pd.Timestamp] = []
    us_date_mapping: dict[pd.Timestamp, pd.Timestamp] = {}
    ticker_avail: dict[pd.Timestamp, list[str]] = {}

    for us_date in us_dates_sorted:
        jp_date = date_map[us_date]

        # 1. Determine available US tickers at this date
        us_avail = get_available_us_tickers(us_date)
        n_us = len(us_avail)
        n_jp = len(jp_tickers)

        # 2. Get combined returns and z-scores for this configuration
        combined, zscores = _get_combined_and_zscores(us_avail)

        # Find the position of us_date in the combined index
        if us_date not in combined.index:
            # US date not in combined data (possible holiday mismatch)
            continue
        t_loc = combined.index.get_loc(us_date)

        # Need at least `window` prior observations for the rolling window
        if t_loc < window:
            continue

        # Check that z-score at t_loc is valid (not NaN)
        z_t = zscores[t_loc]
        if np.any(np.isnan(z_t)):
            continue

        # 3. Build prior subspace for current ticker config
        V0, C0, _ = _get_prior(us_avail)

        # 4. Extract rolling window of z-scores for regularized PCA
        #    Window = [t_loc - window, ..., t_loc - 1]
        z_window = zscores[t_loc - window : t_loc]
        if np.any(np.isnan(z_window)):
            # Fall back to raw returns window if z-scores have NaN
            raw_window = combined.values[t_loc - window : t_loc]
            if np.any(np.isnan(raw_window)):
                continue
            z_window = raw_window  # Will be re-standardized inside regularized_pca

        # 5. Regularized PCA
        V_K = regularized_pca(z_window, C0, lam=lam, K=K)

        # 6. Split loadings into US and JP blocks
        V_U = V_K[:n_us, :]    # (N_US, K)
        V_J = V_K[n_us:, :]    # (N_JP, K)

        # 7. Factor scores from US z-scores at time t
        zU_t = z_t[:n_us]       # (N_US,)
        f_t = V_U.T @ zU_t      # (K,)

        # 8. JP signal: predicted z-score
        z_hat_J = V_J @ f_t     # (N_JP,)

        # 9. Store the signal mapped to JP date
        row = {ticker: z_hat_J[j] for j, ticker in enumerate(jp_tickers)}
        signal_rows.append(row)
        signal_dates.append(jp_date)
        us_date_mapping[us_date] = jp_date
        ticker_avail[us_date] = us_avail

    # --- Build output DataFrame ---
    if signal_rows:
        signals_df = pd.DataFrame(signal_rows, index=pd.DatetimeIndex(signal_dates, name="jp_date"))
        signals_df = signals_df[jp_tickers]  # enforce column order
        # If multiple US dates map to the same JP date, keep the last signal
        signals_df = signals_df[~signals_df.index.duplicated(keep="last")]
        signals_df.sort_index(inplace=True)
    else:
        signals_df = pd.DataFrame(columns=jp_tickers)
        signals_df.index.name = "jp_date"

    return SignalResult(
        signals=signals_df,
        us_date_map=us_date_mapping,
        ticker_availability=ticker_avail,
    )


if __name__ == "__main__":
    # Smoke test with synthetic data
    np.random.seed(42)

    T = 200
    dates_us = pd.bdate_range("2015-01-01", periods=T, freq="B")
    dates_jp = pd.bdate_range("2015-01-02", periods=T, freq="B")

    # Use tickers available in 2015 (no XLC, yes XLRE from Oct 2015)
    us_avail_2015 = [t for t in US_TICKERS if t != "XLC"]
    n_us = len(us_avail_2015)
    n_jp = len(JP_TICKERS)

    us_ret = pd.DataFrame(
        np.random.randn(T, n_us) * 0.01,
        index=dates_us,
        columns=us_avail_2015,
    )
    jp_ret = pd.DataFrame(
        np.random.randn(T, n_jp) * 0.01,
        index=dates_us,  # use same dates for simplicity
        columns=JP_TICKERS,
    )

    # Simple date map: each US date -> next business day
    dmap = {d: d + pd.offsets.BDay(1) for d in dates_us}

    result = generate_signals(
        us_returns=us_ret,
        jp_returns=jp_ret,
        date_map=dmap,
        params={"window": 60, "lam": 0.9, "K": 3},
    )

    print(f"Signals shape: {result.signals.shape}")
    print(f"Date range: {result.signals.index.min()} to {result.signals.index.max()}")
    print(f"Sample signals (first 3 rows):\n{result.signals.head(3)}")
    print(f"\nUS tickers tracked at {len(result.ticker_availability)} dates")
