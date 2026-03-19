"""
Baseline strategy signal generators.

Provides momentum and plain-PCA signals, plus a double-sort portfolio
constructor for comparison against the regularized PCA SUB strategy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import JP_TICKERS, ROLLING_WINDOW, NUM_FACTORS
from src.signal.regularized_pca import regularized_pca, rolling_standardize


def momentum_signal(
    jp_cc_returns: pd.DataFrame,
    window: int = ROLLING_WINDOW,
) -> pd.DataFrame:
    """Compute a simple momentum signal from rolling mean JP returns.

    Parameters
    ----------
    jp_cc_returns : pd.DataFrame
        Daily close-to-close returns for JP tickers.
        index = Date, columns = JP tickers.
    window : int
        Rolling window in trading days (default 60).

    Returns
    -------
    pd.DataFrame
        Momentum signal indexed by date, columns = JP tickers.
        Each cell is the trailing *window*-day mean return.
    """
    return jp_cc_returns.rolling(window=window, min_periods=window).mean()


def pca_plain_signal(
    us_cc_returns: pd.DataFrame,
    jp_cc_returns: pd.DataFrame,
    combined_returns: np.ndarray,
    C0: np.ndarray,
    window: int = ROLLING_WINDOW,
    K: int = NUM_FACTORS,
) -> pd.DataFrame:
    """PCA signal with lambda=0 (no regularization toward prior).

    This mirrors the PCA SUB pipeline but sets the regularization
    parameter to zero, so the factor loadings come purely from the
    sample correlation matrix.

    Parameters
    ----------
    us_cc_returns : pd.DataFrame
        US close-to-close returns.
    jp_cc_returns : pd.DataFrame
        JP close-to-close returns.
    combined_returns : np.ndarray
        Stacked returns array (T, N_us + N_jp), already aligned.
    C0 : np.ndarray
        Prior target correlation matrix (passed for interface
        consistency; ignored when lambda=0).
    window : int
        Rolling window length.
    K : int
        Number of principal components to extract.

    Returns
    -------
    pd.DataFrame
        PCA-plain signal for JP tickers, indexed by date.
    """
    n_us = us_cc_returns.shape[1]
    n_jp = jp_cc_returns.shape[1]
    jp_tickers = list(jp_cc_returns.columns)
    dates = jp_cc_returns.index

    T = combined_returns.shape[0]
    signals = pd.DataFrame(np.nan, index=dates[:T], columns=jp_tickers)

    z_scores = rolling_standardize(combined_returns, window=window)

    for t in range(window, T):
        z_win = z_scores[t - window : t]  # [t-L, ..., t-1], exclude t

        if np.isnan(z_win).any():
            continue

        # lambda=0 → plain sample PCA
        V_K = regularized_pca(z_win, C0, lam=0.0, K=K)

        # Lead-lag signal: US factor scores → JP predicted z-scores
        V_U = V_K[:n_us, :]
        V_J = V_K[n_us:, :]
        z_US_t = z_scores[t, :n_us]
        f_t = V_U.T @ z_US_t
        signals.iloc[t] = V_J @ f_t

    return signals


def double_sort_portfolio(
    mom_signal: pd.DataFrame,
    pca_signal: pd.DataFrame,
    jp_oc_returns: pd.DataFrame,
) -> pd.Series:
    """Construct a 2x2 double-sort portfolio from MOM and PCA signals.

    On each date, tickers are split at the median on both the MOM and
    PCA signal dimensions.  The HH (high MOM, high PCA) group is the
    long leg and the LL (low MOM, low PCA) group is the short leg.

    Parameters
    ----------
    mom_signal : pd.DataFrame
        Momentum signal, index = Date, columns = JP tickers.
    pca_signal : pd.DataFrame
        PCA signal, index = Date, columns = JP tickers.
    jp_oc_returns : pd.DataFrame
        JP open-to-close returns, index = Date, columns = JP tickers.

    Returns
    -------
    pd.Series
        Daily strategy returns from the double-sort portfolio.
    """
    common_dates = mom_signal.index.intersection(pca_signal.index)
    common_dates = common_dates.intersection(jp_oc_returns.index)

    daily_returns = pd.Series(np.nan, index=common_dates)

    for date in common_dates:
        mom_row = mom_signal.loc[date].dropna()
        pca_row = pca_signal.loc[date].dropna()

        # Use only tickers present in both signals and returns
        tickers = mom_row.index.intersection(pca_row.index)
        tickers = tickers.intersection(jp_oc_returns.columns)
        if len(tickers) < 4:
            continue

        mom_vals = mom_row[tickers]
        pca_vals = pca_row[tickers]

        mom_median = mom_vals.median()
        pca_median = pca_vals.median()

        high_mom = mom_vals >= mom_median
        low_mom = mom_vals < mom_median
        high_pca = pca_vals >= pca_median
        low_pca = pca_vals < pca_median

        # HH = High MOM & High PCA → Long
        hh_tickers = tickers[high_mom & high_pca]
        # LL = Low MOM & Low PCA → Short
        ll_tickers = tickers[low_mom & low_pca]

        if len(hh_tickers) == 0 or len(ll_tickers) == 0:
            continue

        oc = jp_oc_returns.loc[date]

        long_ret = oc[hh_tickers].mean()
        short_ret = oc[ll_tickers].mean()

        daily_returns[date] = long_ret - short_ret

    return daily_returns.dropna()
