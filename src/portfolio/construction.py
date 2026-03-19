"""
Portfolio construction from cross-sectional signals.

Converts signal scores into dollar-neutral long/short portfolio weights
using quantile-based sorting.
"""

from __future__ import annotations

import pandas as pd


def construct_portfolio(signals: pd.Series, q: float = 0.3) -> pd.Series:
    """Construct a dollar-neutral long/short portfolio from signal scores.

    Parameters
    ----------
    signals : pd.Series
        Signal values indexed by ticker.  Higher values indicate a
        stronger long signal.
    q : float
        Quantile threshold.  Top *q* fraction of tickers is assigned to
        the long leg; bottom *q* fraction to the short leg.
        Default 0.3 (top/bottom 30 %).

    Returns
    -------
    pd.Series
        Portfolio weights indexed by ticker.
        - Long tickers  :  +1/n_long  each
        - Short tickers :  -1/n_short each
        - Others        :  0
        Satisfies sum(w) = 0 and sum(|w|) = 2.
    """
    if signals.empty:
        return pd.Series(dtype=float)

    n = len(signals)
    n_long = max(1, int(round(n * q)))
    n_short = max(1, int(round(n * q)))

    ranked = signals.rank(method="first", ascending=True)

    weights = pd.Series(0.0, index=signals.index)

    # Bottom n_short by rank → short leg
    short_mask = ranked <= n_short
    # Top n_long by rank → long leg
    long_mask = ranked > (n - n_long)

    weights[long_mask] = 1.0 / n_long
    weights[short_mask] = -1.0 / n_short

    return weights
