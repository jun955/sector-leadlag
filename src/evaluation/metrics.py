"""
Performance metrics and factor regression analysis.

Computes annualized return, risk, Sharpe-like ratio, maximum drawdown,
and runs Fama-French factor regressions with Newey-West standard errors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def compute_metrics(daily_returns: pd.Series) -> dict:
    """Compute standard performance metrics from a daily return series.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily portfolio returns (not cumulative).

    Returns
    -------
    dict
        AR : float
            Annualized return (mean * 252).
        RISK : float
            Annualized risk (std * sqrt(252)).
        RR : float
            Risk-return ratio (AR / RISK).  NaN if RISK is zero.
        MDD : float
            Maximum drawdown as a positive fraction (e.g. 0.15 = 15 %).
        TOTAL_RETURN : float
            Cumulative total return ((1+r).prod() - 1).
    """
    if daily_returns.empty:
        return {
            "AR": np.nan,
            "RISK": np.nan,
            "RR": np.nan,
            "MDD": np.nan,
            "TOTAL_RETURN": np.nan,
        }

    mean_ret = daily_returns.mean()
    std_ret = daily_returns.std(ddof=1)

    ar = mean_ret * 252
    risk = std_ret * np.sqrt(252)
    rr = ar / risk if risk > 1e-12 else np.nan

    # Maximum drawdown
    cumulative = (1.0 + daily_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    mdd = -drawdown.min()  # positive value

    total_return = cumulative.iloc[-1] - 1.0

    return {
        "AR": ar,
        "RISK": risk,
        "RR": rr,
        "MDD": mdd,
        "TOTAL_RETURN": total_return,
    }


def factor_regression(
    daily_returns: pd.Series,
    ff_factors: pd.DataFrame,
    n_factors: int = 3,
) -> dict:
    """Run an OLS factor regression with Newey-West standard errors.

    Model (3-factor):
        R_t = alpha + beta_MKT * MKT_t + beta_SMB * SMB_t
              + beta_HML * HML_t + epsilon_t

    Model (4-factor, n_factors=4):
        Adds + beta_WML * WML_t

    Parameters
    ----------
    daily_returns : pd.Series
        Daily excess portfolio returns.
    ff_factors : pd.DataFrame
        Fama-French factor returns with columns MKT, SMB, HML, RF,
        and optionally WML.
    n_factors : int
        3 or 4.  If 4, the WML (momentum) factor is included.

    Returns
    -------
    dict
        alpha : float
            Annualized alpha (daily alpha * 252).
        alpha_tstat : float
            t-statistic for alpha.
        betas : dict[str, float]
            Factor betas keyed by factor name.
        beta_tstats : dict[str, float]
            t-statistics for each beta.
        adj_r2 : float
            Adjusted R-squared.
    """
    factor_cols = ["MKT", "SMB", "HML"]
    if n_factors >= 4 and "WML" in ff_factors.columns:
        factor_cols.append("WML")

    # Align dates
    common = daily_returns.index.intersection(ff_factors.index)
    if len(common) == 0:
        return _empty_regression_result(factor_cols)

    y = daily_returns.loc[common].values.astype(float)
    X = ff_factors.loc[common, factor_cols].values.astype(float)

    # Add constant for alpha
    X_const = sm.add_constant(X)

    # Newey-West lag selection: int(4 * (T/100)^(2/9))
    T = len(y)
    nw_lag = int(4 * (T / 100) ** (2.0 / 9.0))
    nw_lag = max(1, nw_lag)

    model = sm.OLS(y, X_const, missing="drop")
    result = model.fit(cov_type="HAC", cov_kwds={"maxlags": nw_lag})

    alpha_daily = result.params[0]
    alpha_ann = alpha_daily * 252
    alpha_tstat = result.tvalues[0]

    betas = {}
    beta_tstats = {}
    for i, col in enumerate(factor_cols):
        betas[col] = result.params[i + 1]
        beta_tstats[col] = result.tvalues[i + 1]

    return {
        "alpha": alpha_ann,
        "alpha_tstat": alpha_tstat,
        "betas": betas,
        "beta_tstats": beta_tstats,
        "adj_r2": result.rsquared_adj,
    }


def summary_table(
    results: dict[str, pd.Series],
    ff_factors: pd.DataFrame,
    n_factors: int = 3,
) -> pd.DataFrame:
    """Create a summary comparison table for all strategies.

    Mirrors the structure of the paper's Table 2.

    Parameters
    ----------
    results : dict[str, pd.Series]
        Strategy name -> daily return series.
    ff_factors : pd.DataFrame
        Fama-French factors for regression.
    n_factors : int
        Number of factors for the regression (3 or 4).

    Returns
    -------
    pd.DataFrame
        Rows = strategies, columns = metrics.
    """
    rows = []

    for name, daily_ret in results.items():
        metrics = compute_metrics(daily_ret)
        reg = factor_regression(daily_ret, ff_factors, n_factors=n_factors)

        row = {
            "Strategy": name,
            "AR (%)": metrics["AR"] * 100,
            "Risk (%)": metrics["RISK"] * 100,
            "RR": metrics["RR"],
            "MDD (%)": metrics["MDD"] * 100,
            "Total Return (%)": metrics["TOTAL_RETURN"] * 100,
            "Alpha (%, ann.)": reg["alpha"] * 100,
            "Alpha t-stat": reg["alpha_tstat"],
            "Adj R2": reg["adj_r2"],
        }

        # Add factor betas
        for factor in reg.get("betas", {}):
            row[f"Beta_{factor}"] = reg["betas"][factor]
            row[f"t({factor})"] = reg["beta_tstats"][factor]

        rows.append(row)

    table = pd.DataFrame(rows).set_index("Strategy")
    return table


def _empty_regression_result(factor_cols: list[str]) -> dict:
    """Return a regression result dict filled with NaN."""
    return {
        "alpha": np.nan,
        "alpha_tstat": np.nan,
        "betas": {c: np.nan for c in factor_cols},
        "beta_tstats": {c: np.nan for c in factor_cols},
        "adj_r2": np.nan,
    }
