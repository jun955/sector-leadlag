"""
Visualization utilities for strategy evaluation.

Produces publication-quality charts for cumulative returns, drawdowns,
and factor exposures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Style defaults
_COLORS = {
    "MOM": "#1f77b4",
    "PCA_PLAIN": "#ff7f0e",
    "PCA_SUB": "#2ca02c",
    "DOUBLE": "#d62728",
}

_DEFAULT_FIGSIZE = (12, 6)


def _apply_style(ax: plt.Axes) -> None:
    """Apply a clean, professional style to an axes object."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10, frameon=False)


def _save_or_show(fig: plt.Figure, save_path: str | Path | None) -> None:
    """Save figure to file or display interactively."""
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_cumulative_returns(
    results: dict[str, pd.Series],
    save_path: str | Path | None = None,
) -> None:
    """Plot cumulative returns for all strategies on the same chart.

    Parameters
    ----------
    results : dict[str, pd.Series]
        Strategy name -> daily return series.
    save_path : str | Path | None
        If provided, save the figure to this path instead of displaying.
    """
    fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)

    for name, daily_ret in results.items():
        cumulative = (1.0 + daily_ret).cumprod()
        color = _COLORS.get(name, None)
        ax.plot(
            cumulative.index,
            cumulative.values,
            label=name,
            color=color,
            linewidth=1.2,
        )

    ax.set_title("Cumulative Returns", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Cumulative Return", fontsize=11)
    ax.axhline(y=1.0, color="grey", linestyle="--", linewidth=0.7, alpha=0.5)

    _apply_style(ax)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_drawdown(
    results: dict[str, pd.Series],
    save_path: str | Path | None = None,
) -> None:
    """Plot drawdown over time for all strategies.

    Parameters
    ----------
    results : dict[str, pd.Series]
        Strategy name -> daily return series.
    save_path : str | Path | None
        If provided, save the figure to this path instead of displaying.
    """
    fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE)

    for name, daily_ret in results.items():
        cumulative = (1.0 + daily_ret).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        color = _COLORS.get(name, None)
        ax.fill_between(
            drawdown.index,
            drawdown.values,
            0,
            alpha=0.25,
            color=color,
        )
        ax.plot(
            drawdown.index,
            drawdown.values,
            label=name,
            color=color,
            linewidth=0.8,
        )

    ax.set_title("Drawdown", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Drawdown", fontsize=11)

    _apply_style(ax)
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_factor_exposure(
    regression_results: dict[str, dict],
    save_path: str | Path | None = None,
) -> None:
    """Bar chart of factor betas for each strategy.

    Parameters
    ----------
    regression_results : dict[str, dict]
        Strategy name -> regression result dict (as returned by
        ``evaluation.metrics.factor_regression``).
    save_path : str | Path | None
        If provided, save the figure to this path instead of displaying.
    """
    strategies = list(regression_results.keys())
    if not strategies:
        return

    # Collect all factor names across strategies
    all_factors: list[str] = []
    for res in regression_results.values():
        for f in res.get("betas", {}):
            if f not in all_factors:
                all_factors.append(f)

    n_strategies = len(strategies)
    n_factors = len(all_factors)
    x = np.arange(n_factors)
    bar_width = 0.8 / max(n_strategies, 1)

    fig, ax = plt.subplots(figsize=(_DEFAULT_FIGSIZE[0], 5))

    for i, name in enumerate(strategies):
        betas = regression_results[name].get("betas", {})
        values = [betas.get(f, 0.0) for f in all_factors]
        color = _COLORS.get(name, None)
        ax.bar(
            x + i * bar_width,
            values,
            width=bar_width,
            label=name,
            color=color,
            alpha=0.85,
        )

    ax.set_xticks(x + bar_width * (n_strategies - 1) / 2)
    ax.set_xticklabels(all_factors, fontsize=11)
    ax.set_title("Factor Exposures (Betas)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Beta", fontsize=11)
    ax.axhline(y=0, color="grey", linestyle="-", linewidth=0.7, alpha=0.5)

    _apply_style(ax)
    fig.tight_layout()
    _save_or_show(fig, save_path)
