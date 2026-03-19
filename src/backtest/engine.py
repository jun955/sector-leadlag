"""
Backtest engine for Japan-US sector lead-lag strategies.

Handles the date-mapping logic (US signal date -> JP next business day)
and runs all strategy variants through a unified framework.

Supports time-varying US ticker sets: XLC (listed 2018-06-18) and XLRE
(listed 2015-10-07) are included only for dates on or after their
listing.  PCA is run using only tickers available at each date, with
Cfull/C0/V0 computed per ticker configuration and cached.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.config import (
    CFULL_END,
    CFULL_START,
    JP_TICKERS,
    NUM_FACTORS,
    QUANTILE_THRESHOLD,
    REGULARIZATION_LAMBDA,
    ROLLING_WINDOW,
    TEST_END,
    TEST_START,
    US_LATE_LISTING,
)
from src.portfolio.baselines import (
    double_sort_portfolio,
    momentum_signal,
    pca_plain_signal,
)
from src.portfolio.construction import construct_portfolio
from src.signal.prior_subspace import (
    build_prior_subspace,
    compute_cfull,
    get_available_us_tickers,
)
from src.signal.regularized_pca import regularized_pca, rolling_standardize

logger = logging.getLogger(__name__)

# Minimum number of overlapping observations required to estimate Cfull
_MIN_CFULL_ROWS = 250


@dataclass
class BacktestResult:
    """Container for backtest output of a single strategy."""

    daily_returns: pd.Series
    """Daily portfolio returns indexed by JP trade date."""

    cumulative_returns: pd.Series
    """Cumulative (1 + r).cumprod() time series."""

    weights_history: pd.DataFrame
    """Portfolio weights for each rebalance date, columns = JP tickers."""


@dataclass(frozen=True)
class _TickerConfig:
    """Immutable identifier for a US ticker configuration."""

    us_tickers: tuple[str, ...]
    """Sorted tuple of US ticker symbols available in this config."""


@dataclass
class _ConfigCache:
    """Cached PCA artefacts for a single ticker configuration."""

    us_tickers: list[str]
    cfull_corr: np.ndarray
    C0: np.ndarray
    V0: np.ndarray
    combined: np.ndarray          # (T, N_us + N_jp)
    combined_dates: pd.DatetimeIndex
    z_scores: np.ndarray          # (T, N_us + N_jp)
    date_to_idx: dict[pd.Timestamp, int]
    n_us: int


class BacktestEngine:
    """Run backtests for all strategy variants.

    Parameters
    ----------
    jp_oc_returns : pd.DataFrame
        JP open-to-close returns.  index = Date, columns = JP tickers.
    us_cc_returns : pd.DataFrame
        US close-to-close returns.  index = Date, columns = US tickers.
    jp_cc_returns : pd.DataFrame
        JP close-to-close returns.  index = Date, columns = JP tickers.
    date_map : pd.DataFrame
        Mapping from US date to next JP business day.
        columns = [us_date, jp_next_date].
    ff_factors : pd.DataFrame
        Fama-French factor returns.  index = Date, columns include
        MKT, SMB, HML, RF, WML.
    """

    def __init__(
        self,
        jp_oc_returns: pd.DataFrame,
        us_cc_returns: pd.DataFrame,
        jp_cc_returns: pd.DataFrame,
        date_map: pd.DataFrame,
        ff_factors: pd.DataFrame,
    ) -> None:
        self.jp_oc_returns = jp_oc_returns
        self.us_cc_returns = us_cc_returns
        self.jp_cc_returns = jp_cc_returns
        self.date_map = date_map
        self.ff_factors = ff_factors

        # Pre-build the US->JP date lookup
        self._us_to_jp: dict[pd.Timestamp, pd.Timestamp] = {}
        for _, row in date_map.iterrows():
            us_d = pd.Timestamp(row["us_date"])
            jp_d = pd.Timestamp(row["jp_next_date"])
            self._us_to_jp[us_d] = jp_d

        # Per-config cache: _TickerConfig -> _ConfigCache
        self._config_cache: dict[_TickerConfig, _ConfigCache] = {}

    # ------------------------------------------------------------------
    # Time-varying ticker configuration helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_ticker_config(us_date: pd.Timestamp) -> _TickerConfig:
        """Return the ticker configuration key for a given US date.

        Uses only tickers available in the Cfull estimation period
        (2010-2014).  XLC/XLRE are excluded because their Cfull
        would need a different, shorter estimation window, degrading
        the prior subspace quality.
        """
        cfull_end = pd.Timestamp(CFULL_END)
        tickers = [
            t for t in get_available_us_tickers(us_date)
            if t not in US_LATE_LISTING
            or pd.Timestamp(US_LATE_LISTING[t]) <= cfull_end
        ]
        return _TickerConfig(us_tickers=tuple(tickers))

    def _get_or_build_config(self, config: _TickerConfig) -> _ConfigCache:
        """Return cached artefacts for *config*, building them if needed."""
        if config in self._config_cache:
            return self._config_cache[config]

        us_tickers = list(config.us_tickers)
        jp_cols = list(JP_TICKERS)
        window = ROLLING_WINDOW

        logger.info(
            "Building config for %d US tickers: %s",
            len(us_tickers),
            us_tickers,
        )

        # --- Determine the Cfull estimation period for this config ---
        cfull_start = pd.Timestamp(CFULL_START)
        cfull_end = pd.Timestamp(CFULL_END)

        # The latest listing date among tickers in this config determines
        # the earliest date from which all tickers have data.
        latest_listing = cfull_start
        for t in us_tickers:
            if t in US_LATE_LISTING:
                listing_ts = pd.Timestamp(US_LATE_LISTING[t])
                if listing_ts > latest_listing:
                    latest_listing = listing_ts

        effective_start = max(cfull_start, latest_listing)

        # If effective_start > cfull_end the standard window has no data
        # for the late-listed tickers.  Slide the window forward: start
        # from effective_start and extend cfull_end until we have at
        # least _MIN_CFULL_ROWS common observations.
        if effective_start > cfull_end:
            effective_start_str = effective_start.strftime("%Y-%m-%d")
            # Slice all data from effective_start onward and find the
            # date at which we accumulate enough rows.
            us_slice = self.us_cc_returns[us_tickers].loc[effective_start_str:]
            jp_slice = self.jp_cc_returns[jp_cols].loc[effective_start_str:]
            merged_tmp = pd.concat([us_slice, jp_slice], axis=1).dropna()
            if len(merged_tmp) < _MIN_CFULL_ROWS:
                # Use all available data (best-effort)
                logger.warning(
                    "Only %d rows available for Cfull with tickers %s "
                    "(need %d). Using all available data.",
                    len(merged_tmp),
                    us_tickers,
                    _MIN_CFULL_ROWS,
                )
                new_end = merged_tmp.index[-1] if len(merged_tmp) > 0 else effective_start
            else:
                new_end = merged_tmp.index[_MIN_CFULL_ROWS - 1]
            cfull_end = new_end
            cfull_start = effective_start
        else:
            cfull_start = effective_start

        cfull_start_str = cfull_start.strftime("%Y-%m-%d")
        cfull_end_str = cfull_end.strftime("%Y-%m-%d")

        logger.info(
            "  Cfull period: %s to %s (effective start shifted for late listings)",
            cfull_start_str,
            cfull_end_str,
        )

        # --- Compute Cfull and prior subspace ---
        cfull_corr, us_used = compute_cfull(
            self.us_cc_returns,
            self.jp_cc_returns,
            cfull_start=cfull_start_str,
            cfull_end=cfull_end_str,
            us_tickers_available=us_tickers,
        )
        V0, C0 = build_prior_subspace(us_used, jp_cols, cfull_corr)

        logger.info(
            "  Prior subspace ready: %d US + %d JP = %d sectors",
            len(us_used),
            len(jp_cols),
            V0.shape[0],
        )

        # --- Build combined returns matrix for this ticker set ---
        us_data = self.us_cc_returns[us_tickers]
        jp_data = self.jp_cc_returns[jp_cols]
        merged = pd.concat([us_data, jp_data], axis=1).dropna()
        combined = merged.values
        combined_dates = merged.index

        z_scores = rolling_standardize(combined, window=window)
        date_to_idx = {d: i for i, d in enumerate(combined_dates)}

        cache = _ConfigCache(
            us_tickers=us_tickers,
            cfull_corr=cfull_corr,
            C0=C0,
            V0=V0,
            combined=combined,
            combined_dates=combined_dates,
            z_scores=z_scores,
            date_to_idx=date_to_idx,
            n_us=len(us_tickers),
        )
        self._config_cache[config] = cache
        return cache

    def _ensure_all_configs(self) -> None:
        """Pre-build all ticker configs that will be needed.

        Scans US signal dates to discover distinct configs and builds
        them upfront so the per-date loop does not trigger builds.
        """
        test_start = pd.Timestamp(TEST_START)
        configs_seen: set[_TickerConfig] = set()
        for us_date in self._us_to_jp:
            jp_date = self._us_to_jp[us_date]
            if jp_date < test_start:
                continue
            cfg = self._get_ticker_config(us_date)
            if cfg not in configs_seen:
                configs_seen.add(cfg)
                self._get_or_build_config(cfg)

    # ------------------------------------------------------------------
    # Strategy runners
    # ------------------------------------------------------------------

    def _run_pca_strategy(
        self, lam: float, strategy_name: str
    ) -> BacktestResult:
        """Run a PCA-based strategy (SUB or PLAIN) with given lambda.

        Uses time-varying US ticker sets: for each US signal date the
        appropriate ticker configuration is selected and the matching
        combined-returns matrix, z-scores and C0 are used.
        """
        self._ensure_all_configs()

        jp_tickers = list(JP_TICKERS)
        window = ROLLING_WINDOW
        K = NUM_FACTORS

        test_start = pd.Timestamp(TEST_START)
        test_end = pd.Timestamp(TEST_END) if TEST_END else None
        daily_rets: list[tuple[pd.Timestamp, float]] = []
        weights_list: list[tuple[pd.Timestamp, pd.Series]] = []

        # Iterate over US signal dates
        us_signal_dates = sorted(self._us_to_jp.keys())
        for us_date in us_signal_dates:
            jp_date = self._us_to_jp[us_date]
            if jp_date < test_start:
                continue
            if test_end and jp_date > test_end:
                continue

            cfg = self._get_ticker_config(us_date)
            cache = self._get_or_build_config(cfg)

            # Need the US date row in combined returns for signal generation
            if us_date not in cache.date_to_idx:
                continue
            t = cache.date_to_idx[us_date]
            if t < window:
                continue

            z_win = cache.z_scores[t - window : t]  # [t-L, ..., t-1]
            if np.isnan(z_win).any():
                continue

            V_K = regularized_pca(z_win, cache.C0, lam=lam, K=K)

            # Lead-lag signal: US factor scores -> JP predicted z-scores
            n_us = cache.n_us
            V_U = V_K[:n_us, :]    # (N_US, K)
            V_J = V_K[n_us:, :]    # (N_JP, K)

            z_US_t = cache.z_scores[t, :n_us]  # US z-scores at date t
            f_t = V_U.T @ z_US_t                # factor scores from US
            z_hat_J = V_J @ f_t                 # predicted JP z-scores

            signal = pd.Series(z_hat_J, index=jp_tickers)

            weights = construct_portfolio(signal, q=QUANTILE_THRESHOLD)

            # Apply weights to JP OC return on jp_date
            if jp_date not in self.jp_oc_returns.index:
                continue

            oc_ret = self.jp_oc_returns.loc[jp_date, jp_tickers]
            port_ret = (weights * oc_ret).sum()

            daily_rets.append((jp_date, port_ret))
            weights_list.append((jp_date, weights))

        return self._build_result(daily_rets, weights_list, strategy_name)

    def _run_momentum(self) -> BacktestResult:
        """Run the simple momentum strategy."""
        jp_tickers = list(JP_TICKERS)
        mom = momentum_signal(self.jp_cc_returns[jp_tickers])

        test_start = pd.Timestamp(TEST_START)
        test_end = pd.Timestamp(TEST_END) if TEST_END else None
        daily_rets: list[tuple[pd.Timestamp, float]] = []
        weights_list: list[tuple[pd.Timestamp, pd.Series]] = []

        # For MOM, we use JP dates from the date_map (signal from prior day)
        us_signal_dates = sorted(self._us_to_jp.keys())
        for us_date in us_signal_dates:
            jp_date = self._us_to_jp[us_date]
            if jp_date < test_start:
                continue
            if test_end and jp_date > test_end:
                continue

            # Use the most recent available momentum signal up to us_date
            # The signal reflects trailing JP CC returns ending before jp_date
            available = mom.index[mom.index <= us_date]
            if len(available) == 0:
                continue
            signal_date = available[-1]

            signal_row = mom.loc[signal_date].dropna()
            if len(signal_row) < 4:
                continue

            signal = signal_row[signal_row.index.isin(jp_tickers)]
            weights = construct_portfolio(signal, q=QUANTILE_THRESHOLD)

            if jp_date not in self.jp_oc_returns.index:
                continue

            oc_ret = self.jp_oc_returns.loc[jp_date, jp_tickers]
            port_ret = (weights * oc_ret).sum()

            daily_rets.append((jp_date, port_ret))
            weights_list.append((jp_date, weights))

        return self._build_result(daily_rets, weights_list, "MOM")

    def _run_double_sort(self) -> BacktestResult:
        """Run the double-sort (MOM x PCA SUB) strategy.

        Uses time-varying US ticker sets: PCA signal at each date is
        computed with the ticker configuration available on that date.
        """
        self._ensure_all_configs()

        jp_tickers = list(JP_TICKERS)
        mom = momentum_signal(self.jp_cc_returns[jp_tickers])
        window = ROLLING_WINDOW
        K = NUM_FACTORS

        # We need to produce a PCA signal for every date across all
        # configs.  Collect the union of all combined_dates and fill in
        # the PCA signal using the right config per date.
        all_dates_set: set[pd.Timestamp] = set()
        for cache in self._config_cache.values():
            all_dates_set.update(cache.combined_dates.tolist())
        all_dates = pd.DatetimeIndex(sorted(all_dates_set))

        pca_signal = pd.DataFrame(np.nan, index=all_dates, columns=jp_tickers)

        # For efficiency, iterate per config over its date range
        # Build a mapping: date -> config key, for dates after window
        # We process each config's combined_dates in order.
        for cfg, cache in self._config_cache.items():
            n_us = cache.n_us
            for t in range(window, len(cache.combined_dates)):
                z_win = cache.z_scores[t - window + 1 : t + 1]
                if np.isnan(z_win).any():
                    continue

                date_t = cache.combined_dates[t]
                # Only use this config if the date belongs to it
                actual_cfg = self._get_ticker_config(date_t)
                if actual_cfg != cfg:
                    continue

                V_K = regularized_pca(
                    z_win, cache.C0, lam=REGULARIZATION_LAMBDA, K=K
                )
                V_U = V_K[:n_us, :]
                V_J = V_K[n_us:, :]
                z_US_t = cache.z_scores[t, :n_us]
                f_t = V_U.T @ z_US_t
                pca_signal.loc[date_t] = V_J @ f_t

        # Use the double-sort constructor
        test_start = pd.Timestamp(TEST_START)
        test_end = pd.Timestamp(TEST_END) if TEST_END else None
        jp_oc = self.jp_oc_returns[jp_tickers]

        ds_returns = double_sort_portfolio(mom, pca_signal, jp_oc)
        ds_returns = ds_returns[ds_returns.index >= test_start]
        if test_end:
            ds_returns = ds_returns[ds_returns.index <= test_end]

        # Build weight history (not tracked per-ticker for double sort)
        weights_df = pd.DataFrame(
            0.0, index=ds_returns.index, columns=jp_tickers
        )
        cumrets = (1.0 + ds_returns).cumprod()

        return BacktestResult(
            daily_returns=ds_returns,
            cumulative_returns=cumrets,
            weights_history=weights_df,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        strategy_name: str,
        signal_func: Callable | None = None,
        **params,
    ) -> BacktestResult:
        """Run a named strategy and return the backtest result.

        Parameters
        ----------
        strategy_name : str
            One of 'MOM', 'PCA_PLAIN', 'PCA_SUB', 'DOUBLE'.
        signal_func : Callable | None
            Custom signal function (reserved for extensibility).
        **params
            Extra parameters forwarded to the strategy runner.

        Returns
        -------
        BacktestResult
        """
        name_upper = strategy_name.upper()
        logger.info("Running strategy: %s", name_upper)

        if name_upper == "MOM":
            result = self._run_momentum()
        elif name_upper == "PCA_PLAIN":
            result = self._run_pca_strategy(lam=0.0, strategy_name="PCA_PLAIN")
        elif name_upper == "PCA_SUB":
            result = self._run_pca_strategy(
                lam=REGULARIZATION_LAMBDA, strategy_name="PCA_SUB"
            )
        elif name_upper == "DOUBLE":
            result = self._run_double_sort()
        elif signal_func is not None:
            # Extensibility: custom signal function
            result = self._run_custom(signal_func, **params)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        logger.info(
            "  %s done: %d trading days, total return %.2f%%",
            name_upper,
            len(result.daily_returns),
            (result.cumulative_returns.iloc[-1] - 1) * 100
            if len(result.cumulative_returns) > 0
            else 0.0,
        )
        return result

    def run_all_strategies(self) -> dict[str, BacktestResult]:
        """Run all four strategy variants.

        Returns
        -------
        dict[str, BacktestResult]
            Keys: 'MOM', 'PCA_PLAIN', 'PCA_SUB', 'DOUBLE'.
        """
        results: dict[str, BacktestResult] = {}
        for name in ["MOM", "PCA_PLAIN", "PCA_SUB", "DOUBLE"]:
            results[name] = self.run(name)
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_result(
        daily_rets: list[tuple[pd.Timestamp, float]],
        weights_list: list[tuple[pd.Timestamp, pd.Series]],
        strategy_name: str,
    ) -> BacktestResult:
        """Assemble a BacktestResult from raw lists."""
        if not daily_rets:
            logger.warning("No trades for strategy %s", strategy_name)
            empty_ret = pd.Series(dtype=float)
            empty_cum = pd.Series(dtype=float)
            empty_wt = pd.DataFrame()
            return BacktestResult(empty_ret, empty_cum, empty_wt)

        dates, rets = zip(*daily_rets)
        daily_returns = pd.Series(rets, index=pd.DatetimeIndex(dates), name=strategy_name)
        daily_returns = daily_returns.sort_index()

        cumulative_returns = (1.0 + daily_returns).cumprod()

        wt_dates, wt_series = zip(*weights_list)
        weights_history = pd.DataFrame(
            {d: w for d, w in zip(wt_dates, wt_series)}
        ).T
        weights_history.index = pd.DatetimeIndex(wt_dates)
        weights_history = weights_history.sort_index()

        return BacktestResult(daily_returns, cumulative_returns, weights_history)

    def _run_custom(
        self, signal_func: Callable, **params
    ) -> BacktestResult:
        """Run a custom signal function through the standard pipeline."""
        jp_tickers = list(JP_TICKERS)
        test_start = pd.Timestamp(TEST_START)
        daily_rets: list[tuple[pd.Timestamp, float]] = []
        weights_list: list[tuple[pd.Timestamp, pd.Series]] = []

        us_signal_dates = sorted(self._us_to_jp.keys())
        for us_date in us_signal_dates:
            jp_date = self._us_to_jp[us_date]
            if jp_date < test_start:
                continue

            signal = signal_func(us_date, **params)
            if signal is None or signal.empty:
                continue

            weights = construct_portfolio(signal, q=QUANTILE_THRESHOLD)

            if jp_date not in self.jp_oc_returns.index:
                continue

            oc_ret = self.jp_oc_returns.loc[jp_date, jp_tickers]
            port_ret = (weights * oc_ret).sum()

            daily_rets.append((jp_date, port_ret))
            weights_list.append((jp_date, weights))

        return self._build_result(daily_rets, weights_list, "CUSTOM")
