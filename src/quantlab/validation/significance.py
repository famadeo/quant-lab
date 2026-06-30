"""Significance utilities for guarding strategy Sharpe ratios against luck.

Implements the probabilistic and deflated Sharpe ratio of Bailey and
Lopez de Prado (2014), plus a fixed-universe portfolio aggregation that
avoids inflating headline metrics with a time-varying cross-section.

All Sharpe ratios here are expressed in *per-observation* units (mean over
standard deviation of the return series), not annualized, so that the
sample-size term ``sqrt(n - 1)`` is dimensionally consistent.
"""

from __future__ import annotations

import math
from typing import cast

import pandas as pd
from scipy.stats import norm

# Euler-Mascheroni constant, used for the expected maximum of N draws.
_EULER_MASCHERONI = 0.5772156649015329
# Minimum observations required to estimate a Sharpe standard error.
_MIN_OBSERVATIONS = 2


def probabilistic_sharpe_ratio(
    sharpe: float,
    *,
    n_observations: int,
    benchmark_sharpe: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probability that the true per-observation Sharpe exceeds ``benchmark_sharpe``.

    ``sharpe`` is the observed per-observation Sharpe ratio. ``kurtosis`` is the
    non-excess kurtosis (3.0 for a normal distribution).
    """
    if n_observations < _MIN_OBSERVATIONS:
        raise ValueError("n_observations must be at least 2 to estimate a standard error")

    denominator = math.sqrt(max(1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2, 1e-12))
    z = (sharpe - benchmark_sharpe) * math.sqrt(n_observations - 1) / denominator
    return float(norm.cdf(z))


def expected_max_sharpe(*, n_trials: int, trial_sharpe_variance: float) -> float:
    """Expected maximum per-observation Sharpe across ``n_trials`` independent trials.

    This is the multiple-testing benchmark SR* of the deflated Sharpe ratio. With a
    single trial there is no selection bias and the benchmark is zero.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if trial_sharpe_variance < 0.0:
        raise ValueError("trial_sharpe_variance must be non-negative")
    if n_trials == 1 or trial_sharpe_variance == 0.0:
        return 0.0

    std = math.sqrt(trial_sharpe_variance)
    gamma = _EULER_MASCHERONI
    term = (1.0 - gamma) * norm.ppf(1.0 - 1.0 / n_trials) + gamma * norm.ppf(
        1.0 - 1.0 / (n_trials * math.e)
    )
    return float(std * term)


def deflated_sharpe_ratio(
    sharpe: float,
    *,
    n_observations: int,
    n_trials: int,
    trial_sharpe_variance: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Probabilistic Sharpe ratio deflated by the multiple-testing benchmark SR*."""
    benchmark = expected_max_sharpe(n_trials=n_trials, trial_sharpe_variance=trial_sharpe_variance)
    return probabilistic_sharpe_ratio(
        sharpe,
        n_observations=n_observations,
        benchmark_sharpe=benchmark,
        skew=skew,
        kurtosis=kurtosis,
    )


def fixed_universe_portfolio_returns(pair_returns: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-pair returns with constant 1/N weights over a fixed universe.

    The default pairs aggregation averages only the pairs *observed* at each
    timestamp, so sparse-tail timestamps (few live pairs) carry the same weight
    as dense ones. This divides instead by the full per-method pair count, treating
    a non-observed pair as flat (zero return) for that bar.
    """
    counts = pair_returns.groupby("method")["pair"].nunique()
    universe = pd.DataFrame({"method": counts.index, "universe_size": counts.to_numpy()})
    grouped = pair_returns.groupby(["method", "ts"], as_index=False).agg(
        summed_return=("pair_return", "sum"),
        active_pairs=("active", "sum"),
        observed_pairs=("pair", "nunique"),
        summed_turnover=("turnover", "sum"),
    )
    merged = grouped.merge(universe, on="method", how="left")
    size = merged["universe_size"].astype(float)
    result = merged.assign(
        portfolio_return=merged["summed_return"] / size,
        turnover=merged["summed_turnover"] / size,
    ).drop(columns=["summed_return", "summed_turnover"])
    result = result.sort_values(["method", "ts"]).reset_index(drop=True)
    return cast(pd.DataFrame, result)
