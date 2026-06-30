# ruff: noqa: PLR2004
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.validation.significance import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    fixed_universe_portfolio_returns,
    probabilistic_sharpe_ratio,
)


def test_probabilistic_sharpe_ratio_is_half_at_zero_sharpe() -> None:
    assert probabilistic_sharpe_ratio(0.0, n_observations=500) == pytest.approx(0.5)


def test_probabilistic_sharpe_ratio_increases_with_sharpe() -> None:
    low = probabilistic_sharpe_ratio(0.1, n_observations=500)
    high = probabilistic_sharpe_ratio(0.2, n_observations=500)
    assert high > low > 0.5


def test_probabilistic_sharpe_ratio_known_value() -> None:
    # SR=0.1 per obs, n=101 -> z = 1 / sqrt(1.005) = 0.99751, Phi(z) = 0.8407
    psr = probabilistic_sharpe_ratio(0.1, n_observations=101)
    assert psr == pytest.approx(0.8407, abs=5e-4)


def test_probabilistic_sharpe_ratio_penalizes_negative_skew() -> None:
    symmetric = probabilistic_sharpe_ratio(0.1, n_observations=101, skew=0.0)
    left_tailed = probabilistic_sharpe_ratio(0.1, n_observations=101, skew=-1.0)
    assert left_tailed < symmetric


def test_probabilistic_sharpe_ratio_requires_two_observations() -> None:
    with pytest.raises(ValueError, match="n_observations"):
        probabilistic_sharpe_ratio(0.1, n_observations=1)


def test_expected_max_sharpe_zero_with_single_trial() -> None:
    assert expected_max_sharpe(n_trials=1, trial_sharpe_variance=1.0) == 0.0


def test_expected_max_sharpe_increases_with_trials() -> None:
    few = expected_max_sharpe(n_trials=2, trial_sharpe_variance=1.0)
    many = expected_max_sharpe(n_trials=50, trial_sharpe_variance=1.0)
    assert many > few > 0.0


def test_expected_max_sharpe_known_value() -> None:
    # N=2, var=1 -> SR* = gamma * Phi^-1(1 - 1/(2e)) ~= 0.5772 * 0.9006 = 0.520
    assert expected_max_sharpe(n_trials=2, trial_sharpe_variance=1.0) == pytest.approx(
        0.520, abs=1e-2
    )


def test_deflated_sharpe_ratio_equals_psr_with_single_trial() -> None:
    dsr = deflated_sharpe_ratio(0.1, n_observations=101, n_trials=1, trial_sharpe_variance=1.0)
    psr = probabilistic_sharpe_ratio(0.1, n_observations=101)
    assert dsr == pytest.approx(psr)


def test_deflated_sharpe_ratio_below_psr_with_many_trials() -> None:
    dsr = deflated_sharpe_ratio(0.1, n_observations=101, n_trials=50, trial_sharpe_variance=1.0)
    psr = probabilistic_sharpe_ratio(0.1, n_observations=101)
    assert dsr < psr


def test_fixed_universe_portfolio_downweights_sparse_timestamps() -> None:
    # method m has a 2-pair universe; P2 only trades at ts2.
    pair_returns = pd.DataFrame(
        {
            "method": ["m", "m", "m"],
            "ts": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-02"], utc=True),
            "pair": ["P1", "P1", "P2"],
            "pair_return": [0.10, 0.20, 0.40],
            "active": [True, True, True],
            "turnover": [1.0, 1.0, 1.0],
        }
    )

    fixed = fixed_universe_portfolio_returns(pair_returns)

    ts1 = pd.Timestamp("2026-01-01", tz="UTC")
    ts2 = pd.Timestamp("2026-01-02", tz="UTC")
    row1 = fixed.loc[fixed["ts"] == ts1].iloc[0]
    row2 = fixed.loc[fixed["ts"] == ts2].iloc[0]
    # Fixed universe divides by 2 even when only P1 is observed at ts1.
    assert row1["portfolio_return"] == pytest.approx(0.05)
    assert row2["portfolio_return"] == pytest.approx(0.30)
    assert (fixed["universe_size"] == 2).all()


def test_fixed_universe_portfolio_matches_observed_when_complete() -> None:
    pair_returns = pd.DataFrame(
        {
            "method": ["m", "m"],
            "ts": pd.to_datetime(["2026-01-01", "2026-01-01"], utc=True),
            "pair": ["P1", "P2"],
            "pair_return": [0.10, 0.20],
            "active": [True, True],
            "turnover": [1.0, 1.0],
        }
    )

    fixed = fixed_universe_portfolio_returns(pair_returns)

    assert fixed.iloc[0]["portfolio_return"] == pytest.approx(np.mean([0.10, 0.20]))
