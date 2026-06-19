from __future__ import annotations

from datetime import datetime

import polars as pl

from quantlab.backtests import run_long_only_backtest
from quantlab.config import BacktestConfig


def test_backtest_lags_position_by_one_bar() -> None:
    frame = pl.DataFrame(
        {
            "timestamp": [
                datetime(2021, 1, 1),
                datetime(2021, 1, 4),
                datetime(2021, 1, 5),
                datetime(2021, 1, 6),
            ],
            "symbol": ["A", "A", "A", "A"],
            "close": [100.0, 101.0, 102.0, 103.0],
            "raw_signal": [0.0, 1.0, 1.0, 0.0],
        }
    )

    result, _ = run_long_only_backtest(
        frame,
        BacktestConfig(fee_bps=0.0, slippage_bps=0.0, signal_lag=1),
    )

    assert result["position"].to_list() == [0.0, 0.0, 1.0, 1.0]


def test_backtest_applies_turnover_costs() -> None:
    frame = pl.DataFrame(
        {
            "timestamp": [
                datetime(2021, 1, 1),
                datetime(2021, 1, 4),
                datetime(2021, 1, 5),
            ],
            "symbol": ["A", "A", "A"],
            "close": [100.0, 100.0, 100.0],
            "raw_signal": [1.0, 0.0, 1.0],
        }
    )

    result, metrics = run_long_only_backtest(
        frame,
        BacktestConfig(fee_bps=10.0, slippage_bps=0.0, signal_lag=1),
    )

    assert result["cost"].sum() > 0
    assert metrics.total_return < 0
