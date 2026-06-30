from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.metals_flow.data_quality import (
    align_continuous_marks_to_bars,
    bar_log_returns_with_validity,
    summarize_continuous_frame,
    summarize_trade_frame,
    validate_trade_frame,
)


def test_validate_trade_frame_rejects_missing_required_columns() -> None:
    frame = pd.DataFrame({"ts_event": ["2026-01-01T00:00:00Z"], "price": [2000.0]})

    with pytest.raises(ValueError, match="missing required columns"):
        validate_trade_frame(frame, root="GC")


def test_validate_trade_frame_rejects_invalid_aggressor_side() -> None:
    frame = pd.DataFrame(
        {
            "ts_event": ["2026-01-01T00:00:00Z"],
            "symbol": ["GCM6"],
            "price": [2000.0],
            "size": [1],
            "side": ["X"],
        }
    )

    with pytest.raises(ValueError, match="invalid aggressor sides"):
        validate_trade_frame(frame, root="GC")


def test_summarize_trade_frame_counts_duplicates_and_side_notional() -> None:
    expected_rows = 3
    expected_symbols = 2
    frame = pd.DataFrame(
        {
            "ts_event": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:01:00Z",
            ],
            "symbol": ["GCM6", "GCM6", "GCQ6"],
            "price": [2000.0, 2000.0, 2010.0],
            "size": [1, 1, 2],
            "side": ["B", "B", "A"],
        }
    )

    summary = summarize_trade_frame(frame, root="GC", multiplier=100.0)

    assert summary.rows == expected_rows
    assert summary.symbol_count == expected_symbols
    assert summary.duplicate_rows == 1
    assert summary.notional == pytest.approx(802_000.0)
    assert summary.buy_notional_share == pytest.approx(400_000.0 / 802_000.0)
    assert summary.sell_notional_share == pytest.approx(402_000.0 / 802_000.0)
    assert summary.neutral_notional_share == 0.0


def test_summarize_continuous_frame_counts_rolls_switches_and_duplicates() -> None:
    expected_active_contracts = 2
    frame = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:01:00Z",
                    "2026-01-01T00:01:00Z",
                    "2026-01-01T00:02:00Z",
                ],
                utc=True,
            ),
            "active": ["GCM6", "GCM6", "GCQ6", "GCQ6"],
            "cont_close": [2000.0, 2001.0, 2002.0, 2004.0],
            "cont_logprice": [0.0, 0.0005, 0.001, 0.002],
            "is_roll": [False, False, True, False],
        }
    )

    summary = summarize_continuous_frame(frame, root="GC")

    assert summary.active_contracts == expected_active_contracts
    assert summary.active_switches == 1
    assert summary.roll_rows == 1
    assert summary.duplicate_timestamps == 1
    assert summary.max_abs_log_return == pytest.approx(np.log(2004.0 / 2002.0))


def test_align_continuous_marks_to_bars_masks_stale_and_roll_adjacent_rows() -> None:
    bars = pd.DataFrame(
        {
            "end_ts": pd.to_datetime(
                [
                    "2026-01-01T00:00:30Z",
                    "2026-01-01T00:01:30Z",
                    "2026-01-01T00:02:30Z",
                    "2026-01-01T02:00:00Z",
                ],
                utc=True,
            )
        }
    )
    continuous = {
        "GC": pd.DataFrame(
            {
                "ts": pd.to_datetime(
                    [
                        "2026-01-01T00:00:00Z",
                        "2026-01-01T00:01:00Z",
                        "2026-01-01T00:02:00Z",
                    ],
                    utc=True,
                ),
                "active": ["GCM6", "GCQ6", "GCQ6"],
                "cont_close": [2000.0, 2001.0, 2002.0],
                "cont_logprice": [0.0, 0.001, 0.002],
                "is_roll": [False, True, False],
            }
        )
    }

    panel = align_continuous_marks_to_bars(
        continuous,
        bars,
        ("GC",),
        max_staleness_seconds=60.0,
        roll_cooldown_bars=1,
    )

    validity = panel["price_validity"]
    assert validity["GC_fresh"].tolist() == [True, True, True, False]
    assert validity["roll_invalid"].tolist() == [True, True, True, False]
    assert validity["valid_price_mask"].tolist() == [False, False, False, False]
    assert panel["log_prices"]["GC"].isna().all()


def test_bar_log_returns_with_validity_requires_two_valid_price_marks() -> None:
    log_prices = pd.DataFrame({"GC": [np.log(2000.0), np.log(2001.0), np.log(2003.0)]})
    valid = pd.Series([True, False, True])

    returns = bar_log_returns_with_validity(log_prices, valid)

    assert np.isnan(returns["GC"].iloc[0])
    assert np.isnan(returns["GC"].iloc[1])
    assert np.isnan(returns["GC"].iloc[2])
