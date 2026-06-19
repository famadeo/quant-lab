from __future__ import annotations

import polars as pl
import pytest

from quantlab.data import make_synthetic_bars
from quantlab.validation import validate_bars


def test_synthetic_bars_have_required_schema() -> None:
    bars = make_synthetic_bars("SYNTH", "2021-01-01", "2021-03-01", seed=42)

    assert {"timestamp", "symbol", "open", "high", "low", "close", "volume"}.issubset(bars.columns)
    assert bars.height > 0
    validate_bars(bars)


def test_validate_bars_rejects_duplicate_timestamps() -> None:
    bars = pl.DataFrame(
        {
            "timestamp": ["2021-01-01", "2021-01-01"],
            "symbol": ["A", "A"],
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [1.0, 1.0],
        }
    ).with_columns(pl.col("timestamp").str.to_datetime())

    with pytest.raises(ValueError, match="duplicate"):
        validate_bars(bars)
