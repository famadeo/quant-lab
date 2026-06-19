from __future__ import annotations

import polars as pl


def validate_bars(bars: pl.DataFrame) -> None:
    required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"bars missing required columns: {sorted(missing)}")

    if bars.is_empty():
        raise ValueError("bars are empty")

    duplicates = bars.group_by(["symbol", "timestamp"]).len().filter(pl.col("len") > 1)
    if not duplicates.is_empty():
        raise ValueError("bars contain duplicate symbol/timestamp rows")

    invalid_prices = bars.filter(
        (pl.col("open") <= 0)
        | (pl.col("high") <= 0)
        | (pl.col("low") <= 0)
        | (pl.col("close") <= 0)
        | (pl.col("high") < pl.col("low"))
    )
    if not invalid_prices.is_empty():
        raise ValueError("bars contain invalid OHLC prices")
