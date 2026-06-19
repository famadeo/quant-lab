from __future__ import annotations

import polars as pl


def moving_average_crossover(
    bars: pl.DataFrame,
    *,
    fast_window: int,
    slow_window: int,
    price_column: str = "close",
) -> pl.DataFrame:
    if fast_window >= slow_window:
        raise ValueError("fast_window must be less than slow_window")

    return bars.with_columns(
        pl.col(price_column)
        .rolling_mean(window_size=fast_window, min_samples=fast_window)
        .over("symbol")
        .alias("fast_ma"),
        pl.col(price_column)
        .rolling_mean(window_size=slow_window, min_samples=slow_window)
        .over("symbol")
        .alias("slow_ma"),
    ).with_columns(
        pl.when(pl.col("fast_ma") > pl.col("slow_ma")).then(1.0).otherwise(0.0).alias("raw_signal")
    )
