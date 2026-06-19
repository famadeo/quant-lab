from __future__ import annotations

import polars as pl


def add_returns(
    bars: pl.DataFrame,
    *,
    price_column: str = "close",
    return_column: str = "asset_return",
) -> pl.DataFrame:
    return bars.with_columns(
        pl.col(price_column).pct_change().over("symbol").fill_null(0.0).alias(return_column)
    )
