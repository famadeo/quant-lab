from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl

from quantlab.config import DataConfig
from quantlab.validation.checks import validate_bars

REQUIRED_BAR_COLUMNS = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]


def load_bars(config: DataConfig) -> pl.DataFrame:
    if config.source == "synthetic":
        bars = make_synthetic_bars(
            symbol=config.symbol,
            start=config.start,
            end=config.end,
            seed=config.seed,
        )
    elif config.source == "csv":
        assert config.path is not None
        bars = pl.read_csv(config.path, try_parse_dates=True)
    elif config.source == "parquet":
        assert config.path is not None
        bars = pl.read_parquet(config.path)
    else:
        raise ValueError(f"unsupported data source: {config.source}")

    bars = _normalize_columns(
        bars,
        timestamp_column=config.timestamp_column,
        price_column=config.price_column,
        symbol=config.symbol,
    )
    validate_bars(bars)
    return bars


def make_synthetic_bars(symbol: str, start: str, end: str, seed: int = 7) -> pl.DataFrame:
    dates = pd.bdate_range(start=start, end=end, inclusive="both")
    if dates.empty:
        raise ValueError("synthetic date range produced no business days")

    rng = np.random.default_rng(seed)
    seasonal = 0.0002 * np.sin(np.linspace(0, 8 * np.pi, len(dates)))
    shock = rng.normal(loc=0.00015, scale=0.01, size=len(dates))
    returns = seasonal + shock
    close = 100 * np.cumprod(1 + returns)
    open_ = close * (1 + rng.normal(0, 0.0015, len(dates)))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.0, 0.004, len(dates)))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.0, 0.004, len(dates)))
    volume = rng.integers(100_000, 500_000, len(dates))

    return pl.DataFrame(
        {
            "timestamp": [datetime.combine(date.date(), datetime.min.time()) for date in dates],
            "symbol": symbol,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _normalize_columns(
    bars: pl.DataFrame,
    *,
    timestamp_column: str,
    price_column: str,
    symbol: str,
) -> pl.DataFrame:
    rename_map: dict[str, str] = {}
    if timestamp_column != "timestamp":
        rename_map[timestamp_column] = "timestamp"
    if price_column != "close":
        rename_map[price_column] = "close"
    if rename_map:
        bars = bars.rename(rename_map)

    if "symbol" not in bars.columns:
        bars = bars.with_columns(pl.lit(symbol).alias("symbol"))

    for column in ["open", "high", "low"]:
        if column not in bars.columns:
            bars = bars.with_columns(pl.col("close").alias(column))
    if "volume" not in bars.columns:
        bars = bars.with_columns(pl.lit(0.0).alias("volume"))

    return (
        bars.select(REQUIRED_BAR_COLUMNS)
        .with_columns(
            pl.col("timestamp").cast(pl.Datetime),
            pl.col("symbol").cast(pl.Utf8),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
        )
        .sort(["symbol", "timestamp"])
    )
