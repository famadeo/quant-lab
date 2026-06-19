from __future__ import annotations

from itertools import combinations
from typing import cast

import pandas as pd
import polars as pl

from quantlab.pairs.config import PairsDataConfig


def load_continuous_5m_roots(config: PairsDataConfig) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for root in config.roots:
        path = config.root_dir / f"{root}.parquet"
        if not path.exists():
            continue

        lazy = pl.scan_parquet(path).select(
            pl.col(config.timestamp_column).alias("ts"),
            pl.col(config.price_column).alias("log_price"),
            pl.col(config.return_column).alias("source_log_return"),
            pl.col("volume").cast(pl.Float64).alias("volume"),
        )
        if config.start is not None:
            lazy = lazy.filter(pl.col("ts") >= config.start)
        if config.end is not None:
            lazy = lazy.filter(pl.col("ts") <= config.end)

        frame = lazy.collect().sort("ts")
        if frame.is_empty():
            continue

        raw_pandas_frame = frame.to_pandas()
        pandas_frame = pd.DataFrame(
            {
                "ts": raw_pandas_frame["ts"],
                "log_price": raw_pandas_frame["log_price"].astype(float),
                "log_return": raw_pandas_frame["log_price"].astype(float).diff().fillna(0.0),
                "volume": raw_pandas_frame["volume"].astype(float),
            }
        ).dropna()
        if not pandas_frame.empty:
            frames[root] = pandas_frame.reset_index(drop=True)

    return frames


def build_intra_asset_class_pairs(
    asset_classes: dict[str, list[str]],
    available_roots: set[str],
) -> list[tuple[str, str, str]]:
    pairs: list[tuple[str, str, str]] = []
    for asset_class, roots in asset_classes.items():
        available = [root for root in roots if root in available_roots]
        for root_a, root_b in combinations(available, 2):
            pairs.append((asset_class, root_a, root_b))
    return pairs


def align_pair(
    root_a: str,
    root_b: str,
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    left = frames[root_a].rename(
        columns={
            "log_price": "log_price_a",
            "log_return": "log_return_a",
            "volume": "volume_a",
        }
    )
    right = frames[root_b].rename(
        columns={
            "log_price": "log_price_b",
            "log_return": "log_return_b",
            "volume": "volume_b",
        }
    )
    merged = left.merge(right, on="ts", how="inner")
    return cast(pd.DataFrame, merged.sort_values("ts").reset_index(drop=True))
