from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl


def mbp1_availability(mbp1_dir: Path) -> pd.DataFrame:
    manifest = mbp1_dir.parent / "manifest.csv"
    if manifest.exists():
        frame = pd.read_csv(manifest)
        if frame.empty:
            return frame
        return (
            frame.groupby("root", as_index=False)
            .agg(
                chunks=("rows", "size"),
                rows=("rows", "sum"),
                raw_rows=("raw_rows", "sum"),
                first_start=("start", "min"),
                last_end=("end", "max"),
                downloaded=("status", lambda values: bool((values == "downloaded").all())),
            )
            .sort_values("root")
        )

    rows = []
    for root_dir in sorted(mbp1_dir.glob("*")):
        if not root_dir.is_dir():
            continue
        files = sorted(root_dir.glob("*.parquet"))
        rows.append(
            {
                "root": root_dir.name,
                "chunks": len(files),
                "rows": np.nan,
                "raw_rows": np.nan,
                "first_start": np.nan,
                "last_end": np.nan,
                "downloaded": bool(files),
            }
        )
    return pd.DataFrame(rows)


def top_of_book_minute_features(
    mbp1_dir: Path,
    roots: tuple[str, ...],
    *,
    start: str,
    end: str,
    every: str = "1m",
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    start_ts = pd.Timestamp(start).to_pydatetime()
    end_ts = pd.Timestamp(end).to_pydatetime()
    for root in roots:
        root_dir = mbp1_dir / root
        files = sorted(str(path) for path in root_dir.glob("*.parquet"))
        if not files:
            continue
        lazy = (
            pl.scan_parquet(files)
            .filter(
                (pl.col("ts_event") >= pl.lit(start_ts)) & (pl.col("ts_event") < pl.lit(end_ts))
            )
            .select(
                "ts_event",
                "bid_px_00",
                "ask_px_00",
                "bid_sz_00",
                "ask_sz_00",
                "bid_ct_00",
                "ask_ct_00",
            )
            .with_columns(
                mid=((pl.col("bid_px_00") + pl.col("ask_px_00")) / 2.0),
                spread=(pl.col("ask_px_00") - pl.col("bid_px_00")),
                top_depth=(pl.col("bid_sz_00") + pl.col("ask_sz_00")),
                top_imbalance=(
                    (pl.col("bid_sz_00") - pl.col("ask_sz_00"))
                    / (pl.col("bid_sz_00") + pl.col("ask_sz_00"))
                ),
            )
            .sort("ts_event")
            .group_by_dynamic("ts_event", every=every)
            .agg(
                pl.col("mid").last(),
                pl.col("spread").last(),
                pl.col("top_depth").last(),
                pl.col("top_imbalance").last(),
                pl.col("bid_ct_00").last().alias("bid_count"),
                pl.col("ask_ct_00").last().alias("ask_count"),
            )
            .with_columns(pl.lit(root).alias("root"))
        )
        frames.append(lazy.collect().to_pandas())
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["root", "ts_event"])


def merge_book_features_to_bars(
    book_features: pd.DataFrame,
    bars: pd.DataFrame,
    roots: tuple[str, ...],
) -> pd.DataFrame:
    if book_features.empty:
        return pd.DataFrame(index=bars.index)
    bar_times = pd.DataFrame(
        {"bar_index": bars.index, "end_ts": pd.to_datetime(bars["end_ts"], utc=True)}
    )
    outputs = []
    for root in roots:
        root_book = book_features[book_features["root"] == root].copy()
        if root_book.empty:
            continue
        root_book["ts_event"] = pd.to_datetime(root_book["ts_event"], utc=True)
        merged = pd.merge_asof(
            bar_times.sort_values("end_ts"),
            root_book.sort_values("ts_event"),
            left_on="end_ts",
            right_on="ts_event",
            direction="backward",
        ).set_index("bar_index")
        keep = ["mid", "spread", "top_depth", "top_imbalance", "bid_count", "ask_count"]
        merged = merged.loc[:, keep].add_prefix(f"{root}_book_")
        outputs.append(merged)
    if not outputs:
        return pd.DataFrame(index=bars.index)
    return pd.concat(outputs, axis=1).reindex(bars.index)
