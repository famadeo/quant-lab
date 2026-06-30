from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import databento as db
import pandas as pd
import polars as pl
from dotenv import load_dotenv

RESEARCH_ROOT = Path("/home/famadeo/research/databento-asset-browser")
sys.path.insert(0, str(RESEARCH_ROOT / "src"))

from databento_asset_browser.continuous_5m import (  # noqa: E402
    MONTHS,
    continuous_from_5m,
)

ROOTS = ["GC", "SI", "HG", "PL", "PA", "ALI"]
DATASET = "GLBX.MDP3"
SCHEMA = "ohlcv-1m"
DEFAULT_START = "2016-06-22T00:00:00Z"
DEFAULT_END = "2026-06-22T00:00:00Z"
DEFAULT_OUTPUT_DIR = RESEARCH_ROOT / "data" / "metals_1m_10y"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull 10 years of Databento GLBX.MDP3 ohlcv-1m data for metals roots."
    )
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--roots", default=",".join(ROOTS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--env-file", type=Path, default=RESEARCH_ROOT / ".env")
    parser.add_argument("--force", action="store_true", help="Re-download existing chunk files.")
    return parser.parse_args()


def utc_timestamp(value: str) -> pd.Timestamp:
    return (
        pd.Timestamp(value).tz_convert("UTC")
        if pd.Timestamp(value).tzinfo
        else pd.Timestamp(value, tz="UTC")
    )


def yearly_chunks(start: str, end: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start_ts = utc_timestamp(start)
    end_ts = utc_timestamp(end)
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    left = start_ts
    while left < end_ts:
        right = min(left + pd.DateOffset(years=1), end_ts)
        chunks.append((left, right))
        left = right
    return chunks


def chunk_name(start: pd.Timestamp, end: pd.Timestamp) -> str:
    left = start.strftime("%Y%m%d")
    right = end.strftime("%Y%m%d")
    return f"{left}_{right}.parquet"


def outright_pattern(root: str) -> str:
    return rf"^{root}[{MONTHS}]\d{{1,2}}$"


def download_chunk(
    client: db.Historical,
    root: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    path: Path,
    force: bool,
) -> pl.DataFrame:
    if path.exists() and not force:
        return pl.read_parquet(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    empty = pl.DataFrame(
        schema={
            "ts_event": pl.Datetime("ns", "UTC"),
            "symbol": pl.Utf8,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        }
    )
    start_arg = start.isoformat().replace("+00:00", "Z")
    end_arg = end.isoformat().replace("+00:00", "Z")
    try:
        data = client.timeseries.get_range(
            dataset=DATASET,
            schema=SCHEMA,
            symbols=[f"{root}.FUT"],
            stype_in="parent",
            start=start_arg,
            end=end_arg,
        )
    except db.BentoClientError as exc:
        if "Could not resolve smart symbols" not in str(exc):
            raise
        data = client.timeseries.get_range(
            dataset=DATASET,
            schema=SCHEMA,
            symbols=[f"{root}.c.0"],
            stype_in="continuous",
            start=start_arg,
            end=end_arg,
        )

    pdf = data.to_df(map_symbols=True).reset_index()
    if pdf.empty:
        empty.write_parquet(path)
        return empty

    keep = [
        col
        for col in ["ts_event", "symbol", "open", "high", "low", "close", "volume"]
        if col in pdf
    ]
    if "ts_event" not in keep or "symbol" not in keep:
        empty.write_parquet(path)
        return empty

    frame = (
        pl.from_pandas(pdf[keep])
        .with_columns(
            pl.col("ts_event").cast(pl.Datetime("ns", "UTC")),
            pl.col("symbol").cast(pl.Utf8),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
        )
        .sort(["symbol", "ts_event"])
    )
    frame.write_parquet(path)
    return frame


def write_manifest(
    output_dir: Path,
    rows: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    manifest_path = output_dir / "manifest.csv"
    new_manifest = pd.DataFrame(rows)
    if manifest_path.exists():
        existing = pd.read_csv(manifest_path)
        manifest = pd.concat([existing, new_manifest], ignore_index=True)
        manifest = manifest.drop_duplicates(subset=["root"], keep="last")
    else:
        manifest = new_manifest
    manifest = manifest.sort_values("root").reset_index(drop=True)
    manifest.to_csv(manifest_path, index=False)

    metadata["roots"] = manifest["root"].dropna().astype(str).tolist()
    metadata["manifest"] = str(manifest_path)
    metadata_path = output_dir / "manifest.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nwrote {manifest_path}", flush=True)
    print(f"wrote {metadata_path}", flush=True)
    print(manifest.to_string(index=False), flush=True)


def to_1m_per_contract(raw: pl.DataFrame, root: str) -> pl.DataFrame:
    bars = raw.filter(pl.col("symbol").str.contains(outright_pattern(root)))
    if bars.is_empty():
        return pl.DataFrame(
            schema={
                "symbol": pl.Utf8,
                "ts": pl.Datetime("ns", "UTC"),
                "close": pl.Float64,
                "volume": pl.Float64,
            }
        )
    return (
        bars.select(
            "symbol",
            pl.col("ts_event").alias("ts"),
            "close",
            "volume",
        )
        .filter(pl.col("close").is_not_null())
        .sort(["symbol", "ts"])
    )


def continuous_from_databento_continuous(raw: pl.DataFrame) -> pl.DataFrame:
    if raw.is_empty():
        return pl.DataFrame(
            schema={
                "ts": pl.Datetime("ns", "UTC"),
                "active": pl.Utf8,
                "cont_logret": pl.Float64,
                "cont_close": pl.Float64,
                "volume": pl.Float64,
                "is_roll": pl.Boolean,
                "cont_logprice": pl.Float64,
            }
        )

    base = (
        raw.select(
            pl.col("ts_event").alias("ts"),
            pl.col("symbol").alias("active"),
            pl.col("close").alias("cont_close"),
            "volume",
        )
        .filter(pl.col("cont_close").is_not_null())
        .sort("ts")
        .with_columns(
            (pl.col("active") != pl.col("active").shift(1)).fill_null(False).alias("is_roll")
        )
    )
    return (
        base.with_columns(
            pl.when(pl.col("is_roll"))
            .then(None)
            .otherwise(pl.col("cont_close").log() - pl.col("cont_close").log().shift(1))
            .alias("cont_logret")
        )
        .with_columns(pl.col("cont_logret").fill_null(0.0).cum_sum().alias("cont_logprice"))
        .select(
            "ts",
            "active",
            "cont_logret",
            "cont_close",
            "volume",
            "is_roll",
            "cont_logprice",
        )
    )


def build_root(
    client: db.Historical,
    root: str,
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]],
    output_dir: Path,
    force: bool,
) -> dict[str, object]:
    chunk_dir = output_dir / "raw_chunks" / root
    raw_dir = output_dir / "raw"
    continuous_dir = output_dir / "continuous"
    raw_dir.mkdir(parents=True, exist_ok=True)
    continuous_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for start, end in chunks:
        path = chunk_dir / chunk_name(start, end)
        frame = download_chunk(client, root, start, end, path, force)
        frames.append(frame)
        print(f"{root} {path.name}: {frame.height:,} rows", flush=True)

    raw = pl.concat(frames, how="vertical").unique().sort(["symbol", "ts_event"])
    raw_path = raw_dir / f"{root}.parquet"
    raw.write_parquet(raw_path)

    bars = to_1m_per_contract(raw, root)
    if not bars.is_empty():
        continuous = continuous_from_5m(bars)
    else:
        continuous = continuous_from_databento_continuous(raw)
    continuous_path = continuous_dir / f"{root}.parquet"
    continuous.write_parquet(continuous_path)

    row: dict[str, object] = {
        "root": root,
        "chunk_count": len(chunks),
        "raw_rows": raw.height,
        "outright_rows": bars.height,
        "continuous_rows": continuous.height,
        "raw_path": str(raw_path),
        "continuous_path": str(continuous_path),
    }
    if raw.height:
        row.update(
            {
                "raw_start": str(raw["ts_event"].min()),
                "raw_end": str(raw["ts_event"].max()),
                "symbol_count": raw["symbol"].n_unique(),
            }
        )
    if continuous.height:
        row.update(
            {
                "continuous_start": str(continuous["ts"].min()),
                "continuous_end": str(continuous["ts"].max()),
                "active_contract_count": continuous["active"].n_unique(),
                "roll_count": int(continuous["is_roll"].sum()),
            }
        )
    return row


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise SystemExit("DATABENTO_API_KEY is not set.")

    roots = [root.strip().upper() for root in args.roots.split(",") if root.strip()]
    chunks = yearly_chunks(args.start, args.end)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    client = db.Historical(api_key)
    manifest_rows = []
    for root in roots:
        print(f"\\n=== {root} ===", flush=True)
        manifest_rows.append(build_root(client, root, chunks, args.output_dir, args.force))

    metadata = {
        "dataset": DATASET,
        "schema": SCHEMA,
        "start": args.start,
        "end": args.end,
        "roots": roots,
        "created_at": datetime.now(UTC).isoformat(),
        "output_dir": str(args.output_dir),
    }
    write_manifest(args.output_dir, manifest_rows, metadata)


if __name__ == "__main__":
    main()
