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

from databento_asset_browser.continuous_5m import _outright_pattern  # noqa: E402

ROOTS = ["GC", "SI", "HG", "PL", "PA", "ALI"]
DATASET = "GLBX.MDP3"
SCHEMA = "trades"
DEFAULT_START = "2025-06-22T00:00:00Z"
DEFAULT_END = "2026-06-22T00:00:00Z"
DEFAULT_OUTPUT_DIR = RESEARCH_ROOT / "data" / "metals_trades_12m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull 12 months of Databento GLBX.MDP3 trades for metals roots."
    )
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--roots", default=",".join(ROOTS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--env-file", type=Path, default=RESEARCH_ROOT / ".env")
    parser.add_argument("--force", action="store_true", help="Re-download existing chunk files.")
    return parser.parse_args()


def utc_timestamp(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_convert("UTC") if ts.tzinfo else pd.Timestamp(value, tz="UTC")


def monthly_chunks(start: str, end: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start_ts = utc_timestamp(start)
    end_ts = utc_timestamp(end)
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    left = start_ts
    while left < end_ts:
        right = min(left + pd.DateOffset(months=1), end_ts)
        chunks.append((left, right))
        left = right
    return chunks


def chunk_name(start: pd.Timestamp, end: pd.Timestamp) -> str:
    left = start.strftime("%Y%m%d")
    right = end.strftime("%Y%m%d")
    return f"{left}_{right}.parquet"


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
    data = client.timeseries.get_range(
        dataset=DATASET,
        schema=SCHEMA,
        symbols=[f"{root}.FUT"],
        stype_in="parent",
        start=start.isoformat().replace("+00:00", "Z"),
        end=end.isoformat().replace("+00:00", "Z"),
    )
    pdf = data.to_df(map_symbols=True).reset_index()
    frame = (
        pl.from_pandas(pdf[["ts_event", "symbol", "price", "size", "side"]])
        .with_columns(
            pl.col("ts_event").cast(pl.Datetime("ns", "UTC")),
            pl.col("symbol").cast(pl.Utf8),
            pl.col("price").cast(pl.Float64),
            pl.col("size").cast(pl.Float64),
            pl.col("side").cast(pl.Utf8),
        )
        .sort(["symbol", "ts_event"])
    )
    frame.write_parquet(path)
    return frame


def build_root(
    client: db.Historical,
    root: str,
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]],
    output_dir: Path,
    force: bool,
) -> dict[str, object]:
    chunk_dir = output_dir / "raw_chunks" / root
    raw_dir = output_dir / "raw"
    outright_dir = output_dir / "outright"
    raw_dir.mkdir(parents=True, exist_ok=True)
    outright_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for start, end in chunks:
        path = chunk_dir / chunk_name(start, end)
        frame = download_chunk(client, root, start, end, path, force)
        frames.append(frame)
        print(f"{root} {path.name}: {frame.height:,} rows", flush=True)

    raw = pl.concat(frames, how="vertical").unique().sort(["symbol", "ts_event"])
    raw_path = raw_dir / f"{root}.parquet"
    raw.write_parquet(raw_path)

    outright = raw.filter(pl.col("symbol").str.contains(_outright_pattern(root)))
    outright_path = outright_dir / f"{root}.parquet"
    outright.write_parquet(outright_path)

    row: dict[str, object] = {
        "root": root,
        "chunk_count": len(chunks),
        "raw_rows": raw.height,
        "outright_rows": outright.height,
        "raw_path": str(raw_path),
        "outright_path": str(outright_path),
    }
    if raw.height:
        row.update(
            {
                "raw_start": str(raw["ts_event"].min()),
                "raw_end": str(raw["ts_event"].max()),
                "symbol_count": raw["symbol"].n_unique(),
            }
        )
    if outright.height:
        row.update(
            {
                "outright_start": str(outright["ts_event"].min()),
                "outright_end": str(outright["ts_event"].max()),
                "outright_symbol_count": outright["symbol"].n_unique(),
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
    chunks = monthly_chunks(args.start, args.end)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    client = db.Historical(api_key)
    manifest_rows = []
    for root in roots:
        print(f"\n=== {root} ===", flush=True)
        manifest_rows.append(build_root(client, root, chunks, args.output_dir, args.force))

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = args.output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    metadata = {
        "dataset": DATASET,
        "schema": SCHEMA,
        "start": args.start,
        "end": args.end,
        "roots": roots,
        "created_at": datetime.now(UTC).isoformat(),
        "output_dir": str(args.output_dir),
        "manifest": str(manifest_path),
    }
    metadata_path = args.output_dir / "manifest.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nwrote {manifest_path}", flush=True)
    print(f"wrote {metadata_path}", flush=True)
    print(manifest.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
