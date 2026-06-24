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
from databento.common.error import BentoError
from dotenv import load_dotenv

RESEARCH_ROOT = Path("/home/famadeo/research/databento-asset-browser")
sys.path.insert(0, str(RESEARCH_ROOT / "src"))

from databento_asset_browser.continuous_5m import _outright_pattern  # noqa: E402

ROOTS = ["GC", "SI", "HG", "PL", "PA"]
DATASET = "GLBX.MDP3"
SCHEMA = "mbp-1"
DEFAULT_START = "2026-05-24T00:00:00Z"
DEFAULT_END = "2026-06-23T00:00:00Z"
DEFAULT_OUTPUT_DIR = RESEARCH_ROOT / "data" / "metals_mbp1_30d"
MAX_DATABENTO_RETRIES = 3

KEEP_COLUMNS = [
    "ts_recv",
    "ts_event",
    "symbol",
    "instrument_id",
    "action",
    "side",
    "depth",
    "price",
    "size",
    "bid_px_00",
    "ask_px_00",
    "bid_sz_00",
    "ask_sz_00",
    "bid_ct_00",
    "ask_ct_00",
    "flags",
    "sequence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull Databento GLBX.MDP3 mbp-1 data for liquid metals roots."
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


def daily_chunks(start: str, end: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start_ts = utc_timestamp(start)
    end_ts = utc_timestamp(end)
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    left = start_ts
    while left < end_ts:
        right = min(left + pd.DateOffset(days=1), end_ts)
        chunks.append((left, right))
        left = right
    return chunks


def chunk_name(start: pd.Timestamp, end: pd.Timestamp) -> str:
    left = start.strftime("%Y%m%d")
    right = end.strftime("%Y%m%d")
    return f"{left}_{right}.parquet"


def empty_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ts_recv": pl.Datetime("ns", "UTC"),
            "ts_event": pl.Datetime("ns", "UTC"),
            "symbol": pl.Utf8,
            "instrument_id": pl.UInt32,
            "action": pl.Utf8,
            "side": pl.Utf8,
            "depth": pl.UInt8,
            "price": pl.Float64,
            "size": pl.UInt32,
            "bid_px_00": pl.Float64,
            "ask_px_00": pl.Float64,
            "bid_sz_00": pl.UInt32,
            "ask_sz_00": pl.UInt32,
            "bid_ct_00": pl.UInt32,
            "ask_ct_00": pl.UInt32,
            "flags": pl.UInt8,
            "sequence": pl.UInt32,
        }
    )


def normalize_frame(pdf: pd.DataFrame, root: str) -> pl.DataFrame:
    if pdf.empty:
        return empty_frame()
    keep = [col for col in KEEP_COLUMNS if col in pdf.columns]
    if "ts_event" not in keep or "symbol" not in keep:
        return empty_frame()

    frame = pl.from_pandas(pdf[keep])
    for col in ["ts_recv", "ts_event"]:
        if col in frame.columns:
            frame = frame.with_columns(pl.col(col).cast(pl.Datetime("ns", "UTC")))
    for col in ["symbol", "action", "side"]:
        if col in frame.columns:
            frame = frame.with_columns(pl.col(col).cast(pl.Utf8))
    for col in ["price", "size", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]:
        if col in frame.columns:
            frame = frame.with_columns(pl.col(col).cast(pl.Float64))

    frame = frame.filter(pl.col("symbol").str.contains(_outright_pattern(root)))
    if frame.is_empty():
        return empty_frame()
    return frame.select([col for col in KEEP_COLUMNS if col in frame.columns]).sort(
        ["symbol", "ts_event", "sequence"]
    )


def download_chunk(
    client: db.Historical,
    root: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    path: Path,
    force: bool,
) -> dict[str, object]:
    if path.exists() and not force:
        frame = pl.read_parquet(path)
        return {
            "root": root,
            "chunk": path.name,
            "path": str(path),
            "rows": frame.height,
            "status": "cached",
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    start_arg = start.isoformat().replace("+00:00", "Z")
    end_arg = end.isoformat().replace("+00:00", "Z")
    data = None
    for attempt in range(1, MAX_DATABENTO_RETRIES + 1):
        try:
            data = client.timeseries.get_range(
                dataset=DATASET,
                schema=SCHEMA,
                symbols=[f"{root}.FUT"],
                stype_in="parent",
                start=start_arg,
                end=end_arg,
            )
            break
        except BentoError as exc:
            if attempt == MAX_DATABENTO_RETRIES:
                raise
            message = (
                f"{root} {path.name}: retry {attempt}/{MAX_DATABENTO_RETRIES} "
                f"after Databento error: {exc}"
            )
            print(
                message,
                flush=True,
            )

    if data is None:
        raise RuntimeError(f"No Databento response for {root} {path.name}.")

    pdf = data.to_df(map_symbols=True).reset_index()
    raw_rows = len(pdf)
    frame = normalize_frame(pdf, root)
    frame.write_parquet(path)
    return {
        "root": root,
        "chunk": path.name,
        "start": start_arg,
        "end": end_arg,
        "path": str(path),
        "raw_rows": raw_rows,
        "rows": frame.height,
        "status": "downloaded",
    }


def write_manifest(
    output_dir: Path,
    rows: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    manifest_path = output_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    metadata_path = output_dir / "manifest.json"
    metadata["manifest"] = str(manifest_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}", flush=True)
    print(f"wrote {metadata_path}", flush=True)


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file)
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise SystemExit("DATABENTO_API_KEY is not set.")

    roots = [root.strip().upper() for root in args.roots.split(",") if root.strip()]
    chunks = daily_chunks(args.start, args.end)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    client = db.Historical(api_key)
    manifest_rows: list[dict[str, object]] = []
    for root in roots:
        print(f"\n=== {root} ===", flush=True)
        for start, end in chunks:
            path = args.output_dir / "outright_chunks" / root / chunk_name(start, end)
            row = download_chunk(client, root, start, end, path, args.force)
            manifest_rows.append(row)
            print(
                f"{root} {path.name}: {row.get('rows', 0):,} outright rows "
                f"({row.get('raw_rows', 'cached')} raw) [{row['status']}]",
                flush=True,
            )
            write_manifest(
                args.output_dir,
                manifest_rows,
                {
                    "dataset": DATASET,
                    "schema": SCHEMA,
                    "start": args.start,
                    "end": args.end,
                    "roots": roots,
                    "created_at": datetime.now(UTC).isoformat(),
                    "output_dir": str(args.output_dir),
                    "filter": "outright symbols only",
                },
            )


if __name__ == "__main__":
    main()
