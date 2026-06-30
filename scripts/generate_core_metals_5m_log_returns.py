from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTINUOUS_DIR = Path(
    "/home/famadeo/research/databento-asset-browser/data/metals_1m_10y/continuous"
)
OUTPUT_DIR = REPO_ROOT / "experiments" / "HYP-0041-core-metals-5m-log-returns"

ROOTS = ["GC", "SI", "HG", "PL", "PA"]
FILL_LEADING_RETURNS_WITH = 0.0


def load_root_5m_observed(root: str) -> pd.DataFrame:
    path = CONTINUOUS_DIR / f"{root}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)

    frame = (
        pl.scan_parquet(path)
        .select("ts", "active", "cont_close", "volume", "cont_logprice")
        .with_columns(pl.col("ts").dt.truncate("5m").alias("bar_ts"))
        .group_by("bar_ts")
        .agg(
            [
                pl.col("cont_logprice").sort_by("ts").last().alias("log_price"),
                pl.col("cont_close").sort_by("ts").last().alias("close"),
                pl.col("active").sort_by("ts").last().alias("active_contract"),
                pl.col("volume").sum().alias("volume"),
                pl.col("ts").max().alias("last_1m_ts"),
                pl.len().alias("obs_1m"),
            ]
        )
        .rename({"bar_ts": "ts"})
        .sort("ts")
        .collect()
        .to_pandas()
    )
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    frame["last_1m_ts"] = pd.to_datetime(frame["last_1m_ts"], utc=True)
    frame["root"] = root
    frame["observed_gap_minutes"] = frame["ts"].diff().dt.total_seconds() / 60.0
    frame["log_return_5m_raw"] = frame["log_price"].diff()
    frame["had_observed_5m_bar"] = True
    return frame[
        [
            "root",
            "ts",
            "log_return_5m_raw",
            "log_price",
            "close",
            "active_contract",
            "volume",
            "obs_1m",
            "observed_gap_minutes",
            "last_1m_ts",
            "had_observed_5m_bar",
        ]
    ]


def align_root_to_common_grid(
    root: str,
    observed: pd.DataFrame,
    common_ts: pd.Series,
) -> pd.DataFrame:
    aligned = pd.DataFrame({"ts": common_ts}).merge(
        observed.drop(columns="root"),
        on="ts",
        how="left",
        sort=True,
    )
    aligned["root"] = root
    aligned["had_observed_5m_bar"] = aligned["had_observed_5m_bar"].eq(True)

    fill_level_cols = ["log_price", "close", "active_contract", "last_1m_ts"]
    aligned[fill_level_cols] = aligned[fill_level_cols].ffill()
    aligned["volume"] = aligned["volume"].fillna(0.0)
    aligned["obs_1m"] = aligned["obs_1m"].fillna(0).astype("int64")
    aligned["last_observed_5m_ts"] = aligned["ts"].where(aligned["had_observed_5m_bar"]).ffill()
    aligned["minutes_since_observed_5m"] = (
        (aligned["ts"] - aligned["last_observed_5m_ts"]).dt.total_seconds() / 60.0
    ).fillna(0.0)
    aligned["was_price_forward_filled"] = ~aligned["had_observed_5m_bar"]
    aligned["log_return_5m"] = aligned["log_price"].diff().fillna(FILL_LEADING_RETURNS_WITH)
    return aligned[
        [
            "root",
            "ts",
            "log_return_5m",
            "log_return_5m_raw",
            "log_price",
            "close",
            "active_contract",
            "volume",
            "obs_1m",
            "observed_gap_minutes",
            "last_1m_ts",
            "had_observed_5m_bar",
            "was_price_forward_filled",
            "last_observed_5m_ts",
            "minutes_since_observed_5m",
        ]
    ]


def build_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observed_frames = []
    inventory_rows = []
    for root in ROOTS:
        print(f"Generating 5m returns for {root}", flush=True)
        data = load_root_5m_observed(root)
        observed_frames.append(data)
        inventory_rows.append(
            {
                "root": root,
                "observed_rows_5m": len(data),
                "valid_raw_returns": data["log_return_5m_raw"].notna().sum(),
                "first_ts": data["ts"].min(),
                "last_ts": data["ts"].max(),
                "median_obs_1m_per_5m": data["obs_1m"].median(),
                "max_observed_gap_minutes": data["observed_gap_minutes"].max(),
            }
        )

    common_ts = (
        pd.concat([data["ts"] for data in observed_frames], ignore_index=True)
        .drop_duplicates()
        .sort_values(ignore_index=True)
    )
    frames = [
        align_root_to_common_grid(root, data, common_ts)
        for root, data in zip(ROOTS, observed_frames, strict=True)
    ]
    long = pd.concat(frames, ignore_index=True).sort_values(["root", "ts"])
    wide = (
        long.pivot(index="ts", columns="root", values="log_return_5m")
        .reindex(columns=ROOTS)
        .sort_index()
        .ffill()
        .fillna(FILL_LEADING_RETURNS_WITH)
        .reset_index()
    )
    inventory = pd.DataFrame(inventory_rows)
    inventory["aligned_rows_5m"] = len(common_ts)
    fill_counts = long.groupby("root")["was_price_forward_filled"].sum()
    inventory["price_forward_filled_rows"] = inventory["root"].map(fill_counts).astype("int64")
    return long, wide, inventory


def write_report(inventory: pd.DataFrame) -> None:
    lines = [
        "# Core Metals 5-Minute Log Returns",
        "",
        "Generated from the 10-year continuous 1-minute series.",
        "",
        "Method:",
        "",
        "- Sample each metal to 5-minute bars using the last continuous log price in each bucket.",
        "- Align all metals to the common observed 5-minute timestamp grid.",
        "- Forward-fill missing price, close, active-contract, and last-observation fields "
        "within each metal.",
        "- Compute `log_return_5m = log_price_t - log_price_{t-1}` from the forward-filled "
        f"log prices. Leading returns are set to `{FILL_LEADING_RETURNS_WITH:g}`.",
        "- Missing aligned bars therefore produce zero return until the next observed price "
        "update, instead of repeating the prior return.",
        "- The long file preserves observed-only returns in `log_return_5m_raw` and marks "
        "aligned bars whose price was forward-filled with `was_price_forward_filled`.",
        "",
        "## Inventory",
        "",
        inventory.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Files",
        "",
        "- `core_metals_5m_log_returns_long.parquet`",
        "- `core_metals_5m_log_returns_wide.parquet`",
        "- `core_metals_5m_log_returns_wide.csv.gz`",
        "- `data_inventory.csv`",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    long, wide, inventory = build_outputs()
    long.to_parquet(OUTPUT_DIR / "core_metals_5m_log_returns_long.parquet", index=False)
    wide.to_parquet(OUTPUT_DIR / "core_metals_5m_log_returns_wide.parquet", index=False)
    wide.to_csv(OUTPUT_DIR / "core_metals_5m_log_returns_wide.csv.gz", index=False)
    inventory.to_csv(OUTPUT_DIR / "data_inventory.csv", index=False)
    write_report(inventory)
    print(f"Long rows: {len(long):,}", flush=True)
    print(f"Wide rows: {len(wide):,}", flush=True)
    print(f"Wrote {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
