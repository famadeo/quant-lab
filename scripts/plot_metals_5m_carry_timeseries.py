from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

MONTHS = {
    "F": 1,
    "G": 2,
    "H": 3,
    "J": 4,
    "K": 5,
    "M": 6,
    "N": 7,
    "Q": 8,
    "U": 9,
    "V": 10,
    "X": 11,
    "Z": 12,
}
MAX_CONTRACT_MONTHS_OUT = 120
MAX_ANCHOR_MONTHS_OUT = 4
ROOTS = ["GC", "SI", "HG", "PL", "PA"]
COLORS = {
    "GC": "#b8860b",
    "SI": "#6f7f8f",
    "HG": "#b15a2a",
    "PL": "#2f7d8c",
    "PA": "#7a4e9b",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = Path("/home/famadeo/research/databento-asset-browser/data/metals_1m_10y/raw")
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "notebooks" / "explorations" / "assets" / "2026-06-25_metals_5m_carry_timeseries"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 5-minute annualized metals futures carry.")
    parser.add_argument("--start", default="2023-06-22")
    parser.add_argument("--end", default="2026-06-22")
    parser.add_argument("--target-months", type=int, default=3)
    parser.add_argument("--min-volume", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def contract_months_out(symbol: str, date_value: object) -> float | None:
    match = re.match(r"^([A-Z]+)([FGHJKMNQUVXZ])(\d)$", symbol)
    if not match:
        return None
    month = MONTHS[match.group(2)]
    year_digit = int(match.group(3))
    date = pd.Timestamp(date_value)
    candidates = []
    for year in range(date.year - 1, date.year + 11):
        if year % 10 == year_digit:
            months_out = (year - date.year) * 12 + (month - date.month)
            if 0 <= months_out <= MAX_CONTRACT_MONTHS_OUT:
                candidates.append(months_out)
    return min(candidates) if candidates else None


def choose_daily_pairs(
    daily: pd.DataFrame, *, target_months: int, min_volume: float
) -> pd.DataFrame:
    rows = []
    for date, group in daily.groupby("date", sort=True):
        contract_date = date
        day = group.copy()
        day["months_out"] = [contract_months_out(symbol, contract_date) for symbol in day["symbol"]]
        day = day.dropna(subset=["months_out"])
        day = day[(day["volume"] >= min_volume) & (day["close"] > 0)]
        if day.empty:
            continue

        early = day[day["months_out"] <= MAX_ANCHOR_MONTHS_OUT]
        if early.empty:
            continue
        anchor = early.sort_values(["volume", "months_out"], ascending=[False, True]).iloc[0]
        far = day[day["months_out"] - anchor["months_out"] >= target_months]
        if far.empty:
            continue
        far = far.sort_values(["months_out", "volume"], ascending=[True, False]).iloc[0]
        rows.append(
            {
                "date": date,
                "anchor": anchor["symbol"],
                "far": far["symbol"],
                "anchor_months_out": anchor["months_out"],
                "far_months_out": far["months_out"],
                "months_from_anchor": far["months_out"] - anchor["months_out"],
                "anchor_daily_volume": anchor["volume"],
                "far_daily_volume": far["volume"],
            }
        )
    return pd.DataFrame(rows)


def load_root_carry(
    root: str,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    target_months: int,
    min_volume: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = RAW_DIR / f"{root}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)

    lazy = pl.scan_parquet(path).filter((pl.col("ts_event") >= start) & (pl.col("ts_event") < end))
    daily = (
        lazy.sort(["symbol", "ts_event"])
        .with_columns(pl.col("ts_event").dt.date().alias("date"))
        .group_by(["date", "symbol"])
        .agg(
            [
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
            ]
        )
        .collect()
        .to_pandas()
    )
    pairs = choose_daily_pairs(daily, target_months=target_months, min_volume=min_volume)
    if pairs.empty:
        return pd.DataFrame(), pairs
    pairs["date"] = pd.to_datetime(pairs["date"]).dt.date

    bars = (
        lazy.with_columns(
            [
                pl.col("ts_event").dt.truncate("5m").alias("ts"),
                pl.col("ts_event").dt.date().alias("date"),
            ]
        )
        .group_by(["date", "ts", "symbol"])
        .agg(
            [
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume_5m"),
            ]
        )
        .collect()
        .to_pandas()
    )
    bars["date"] = pd.to_datetime(bars["date"]).dt.date

    pair_columns = [
        "date",
        "anchor",
        "far",
        "anchor_months_out",
        "far_months_out",
        "months_from_anchor",
        "anchor_daily_volume",
        "far_daily_volume",
    ]
    anchor = bars.merge(
        pairs[pair_columns], left_on=["date", "symbol"], right_on=["date", "anchor"]
    )
    anchor = anchor.rename(columns={"close": "anchor_close", "volume_5m": "anchor_volume_5m"})
    far = bars.merge(pairs[pair_columns], left_on=["date", "symbol"], right_on=["date", "far"])
    far = far.rename(columns={"close": "far_close", "volume_5m": "far_volume_5m"})

    carry = anchor.merge(
        far[
            [
                "date",
                "ts",
                "anchor",
                "far",
                "months_from_anchor",
                "far_close",
                "far_volume_5m",
            ]
        ],
        on=["date", "ts", "anchor", "far", "months_from_anchor"],
        how="inner",
    )
    carry = carry[
        [
            "date",
            "ts",
            "anchor",
            "far",
            "anchor_months_out",
            "far_months_out",
            "months_from_anchor",
            "anchor_close",
            "far_close",
            "anchor_volume_5m",
            "far_volume_5m",
            "anchor_daily_volume",
            "far_daily_volume",
        ]
    ].copy()
    carry.insert(0, "root", root)
    carry["annualized_log_carry_pct"] = (
        np.log(carry["far_close"] / carry["anchor_close"])
        / (carry["months_from_anchor"] / 12.0)
        * 100.0
    )
    return carry.sort_values("ts"), pairs


def plot_raw_and_daily(carry: pd.DataFrame, output_dir: Path) -> None:
    daily = (
        carry.groupby(["root", "date"], as_index=False)
        .agg(
            annualized_log_carry_pct=("annualized_log_carry_pct", "median"),
            obs=("annualized_log_carry_pct", "size"),
        )
        .sort_values(["root", "date"])
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily["rolling_20d"] = daily.groupby("root")["annualized_log_carry_pct"].transform(
        lambda series: series.rolling(20, min_periods=5).median()
    )

    fig, axes = plt.subplots(len(ROOTS), 1, figsize=(15, 11), sharex=True, constrained_layout=True)
    for ax, root in zip(axes, ROOTS, strict=True):
        raw = carry[carry["root"] == root]
        d = daily[daily["root"] == root]
        ax.scatter(
            raw["ts"],
            raw["annualized_log_carry_pct"],
            s=2,
            alpha=0.08,
            color=COLORS[root],
            rasterized=True,
            label="5m exact observations",
        )
        ax.plot(
            d["date"],
            d["annualized_log_carry_pct"],
            color="#1f2933",
            lw=0.9,
            label="daily median",
        )
        ax.plot(d["date"], d["rolling_20d"], color="#c43d3d", lw=1.2, label="20d median")
        ax.axhline(0, color="#333333", lw=0.8)
        ax.set_ylabel(f"{root}\n% ann.")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=8, ncols=3)
    fig.suptitle(
        "Annualized futures carry from liquid front to first contract >= 3 months out",
        fontsize=14,
    )
    axes[-1].set_xlabel("date")
    fig.savefig(output_dir / "annualized_carry_5m_points_with_daily_median.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(len(ROOTS), 1, figsize=(15, 10), sharex=True, constrained_layout=True)
    for ax, root in zip(axes, ROOTS, strict=True):
        d = daily[daily["root"] == root]
        ax.plot(
            d["date"],
            d["annualized_log_carry_pct"],
            color=COLORS[root],
            lw=1.0,
            label="daily median",
        )
        ax.plot(d["date"], d["rolling_20d"], color="#1f2933", lw=1.4, label="20d median")
        ax.axhline(0, color="#333333", lw=0.8)
        ax.set_ylabel(f"{root}\n% ann.")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Daily median of exact 5-minute annualized carry observations", fontsize=14)
    axes[-1].set_xlabel("date")
    fig.savefig(output_dir / "annualized_carry_daily_median.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")

    frames = []
    pair_frames = []
    for root in ROOTS:
        print(f"Processing {root}", flush=True)
        carry, pairs = load_root_carry(
            root,
            start=start,
            end=end,
            target_months=args.target_months,
            min_volume=args.min_volume,
        )
        if not carry.empty:
            frames.append(carry)
        if not pairs.empty:
            pairs.insert(0, "root", root)
            pair_frames.append(pairs)

    if not frames:
        raise RuntimeError("No carry observations were generated.")

    carry = pd.concat(frames, ignore_index=True).sort_values(["root", "ts"])
    pairs = pd.concat(pair_frames, ignore_index=True).sort_values(["root", "date"])
    carry.to_csv(output_dir / "annualized_carry_5m_exact_observations.csv", index=False)
    pairs.to_csv(output_dir / "daily_selected_carry_pairs.csv", index=False)

    summary = (
        carry.groupby("root")
        .agg(
            observations=("annualized_log_carry_pct", "size"),
            start=("ts", "min"),
            end=("ts", "max"),
            mean_pct_ann=("annualized_log_carry_pct", "mean"),
            median_pct_ann=("annualized_log_carry_pct", "median"),
            p10_pct_ann=("annualized_log_carry_pct", lambda series: series.quantile(0.1)),
            p90_pct_ann=("annualized_log_carry_pct", lambda series: series.quantile(0.9)),
            median_months_from_anchor=("months_from_anchor", "median"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "annualized_carry_5m_summary.csv", index=False)
    plot_raw_and_daily(carry, output_dir)
    print(summary.round(4).to_string(index=False))
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
