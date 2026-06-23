from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import databento as db
import numpy as np
import pandas as pd

DATASET_DIR = Path(
    "data/raw/databento/glbx_mdp3/representative_trend_universe_2025-06-19_2026-06-19"
)
OUTPUT_DIR = Path("reports/generated/databento_glbx_representative_eda_2026-06-20")
PRICE_SCALE = 1_000_000_000.0
NS_PER_MINUTE = 60 * 1_000_000_000
EXPECTED_CONDITION_STATES = {"available", "degraded"}


@dataclass
class SymbolStats:
    category: str
    root: str
    symbol: str
    description: str
    schema: str
    records: int = 0
    files: int = 0
    compressed_bytes: int = 0
    first_ts_ns: int | None = None
    last_ts_ns: int | None = None
    active_days: set[str] = field(default_factory=set)
    daily_records: Counter[str] = field(default_factory=Counter)
    price_min: float | None = None
    price_max: float | None = None
    total_size_or_volume: int = 0
    weighted_price_sum: float = 0.0
    side_counts: Counter[str] = field(default_factory=Counter)
    action_counts: Counter[str] = field(default_factory=Counter)
    size_counts: Counter[int] = field(default_factory=Counter)
    non_positive_price_records: int = 0
    duplicate_ts_records: int = 0
    non_monotonic_ts_records: int = 0
    gap_count_gt_1m: int = 0
    same_day_gap_count_gt_1m: int = 0
    max_gap_minutes: float = 0.0
    high_low_violations: int = 0
    open_outside_hilo: int = 0
    close_outside_hilo: int = 0
    zero_volume_bars: int = 0
    close_values: list[float] = field(default_factory=list)
    previous_ts_ns: int | None = None

    def update_time(self, ts_ns: np.ndarray) -> None:
        if len(ts_ns) == 0:
            return
        first = int(ts_ns[0])
        last = int(ts_ns[-1])
        self.first_ts_ns = first if self.first_ts_ns is None else min(self.first_ts_ns, first)
        self.last_ts_ns = last if self.last_ts_ns is None else max(self.last_ts_ns, last)
        days = pd.to_datetime(ts_ns, utc=True).date.astype(str)
        self.active_days.update(days.tolist())
        self.daily_records.update(days.tolist())

        diffs = np.diff(ts_ns.astype(np.int64))
        if self.previous_ts_ns is not None:
            diffs = np.concatenate(([first - self.previous_ts_ns], diffs))
            prev_day = pd.Timestamp(self.previous_ts_ns, unit="ns", tz="UTC").date().isoformat()
            day_pairs_left = np.concatenate(([prev_day], days[:-1]))
        else:
            day_pairs_left = days[:-1]
        if len(diffs):
            self.duplicate_ts_records += int(np.sum(diffs == 0))
            self.non_monotonic_ts_records += int(np.sum(diffs < 0))
            gap_mask = diffs > NS_PER_MINUTE
            self.gap_count_gt_1m += int(np.sum(gap_mask))
            if np.any(gap_mask):
                max_gap = float(np.max(diffs[gap_mask]) / NS_PER_MINUTE)
                self.max_gap_minutes = max(self.max_gap_minutes, max_gap)
                right_days = days if self.previous_ts_ns is not None else days[1:]
                same_day = np.asarray(day_pairs_left) == np.asarray(right_days)
                self.same_day_gap_count_gt_1m += int(np.sum(gap_mask & same_day))
        self.previous_ts_ns = last

    def update_price_range(self, prices: np.ndarray) -> None:
        if len(prices) == 0:
            return
        positive = prices > 0
        self.non_positive_price_records += int(np.sum(~positive))
        if np.any(positive):
            price_min = float(np.min(prices[positive]) / PRICE_SCALE)
            price_max = float(np.max(prices[positive]) / PRICE_SCALE)
            self.price_min = price_min if self.price_min is None else min(self.price_min, price_min)
            self.price_max = price_max if self.price_max is None else max(self.price_max, price_max)

    def quantile_from_size_counts(self, q: float) -> float | None:
        total = sum(self.size_counts.values())
        if total == 0:
            return None
        target = math.ceil(q * total)
        cumulative = 0
        for value in sorted(self.size_counts):
            cumulative += self.size_counts[value]
            if cumulative >= target:
                return float(value)
        return float(max(self.size_counts))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    universe = _load_universe()
    condition_rows = _load_condition_rows()
    expected_days = {
        row["date"] for row in condition_rows if row.get("condition") in EXPECTED_CONDITION_STATES
    }

    inventory = pd.read_csv(DATASET_DIR / "metadata" / "download_inventory.csv")
    summaries: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []

    for schema in ["ohlcv-1m", "trades"]:
        for item in universe:
            stats = SymbolStats(schema=schema, **item)
            files = inventory[
                (inventory["schema"] == schema) & (inventory["symbol"] == item["symbol"])
            ].sort_values(["start", "filename"])
            for _, file_row in files.iterrows():
                path = Path(str(file_row["path"]))
                stats.files += 1
                stats.compressed_bytes += int(file_row["bytes"])
                _process_dbn_file(stats, path)
            summary = _finalize_summary(stats, expected_days)
            summaries.append(summary)
            daily_rows.extend(_daily_rows(stats, condition_rows))
            print(
                f"{schema:8s} {stats.symbol:7s} records={stats.records:,} "
                f"active_days={len(stats.active_days):,}",
                flush=True,
            )

    summary_df = pd.DataFrame(summaries).sort_values(["schema", "category", "symbol"])
    daily_df = pd.DataFrame(daily_rows).sort_values(["schema", "symbol", "date"])
    condition_df = pd.DataFrame(_condition_summary(condition_rows))

    summary_df.to_csv(OUTPUT_DIR / "contract_schema_summary.csv", index=False)
    daily_df.to_csv(OUTPUT_DIR / "daily_coverage.csv", index=False)
    condition_df.to_csv(OUTPUT_DIR / "databento_condition_summary.csv", index=False)
    _write_report(summary_df, daily_df, condition_df)


def _load_universe() -> list[dict[str, str]]:
    with (DATASET_DIR / "metadata" / "universe.csv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return [
            {
                "category": row["category"],
                "root": row["root"],
                "symbol": row["continuous_symbol"],
                "description": row["description"],
            }
            for row in reader
        ]


def _load_condition_rows() -> list[dict[str, Any]]:
    path = next((DATASET_DIR / "ohlcv-1m").rglob("condition.json"))
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        dt = datetime.fromisoformat(row["date"]).replace(tzinfo=UTC)
        row["weekday"] = dt.strftime("%A")
    return rows


def _process_dbn_file(stats: SymbolStats, path: Path) -> None:
    store = db.DBNStore.from_file(path)
    for chunk in store.to_ndarray(count=2_000_000):
        if len(chunk) == 0:
            continue
        stats.records += len(chunk)
        stats.update_time(chunk["ts_event"])
        if stats.schema == "ohlcv-1m":
            _update_ohlcv(stats, chunk)
        else:
            _update_trades(stats, chunk)


def _update_ohlcv(stats: SymbolStats, chunk: np.ndarray) -> None:
    open_px = chunk["open"]
    high = chunk["high"]
    low = chunk["low"]
    close = chunk["close"]
    volume = chunk["volume"].astype(np.uint64)
    stats.update_price_range(close)
    stats.total_size_or_volume += int(np.sum(volume))
    stats.weighted_price_sum += float(np.sum((close.astype(np.float64) / PRICE_SCALE) * volume))
    stats.high_low_violations += int(np.sum(high < low))
    stats.open_outside_hilo += int(np.sum((open_px < low) | (open_px > high)))
    stats.close_outside_hilo += int(np.sum((close < low) | (close > high)))
    stats.zero_volume_bars += int(np.sum(volume == 0))
    stats.close_values.extend((close.astype(np.float64) / PRICE_SCALE).tolist())


def _update_trades(stats: SymbolStats, chunk: np.ndarray) -> None:
    price = chunk["price"]
    size = chunk["size"].astype(np.uint64)
    stats.update_price_range(price)
    stats.total_size_or_volume += int(np.sum(size))
    stats.weighted_price_sum += float(np.sum((price.astype(np.float64) / PRICE_SCALE) * size))
    sides, side_counts = np.unique(chunk["side"], return_counts=True)
    actions, action_counts = np.unique(chunk["action"], return_counts=True)
    sizes, sizes_counts = np.unique(size, return_counts=True)
    stats.side_counts.update(
        {
            side.decode("ascii", errors="ignore"): int(count)
            for side, count in zip(sides, side_counts, strict=True)
        }
    )
    stats.action_counts.update(
        {
            action.decode("ascii", errors="ignore"): int(count)
            for action, count in zip(actions, action_counts, strict=True)
        }
    )
    stats.size_counts.update(
        {int(value): int(count) for value, count in zip(sizes, sizes_counts, strict=True)}
    )


def _finalize_summary(stats: SymbolStats, expected_days: set[str]) -> dict[str, Any]:
    active_days = len(stats.active_days)
    zero_expected_days = sorted(expected_days - stats.active_days)
    daily_counts = np.asarray(list(stats.daily_records.values()), dtype=float)
    close = np.asarray(stats.close_values, dtype=float)
    returns = np.diff(np.log(close)) if len(close) > 1 else np.asarray([], dtype=float)
    return {
        "schema": stats.schema,
        "category": stats.category,
        "root": stats.root,
        "symbol": stats.symbol,
        "description": stats.description,
        "files": stats.files,
        "compressed_mb": stats.compressed_bytes / 1_000_000,
        "records": stats.records,
        "first_ts": _ts(stats.first_ts_ns),
        "last_ts": _ts(stats.last_ts_ns),
        "active_days": active_days,
        "zero_record_expected_days": len(zero_expected_days),
        "zero_record_expected_days_list": ",".join(zero_expected_days),
        "median_daily_records": _nan_if_empty(
            np.median(daily_counts) if len(daily_counts) else np.nan
        ),
        "min_daily_records": _nan_if_empty(np.min(daily_counts) if len(daily_counts) else np.nan),
        "max_daily_records": _nan_if_empty(np.max(daily_counts) if len(daily_counts) else np.nan),
        "price_min": stats.price_min,
        "price_max": stats.price_max,
        "total_size_or_volume": stats.total_size_or_volume,
        "vwap_or_vw_close": (
            stats.weighted_price_sum / stats.total_size_or_volume
            if stats.total_size_or_volume
            else None
        ),
        "side_A_count": stats.side_counts.get("A", 0),
        "side_B_count": stats.side_counts.get("B", 0),
        "side_N_count": stats.side_counts.get("N", 0),
        "action_T_count": stats.action_counts.get("T", 0),
        "trade_size_p50": stats.quantile_from_size_counts(0.50),
        "trade_size_p95": stats.quantile_from_size_counts(0.95),
        "trade_size_p99": stats.quantile_from_size_counts(0.99),
        "trade_size_max": max(stats.size_counts) if stats.size_counts else None,
        "zero_volume_bars": stats.zero_volume_bars,
        "zero_volume_bar_pct": stats.zero_volume_bars / stats.records if stats.records else None,
        "non_positive_price_records": stats.non_positive_price_records,
        "high_low_violations": stats.high_low_violations,
        "open_outside_hilo": stats.open_outside_hilo,
        "close_outside_hilo": stats.close_outside_hilo,
        "duplicate_ts_records": stats.duplicate_ts_records,
        "non_monotonic_ts_records": stats.non_monotonic_ts_records,
        "gap_count_gt_1m": stats.gap_count_gt_1m,
        "same_day_gap_count_gt_1m": stats.same_day_gap_count_gt_1m,
        "max_gap_minutes": stats.max_gap_minutes,
        "logret_count": len(returns),
        "logret_mean": _nan_if_empty(np.mean(returns) if len(returns) else np.nan),
        "logret_std": _nan_if_empty(np.std(returns) if len(returns) else np.nan),
        "logret_abs_p99": _nan_if_empty(
            np.quantile(np.abs(returns), 0.99) if len(returns) else np.nan
        ),
        "logret_abs_max": _nan_if_empty(np.max(np.abs(returns)) if len(returns) else np.nan),
    }


def _daily_rows(stats: SymbolStats, condition_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "schema": stats.schema,
            "category": stats.category,
            "symbol": stats.symbol,
            "date": row["date"],
            "weekday": row["weekday"],
            "databento_condition": row["condition"],
            "records": stats.daily_records.get(row["date"], 0),
            "has_records": stats.daily_records.get(row["date"], 0) > 0,
        }
        for row in condition_rows
    ]


def _condition_summary(condition_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter((row["condition"], row["weekday"]) for row in condition_rows)
    return [
        {"condition": condition, "weekday": weekday, "days": days}
        for (condition, weekday), days in sorted(counts.items())
    ]


def _write_report(
    summary_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    condition_df: pd.DataFrame,
) -> None:
    ohlcv = summary_df[summary_df["schema"] == "ohlcv-1m"]
    trades = summary_df[summary_df["schema"] == "trades"]
    missing_daily = (
        daily_df[
            daily_df["databento_condition"].isin(EXPECTED_CONDITION_STATES)
            & (~daily_df["has_records"])
        ]
        .groupby(["schema", "symbol"], as_index=False)
        .size()
        .rename(columns={"size": "zero_record_expected_days"})
        .sort_values(["schema", "zero_record_expected_days"], ascending=[True, False])
    )
    zero_by_category = (
        daily_df[
            daily_df["databento_condition"].isin(EXPECTED_CONDITION_STATES)
            & (~daily_df["has_records"])
        ]
        .groupby(["schema", "category", "weekday"], as_index=False)
        .size()
        .rename(columns={"size": "contract_days"})
        .sort_values(["schema", "category", "weekday"])
    )
    zero_unique_dates = (
        daily_df[
            daily_df["databento_condition"].isin(EXPECTED_CONDITION_STATES)
            & (~daily_df["has_records"])
        ]
        .groupby(["schema", "category"], as_index=False)["date"]
        .nunique()
        .rename(columns={"date": "unique_calendar_dates"})
        .sort_values(["schema", "unique_calendar_dates"], ascending=[True, False])
    )
    lines = [
        "# Databento GLBX Representative Universe EDA",
        "",
        "Dataset: `GLBX.MDP3`",
        "",
        "Window: `2025-06-19` through `2026-06-19`",
        "",
        "Symbols: 30 volume-front continuous futures (`ROOT.v.0`).",
        "",
        "## Files",
        "",
        "- `contract_schema_summary.csv`: per-contract summary for `ohlcv-1m` and `trades`.",
        "- `daily_coverage.csv`: per-contract daily record counts joined to "
        "Databento condition state.",
        "- `databento_condition_summary.csv`: Databento dataset condition by weekday.",
        "",
        "## High-Level Counts",
        "",
        f"- `ohlcv-1m`: {int(ohlcv['records'].sum()):,} records across {len(ohlcv):,} contracts.",
        f"- `trades`: {int(trades['records'].sum()):,} records across {len(trades):,} contracts.",
        f"- `ohlcv-1m` compressed size: {ohlcv['compressed_mb'].sum():,.1f} MB.",
        f"- `trades` compressed size: {trades['compressed_mb'].sum():,.1f} MB.",
        "",
        "## Missing Data Classification",
        "",
        "Databento condition states are dataset-level, not contract-specific. In this pull, "
        "the vendor `missing` condition occurs only on Saturdays. Contract-level zero-record "
        "days inside otherwise `available` or `degraded` dataset dates are mostly exchange "
        "calendar and session-structure effects: broad futures show a handful of weekend "
        "UTC dates, energy/metals add Good Friday, and grains/oilseeds have no UTC Sunday "
        "records plus several holiday dates.",
        "",
        "`degraded` days are vendor-marked quality caveats and should be flagged or excluded "
        "in formal experiments.",
        "",
        condition_df.to_markdown(index=False),
        "",
        "## Largest Trade Feeds",
        "",
        trades.sort_values("records", ascending=False)
        .head(10)[["category", "symbol", "records", "compressed_mb", "active_days"]]
        .to_markdown(index=False),
        "",
        "## Ohlcv Contracts With Most Same-Day Gaps",
        "",
        ohlcv.sort_values("same_day_gap_count_gt_1m", ascending=False)
        .head(10)[
            [
                "category",
                "symbol",
                "records",
                "same_day_gap_count_gt_1m",
                "max_gap_minutes",
                "zero_volume_bar_pct",
            ]
        ]
        .to_markdown(index=False),
        "",
        "## Contract-Days With Records Missing Despite Available/Degraded Dataset Condition",
        "",
        missing_daily.head(30).to_markdown(index=False) if not missing_daily.empty else "None.",
        "",
        "## Zero-Record Days By Category And Weekday",
        "",
        zero_by_category.to_markdown(index=False) if not zero_by_category.empty else "None.",
        "",
        "## Unique Zero-Record Calendar Dates By Category",
        "",
        zero_unique_dates.to_markdown(index=False) if not zero_unique_dates.empty else "None.",
        "",
        "## Data Quality Notes",
        "",
        "- Bar-level OHLC consistency checks count `high < low`, open outside "
        "`[low, high]`, and close outside `[low, high]`.",
        "- Trade gaps are not interpreted as missing data because sparse trading is "
        "normal in less liquid contracts.",
        "- Continuous contracts can roll; large close-to-close moves around roll "
        "boundaries require separate roll diagnostics before return modeling.",
        "",
    ]
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _ts(value: int | None) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(value, unit="ns", tz="UTC").isoformat()


def _nan_if_empty(value: float) -> float | None:
    return None if math.isnan(float(value)) else float(value)


if __name__ == "__main__":
    main()
