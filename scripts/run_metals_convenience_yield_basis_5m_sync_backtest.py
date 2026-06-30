from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_metals_convenience_yield_basis_backtest as base  # noqa: E402

RAW_DIR = Path("/home/famadeo/research/databento-asset-browser/data/metals_1m_10y/raw")
OUTPUT_DIR = REPO_ROOT / "experiments" / "HYP-0031-metals-convenience-yield-basis-5m-sync"


def load_daily_and_5m(root: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = RAW_DIR / f"{root}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)

    lazy = (
        pl.scan_parquet(path)
        .filter((pl.col("ts_event") >= base.START) & (pl.col("ts_event") < base.END))
        .filter(~pl.col("symbol").str.contains("-"))
        .filter((pl.col("close") > 0) & (pl.col("volume") > 0))
    )

    daily = (
        lazy.with_columns(pl.col("ts_event").dt.date().alias("date"))
        .group_by(["date", "symbol"])
        .agg(
            [
                pl.col("close").sort_by("ts_event").last().alias("daily_last_close"),
                pl.col("volume").sum().alias("volume"),
                pl.col("ts_event").max().alias("daily_last_ts"),
            ]
        )
        .collect()
        .to_pandas()
    )
    daily["date"] = pd.to_datetime(daily["date"], utc=True)
    daily["daily_last_ts"] = pd.to_datetime(daily["daily_last_ts"], utc=True)
    daily["root"] = root
    daily["months_out"] = [
        base.contract_months_out(symbol, date_value)
        for symbol, date_value in zip(daily["symbol"], daily["date"], strict=True)
    ]
    daily = daily.dropna(subset=["months_out", "daily_last_close", "volume"])
    daily = daily[
        (daily["daily_last_close"] > 0) & (daily["volume"] >= base.BASE_MIN_DAILY_VOLUME)
    ]

    bars = (
        lazy.with_columns(
            [
                pl.col("ts_event").dt.date().alias("date"),
                pl.col("ts_event").dt.truncate("5m").alias("ts"),
            ]
        )
        .group_by(["date", "ts", "symbol"])
        .agg(
            [
                pl.col("close").sort_by("ts_event").last().alias("close"),
                pl.col("volume").sum().alias("volume_5m"),
                pl.col("ts_event").max().alias("last_ts"),
            ]
        )
        .collect()
        .to_pandas()
    )
    bars["date"] = pd.to_datetime(bars["date"], utc=True)
    bars["ts"] = pd.to_datetime(bars["ts"], utc=True)
    bars["last_ts"] = pd.to_datetime(bars["last_ts"], utc=True)
    return daily.sort_values(["date", "symbol"]), bars.sort_values(["date", "symbol", "ts"])


def choose_pairs_for_target(
    daily: pd.DataFrame, *, root: str, target_months: int, min_volume: float
) -> pd.DataFrame:
    rows = []
    dates = sorted(daily["date"].unique())
    next_date = {date: dates[i + 1] for i, date in enumerate(dates[:-1])}
    for date, group in daily.groupby("date", sort=True):
        day = group[group["volume"] >= min_volume].copy()
        early = day[day["months_out"] <= base.MAX_ANCHOR_MONTHS_OUT]
        if early.empty:
            continue
        anchor = early.sort_values(["volume", "months_out"], ascending=[False, True]).iloc[0]
        far_candidates = day[day["months_out"] - anchor["months_out"] >= target_months]
        if far_candidates.empty:
            continue
        far = far_candidates.sort_values(["months_out", "volume"], ascending=[True, False]).iloc[0]
        rows.append(
            {
                "root": root,
                "date": date,
                "next_date": next_date.get(date),
                "target_months": target_months,
                "min_volume": min_volume,
                "anchor": str(anchor["symbol"]),
                "far": str(far["symbol"]),
                "anchor_months_out": float(anchor["months_out"]),
                "far_months_out": float(far["months_out"]),
                "months_from_anchor": float(far["months_out"] - anchor["months_out"]),
                "anchor_daily_volume": float(anchor["volume"]),
                "far_daily_volume": float(far["volume"]),
            }
        )
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pairs
    pairs["row_id"] = np.arange(len(pairs), dtype=np.int64)
    return pairs


def common_5m_marks(
    pairs: pd.DataFrame, bars: pd.DataFrame, *, mark_date_col: str, suffix: str
) -> pd.DataFrame:
    pair_cols = ["row_id", mark_date_col, "anchor", "far"]
    pair_dates = pairs[pair_cols].dropna(subset=[mark_date_col]).copy()
    if pair_dates.empty:
        return pd.DataFrame(columns=["row_id"])

    anchor_bars = bars.rename(
        columns={
            "date": mark_date_col,
            "symbol": "anchor",
            "close": f"anchor_close_{suffix}",
            "volume_5m": f"anchor_volume_5m_{suffix}",
            "last_ts": f"anchor_last_ts_{suffix}",
        }
    )
    far_bars = bars.rename(
        columns={
            "date": mark_date_col,
            "symbol": "far",
            "close": f"far_close_{suffix}",
            "volume_5m": f"far_volume_5m_{suffix}",
            "last_ts": f"far_last_ts_{suffix}",
        }
    )
    anchor = pair_dates.merge(anchor_bars, on=[mark_date_col, "anchor"], how="inner")
    far = pair_dates.merge(far_bars, on=[mark_date_col, "far"], how="inner")
    common = anchor.merge(far, on=["row_id", mark_date_col, "anchor", "far", "ts"], how="inner")
    if common.empty:
        return pd.DataFrame(columns=["row_id"])

    common = common.sort_values(["row_id", "ts"])
    obs = common.groupby("row_id").size().rename(f"common_5m_obs_{suffix}")
    last = common.loc[common.groupby("row_id")["ts"].idxmax()].copy()
    last = last.merge(obs, left_on="row_id", right_index=True, how="left")
    last = last.rename(columns={"ts": f"mark_ts_{suffix}"})
    return last[
        [
            "row_id",
            f"mark_ts_{suffix}",
            f"anchor_close_{suffix}",
            f"far_close_{suffix}",
            f"anchor_volume_5m_{suffix}",
            f"far_volume_5m_{suffix}",
            f"anchor_last_ts_{suffix}",
            f"far_last_ts_{suffix}",
            f"common_5m_obs_{suffix}",
        ]
    ]


def build_root_panel(root: str) -> pd.DataFrame:
    print(f"Building synchronized 5m curve panel for {root}", flush=True)
    daily, bars = load_daily_and_5m(root)
    frames = []
    for min_volume in base.MIN_VOLUME_VARIANTS:
        for target_months in base.TARGET_MONTHS:
            pairs = choose_pairs_for_target(
                daily, root=root, target_months=target_months, min_volume=min_volume
            )
            if pairs.empty:
                continue
            entry = common_5m_marks(pairs, bars, mark_date_col="date", suffix="entry")
            next_mark = common_5m_marks(pairs, bars, mark_date_col="next_date", suffix="next")
            panel = pairs.merge(entry, on="row_id", how="left").merge(
                next_mark, on="row_id", how="left"
            )
            frames.append(panel)
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True).sort_values(
        ["min_volume", "target_months", "date"]
    )
    panel["anchor_close"] = panel["anchor_close_entry"]
    panel["far_close"] = panel["far_close_entry"]
    panel["anchor_next_close"] = panel["anchor_close_next"]
    panel["far_next_close"] = panel["far_close_next"]
    panel["log_spread"] = np.log(panel["far_close"] / panel["anchor_close"])
    panel["carry_pct_ann"] = panel["log_spread"] / (panel["months_from_anchor"] / 12.0) * 100.0
    panel["spread_return"] = (
        np.log(panel["far_next_close"] / panel["far_close"])
        - np.log(panel["anchor_next_close"] / panel["anchor_close"])
    )
    return panel


def build_curve_panel() -> pd.DataFrame:
    frames = [build_root_panel(root) for root in base.ROOTS]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise RuntimeError("No synchronized 5m curve panel rows were generated.")
    panel = pd.concat(frames, ignore_index=True).sort_values(
        ["root", "min_volume", "target_months", "date"]
    )
    panel["date"] = pd.to_datetime(panel["date"], utc=True)
    panel["next_date"] = pd.to_datetime(panel["next_date"], utc=True)
    return panel.reset_index(drop=True)


def write_report(
    *,
    metrics: pd.DataFrame,
    split: pd.DataFrame,
    root_summary: pd.DataFrame,
    volume_cost: pd.DataFrame,
    target_volume: pd.DataFrame,
    panel: pd.DataFrame,
    best_variant: str,
) -> None:
    top_cols = [
        "variant",
        "min_volume",
        "cost_multiplier",
        "net_return",
        "cost_return",
        "sharpe",
        "tstat",
        "max_drawdown",
        "event_count",
        "event_tstat",
        "active_fraction",
    ]
    coverage = (
        panel.groupby(["root", "min_volume", "target_months"])
        .agg(
            rows=("date", "size"),
            entry_marked=("mark_ts_entry", lambda x: x.notna().sum()),
            return_marked=("spread_return", lambda x: x.notna().sum()),
            median_common_5m_obs=("common_5m_obs_entry", "median"),
        )
        .reset_index()
    )
    coverage.to_csv(OUTPUT_DIR / "sync_mark_coverage.csv", index=False)

    lines = [
        "# HYP-0031 Metals Convenience-Yield Basis Backtest With Synchronized 5m Marks",
        "",
        "## Design",
        "",
        "- Universe: `GC`, `SI`, `HG`, `PL`, `PA`.",
        "- Pair selection is still daily: liquid front versus first deferred contract at the "
        "target tenor.",
        "- Entry and next-day marks use the last exact shared 5-minute timestamp for the "
        "selected near/far contracts.",
        "- Signal, event exits, costs, position sizing, and robustness grid match HYP-0030.",
        "- This is stricter than daily last-trade closes and removes asynchronous near/far marks.",
        "",
        "## Best Variant",
        "",
        metrics.head(1)[top_cols].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Volume And Cost Robustness",
        "",
        volume_cost.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 1x Cost Tenor And Volume Robustness",
        "",
        target_volume.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Split Metrics",
        "",
        split.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Event Summary For Best Variant",
        "",
        root_summary.to_markdown(index=False, floatfmt=".4f")
        if not root_summary.empty
        else "No events.",
        "",
        "## Sync Mark Coverage",
        "",
        coverage.to_markdown(index=False, floatfmt=".2f"),
        "",
        "## Interpretation",
        "",
        (
            f"The best synchronized-mark variant is `{best_variant}` with net cumulative log "
            f"return `{metrics.iloc[0]['net_return']:.4f}`, t-stat "
            f"`{metrics.iloc[0]['tstat']:.2f}`, and event t-stat "
            f"`{metrics.iloc[0]['event_tstat']:.2f}`."
        ),
        "",
        "If this result is materially weaker than HYP-0030, the daily-close result was likely "
        "benefiting from asynchronous deferred-contract marks. If it survives, the curve-basis "
        "signal has passed a more realistic pricing gate.",
        "",
        "## Files",
        "",
        "- `curve_panel.parquet`",
        "- `sync_mark_coverage.csv`",
        "- `strategy_metrics.csv`",
        "- `best_strategy_returns.csv`",
        "- `event_log.csv`",
        "- `split_metrics.csv`",
        "- `root_event_summary.csv`",
        "- `volume_cost_robustness.csv`",
        "- `target_volume_robustness_1x.csv`",
        "- `best_strategy_equity.png`",
        "- `root_event_summary.png`",
        "- `top_variant_metrics.png`",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel_path = OUTPUT_DIR / "curve_panel.parquet"
    if panel_path.exists():
        panel = pd.read_parquet(panel_path)
    else:
        panel = build_curve_panel()
        panel.to_parquet(panel_path, index=False)
        panel.head(10_000).to_csv(OUTPUT_DIR / "curve_panel_sample.csv", index=False)

    metrics, returns, events = base.run_backtests(panel)
    metrics.to_csv(OUTPUT_DIR / "strategy_metrics.csv", index=False)
    returns.to_csv(OUTPUT_DIR / "all_strategy_returns.csv", index=False)
    events.to_csv(OUTPUT_DIR / "event_log.csv", index=False)

    best_variant = str(metrics.iloc[0]["variant"])
    best_returns = returns[returns["variant"] == best_variant].copy()
    best_events = events[events["variant"] == best_variant].copy()
    split = base.split_metrics(best_returns, best_events)
    root_summary = base.root_event_summary(events, best_variant)
    volume_cost, target_volume = base.robustness_tables(metrics)

    best_returns.to_csv(OUTPUT_DIR / "best_strategy_returns.csv", index=False)
    split.to_csv(OUTPUT_DIR / "split_metrics.csv", index=False)
    root_summary.to_csv(OUTPUT_DIR / "root_event_summary.csv", index=False)
    volume_cost.to_csv(OUTPUT_DIR / "volume_cost_robustness.csv", index=False)
    target_volume.to_csv(OUTPUT_DIR / "target_volume_robustness_1x.csv", index=False)

    base.plot_best_equity(best_returns, OUTPUT_DIR / "best_strategy_equity.png")
    base.plot_root_events(root_summary, OUTPUT_DIR / "root_event_summary.png")
    base.plot_top_variants(metrics, OUTPUT_DIR / "top_variant_metrics.png")
    write_report(
        metrics=metrics,
        split=split,
        root_summary=root_summary,
        volume_cost=volume_cost,
        target_volume=target_volume,
        panel=panel,
        best_variant=best_variant,
    )
    print(metrics.head(12).round(4).to_string(index=False))
    print(f"Wrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
