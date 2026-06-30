from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
FUNDING_PATH = (
    REPO_ROOT
    / "experiments"
    / "HYP-0036-metals-hourly-funding"
    / "hourly_funding.parquet"
)
CONTINUOUS_DIR = Path(
    "/home/famadeo/research/databento-asset-browser/data/metals_1m_10y/continuous"
)
OUTPUT_DIR = REPO_ROOT / "experiments" / "HYP-0037-metals-funding-vs-realized-returns"

ROOTS = ["GC", "SI", "HG", "PL", "PA"]
TARGET_MONTHS = [1, 3, 6]
HORIZONS = {
    "1h": 1,
    "4h": 4,
    "1d": 24,
    "3d": 72,
    "1w": 168,
    "1m": 720,
}

HOURS_PER_YEAR = 365.0 * 24.0
SECONDS_PER_YEAR = HOURS_PER_YEAR * 60.0 * 60.0
MAX_INTERVAL_HOURS = 96.0
MAX_FUNDING_FFILL_PRICE_BARS = 12
MIN_OBSERVATIONS = 100
MAX_SCATTER_POINTS = 8_000

COLORS = {
    "realized": "#1f77b4",
    "funding": "#c0392b",
    "net": "#1b9e77",
}


def load_funding() -> pd.DataFrame:
    if not FUNDING_PATH.exists():
        raise FileNotFoundError(FUNDING_PATH)
    cols = [
        "root",
        "target_months",
        "ts",
        "funding_rate",
        "funding_pct_ann",
        "funding_z_126d",
        "anchor_symbol",
        "far_symbol",
        "tenor_months",
        "common_hourly_volume",
    ]
    funding = pd.read_parquet(FUNDING_PATH, columns=cols)
    funding["ts"] = pd.to_datetime(funding["ts"], utc=True)
    funding = funding.sort_values(["root", "target_months", "ts"])
    return funding


def load_hourly_price(root: str) -> pd.DataFrame:
    path = CONTINUOUS_DIR / f"{root}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = (
        pl.scan_parquet(path)
        .select("ts", "cont_logprice", "cont_close", "volume")
        .with_columns(pl.col("ts").dt.truncate("1h").alias("hour"))
        .group_by("hour")
        .agg(
            [
                pl.col("cont_logprice").sort_by("ts").last().alias("log_price"),
                pl.col("cont_close").sort_by("ts").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
            ]
        )
        .rename({"hour": "ts"})
        .sort("ts")
        .collect()
        .to_pandas()
    )
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    frame["root"] = root
    return frame[["root", "ts", "log_price", "close", "volume"]]


def forward_sum(series: pd.Series, periods: int) -> pd.Series:
    return series.iloc[::-1].rolling(periods, min_periods=periods).sum().iloc[::-1]


def hac_tstat_mean(values: pd.Series, maxlags: int) -> float:
    data = values.dropna().to_numpy(dtype=float)
    if data.size < MIN_OBSERVATIONS:
        return np.nan
    demeaned = data - data.mean()
    n = data.size
    gamma0 = float(np.dot(demeaned, demeaned) / n)
    variance = gamma0
    for lag in range(1, min(maxlags, n - 1) + 1):
        cov = float(np.dot(demeaned[lag:], demeaned[:-lag]) / n)
        weight = 1.0 - lag / (maxlags + 1.0)
        variance += 2.0 * weight * cov
    if variance <= 0:
        return np.nan
    se = np.sqrt(variance / n)
    return float(data.mean() / se) if se > 0 else np.nan


def build_root_target_panel(
    price: pd.DataFrame, funding: pd.DataFrame, root: str, target_months: int
) -> pd.DataFrame:
    root_funding = funding[
        (funding["root"] == root) & (funding["target_months"] == target_months)
    ].copy()
    if root_funding.empty:
        return pd.DataFrame()

    rate = (
        root_funding.groupby("ts", as_index=True)
        .agg(
            funding_rate=("funding_rate", "median"),
            funding_pct_ann=("funding_pct_ann", "median"),
            funding_z_126d=("funding_z_126d", "median"),
            common_hourly_volume=("common_hourly_volume", "median"),
            anchor_symbol=("anchor_symbol", lambda x: x.mode().iloc[0]),
            far_symbol=("far_symbol", lambda x: x.mode().iloc[0]),
            tenor_months=("tenor_months", "median"),
        )
        .sort_index()
    )
    base = price[price["root"] == root].copy().set_index("ts").sort_index()
    merged = base.join(rate, how="left")
    merged["funding_observed"] = merged["funding_rate"].notna()
    fill_cols = [
        "funding_rate",
        "funding_pct_ann",
        "funding_z_126d",
        "common_hourly_volume",
        "anchor_symbol",
        "far_symbol",
        "tenor_months",
    ]
    merged[fill_cols] = merged[fill_cols].ffill(limit=MAX_FUNDING_FFILL_PRICE_BARS)

    next_ts = merged.index.to_series().shift(-1)
    dt_hours = (next_ts - merged.index.to_series()).dt.total_seconds() / 3600.0
    valid_interval = dt_hours.le(MAX_INTERVAL_HOURS)
    dt_years = dt_hours / HOURS_PER_YEAR

    merged["target_months"] = target_months
    merged["dt_hours_to_next_bar"] = dt_hours.where(valid_interval)
    merged["log_return_next_bar"] = (
        merged["log_price"].shift(-1) - merged["log_price"]
    ).where(valid_interval)
    merged["funding_paid_next_bar"] = (
        merged["funding_rate"] * dt_years
    ).where(valid_interval)
    merged["excess_after_funding_next_bar"] = (
        merged["log_return_next_bar"] - merged["funding_paid_next_bar"]
    )
    merged = merged.reset_index()
    merged["root"] = root
    return merged


def build_accounting_panels(
    funding: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    price = pd.concat([load_hourly_price(root) for root in ROOTS], ignore_index=True)
    cumulative_frames = []
    horizon_frames = []
    coverage_rows = []

    for root in ROOTS:
        print(f"Accounting {root}", flush=True)
        root_price = price[price["root"] == root].copy()
        for target_months in TARGET_MONTHS:
            panel = build_root_target_panel(root_price, funding, root, target_months)
            if panel.empty:
                continue

            valid = panel.dropna(
                subset=[
                    "log_return_next_bar",
                    "funding_paid_next_bar",
                    "excess_after_funding_next_bar",
                ]
            ).copy()
            if valid.empty:
                continue

            valid["cum_realized_log_return"] = valid["log_return_next_bar"].cumsum()
            valid["cum_funding_paid_log"] = valid["funding_paid_next_bar"].cumsum()
            valid["cum_excess_after_funding_log"] = valid[
                "excess_after_funding_next_bar"
            ].cumsum()
            cumulative_frames.append(
                valid[
                    [
                        "root",
                        "target_months",
                        "ts",
                        "funding_rate",
                        "funding_pct_ann",
                        "funding_z_126d",
                        "dt_hours_to_next_bar",
                        "log_return_next_bar",
                        "funding_paid_next_bar",
                        "excess_after_funding_next_bar",
                        "cum_realized_log_return",
                        "cum_funding_paid_log",
                        "cum_excess_after_funding_log",
                        "anchor_symbol",
                        "far_symbol",
                        "tenor_months",
                        "funding_observed",
                    ]
                ]
            )

            coverage_rows.append(
                {
                    "root": root,
                    "target_months": target_months,
                    "price_bars": len(panel),
                    "observed_funding_bars": int(panel["funding_observed"].sum()),
                    "usable_return_bars": len(valid),
                    "observed_funding_fraction": float(panel["funding_observed"].mean()),
                    "usable_fraction": float(len(valid) / len(panel)),
                    "first_ts": panel["ts"].min(),
                    "last_ts": panel["ts"].max(),
                }
            )

            indexed = panel.set_index("ts").sort_index()
            ts_values = indexed.index.to_series()
            for horizon_label, periods in HORIZONS.items():
                future_log_price = indexed["log_price"].shift(-periods)
                forward_return = future_log_price - indexed["log_price"]
                target_ts = ts_values.shift(-periods)
                elapsed_years = (
                    (target_ts - ts_values).dt.total_seconds() / SECONDS_PER_YEAR
                )
                initial_paid = indexed["funding_rate"] * elapsed_years
                path_paid = forward_sum(indexed["funding_paid_next_bar"], periods)
                aligned = pd.DataFrame(
                    {
                        "root": root,
                        "target_months": target_months,
                        "horizon": horizon_label,
                        "horizon_bars": periods,
                        "ts": indexed.index,
                        "funding_rate": indexed["funding_rate"].to_numpy(),
                        "funding_pct_ann": indexed["funding_pct_ann"].to_numpy(),
                        "funding_z_126d": indexed["funding_z_126d"].to_numpy(),
                        "forward_log_return": forward_return.to_numpy(),
                        "initial_funding_paid_log": initial_paid.to_numpy(),
                        "path_funding_paid_log": path_paid.to_numpy(),
                    }
                )
                aligned["excess_after_initial_funding_log"] = (
                    aligned["forward_log_return"] - aligned["initial_funding_paid_log"]
                )
                aligned["excess_after_path_funding_log"] = (
                    aligned["forward_log_return"] - aligned["path_funding_paid_log"]
                )
                aligned = aligned.dropna(
                    subset=[
                        "funding_rate",
                        "forward_log_return",
                        "initial_funding_paid_log",
                        "path_funding_paid_log",
                        "excess_after_initial_funding_log",
                        "excess_after_path_funding_log",
                    ]
                )
                if not aligned.empty:
                    horizon_frames.append(aligned)

    cumulative = (
        pd.concat(cumulative_frames, ignore_index=True)
        if cumulative_frames
        else pd.DataFrame()
    )
    horizons = (
        pd.concat(horizon_frames, ignore_index=True) if horizon_frames else pd.DataFrame()
    )
    coverage = pd.DataFrame(coverage_rows)
    return cumulative, horizons, coverage


def summarize_horizons(horizons: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (root, target, horizon), group in horizons.groupby(
        ["root", "target_months", "horizon"], sort=True
    ):
        periods = int(group["horizon_bars"].iloc[0])
        rows.append(
            {
                "root": root,
                "target_months": target,
                "horizon": horizon,
                "horizon_bars": periods,
                "n": len(group),
                "mean_forward_return_bp": group["forward_log_return"].mean() * 10_000.0,
                "median_forward_return_bp": group["forward_log_return"].median()
                * 10_000.0,
                "mean_initial_funding_paid_bp": group[
                    "initial_funding_paid_log"
                ].mean()
                * 10_000.0,
                "mean_path_funding_paid_bp": group["path_funding_paid_log"].mean()
                * 10_000.0,
                "mean_excess_after_initial_funding_bp": group[
                    "excess_after_initial_funding_log"
                ].mean()
                * 10_000.0,
                "mean_excess_after_path_funding_bp": group[
                    "excess_after_path_funding_log"
                ].mean()
                * 10_000.0,
                "median_excess_after_path_funding_bp": group[
                    "excess_after_path_funding_log"
                ].median()
                * 10_000.0,
                "path_excess_hac_t": hac_tstat_mean(
                    group["excess_after_path_funding_log"], maxlags=max(1, periods - 1)
                ),
                "initial_excess_hac_t": hac_tstat_mean(
                    group["excess_after_initial_funding_log"],
                    maxlags=max(1, periods - 1),
                ),
                "positive_path_excess_fraction": float(
                    (group["excess_after_path_funding_log"] > 0).mean()
                ),
                "contango_fraction": float((group["funding_rate"] > 0).mean()),
                "avg_abs_funding_share_of_abs_return": (
                    group["path_funding_paid_log"].abs()
                    / group["forward_log_return"].abs().replace(0.0, np.nan)
                ).mean(),
            }
        )
    out = pd.DataFrame(rows)
    order = {label: idx for idx, label in enumerate(HORIZONS)}
    out["horizon_order"] = out["horizon"].map(order)
    return out.sort_values(["root", "target_months", "horizon_order"]).drop(
        columns=["horizon_order"]
    )


def summarize_cumulative(cumulative: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (root, target), group in cumulative.groupby(["root", "target_months"], sort=True):
        last = group.sort_values("ts").iloc[-1]
        years = (
            group["dt_hours_to_next_bar"].sum() / HOURS_PER_YEAR
            if "dt_hours_to_next_bar" in group
            else np.nan
        )
        rows.append(
            {
                "root": root,
                "target_months": target,
                "first_ts": group["ts"].min(),
                "last_ts": group["ts"].max(),
                "accounted_years": years,
                "intervals": len(group),
                "cum_realized_log_return": last["cum_realized_log_return"],
                "cum_funding_paid_log": last["cum_funding_paid_log"],
                "cum_excess_after_funding_log": last["cum_excess_after_funding_log"],
                "ann_realized_log_return_pct": last["cum_realized_log_return"]
                / years
                * 100.0
                if years and years > 0
                else np.nan,
                "ann_funding_paid_pct": last["cum_funding_paid_log"] / years * 100.0
                if years and years > 0
                else np.nan,
                "ann_excess_after_funding_pct": last["cum_excess_after_funding_log"]
                / years
                * 100.0
                if years and years > 0
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def sample_cumulative_daily(cumulative: pd.DataFrame) -> pd.DataFrame:
    if cumulative.empty:
        return cumulative
    frames = []
    for (root, target), group in cumulative.groupby(["root", "target_months"], sort=True):
        sampled = (
            group.sort_values("ts")
            .set_index("ts")
            .resample("1D")
            .last()
            .dropna(subset=["cum_realized_log_return"])
            .reset_index()
        )
        sampled["root"] = root
        sampled["target_months"] = target
        frames.append(sampled)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_cumulative(cumulative_daily: pd.DataFrame, target_months: int) -> None:
    data = cumulative_daily[cumulative_daily["target_months"] == target_months]
    if data.empty:
        return
    fig, axes = plt.subplots(len(ROOTS), 1, figsize=(14, 10), sharex=True, constrained_layout=True)
    for ax, root in zip(axes, ROOTS, strict=True):
        root_data = data[data["root"] == root].sort_values("ts")
        if root_data.empty:
            continue
        ax.plot(
            root_data["ts"],
            root_data["cum_realized_log_return"] * 100.0,
            color=COLORS["realized"],
            lw=1.1,
            label="realized log return",
        )
        ax.plot(
            root_data["ts"],
            root_data["cum_funding_paid_log"] * 100.0,
            color=COLORS["funding"],
            lw=1.1,
            label="funding paid",
        )
        ax.plot(
            root_data["ts"],
            root_data["cum_excess_after_funding_log"] * 100.0,
            color=COLORS["net"],
            lw=1.1,
            label="realized minus funding",
        )
        ax.axhline(0.0, color="#333333", lw=0.8)
        ax.set_ylabel(f"{root}\nlog %")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(
        f"Cumulative continuous return vs curve-implied funding paid, target {target_months}M",
        fontsize=13,
    )
    fig.savefig(OUTPUT_DIR / f"cumulative_realized_vs_funding_target{target_months}m.png", dpi=160)
    plt.close(fig)


def plot_horizon_heatmap(summary: pd.DataFrame, target_months: int) -> None:
    data = summary[summary["target_months"] == target_months].copy()
    if data.empty:
        return
    metrics = [
        (
            "mean_forward_return_bp",
            f"Mean Forward Realized Return, Target {target_months}M (bp)",
            f"mean_forward_return_bp_target{target_months}m.png",
        ),
        (
            "mean_path_funding_paid_bp",
            f"Mean Realized Funding Paid, Target {target_months}M (bp)",
            f"mean_path_funding_paid_bp_target{target_months}m.png",
        ),
        (
            "mean_excess_after_path_funding_bp",
            f"Mean Realized Return Minus Funding Paid, Target {target_months}M (bp)",
            f"mean_excess_after_path_funding_bp_target{target_months}m.png",
        ),
    ]
    horizons = list(HORIZONS)
    for metric, title, filename in metrics:
        matrix = data.pivot(index="root", columns="horizon", values=metric).reindex(
            index=ROOTS, columns=horizons
        )
        values = matrix.to_numpy(dtype=float)
        finite = np.isfinite(values)
        vmax = np.nanpercentile(np.abs(values[finite]), 95) if finite.any() else 1.0
        vmax = max(vmax, 1e-9)
        fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
        image = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(horizons)), labels=horizons)
        ax.set_yticks(np.arange(len(ROOTS)), labels=ROOTS)
        for i, root in enumerate(ROOTS):
            for j, horizon in enumerate(horizons):
                value = matrix.loc[root, horizon]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, label="bp")
        fig.savefig(OUTPUT_DIR / filename, dpi=160)
        plt.close(fig)


def plot_scatter(horizons: pd.DataFrame, target_months: int, horizon: str) -> None:
    data = horizons[
        (horizons["target_months"] == target_months) & (horizons["horizon"] == horizon)
    ].copy()
    if data.empty:
        return
    fig, axes = plt.subplots(1, len(ROOTS), figsize=(17, 3.8), sharex=False, sharey=False)
    for ax, root in zip(axes, ROOTS, strict=True):
        root_data = data[data["root"] == root]
        if root_data.empty:
            continue
        if len(root_data) > MAX_SCATTER_POINTS:
            root_data = root_data.sample(MAX_SCATTER_POINTS, random_state=42)
        ax.scatter(
            root_data["path_funding_paid_log"] * 10_000.0,
            root_data["forward_log_return"] * 10_000.0,
            s=5,
            alpha=0.18,
            color="#2f5597",
            edgecolors="none",
        )
        ax.axhline(0.0, color="#333333", lw=0.8)
        ax.axvline(0.0, color="#333333", lw=0.8)
        ax.set_title(root)
        ax.set_xlabel("funding paid bp")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("realized return bp")
    fig.suptitle(f"{horizon} realized return vs realized funding paid, target {target_months}M")
    fig.savefig(
        OUTPUT_DIR / f"scatter_return_vs_funding_target{target_months}m_{horizon}.png",
        dpi=160,
    )
    plt.close(fig)


def write_report(
    *,
    cumulative_summary: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    lines = [
        "# HYP-0037 Metals Funding Paid vs Realized Returns",
        "",
        "## Definition",
        "",
        "For a long metal exposure, hourly funding paid is computed from the curve-implied "
        "annualized funding rate:",
        "",
        "`funding_paid = funding_rate * elapsed_years`",
        "",
        "Then:",
        "",
        "`excess_after_funding = realized_log_return - funding_paid`",
        "",
        "Positive funding is contango and is a cost to longs. Negative funding is "
        "backwardation and is a benefit to longs.",
        "",
        "## Method",
        "",
        "- Funding input: `experiments/HYP-0036-metals-hourly-funding/hourly_funding.parquet`.",
        "- Realized returns: continuous hourly log-price series for `GC`, `SI`, `HG`, `PL`, `PA`.",
        "- Funding is forward-filled for at most "
        f"{MAX_FUNDING_FFILL_PRICE_BARS} observed price bars to avoid overusing stale "
        "far-contract marks.",
        "- Funding accrual uses actual elapsed calendar time between consecutive observed "
        "price bars, "
        f"excluding intervals longer than {MAX_INTERVAL_HOURS:g} hours.",
        "- Forward-horizon tables include both initial-rate funding and realized path funding. "
        "Initial-rate funding is ex ante; path funding is realized accounting.",
        "",
        "## Coverage",
        "",
        coverage.to_markdown(index=False, floatfmt=".4f")
        if not coverage.empty
        else "No coverage rows.",
        "",
        "## Cumulative Long Accounting",
        "",
        cumulative_summary.to_markdown(index=False, floatfmt=".4f")
        if not cumulative_summary.empty
        else "No cumulative rows.",
        "",
        "## Forward Horizon Summary",
        "",
        horizon_summary.to_markdown(index=False, floatfmt=".4f")
        if not horizon_summary.empty
        else "No horizon rows.",
        "",
        "## Files",
        "",
        "- `cumulative_hourly_accounting.parquet`",
        "- `cumulative_daily_accounting.csv`",
        "- `cumulative_summary.csv`",
        "- `forward_horizon_accounting.parquet`",
        "- `forward_horizon_summary.csv`",
        "- `coverage_summary.csv`",
        "- `cumulative_realized_vs_funding_target1m.png`",
        "- `cumulative_realized_vs_funding_target3m.png`",
        "- `cumulative_realized_vs_funding_target6m.png`",
        "- heatmaps of mean realized return, funding paid, and excess return by horizon",
        "",
        "## Caveats",
        "",
        "- The funding input is front-futures-to-deferred futures carry, not true "
        "cash-to-futures funding.",
        "- Continuous realized returns remove roll jumps, so this is a clean price-return "
        "comparison, "
        "not a full executable roll PnL simulation.",
        "- Overlapping forward horizons require cautious inference; HAC t-statistics are "
        "included as "
        "diagnostics, not proof of alpha.",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    funding = load_funding()
    cumulative, horizons, coverage = build_accounting_panels(funding)
    if cumulative.empty or horizons.empty:
        raise RuntimeError("No accounting rows were generated.")

    cumulative_summary = summarize_cumulative(cumulative)
    horizon_summary = summarize_horizons(horizons)
    cumulative_daily = sample_cumulative_daily(cumulative)

    cumulative.to_parquet(OUTPUT_DIR / "cumulative_hourly_accounting.parquet", index=False)
    horizons.to_parquet(OUTPUT_DIR / "forward_horizon_accounting.parquet", index=False)
    cumulative_daily.to_csv(OUTPUT_DIR / "cumulative_daily_accounting.csv", index=False)
    cumulative_summary.to_csv(OUTPUT_DIR / "cumulative_summary.csv", index=False)
    horizon_summary.to_csv(OUTPUT_DIR / "forward_horizon_summary.csv", index=False)
    coverage.to_csv(OUTPUT_DIR / "coverage_summary.csv", index=False)

    for target_months in TARGET_MONTHS:
        plot_cumulative(cumulative_daily, target_months)
        plot_horizon_heatmap(horizon_summary, target_months)
    plot_scatter(horizons, target_months=3, horizon="1d")
    write_report(
        cumulative_summary=cumulative_summary,
        horizon_summary=horizon_summary,
        coverage=coverage,
    )
    print(f"Cumulative accounting rows: {len(cumulative):,}", flush=True)
    print(f"Forward horizon rows: {len(horizons):,}", flush=True)
    print(f"Wrote {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
