from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTINUOUS_DIR = Path(
    "/home/famadeo/research/databento-asset-browser/data/metals_1m_10y/continuous"
)
OUTPUT_DIR = REPO_ROOT / "experiments" / "HYP-0040-core-metals-pairwise-covariance"

ROOTS = ["GC", "SI", "HG", "PL", "PA"]
ROLLING_SMOOTH_DAYS = 20
ANNUALIZATION_DAYS = 252.0

COLORS = {
    "GC": "#b8860b",
    "SI": "#6f7f8f",
    "HG": "#b15a2a",
    "PL": "#2f7d8c",
    "PA": "#7a4e9b",
}


def load_returns() -> pd.DataFrame:
    frames = []
    inventory_rows = []
    for root in ROOTS:
        path = CONTINUOUS_DIR / f"{root}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = (
            pl.scan_parquet(path)
            .select("ts", pl.col("cont_logret").alias(root))
            .filter(pl.col(root).is_not_null())
            .collect()
            .to_pandas()
        )
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
        inventory_rows.append(
            {
                "root": root,
                "rows": len(frame),
                "first_ts": frame["ts"].min(),
                "last_ts": frame["ts"].max(),
            }
        )
        frames.append(frame.set_index("ts"))

    returns = pd.concat(frames, axis=1).sort_index()
    returns = returns.replace([np.inf, -np.inf], np.nan)
    pd.DataFrame(inventory_rows).to_csv(OUTPUT_DIR / "data_inventory.csv", index=False)
    return returns


def compute_daily_realized_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    data = returns.copy()
    data["date"] = data.index.normalize()
    rows = []
    for date, group in data.groupby("date", sort=True):
        for left, right in combinations(ROOTS, 2):
            pair = group[[left, right]].dropna()
            if pair.empty:
                continue
            realized_cov = float((pair[left] * pair[right]).sum())
            rows.append(
                {
                    "date": date,
                    "pair": f"{left}-{right}",
                    "left": left,
                    "right": right,
                    "obs": len(pair),
                    "daily_realized_cov": realized_cov,
                    "annualized_realized_cov": realized_cov * ANNUALIZATION_DAYS,
                }
            )
    cov = pd.DataFrame(rows).sort_values(["pair", "date"])
    cov["smooth_annualized_realized_cov"] = (
        cov.groupby("pair")["annualized_realized_cov"]
        .transform(lambda values: values.rolling(ROLLING_SMOOTH_DAYS, min_periods=5).median())
    )
    return cov


def plot_pairwise_covariance(cov: pd.DataFrame) -> None:
    pairs = [f"{left}-{right}" for left, right in combinations(ROOTS, 2)]
    fig, axes = plt.subplots(5, 2, figsize=(16, 15), sharex=True, constrained_layout=True)
    for ax, pair in zip(axes.ravel(), pairs, strict=True):
        pair_data = cov[cov["pair"].eq(pair)].sort_values("date")
        if pair_data.empty:
            ax.set_title(pair)
            continue
        left, _right = pair.split("-")
        ax.plot(
            pair_data["date"],
            pair_data["annualized_realized_cov"],
            color="#c9ced6",
            lw=0.45,
            alpha=0.6,
            label="daily realized covariance",
        )
        ax.plot(
            pair_data["date"],
            pair_data["smooth_annualized_realized_cov"],
            color=COLORS[left],
            lw=1.2,
            label=f"{ROLLING_SMOOTH_DAYS}d median",
        )
        ax.axhline(0.0, color="#333333", lw=0.8)
        ax.set_title(pair)
        ax.set_ylabel("ann. cov")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Core Metals Pairwise Daily Realized Covariance From 1-Min Continuous Returns")
    fig.savefig(OUTPUT_DIR / "pairwise_daily_realized_covariance.png", dpi=160)
    plt.close(fig)


def plot_covariance_heatmap(cov: pd.DataFrame) -> None:
    latest = (
        cov.dropna(subset=["smooth_annualized_realized_cov"])
        .sort_values("date")
        .groupby("pair")
        .tail(1)
    )
    matrix = pd.DataFrame(np.nan, index=ROOTS, columns=ROOTS, dtype=float)
    for root in ROOTS:
        matrix.loc[root, root] = np.nan
    for row in latest.itertuples(index=False):
        matrix.loc[row.left, row.right] = row.smooth_annualized_realized_cov
        matrix.loc[row.right, row.left] = row.smooth_annualized_realized_cov
    values = matrix.to_numpy(dtype=float)
    finite = np.isfinite(values)
    vmax = np.nanpercentile(np.abs(values[finite]), 95) if finite.any() else 1.0
    vmax = max(vmax, 1e-9)
    fig, ax = plt.subplots(figsize=(7, 5.6), constrained_layout=True)
    image = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(f"Latest {ROLLING_SMOOTH_DAYS}d Median Annualized Covariance")
    ax.set_xticks(np.arange(len(ROOTS)), labels=ROOTS)
    ax.set_yticks(np.arange(len(ROOTS)), labels=ROOTS)
    for i, left in enumerate(ROOTS):
        for j, right in enumerate(ROOTS):
            value = matrix.loc[left, right]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.4f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="annualized covariance")
    fig.savefig(OUTPUT_DIR / "latest_pairwise_covariance_heatmap.png", dpi=160)
    plt.close(fig)


def summarize(cov: pd.DataFrame) -> pd.DataFrame:
    summary = (
        cov.groupby("pair", as_index=False)
        .agg(
            days=("date", "nunique"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            median_obs_per_day=("obs", "median"),
            median_ann_cov=("annualized_realized_cov", "median"),
            p10_ann_cov=("annualized_realized_cov", lambda values: values.quantile(0.10)),
            p90_ann_cov=("annualized_realized_cov", lambda values: values.quantile(0.90)),
            latest_smooth_ann_cov=("smooth_annualized_realized_cov", "last"),
        )
        .sort_values("pair")
    )
    return summary


def write_report(summary: pd.DataFrame) -> None:
    lines = [
        "# Core Metals Pairwise Covariance",
        "",
        "Daily realized covariance is computed from synchronized 1-minute continuous log returns:",
        "",
        "`daily_cov(i, j) = sum_t r_i,t * r_j,t`",
        "",
        f"The plotted value is annualized by multiplying by `{ANNUALIZATION_DAYS:g}`.",
        f"The dark line is a `{ROLLING_SMOOTH_DAYS}`-day rolling median.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Files",
        "",
        "- `pairwise_daily_realized_covariance.png`",
        "- `latest_pairwise_covariance_heatmap.png`",
        "- `daily_pairwise_realized_covariance.csv`",
        "- `pairwise_covariance_summary.csv`",
        "- `data_inventory.csv`",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    returns = load_returns()
    cov = compute_daily_realized_covariance(returns)
    summary = summarize(cov)
    cov.to_csv(OUTPUT_DIR / "daily_pairwise_realized_covariance.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "pairwise_covariance_summary.csv", index=False)
    plot_pairwise_covariance(cov)
    plot_covariance_heatmap(cov)
    write_report(summary)
    print(f"Wrote {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
