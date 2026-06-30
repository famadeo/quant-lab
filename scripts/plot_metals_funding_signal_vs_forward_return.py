from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from run_metals_funding_adjusted_drift_backtest import (  # noqa: E402
    OUTPUT_DIR,
    ROOTS,
    add_pressure_signals,
    load_source,
)

PLOT_DIR = OUTPUT_DIR / "signal_vs_forward_return"

HORIZONS = {"1h": 1, "4h": 4, "1d": 24, "3d": 72}
MAX_SCATTER_POINTS_PER_ROOT = 8_000
DECILES = 10

COLORS = {
    "GC": "#b8860b",
    "SI": "#6f7f8f",
    "HG": "#b15a2a",
    "PL": "#2f7d8c",
    "PA": "#7a4e9b",
}


@dataclass(frozen=True)
class SignalConfig:
    name: str
    target_months: int
    lookback: str
    score_method: str
    funding_scaled_filter: float

    @property
    def score_col(self) -> str:
        return f"{self.lookback}_{self.score_method}"

    @property
    def filter_col(self) -> str:
        return f"{self.lookback}_funding_scaled"


CONFIGS = [
    SignalConfig(
        name="robust_1m_3d_pressure_vol_scaled_filt0p02",
        target_months=1,
        lookback="3d",
        score_method="pressure_vol_scaled",
        funding_scaled_filter=0.02,
    ),
    SignalConfig(
        name="robust_3m_3d_pressure_vol_scaled_filt0p02",
        target_months=3,
        lookback="3d",
        score_method="pressure_vol_scaled",
        funding_scaled_filter=0.02,
    ),
    SignalConfig(
        name="full_best_6m_3d_pressure_vol_scaled_filt0",
        target_months=6,
        lookback="3d",
        score_method="pressure_vol_scaled",
        funding_scaled_filter=0.00,
    ),
]


def forward_sum(series: pd.Series, periods: int) -> pd.Series:
    return series.iloc[::-1].rolling(periods, min_periods=periods).sum().iloc[::-1]


def add_forward_returns(panel: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for (_root, _target), group in panel.groupby(["root", "target_months"], sort=False):
        data = group.sort_values("ts").copy()
        for horizon, periods in HORIZONS.items():
            data[f"fwd_excess_{horizon}"] = forward_sum(
                data["excess_after_funding_next_bar"], periods
            )
            data[f"fwd_price_{horizon}"] = forward_sum(data["log_return_next_bar"], periods)
            data[f"fwd_funding_paid_{horizon}"] = forward_sum(
                data["funding_paid_next_bar"], periods
            )
        frames.append(data)
    return pd.concat(frames, ignore_index=True)


def config_data(panel: pd.DataFrame, config: SignalConfig, horizon: str) -> pd.DataFrame:
    cols = [
        "root",
        "target_months",
        "ts",
        config.score_col,
        config.filter_col,
        f"fwd_excess_{horizon}",
        f"fwd_price_{horizon}",
        f"fwd_funding_paid_{horizon}",
    ]
    data = panel.loc[panel["target_months"].eq(config.target_months), cols].copy()
    data = data.rename(
        columns={
            config.score_col: "signal",
            config.filter_col: "funding_scaled",
            f"fwd_excess_{horizon}": "forward_excess",
            f"fwd_price_{horizon}": "forward_price",
            f"fwd_funding_paid_{horizon}": "forward_funding_paid",
        }
    )
    data = data[data["funding_scaled"] >= config.funding_scaled_filter]
    data = data.dropna(subset=["signal", "forward_excess", "forward_price"])
    data = data[np.isfinite(data["signal"]) & np.isfinite(data["forward_excess"])]
    data["horizon"] = horizon
    data["config"] = config.name
    return data


def decile_summary_for(data: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for root, group in data.groupby("root", sort=True):
        if group["signal"].nunique() < DECILES:
            continue
        work = group.copy()
        work["decile"] = pd.qcut(work["signal"], DECILES, labels=False, duplicates="drop")
        work = work.dropna(subset=["decile"])
        if work.empty:
            continue
        summary = (
            work.groupby("decile", as_index=False)
            .agg(
                n=("forward_excess", "size"),
                signal_mean=("signal", "mean"),
                forward_excess_mean_bp=("forward_excess", lambda values: values.mean() * 10_000.0),
                forward_excess_median_bp=(
                    "forward_excess",
                    lambda values: values.median() * 10_000.0,
                ),
                forward_price_mean_bp=("forward_price", lambda values: values.mean() * 10_000.0),
                forward_funding_paid_mean_bp=(
                    "forward_funding_paid",
                    lambda values: values.mean() * 10_000.0,
                ),
                positive_excess_fraction=("forward_excess", lambda values: (values > 0).mean()),
            )
        )
        summary["decile"] = summary["decile"].astype(int) + 1
        summary.insert(0, "root", root)
        frames.append(summary)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_config_horizon(data: pd.DataFrame, deciles: pd.DataFrame) -> dict[str, object]:
    rows = []
    for root, group in data.groupby("root", sort=True):
        root_deciles = deciles[deciles["root"].eq(root)]
        low = root_deciles[root_deciles["decile"].eq(1)]["forward_excess_mean_bp"]
        high = root_deciles[root_deciles["decile"].eq(root_deciles["decile"].max())][
            "forward_excess_mean_bp"
        ]
        rows.append(
            {
                "root": root,
                "n": len(group),
                "corr_signal_forward_excess": group["signal"].corr(group["forward_excess"]),
                "mean_forward_excess_bp": group["forward_excess"].mean() * 10_000.0,
                "high_minus_low_decile_bp": float(high.iloc[0] - low.iloc[0])
                if len(low) and len(high)
                else np.nan,
                "positive_excess_fraction": float((group["forward_excess"] > 0).mean()),
            }
        )
    return rows


def plot_scatter_with_bins(
    data: pd.DataFrame, deciles: pd.DataFrame, config: SignalConfig, horizon: str
) -> None:
    fig, axes = plt.subplots(
        len(ROOTS), 1, figsize=(12, 12), sharex=True, constrained_layout=True
    )
    for ax, root in zip(axes, ROOTS, strict=True):
        root_data = data[data["root"].eq(root)].copy()
        if root_data.empty:
            ax.set_ylabel(root)
            continue
        if len(root_data) > MAX_SCATTER_POINTS_PER_ROOT:
            root_data = root_data.sample(MAX_SCATTER_POINTS_PER_ROOT, random_state=42)
        ax.scatter(
            root_data["signal"],
            root_data["forward_excess"] * 10_000.0,
            s=5,
            alpha=0.15,
            color=COLORS[root],
            edgecolors="none",
        )
        root_deciles = deciles[deciles["root"].eq(root)].sort_values("decile")
        if not root_deciles.empty:
            ax.plot(
                root_deciles["signal_mean"],
                root_deciles["forward_excess_mean_bp"],
                color="#111111",
                lw=1.6,
                marker="o",
                ms=3,
                label="decile mean",
            )
        ax.axhline(0.0, color="#333333", lw=0.8)
        ax.axvline(0.0, color="#333333", lw=0.8)
        ax.set_ylabel(f"{root}\nfwd bp")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel(config.score_col)
    fig.suptitle(f"{config.name}: signal vs {horizon} forward excess return")
    fig.savefig(PLOT_DIR / f"{config.name}_{horizon}_scatter_decile_overlay.png", dpi=160)
    plt.close(fig)


def plot_decile_profiles(decile_frames: pd.DataFrame, config: SignalConfig) -> None:
    data = decile_frames[decile_frames["config"].eq(config.name)].copy()
    if data.empty:
        return
    fig, axes = plt.subplots(
        len(HORIZONS), 1, figsize=(12, 10), sharex=True, constrained_layout=True
    )
    for ax, horizon in zip(axes, HORIZONS, strict=True):
        horizon_data = data[data["horizon"].eq(horizon)]
        for root in ROOTS:
            root_data = horizon_data[horizon_data["root"].eq(root)].sort_values("decile")
            if root_data.empty:
                continue
            ax.plot(
                root_data["decile"],
                root_data["forward_excess_mean_bp"],
                lw=1.2,
                marker="o",
                ms=3,
                color=COLORS[root],
                label=root,
            )
        ax.axhline(0.0, color="#333333", lw=0.8)
        ax.set_ylabel(f"{horizon}\nbp")
        ax.grid(True, alpha=0.25)
    axes[0].legend(ncol=len(ROOTS), loc="upper left", fontsize=8)
    axes[-1].set_xlabel("signal decile")
    fig.suptitle(f"{config.name}: decile mean forward excess return")
    fig.savefig(PLOT_DIR / f"{config.name}_decile_profiles.png", dpi=160)
    plt.close(fig)


def plot_high_low_heatmap(summary: pd.DataFrame, config: SignalConfig) -> None:
    data = summary[summary["config"].eq(config.name)]
    if data.empty:
        return
    matrix = data.pivot(index="root", columns="horizon", values="high_minus_low_decile_bp").reindex(
        index=ROOTS,
        columns=list(HORIZONS),
    )
    values = matrix.to_numpy(dtype=float)
    finite = np.isfinite(values)
    vmax = np.nanpercentile(np.abs(values[finite]), 95) if finite.any() else 1.0
    vmax = max(vmax, 1e-9)
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    image = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title(f"{config.name}: high-minus-low signal decile fwd excess bp")
    ax.set_xticks(np.arange(len(HORIZONS)), labels=list(HORIZONS))
    ax.set_yticks(np.arange(len(ROOTS)), labels=ROOTS)
    for i, root in enumerate(ROOTS):
        for j, horizon in enumerate(HORIZONS):
            value = matrix.loc[root, horizon]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="bp")
    fig.savefig(PLOT_DIR / f"{config.name}_high_minus_low_heatmap.png", dpi=160)
    plt.close(fig)


def write_report(summary: pd.DataFrame) -> None:
    lines = [
        "# Signal vs Forward Return Plots",
        "",
        "The x-axis signal is the funding-adjusted drift score used in `HYP-0039`.",
        "The y-axis is forward realized return after realized path funding paid.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4f") if not summary.empty else "No rows.",
        "",
        "## Files",
        "",
        "- `*_scatter_decile_overlay.png`: root-level scatter plus decile mean overlay.",
        "- `*_decile_profiles.png`: decile mean forward excess by root and horizon.",
        "- `*_high_minus_low_heatmap.png`: top-minus-bottom signal decile by root/horizon.",
        "- `signal_vs_forward_summary.csv`",
        "- `signal_vs_forward_deciles.csv`",
    ]
    (PLOT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    panel = add_forward_returns(add_pressure_signals(load_source()))
    summary_rows = []
    decile_frames = []

    for config in CONFIGS:
        print(f"Plotting {config.name}", flush=True)
        config_deciles = []
        for horizon in HORIZONS:
            data = config_data(panel, config, horizon)
            deciles = decile_summary_for(data)
            if deciles.empty:
                continue
            deciles.insert(0, "horizon", horizon)
            deciles.insert(0, "config", config.name)
            decile_frames.append(deciles)
            config_deciles.append(deciles)
            summary_rows.extend(
                {
                    "config": config.name,
                    "target_months": config.target_months,
                    "lookback": config.lookback,
                    "score_method": config.score_method,
                    "funding_scaled_filter": config.funding_scaled_filter,
                    "horizon": horizon,
                    **row,
                }
                for row in summarize_config_horizon(data, deciles)
            )
            if horizon in {"1h", "1d"}:
                plot_scatter_with_bins(data, deciles, config, horizon)
        if config_deciles:
            plot_decile_profiles(pd.concat(config_deciles, ignore_index=True), config)

    summary = pd.DataFrame(summary_rows)
    decile_summary = (
        pd.concat(decile_frames, ignore_index=True) if decile_frames else pd.DataFrame()
    )
    summary.to_csv(PLOT_DIR / "signal_vs_forward_summary.csv", index=False)
    decile_summary.to_csv(PLOT_DIR / "signal_vs_forward_deciles.csv", index=False)
    for config in CONFIGS:
        plot_high_low_heatmap(summary, config)
    write_report(summary)
    print(f"Wrote {PLOT_DIR}", flush=True)


if __name__ == "__main__":
    main()
