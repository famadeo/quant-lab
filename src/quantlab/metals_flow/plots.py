# ruff: noqa: PLR2004
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_threshold_sensitivity(summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    x = summary["threshold_m"].astype(float)
    panels = [
        ("bars_per_day", "Bars per day"),
        ("median_duration_seconds", "Median duration, seconds"),
        ("median_trades", "Median trades"),
        ("median_dominant_share", "Median dominant share"),
    ]
    for axis, (column, title) in zip(axes.ravel(), panels, strict=True):
        axis.plot(x, summary[column], marker="o")
        axis.set_title(title)
        axis.set_xlabel("Threshold ($M)")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_rolling_contribution(
    shares: pd.DataFrame,
    bars: pd.DataFrame,
    output_path: Path,
    *,
    window: int = 100,
) -> None:
    times = pd.to_datetime(bars["end_ts"], utc=True)
    rolling = shares.rolling(window, min_periods=max(10, window // 5)).mean()
    fig, axis = plt.subplots(figsize=(12, 5))
    for root in shares.columns:
        axis.plot(times, rolling[root], label=root, linewidth=1.2)
    axis.set_title(f"Rolling {window}-bar notional contribution")
    axis.set_ylabel("Contribution share")
    axis.legend(ncol=3)
    axis.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_mahalanobis(
    anomalies: pd.DataFrame,
    bars: pd.DataFrame,
    output_path: Path,
    *,
    column: str = "md_rolling",
) -> None:
    times = pd.to_datetime(bars["end_ts"], utc=True)
    series = anomalies[column]
    fig, axis = plt.subplots(figsize=(12, 4.5))
    axis.plot(times, series, linewidth=0.9, label=column)
    for quantile in (0.90, 0.95, 0.99):
        threshold = float(series.dropna().quantile(quantile))
        axis.axhline(threshold, linestyle="--", linewidth=0.9, label=f"q{int(quantile * 100)}")
    axis.set_title("Rolling Mahalanobis flow anomaly distance")
    axis.set_ylabel("Distance")
    axis.legend(ncol=4)
    axis.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_concentration(features: pd.DataFrame, bars: pd.DataFrame, output_path: Path) -> None:
    times = pd.to_datetime(bars["end_ts"], utc=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    panels = [
        ("entropy_normalized", "Normalized entropy"),
        ("hhi", "Herfindahl index"),
        ("effective_metals", "Effective active metals"),
    ]
    for axis, (column, title) in zip(axes, panels, strict=True):
        axis.plot(times, features[column], linewidth=0.9)
        axis.set_title(title)
        axis.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_geometry(
    coords: pd.DataFrame,
    color: pd.Series,
    output_path: Path,
    *,
    title: str,
) -> None:
    if coords.shape[1] < 2:
        return
    aligned_color = color.reindex(coords.index)
    fig, axis = plt.subplots(figsize=(8, 6))
    scatter = axis.scatter(
        coords.iloc[:, 0],
        coords.iloc[:, 1],
        c=aligned_color,
        s=12,
        cmap="viridis",
        alpha=0.8,
    )
    axis.set_title(title)
    axis.set_xlabel(coords.columns[0])
    axis.set_ylabel(coords.columns[1])
    fig.colorbar(scatter, ax=axis, label=color.name or "color")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_event_study(event_summary: pd.DataFrame, output_path: Path) -> None:
    if event_summary.empty:
        return
    fig, axis = plt.subplots(figsize=(10, 5))
    for root, group in event_summary.groupby("root"):
        axis.plot(group["relative_bar"], group["mean"] * 10_000.0, label=root)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_title("Mean cumulative log return around flow anomalies")
    axis.set_xlabel("Relative bar")
    axis.set_ylabel("Mean cumulative return, bps")
    axis.legend(ncol=3)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_forward_heatmap(
    study: pd.DataFrame,
    output_path: Path,
    *,
    feature: str,
    horizon: int,
) -> None:
    frame = study[(study["feature"] == feature) & (study["horizon"] == horizon)]
    if frame.empty:
        return
    pivot = frame.pivot_table(index="root", columns="bucket", values="mean_bps")
    fig, axis = plt.subplots(figsize=(10, 4))
    values = pivot.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
    image = axis.imshow(values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axis.set_xticks(range(len(pivot.columns)), labels=pivot.columns)
    axis.set_yticks(range(len(pivot.index)), labels=pivot.index)
    axis.set_title(f"{feature}, horizon {horizon}: future return by bucket")
    axis.set_xlabel("Bucket")
    axis.set_ylabel("Root")
    fig.colorbar(image, ax=axis, label="Mean future return, bps")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_trade_size_disagreement(
    disagreement: pd.DataFrame,
    bars: pd.DataFrame,
    output_path: Path,
) -> None:
    times = pd.to_datetime(bars["end_ts"], utc=True)
    columns = [column for column in disagreement.columns if column.endswith("share")]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(times, disagreement["large_small_l1_distance"], linewidth=0.9)
    axes[0].set_title("Large-flow minus small-flow L1 distance")
    axes[0].grid(alpha=0.25)
    for column in columns:
        label = column.split("_", maxsplit=1)[0]
        axes[1].plot(
            times,
            disagreement[column].rolling(100, min_periods=20).mean(),
            label=label,
        )
    axes[1].set_title("Rolling large-flow minus small-flow contribution by root")
    axes[1].legend(ncol=3)
    axes[1].grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_fair_value_zscores(zscores: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(
        len(zscores.columns),
        1,
        figsize=(12, 1.8 * len(zscores.columns)),
        sharex=True,
    )
    if len(zscores.columns) == 1:
        axes = [axes]
    for axis, root in zip(axes, zscores.columns, strict=True):
        axis.plot(zscores.index, zscores[root], linewidth=0.8)
        axis.axhline(2.0, color="red", linestyle="--", linewidth=0.7)
        axis.axhline(-2.0, color="red", linestyle="--", linewidth=0.7)
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_ylabel(root)
        axis.grid(alpha=0.25)
    axes[0].set_title("Rolling relative-value residual z-scores")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
