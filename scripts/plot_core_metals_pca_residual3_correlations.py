"""Plot rolling correlations after removing the first three robust PCA components."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "experiments" / "HYP-0042-core-metals-robust-ewma-pca"
STATE_PATH = OUTPUT_DIR / "robust_ewma_pca_state.parquet"
RESIDUAL_PC12_PATH = OUTPUT_DIR / "robust_ewma_pca_residuals.parquet"

ROOTS = ["GC", "SI", "HG", "PL", "PA"]
ROLLING_WINDOW = "30D"
MIN_PAIR_OBS = 240

COLORS = {
    "GC": "#b68b00",
    "SI": "#7a8591",
    "HG": "#b35c2e",
    "PL": "#3b6ea8",
    "PA": "#5f8f5f",
}


def load_pc13_residuals() -> pd.DataFrame:
    state = pd.read_parquet(STATE_PATH)
    residuals = pd.read_parquet(RESIDUAL_PC12_PATH)
    state["ts"] = pd.to_datetime(state["ts"], utc=True)
    residuals["ts"] = pd.to_datetime(residuals["ts"], utc=True)

    frame = residuals.merge(
        state[["ts", "pc3_score", *[f"pc3_loading_{root}" for root in ROOTS]]],
        on="ts",
        how="left",
        validate="one_to_one",
    )
    output = pd.DataFrame({"ts": frame["ts"], "observed_count": frame["observed_count"]})
    for root in ROOTS:
        output[root] = frame[f"residual_{root}"] - frame["pc3_score"] * frame[
            f"pc3_loading_{root}"
        ]
        output.loc[output["observed_count"].lt(4), root] = np.nan

    output = output.sort_values("ts").set_index("ts")
    output.to_parquet(
        OUTPUT_DIR / "robust_ewma_pca_residuals_pc1_pc3.parquet",
        index=True,
    )
    return output[ROOTS].replace([np.inf, -np.inf], np.nan)


def compute_rolling_correlations(residuals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for left, right in combinations(ROOTS, 2):
        pair = residuals[[left, right]]
        corr = pair[left].rolling(ROLLING_WINDOW, min_periods=MIN_PAIR_OBS).corr(pair[right])
        obs = pair.notna().all(axis=1).rolling(ROLLING_WINDOW).sum()
        rows.append(
            pd.DataFrame(
                {
                    "ts": corr.index,
                    "pair": f"{left}-{right}",
                    "left": left,
                    "right": right,
                    "rolling_corr": corr.to_numpy(dtype="float64"),
                    "paired_obs": obs.to_numpy(dtype="float64"),
                }
            )
        )
    return pd.concat(rows, ignore_index=True).dropna(subset=["rolling_corr"])


def daily_last_correlations(correlations: pd.DataFrame) -> pd.DataFrame:
    daily_frames = []
    for pair, pair_data in correlations.groupby("pair", sort=True):
        sampled = (
            pair_data.sort_values("ts")
            .set_index("ts")
            .resample("1D")
            .last()
            .dropna(subset=["rolling_corr"])
            .reset_index()
        )
        sampled["pair"] = pair
        daily_frames.append(sampled)
    return pd.concat(daily_frames, ignore_index=True)


def latest_correlation_matrix(correlations: pd.DataFrame) -> pd.DataFrame:
    latest = correlations.sort_values("ts").groupby("pair", as_index=False).tail(1)
    matrix = pd.DataFrame(np.eye(len(ROOTS)), index=ROOTS, columns=ROOTS, dtype=float)
    for row in latest.itertuples(index=False):
        matrix.loc[row.left, row.right] = row.rolling_corr
        matrix.loc[row.right, row.left] = row.rolling_corr
    return matrix


def summarize(correlations: pd.DataFrame) -> pd.DataFrame:
    return (
        correlations.groupby("pair", as_index=False)
        .agg(
            first_ts=("ts", "min"),
            last_ts=("ts", "max"),
            observations=("rolling_corr", "count"),
            median_corr=("rolling_corr", "median"),
            p10_corr=("rolling_corr", lambda values: values.quantile(0.10)),
            p90_corr=("rolling_corr", lambda values: values.quantile(0.90)),
            latest_corr=("rolling_corr", "last"),
            latest_paired_obs=("paired_obs", "last"),
        )
        .sort_values("pair")
    )


def plot_pairwise_correlations(correlations: pd.DataFrame) -> None:
    pairs = [f"{left}-{right}" for left, right in combinations(ROOTS, 2)]
    daily = daily_last_correlations(correlations)

    fig, axes = plt.subplots(5, 2, figsize=(16, 15), sharex=True, constrained_layout=True)
    for ax, pair in zip(axes.ravel(), pairs, strict=True):
        pair_data = daily[daily["pair"].eq(pair)]
        left, _right = pair.split("-")
        ax.plot(
            pair_data["ts"],
            pair_data["rolling_corr"],
            color=COLORS[left],
            linewidth=1.1,
        )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylim(-1.0, 1.0)
        ax.set_title(pair)
        ax.set_ylabel("corr")
        ax.grid(True, alpha=0.25)
    fig.suptitle(
        f"Core Metals PCA Residual Rolling Correlations After Removing PC1-PC3 "
        f"({ROLLING_WINDOW}, min {MIN_PAIR_OBS} paired obs)"
    )
    fig.savefig(OUTPUT_DIR / "residual_pc1_pc3_rolling_pairwise_correlations.png", dpi=170)
    plt.close(fig)


def plot_latest_heatmap(matrix: pd.DataFrame) -> None:
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7, 5.8), constrained_layout=True)
    image = ax.imshow(values, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_title(f"Latest PCA Residual Correlation After PC1-PC3 ({ROLLING_WINDOW})")
    ax.set_xticks(np.arange(len(ROOTS)), labels=ROOTS)
    ax.set_yticks(np.arange(len(ROOTS)), labels=ROOTS)
    for i, left in enumerate(ROOTS):
        for j, right in enumerate(ROOTS):
            ax.text(j, i, f"{matrix.loc[left, right]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, label="correlation")
    fig.savefig(OUTPUT_DIR / "latest_residual_pc1_pc3_correlation_heatmap.png", dpi=170)
    plt.close(fig)


def write_report(summary: pd.DataFrame, latest_matrix: pd.DataFrame) -> None:
    lines = [
        "# Core Metals PCA Residual Rolling Correlations After PC1-PC3",
        "",
        f"State input: `{STATE_PATH}`.",
        f"PC1-PC2 residual input: `{RESIDUAL_PC12_PATH}`.",
        "",
        "Method:",
        "",
        "- Start from robust EWMA PCA residuals after removing PC1-PC2.",
        "- Subtract the PC3 loading times PC3 score at each emitted PCA timestamp.",
        "- Require at least four observed assets before retaining a PC1-PC3 residual row.",
        f"- Pairwise correlations use a `{ROLLING_WINDOW}` rolling window.",
        f"- A correlation is emitted only with at least `{MIN_PAIR_OBS}` paired observations.",
        "- Plotted values are daily last observations for readability.",
        "",
        "## Pair Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Latest Matrix",
        "",
        latest_matrix.to_markdown(floatfmt=".4f"),
        "",
        "## Files",
        "",
        "- `robust_ewma_pca_residuals_pc1_pc3.parquet`",
        "- `residual_pc1_pc3_rolling_correlations.parquet`",
        "- `residual_pc1_pc3_rolling_correlations_daily.csv`",
        "- `residual_pc1_pc3_rolling_correlation_summary.csv`",
        "- `residual_pc1_pc3_rolling_pairwise_correlations.png`",
        "- `latest_residual_pc1_pc3_correlation_heatmap.png`",
    ]
    (OUTPUT_DIR / "residual_pc1_pc3_rolling_correlation_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    residuals = load_pc13_residuals()
    correlations = compute_rolling_correlations(residuals)
    daily = daily_last_correlations(correlations)
    summary = summarize(correlations)
    latest_matrix = latest_correlation_matrix(correlations)

    correlations.to_parquet(
        OUTPUT_DIR / "residual_pc1_pc3_rolling_correlations.parquet",
        index=False,
    )
    daily.to_csv(OUTPUT_DIR / "residual_pc1_pc3_rolling_correlations_daily.csv", index=False)
    summary.to_csv(
        OUTPUT_DIR / "residual_pc1_pc3_rolling_correlation_summary.csv",
        index=False,
    )
    plot_pairwise_correlations(correlations)
    plot_latest_heatmap(latest_matrix)
    write_report(summary, latest_matrix)
    print(f"Wrote PC1-PC3 residual rolling correlations to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
