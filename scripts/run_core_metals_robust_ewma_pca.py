"""Run robust EWMA rolling PCA on core metals 5-minute returns."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = REPO_ROOT / "experiments" / "HYP-0041-core-metals-5m-log-returns"
OUTPUT_DIR = REPO_ROOT / "experiments" / "HYP-0042-core-metals-robust-ewma-pca"

WIDE_RETURNS_PATH = INPUT_DIR / "core_metals_5m_log_returns_wide.parquet"
LONG_RETURNS_PATH = INPUT_DIR / "core_metals_5m_log_returns_long.parquet"

ROOTS = ["GC", "SI", "HG", "PL", "PA"]
COLORS = {
    "GC": "#b68b00",
    "SI": "#7a8591",
    "HG": "#b35c2e",
    "PL": "#3b6ea8",
    "PA": "#5f8f5f",
}

BARS_PER_DAY = 288
VOL_HALFLIFE_BARS = 3 * BARS_PER_DAY
VOL_MIN_OBS = BARS_PER_DAY
PCA_HALFLIFE_BARS = 10 * BARS_PER_DAY
PCA_MIN_PAIR_OBS = 5 * BARS_PER_DAY
PCA_EVALUATE_EVERY_BARS = 12
CLIP_Z = 6.0
SHRINKAGE_ALPHA = 0.10
N_COMPONENTS = 3
RESIDUAL_COMPONENTS = 2
EPS = 1e-10


def load_returns_and_observation_mask() -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = pd.read_parquet(WIDE_RETURNS_PATH)
    wide["ts"] = pd.to_datetime(wide["ts"], utc=True)
    returns = wide.sort_values("ts").set_index("ts")[ROOTS].astype("float64")

    long = pd.read_parquet(
        LONG_RETURNS_PATH,
        columns=["root", "ts", "had_observed_5m_bar"],
    )
    long["ts"] = pd.to_datetime(long["ts"], utc=True)
    mask = (
        long.pivot(index="ts", columns="root", values="had_observed_5m_bar")
        .reindex(index=returns.index, columns=ROOTS)
        .fillna(False)
        .astype(bool)
    )
    return returns, mask


def standardized_observed_returns(
    returns: pd.DataFrame,
    observed_mask: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observed_returns = returns.where(observed_mask)
    vol = (
        observed_returns.ewm(
            halflife=VOL_HALFLIFE_BARS,
            min_periods=VOL_MIN_OBS,
            adjust=False,
            ignore_na=True,
        )
        .std()
        .shift(1)
    )
    standardized = (observed_returns / vol).replace([np.inf, -np.inf], np.nan)
    clipped = standardized.clip(lower=-CLIP_Z, upper=CLIP_Z)
    return observed_returns, vol, clipped


def project_to_correlation(matrix: np.ndarray) -> np.ndarray:
    corr = (matrix + matrix.T) * 0.5
    np.fill_diagonal(corr, 1.0)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    clipped_values = np.clip(eigenvalues, EPS, None)
    projected = (eigenvectors * clipped_values) @ eigenvectors.T
    scale = np.sqrt(np.clip(np.diag(projected), EPS, None))
    projected = projected / np.outer(scale, scale)
    projected = (projected + projected.T) * 0.5
    np.fill_diagonal(projected, 1.0)
    return projected


def effective_rank(eigenvalues: np.ndarray) -> float:
    weights = np.clip(eigenvalues, EPS, None)
    weights = weights / weights.sum()
    entropy = -(weights * np.log(weights)).sum()
    return float(np.exp(entropy))


def robust_ewma_pca(clipped: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:  # noqa: PLR0915
    values = clipped.to_numpy(dtype="float64")
    timestamps = clipped.index.to_numpy()
    n_rows, n_assets = values.shape
    lambda_ = float(np.exp(np.log(0.5) / PCA_HALFLIFE_BARS))

    covariance = np.eye(n_assets, dtype="float64")
    pair_obs = np.zeros((n_assets, n_assets), dtype="int64")
    previous_vectors: np.ndarray | None = None

    observed_counts = np.zeros(n_rows, dtype="int16")
    effective_ranks = np.full(n_rows, np.nan, dtype="float64")
    eigenvalue_output = np.full((n_rows, N_COMPONENTS), np.nan, dtype="float64")
    explained_output = np.full((n_rows, N_COMPONENTS), np.nan, dtype="float64")
    loading_output = np.full((n_rows, N_COMPONENTS, n_assets), np.nan, dtype="float64")
    score_output = np.full((n_rows, N_COMPONENTS), np.nan, dtype="float64")
    residual_output = np.full((n_rows, n_assets), np.nan, dtype="float64")
    residual_norms = np.full(n_rows, np.nan, dtype="float64")
    max_abs_residuals = np.full(n_rows, np.nan, dtype="float64")

    identity = np.eye(n_assets, dtype="float64")
    for row_index in range(n_rows):
        row = values[row_index]
        valid = np.isfinite(row)
        valid_count = int(valid.sum())
        observed_counts[row_index] = valid_count

        if valid_count:
            valid_index = np.flatnonzero(valid)
            ix = np.ix_(valid_index, valid_index)
            covariance[ix] = lambda_ * covariance[ix] + (1.0 - lambda_) * np.outer(
                row[valid_index],
                row[valid_index],
            )
            pair_obs[ix] += 1

        if pair_obs.min() < PCA_MIN_PAIR_OBS:
            continue
        if row_index % PCA_EVALUATE_EVERY_BARS != 0 and row_index != n_rows - 1:
            continue

        scale = np.sqrt(np.clip(np.diag(covariance), EPS, None))
        corr = covariance / np.outer(scale, scale)
        corr = project_to_correlation(corr)
        corr = (1.0 - SHRINKAGE_ALPHA) * corr + SHRINKAGE_ALPHA * identity

        eigenvalues, eigenvectors = np.linalg.eigh(corr)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        if previous_vectors is not None:
            for component in range(n_assets):
                if np.dot(eigenvectors[:, component], previous_vectors[:, component]) < 0:
                    eigenvectors[:, component] *= -1.0
        previous_vectors = eigenvectors.copy()

        explained = eigenvalues / np.clip(eigenvalues.sum(), EPS, None)
        effective_ranks[row_index] = effective_rank(eigenvalues)
        eigenvalue_output[row_index] = eigenvalues[:N_COMPONENTS]
        explained_output[row_index] = explained[:N_COMPONENTS]
        loading_output[row_index] = eigenvectors[:, :N_COMPONENTS].T

        if valid_count >= RESIDUAL_COMPONENTS + 1:
            filled_row = np.where(valid, row, 0.0)
            scores = eigenvectors.T @ filled_row
            common = eigenvectors[:, :RESIDUAL_COMPONENTS] @ scores[:RESIDUAL_COMPONENTS]
            residual_values = filled_row - common
            residual_values[~valid] = np.nan
            residual_norm = np.sqrt(
                n_assets / valid_count * np.nansum(np.square(residual_values))
            )
            residual_norms[row_index] = residual_norm
            max_abs_residuals[row_index] = np.nanmax(np.abs(residual_values))
            score_output[row_index] = scores[:N_COMPONENTS]
            residual_output[row_index] = residual_values

    state_data: dict[str, np.ndarray] = {
        "ts": timestamps,
        "observed_count": observed_counts,
        "effective_rank": effective_ranks,
    }
    for component in range(N_COMPONENTS):
        pc_name = f"pc{component + 1}"
        state_data[f"{pc_name}_eigenvalue"] = eigenvalue_output[:, component]
        state_data[f"{pc_name}_explained_variance"] = explained_output[:, component]
        state_data[f"{pc_name}_score"] = score_output[:, component]
        for asset_index, root in enumerate(ROOTS):
            state_data[f"{pc_name}_loading_{root}"] = loading_output[
                :,
                component,
                asset_index,
            ]

    residual_data: dict[str, np.ndarray] = {
        "ts": timestamps,
        "observed_count": observed_counts,
        "residual_norm": residual_norms,
        "max_abs_residual": max_abs_residuals,
    }
    for asset_index, root in enumerate(ROOTS):
        residual_data[f"residual_{root}"] = residual_output[:, asset_index]

    state_frame = pd.DataFrame(state_data)
    residual_frame = pd.DataFrame(residual_data)
    emitted = state_frame["pc1_eigenvalue"].notna()
    state_frame = state_frame.loc[emitted].reset_index(drop=True)
    residual_frame = residual_frame.loc[emitted].reset_index(drop=True)
    return state_frame, residual_frame


def daily_last(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["ts"] = pd.to_datetime(data["ts"], utc=True)
    return data.set_index("ts").resample("1D").last().dropna(how="all")


def plot_explained_variance(state: pd.DataFrame) -> None:
    daily = daily_last(state)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for component in range(N_COMPONENTS):
        column = f"pc{component + 1}_explained_variance"
        axes[0].plot(
            daily.index,
            daily[column],
            label=f"PC{component + 1}",
            linewidth=1.2,
        )
    axes[0].set_title("Robust EWMA PCA explained variance")
    axes[0].set_ylabel("Share of variance")
    axes[0].legend(ncol=N_COMPONENTS, loc="upper left", frameon=False)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(daily.index, daily["effective_rank"], color="#333333", linewidth=1.2)
    axes[1].set_title("Effective rank")
    axes[1].set_ylabel("Effective number of PCs")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "explained_variance_effective_rank.png", dpi=170)
    plt.close(fig)


def plot_loading_panel(state: pd.DataFrame, component: int) -> None:
    daily = daily_last(state)
    pc_name = f"pc{component}"
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for root in ROOTS:
        ax.plot(
            daily.index,
            daily[f"{pc_name}_loading_{root}"],
            label=root,
            color=COLORS[root],
            linewidth=1.2,
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(f"Robust EWMA PCA {pc_name.upper()} loadings")
    ax.set_ylabel("Loading")
    ax.set_xlabel("Date")
    ax.legend(ncol=len(ROOTS), loc="upper left", frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{pc_name}_loadings_over_time.png", dpi=170)
    plt.close(fig)


def plot_pc2_pc3_loadings(state: pd.DataFrame) -> None:
    daily = daily_last(state)
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for axis, component in zip(axes, [2, 3], strict=True):
        pc_name = f"pc{component}"
        for root in ROOTS:
            axis.plot(
                daily.index,
                daily[f"{pc_name}_loading_{root}"],
                label=root,
                color=COLORS[root],
                linewidth=1.1,
            )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(f"{pc_name.upper()} loadings")
        axis.set_ylabel("Loading")
        axis.grid(True, alpha=0.25)
    axes[0].legend(ncol=len(ROOTS), loc="upper left", frameon=False)
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pc2_pc3_loadings_over_time.png", dpi=170)
    plt.close(fig)


def plot_scores(state: pd.DataFrame) -> None:
    data = state.copy()
    data["ts"] = pd.to_datetime(data["ts"], utc=True)
    scores = data.set_index("ts")[[f"pc{i}_score" for i in range(1, N_COMPONENTS + 1)]]
    cumulative_scores = scores.fillna(0.0).resample("1D").sum().cumsum()
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for component in range(N_COMPONENTS):
        column = f"pc{component + 1}_score"
        ax.plot(
            cumulative_scores.index,
            cumulative_scores[column],
            label=f"PC{component + 1}",
            linewidth=1.2,
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Cumulative robust PCA factor scores")
    ax.set_ylabel("Cumulative standardized score")
    ax.set_xlabel("Date")
    ax.legend(ncol=N_COMPONENTS, loc="upper left", frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pc_scores_cumulative.png", dpi=170)
    plt.close(fig)


def plot_residual_dislocation(residuals: pd.DataFrame) -> None:
    data = residuals.copy()
    data["ts"] = pd.to_datetime(data["ts"], utc=True)
    indexed = data.set_index("ts")
    daily_norm = indexed["residual_norm"].resample("1D").agg(["median", "max"])
    daily_norm["p95"] = indexed["residual_norm"].resample("1D").quantile(0.95)

    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.plot(daily_norm.index, daily_norm["median"], label="daily median", linewidth=1.0)
    ax.plot(daily_norm.index, daily_norm["p95"], label="daily p95", linewidth=1.2)
    ax.plot(daily_norm.index, daily_norm["max"], label="daily max", linewidth=0.9, alpha=0.7)
    ax.set_title("Residual dislocation norm after removing PC1-PC2")
    ax.set_ylabel("Residual norm")
    ax.set_xlabel("Date")
    ax.legend(ncol=3, loc="upper left", frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "residual_dislocation_norm.png", dpi=170)
    plt.close(fig)


def plot_asset_residuals(residuals: pd.DataFrame) -> None:
    data = residuals.copy()
    data["ts"] = pd.to_datetime(data["ts"], utc=True)
    residual_columns = [f"residual_{root}" for root in ROOTS]
    daily_p95_abs = (
        data.set_index("ts")[residual_columns]
        .abs()
        .resample("1D")
        .quantile(0.95)
        .rename(columns={f"residual_{root}": root for root in ROOTS})
    )
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for root in ROOTS:
        ax.plot(
            daily_p95_abs.index,
            daily_p95_abs[root],
            label=root,
            color=COLORS[root],
            linewidth=1.1,
        )
    ax.set_title("Daily p95 absolute residual by asset")
    ax.set_ylabel("Absolute residual")
    ax.set_xlabel("Date")
    ax.legend(ncol=len(ROOTS), loc="upper left", frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "asset_residual_abs_p95.png", dpi=170)
    plt.close(fig)


def plot_latest_loadings(state: pd.DataFrame) -> None:
    latest = state.dropna(subset=["pc1_eigenvalue"]).iloc[-1]
    x = np.arange(len(ROOTS))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11, 6))
    for component in range(N_COMPONENTS):
        values = [latest[f"pc{component + 1}_loading_{root}"] for root in ROOTS]
        ax.bar(x + (component - 1) * width, values, width=width, label=f"PC{component + 1}")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ROOTS)
    ax.set_title(f"Latest robust EWMA PCA loadings: {latest['ts']}")
    ax.set_ylabel("Loading")
    ax.legend(ncol=N_COMPONENTS, loc="upper left", frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "latest_loadings_bar.png", dpi=170)
    plt.close(fig)


def summarize_outputs(
    state: pd.DataFrame,
    residuals: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_state = state.dropna(subset=["pc1_eigenvalue"]).copy()
    latest = valid_state.iloc[-1]
    latest_rows = [
        {
            "ts": latest["ts"],
            "component": f"PC{component}",
            "root": root,
            "loading": latest[f"pc{component}_loading_{root}"],
            "explained_variance": latest[f"pc{component}_explained_variance"],
            "eigenvalue": latest[f"pc{component}_eigenvalue"],
            "effective_rank": latest["effective_rank"],
        }
        for component in range(1, N_COMPONENTS + 1)
        for root in ROOTS
    ]
    latest_loadings = pd.DataFrame(latest_rows)

    residual_columns = [f"residual_{root}" for root in ROOTS]
    top = (
        residuals.dropna(subset=["residual_norm"])
        .sort_values("residual_norm", ascending=False)
        .head(50)
        [["ts", "observed_count", "residual_norm", "max_abs_residual", *residual_columns]]
        .reset_index(drop=True)
    )
    return latest_loadings, top


def write_report(
    returns: pd.DataFrame,
    state: pd.DataFrame,
    residuals: pd.DataFrame,
    latest_loadings: pd.DataFrame,
    top_dislocations: pd.DataFrame,
) -> None:
    valid_state = state.dropna(subset=["pc1_eigenvalue"])
    latest_state = valid_state.iloc[-1]
    residual_summary = residuals["residual_norm"].describe(
        percentiles=[0.5, 0.9, 0.95, 0.99]
    )
    report = [
        "# Core Metals Robust EWMA Rolling PCA",
        "",
        f"Completed at `{datetime.now(UTC).isoformat()}`.",
        "",
        "## Data",
        "",
        f"- Input: `{WIDE_RETURNS_PATH}` and `{LONG_RETURNS_PATH}`.",
        f"- Assets: `{', '.join(ROOTS)}`.",
        f"- Span: `{returns.index.min()}` to `{returns.index.max()}`.",
        f"- Rows: `{len(returns):,}` five-minute timestamps.",
        f"- PCA diagnostics: `{len(valid_state):,}` emitted rows from "
        f"`{valid_state['ts'].min()}` to `{valid_state['ts'].max()}`.",
        "",
        "## Method",
        "",
        "- Use observed 5-minute returns only for PCA estimation; stale aligned bars are masked.",
        "- Standardize each asset by lagged EWMA volatility.",
        f"- EWMA volatility half-life: `{VOL_HALFLIFE_BARS}` bars, "
        f"minimum `{VOL_MIN_OBS}` observations.",
        f"- Clip standardized returns to `+/-{CLIP_Z:g}` before correlation estimation.",
        f"- Estimate EWMA correlation with half-life `{PCA_HALFLIFE_BARS}` bars.",
        f"- Require `{PCA_MIN_PAIR_OBS}` pair observations before emitting PCA state.",
        f"- Update EWMA every 5-minute bar; emit PCA diagnostics every "
        f"`{PCA_EVALUATE_EVERY_BARS}` bars.",
        f"- Shrink correlation matrix toward identity with alpha `{SHRINKAGE_ALPHA:g}`.",
        "- Project to nearest positive semi-definite correlation matrix before eigendecomposition.",
        "- Align eigenvector signs to the prior timestamp.",
        f"- Residual dislocation removes the first `{RESIDUAL_COMPONENTS}` PCs.",
        "",
        "## Latest State",
        "",
        f"- Latest PCA timestamp: `{latest_state['ts']}`.",
        f"- PC1 explained variance: `{latest_state['pc1_explained_variance']:.4f}`.",
        f"- PC2 explained variance: `{latest_state['pc2_explained_variance']:.4f}`.",
        f"- PC3 explained variance: `{latest_state['pc3_explained_variance']:.4f}`.",
        f"- Effective rank: `{latest_state['effective_rank']:.4f}`.",
        "",
        "### Latest Loadings",
        "",
        latest_loadings.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Residual Norm Summary",
        "",
        residual_summary.to_frame("value").to_markdown(floatfmt=".4f"),
        "",
        "## Top Residual Dislocations",
        "",
        top_dislocations.head(10).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Files",
        "",
        "- `robust_ewma_pca_state.parquet`",
        "- `robust_ewma_pca_residuals.parquet`",
        "- `latest_loadings.csv`",
        "- `top_residual_dislocations.csv`",
        "- `explained_variance_effective_rank.png`",
        "- `pc1_loadings_over_time.png`",
        "- `pc2_pc3_loadings_over_time.png`",
        "- `pc_scores_cumulative.png`",
        "- `residual_dislocation_norm.png`",
        "- `asset_residual_abs_p95.png`",
        "- `latest_loadings_bar.png`",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    returns, observed_mask = load_returns_and_observation_mask()
    observed_returns, vol, clipped = standardized_observed_returns(returns, observed_mask)

    coverage = pd.DataFrame(
        {
            "root": ROOTS,
            "total_rows": len(returns),
            "observed_rows": [int(observed_mask[root].sum()) for root in ROOTS],
            "vol_valid_rows": [int(vol[root].notna().sum()) for root in ROOTS],
            "standardized_valid_rows": [int(clipped[root].notna().sum()) for root in ROOTS],
            "mean_observed_5m_logret": [float(observed_returns[root].mean()) for root in ROOTS],
            "std_observed_5m_logret": [float(observed_returns[root].std()) for root in ROOTS],
        }
    )

    print("Running robust EWMA PCA", flush=True)
    state, residuals = robust_ewma_pca(clipped)

    state.to_parquet(OUTPUT_DIR / "robust_ewma_pca_state.parquet", index=False)
    residuals.to_parquet(OUTPUT_DIR / "robust_ewma_pca_residuals.parquet", index=False)
    coverage.to_csv(OUTPUT_DIR / "input_coverage.csv", index=False)

    latest_loadings, top_dislocations = summarize_outputs(state, residuals)
    latest_loadings.to_csv(OUTPUT_DIR / "latest_loadings.csv", index=False)
    top_dislocations.to_csv(OUTPUT_DIR / "top_residual_dislocations.csv", index=False)

    print("Writing plots", flush=True)
    plot_explained_variance(state)
    plot_loading_panel(state, component=1)
    plot_pc2_pc3_loadings(state)
    plot_scores(state)
    plot_residual_dislocation(residuals)
    plot_asset_residuals(residuals)
    plot_latest_loadings(state)
    write_report(returns, state, residuals, latest_loadings, top_dislocations)

    print(f"Wrote robust PCA outputs to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
