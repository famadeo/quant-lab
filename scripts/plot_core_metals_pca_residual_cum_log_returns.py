"""Plot cumulative log returns of PC1-PC2 PCA residuals for core metals."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = REPO_ROOT / "experiments" / "HYP-0041-core-metals-5m-log-returns"
OUTPUT_DIR = REPO_ROOT / "experiments" / "HYP-0042-core-metals-robust-ewma-pca"

WIDE_RETURNS_PATH = INPUT_DIR / "core_metals_5m_log_returns_wide.parquet"
LONG_RETURNS_PATH = INPUT_DIR / "core_metals_5m_log_returns_long.parquet"
RESIDUALS_PATH = OUTPUT_DIR / "robust_ewma_pca_residuals.parquet"

ROOTS = ["GC", "SI", "HG", "PL", "PA"]
BARS_PER_DAY = 288
VOL_HALFLIFE_BARS = 3 * BARS_PER_DAY
VOL_MIN_OBS = BARS_PER_DAY

COLORS = {
    "GC": "#b68b00",
    "SI": "#7a8591",
    "HG": "#b35c2e",
    "PL": "#3b6ea8",
    "PA": "#5f8f5f",
}


def load_returns_and_mask() -> tuple[pd.DataFrame, pd.DataFrame]:
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


def lagged_ewma_vol(returns: pd.DataFrame, observed_mask: pd.DataFrame) -> pd.DataFrame:
    observed_returns = returns.where(observed_mask)
    return (
        observed_returns.ewm(
            halflife=VOL_HALFLIFE_BARS,
            min_periods=VOL_MIN_OBS,
            adjust=False,
            ignore_na=True,
        )
        .std()
        .shift(1)
    )


def load_standardized_residuals() -> pd.DataFrame:
    residuals = pd.read_parquet(RESIDUALS_PATH)
    residuals["ts"] = pd.to_datetime(residuals["ts"], utc=True)
    residuals = residuals.sort_values("ts").set_index("ts")
    return residuals[[f"residual_{root}" for root in ROOTS]].rename(
        columns={f"residual_{root}": root for root in ROOTS}
    )


def residual_log_returns() -> pd.DataFrame:
    returns, observed_mask = load_returns_and_mask()
    vol = lagged_ewma_vol(returns, observed_mask)
    residuals = load_standardized_residuals()
    residual_returns = residuals * vol.reindex(residuals.index)
    residual_returns.index.name = "ts"
    return residual_returns


def cumulative_daily(residual_returns: pd.DataFrame) -> pd.DataFrame:
    cumulative = residual_returns.fillna(0.0).cumsum()
    return cumulative.resample("1D").last().dropna(how="all")


def plot_overlay(cumulative: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for root in ROOTS:
        final_value = cumulative[root].iloc[-1]
        ax.plot(
            cumulative.index,
            cumulative[root],
            label=f"{root} ({final_value:+.4f})",
            color=COLORS[root],
            linewidth=1.3,
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Core metals cumulative PC1-PC2 residual log returns")
    ax.set_ylabel("Cumulative residual log return")
    ax.set_xlabel("Date")
    ax.legend(ncol=len(ROOTS), loc="upper left", frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pc12_residual_cumulative_log_returns_overlay.png", dpi=170)
    plt.close(fig)


def plot_panels(cumulative: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(ROOTS), 1, figsize=(13, 10), sharex=True)
    for ax, root in zip(axes, ROOTS, strict=True):
        final_value = cumulative[root].iloc[-1]
        ax.plot(cumulative.index, cumulative[root], color=COLORS[root], linewidth=1.2)
        ax.axhline(0.0, color="black", linewidth=0.7)
        ax.text(
            0.99,
            0.82,
            f"final {final_value:+.4f}",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=9,
        )
        ax.set_ylabel(root)
        ax.grid(True, alpha=0.25)
    axes[0].set_title("Core metals cumulative PC1-PC2 residual log returns by asset")
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pc12_residual_cumulative_log_returns_panels.png", dpi=170)
    plt.close(fig)


def summarize(residual_returns: pd.DataFrame, cumulative: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for root in ROOTS:
        values = residual_returns[root].dropna()
        rows.append(
            {
                "root": root,
                "nobs": len(values),
                "final_cum_residual_log_return": cumulative[root].iloc[-1],
                "mean_residual_return_bps": values.mean() * 10_000.0,
                "std_residual_return_bps": values.std(ddof=1) * 10_000.0,
                "mean_abs_residual_return_bps": values.abs().mean() * 10_000.0,
                "p95_abs_residual_return_bps": values.abs().quantile(0.95) * 10_000.0,
            }
        )
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, residual_returns: pd.DataFrame) -> None:
    lines = [
        "# Core Metals PC1-PC2 Residual Cumulative Log Returns",
        "",
        f"Standardized residual input: `{RESIDUALS_PATH}`.",
        "",
        "Method:",
        "",
        "- Residuals are from the robust EWMA PCA run after removing PC1-PC2.",
        "- Convert standardized residuals back into return units by multiplying by each "
        "asset's lagged EWMA 5-minute volatility.",
        f"- EWMA volatility half-life: `{VOL_HALFLIFE_BARS}` bars.",
        f"- EWMA volatility minimum observations: `{VOL_MIN_OBS}` bars.",
        "- Cumulative paths fill missing residual returns with zero.",
        "- Plots use daily last cumulative values for readability.",
        "",
        "Caveat: these are residual returns at the emitted PCA diagnostic timestamps, "
        "not every raw 5-minute bar.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Coverage",
        "",
        f"- First residual timestamp: `{residual_returns.index.min()}`.",
        f"- Last residual timestamp: `{residual_returns.index.max()}`.",
        f"- Emitted rows: `{len(residual_returns):,}`.",
        "",
        "## Files",
        "",
        "- `pc12_residual_log_returns.parquet`",
        "- `pc12_residual_cumulative_log_returns_daily.csv`",
        "- `pc12_residual_cumulative_log_returns_summary.csv`",
        "- `pc12_residual_cumulative_log_returns_overlay.png`",
        "- `pc12_residual_cumulative_log_returns_panels.png`",
    ]
    (OUTPUT_DIR / "pc12_residual_cumulative_log_returns_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    residual_returns = residual_log_returns()
    cumulative = cumulative_daily(residual_returns)
    summary = summarize(residual_returns, cumulative)

    residual_returns.reset_index().to_parquet(
        OUTPUT_DIR / "pc12_residual_log_returns.parquet",
        index=False,
    )
    cumulative.to_csv(OUTPUT_DIR / "pc12_residual_cumulative_log_returns_daily.csv")
    summary.to_csv(OUTPUT_DIR / "pc12_residual_cumulative_log_returns_summary.csv", index=False)
    plot_overlay(cumulative)
    plot_panels(cumulative)
    write_report(summary, residual_returns)
    print(f"Wrote residual cumulative log returns to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
