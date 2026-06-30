"""Analyze PC1-PC2 residual returns after residual basket carry cost."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
PCA_DIR = REPO_ROOT / "experiments" / "HYP-0042-core-metals-robust-ewma-pca"

RESIDUAL_RETURNS_PATH = PCA_DIR / "pc12_residual_log_returns.parquet"
CARRY_PATH = PCA_DIR / "pc12_residual_carry_cost_pct_ann.parquet"

ROOTS = ["GC", "SI", "HG", "PL", "PA"]
SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60
WINDOWS = [
    ("1D", "1D", 12),
    ("5D", "5D", 48),
    ("20D", "20D", 160),
    ("60D", "60D", 480),
    ("120D", "120D", 960),
    ("252D", "252D", 2_000),
]

COLORS = {
    "GC": "#b68b00",
    "SI": "#7a8591",
    "HG": "#b35c2e",
    "PL": "#3b6ea8",
    "PA": "#5f8f5f",
}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    residual = pd.read_parquet(RESIDUAL_RETURNS_PATH)
    residual["ts"] = pd.to_datetime(residual["ts"], utc=True)
    residual = residual.sort_values("ts").set_index("ts")[ROOTS]

    carry = pd.read_parquet(CARRY_PATH)
    carry["ts"] = pd.to_datetime(carry["ts"], utc=True)
    carry = carry.sort_values("ts").set_index("ts")[
        [f"{root}_carry_pct_ann" for root in ROOTS]
    ]
    carry = carry.rename(columns={f"{root}_carry_pct_ann": root for root in ROOTS})

    common_index = residual.index.intersection(carry.index).sort_values()
    return residual.reindex(common_index), carry.reindex(common_index)


def build_period_accounting(
    residual_returns: pd.DataFrame,
    carry_pct_ann: pd.DataFrame,
) -> pd.DataFrame:
    elapsed_years = (
        residual_returns.index.to_series().diff().dt.total_seconds().fillna(0.0)
        / SECONDS_PER_YEAR
    )
    rows = []
    for root in ROOTS:
        lagged_carry_pct = carry_pct_ann[root].shift(1)
        carry_cost = lagged_carry_pct / 100.0 * elapsed_years
        valid = carry_cost.notna()
        period_residual = residual_returns[root].fillna(0.0).where(valid)
        period_after_carry = period_residual - carry_cost
        rows.append(
            pd.DataFrame(
                {
                    "ts": residual_returns.index,
                    "root": root,
                    "period_residual_log_return": period_residual.to_numpy(dtype="float64"),
                    "period_carry_cost_log": carry_cost.to_numpy(dtype="float64"),
                    "period_after_carry_log_return": period_after_carry.to_numpy(
                        dtype="float64"
                    ),
                    "carry_pct_ann_lagged": lagged_carry_pct.to_numpy(dtype="float64"),
                    "elapsed_years": elapsed_years.to_numpy(dtype="float64"),
                    "valid": valid.to_numpy(dtype=bool),
                }
            )
        )
    accounting = pd.concat(rows, ignore_index=True)
    accounting = accounting.sort_values(["root", "ts"])
    for column in [
        "period_residual_log_return",
        "period_carry_cost_log",
        "period_after_carry_log_return",
    ]:
        accounting[f"cum_{column.removeprefix('period_')}"] = accounting.groupby("root")[
            column
        ].transform(lambda values: values.fillna(0.0).cumsum())
    return accounting


def build_rolling_windows(accounting: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for root, root_data in accounting.groupby("root", sort=True):
        data = root_data.sort_values("ts").set_index("ts")
        valid_obs = data["valid"].astype(int)
        for label, window, min_obs in WINDOWS:
            obs = valid_obs.rolling(window).sum()
            rolling_residual = data["period_residual_log_return"].rolling(window).sum()
            rolling_carry = data["period_carry_cost_log"].rolling(window).sum()
            rolling_after = data["period_after_carry_log_return"].rolling(window).sum()
            frame = pd.DataFrame(
                {
                    "ts": data.index,
                    "root": root,
                    "window": label,
                    "obs": obs,
                    "residual_log_return": rolling_residual,
                    "carry_cost_log": rolling_carry,
                    "after_carry_log_return": rolling_after,
                }
            )
            frame.loc[frame["obs"].lt(min_obs), [
                "residual_log_return",
                "carry_cost_log",
                "after_carry_log_return",
            ]] = np.nan
            rows.append(frame)
    return pd.concat(rows, ignore_index=True).dropna(subset=["after_carry_log_return"])


def daily_last(accounting: pd.DataFrame) -> pd.DataFrame:
    wide = (
        accounting.pivot(
            index="ts",
            columns="root",
            values="cum_after_carry_log_return",
        )
        .sort_index()
        .resample("1D")
        .last()
        .dropna(how="all")
    )
    return wide


def daily_rolling_last(rolling: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for (root, window), group in rolling.groupby(["root", "window"], sort=True):
        sampled = (
            group.sort_values("ts")
            .set_index("ts")
            .resample("1D")
            .last()
            .dropna(subset=["after_carry_log_return"])
            .reset_index()
        )
        sampled["root"] = root
        sampled["window"] = window
        frames.append(sampled)
    return pd.concat(frames, ignore_index=True)


def summarize_full(accounting: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for root, data in accounting.groupby("root", sort=True):
        last = data.sort_values("ts").iloc[-1]
        rows.append(
            {
                "root": root,
                "nobs": int(data["valid"].sum()),
                "cum_residual_log_return": last["cum_residual_log_return"],
                "cum_carry_cost_log": last["cum_carry_cost_log"],
                "cum_after_carry_log_return": last["cum_after_carry_log_return"],
                "cum_after_carry_bp": last["cum_after_carry_log_return"] * 10_000.0,
            }
        )
    return pd.DataFrame(rows)


def summarize_rolling(rolling: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (window, root), data in rolling.groupby(["window", "root"], sort=True):
        values = data["after_carry_log_return"].dropna()
        residual = data["residual_log_return"].dropna()
        carry = data["carry_cost_log"].dropna()
        rows.append(
            {
                "window": window,
                "root": root,
                "nobs": len(values),
                "mean_after_carry_bp": values.mean() * 10_000.0,
                "median_after_carry_bp": values.median() * 10_000.0,
                "p10_after_carry_bp": values.quantile(0.10) * 10_000.0,
                "p90_after_carry_bp": values.quantile(0.90) * 10_000.0,
                "latest_after_carry_bp": values.iloc[-1] * 10_000.0,
                "positive_fraction": values.gt(0).mean(),
                "mean_residual_bp": residual.mean() * 10_000.0,
                "mean_carry_cost_bp": carry.mean() * 10_000.0,
            }
        )
    return pd.DataFrame(rows)


def plot_full_cumulative(daily_after_carry: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for root in ROOTS:
        final_value = daily_after_carry[root].dropna().iloc[-1]
        ax.plot(
            daily_after_carry.index,
            daily_after_carry[root],
            label=f"{root} ({final_value:+.4f})",
            color=COLORS[root],
            linewidth=1.25,
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("PC1-PC2 residual cumulative log return after carry")
    ax.set_ylabel("Cumulative residual return minus carry cost")
    ax.set_xlabel("Date")
    ax.legend(ncol=len(ROOTS), loc="upper left", frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(PCA_DIR / "pc12_residual_after_carry_cumulative_overlay.png", dpi=170)
    plt.close(fig)


def plot_components(accounting: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(ROOTS), 1, figsize=(13, 10), sharex=True)
    for ax, root in zip(axes, ROOTS, strict=True):
        data = accounting[accounting["root"].eq(root)].sort_values("ts")
        daily = (
            data.set_index("ts")[
                [
                    "cum_residual_log_return",
                    "cum_carry_cost_log",
                    "cum_after_carry_log_return",
                ]
            ]
            .resample("1D")
            .last()
            .dropna(how="all")
        )
        ax.plot(
            daily.index,
            daily["cum_residual_log_return"],
            label="residual return",
            color=COLORS[root],
            linewidth=1.1,
        )
        ax.plot(
            daily.index,
            daily["cum_carry_cost_log"],
            label="carry cost",
            color="#777777",
            linewidth=1.0,
            alpha=0.85,
        )
        ax.plot(
            daily.index,
            daily["cum_after_carry_log_return"],
            label="after carry",
            color="black",
            linewidth=1.1,
            alpha=0.85,
        )
        ax.axhline(0.0, color="black", linewidth=0.7)
        ax.set_ylabel(root)
        ax.grid(True, alpha=0.25)
    axes[0].legend(ncol=3, loc="upper left", frameon=False)
    axes[0].set_title("Residual cumulative return, cumulative carry cost, and after-carry net")
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(PCA_DIR / "pc12_residual_after_carry_cumulative_components.png", dpi=170)
    plt.close(fig)


def plot_rolling_windows(rolling_daily: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True, constrained_layout=True)
    for ax, (label, _window, _min_obs) in zip(axes.ravel(), WINDOWS, strict=True):
        window_data = rolling_daily[rolling_daily["window"].eq(label)]
        for root in ROOTS:
            root_data = window_data[window_data["root"].eq(root)]
            ax.plot(
                root_data["ts"],
                root_data["after_carry_log_return"] * 10_000.0,
                label=root,
                color=COLORS[root],
                linewidth=1.0,
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(label)
        ax.set_ylabel("bp")
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(ncol=len(ROOTS), loc="upper left", frameon=False)
    fig.suptitle("Rolling PC1-PC2 residual return minus carry cost")
    fig.savefig(PCA_DIR / "pc12_residual_after_carry_rolling_windows.png", dpi=170)
    plt.close(fig)


def plot_latest_heatmap(summary: pd.DataFrame) -> None:
    matrix = summary.pivot(index="window", columns="root", values="latest_after_carry_bp")
    matrix = matrix.reindex(index=[label for label, _window, _min_obs in WINDOWS], columns=ROOTS)
    values = matrix.to_numpy(dtype=float)
    vmax = np.nanpercentile(np.abs(values), 95)
    vmax = max(vmax, 1e-9)
    fig, ax = plt.subplots(figsize=(8, 5.8), constrained_layout=True)
    image = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title("Latest rolling residual return minus carry cost")
    ax.set_xticks(np.arange(len(ROOTS)), labels=ROOTS)
    ax.set_yticks(np.arange(len(matrix.index)), labels=matrix.index)
    for i, window in enumerate(matrix.index):
        for j, root in enumerate(ROOTS):
            value = matrix.loc[window, root]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="bp")
    fig.savefig(PCA_DIR / "pc12_residual_after_carry_latest_heatmap.png", dpi=170)
    plt.close(fig)


def write_report(full_summary: pd.DataFrame, rolling_summary: pd.DataFrame) -> None:
    lines = [
        "# Core Metals PC1-PC2 Residual Return Minus Carry Cost",
        "",
        "Definition:",
        "",
        "`after_carry = residual_log_return - integrated_residual_carry_cost`",
        "",
        "Method:",
        "",
        f"- Residual returns: `{RESIDUAL_RETURNS_PATH}`.",
        f"- Residual carry rates: `{CARRY_PATH}`.",
        "- Carry is annualized percent paid by a long residual basket.",
        "- Carry cost is integrated over elapsed clock time using the lagged carry rate.",
        "- Missing residual returns are treated as zero, matching the residual cumulative plot.",
        "- Rolling windows require a minimum number of valid emitted PCA observations.",
        "",
        "Caveat: residual returns are only available at emitted PCA diagnostic timestamps, "
        "not every raw 5-minute bar. Carry accrues over clock time.",
        "",
        "## Full Sample Cumulative",
        "",
        full_summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Rolling Window Summary",
        "",
        rolling_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Files",
        "",
        "- `pc12_residual_after_carry_accounting.parquet`",
        "- `pc12_residual_after_carry_rolling.parquet`",
        "- `pc12_residual_after_carry_rolling_daily.csv`",
        "- `pc12_residual_after_carry_full_summary.csv`",
        "- `pc12_residual_after_carry_rolling_summary.csv`",
        "- `pc12_residual_after_carry_cumulative_overlay.png`",
        "- `pc12_residual_after_carry_cumulative_components.png`",
        "- `pc12_residual_after_carry_rolling_windows.png`",
        "- `pc12_residual_after_carry_latest_heatmap.png`",
    ]
    (PCA_DIR / "pc12_residual_after_carry_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    residual_returns, carry_pct_ann = load_inputs()
    accounting = build_period_accounting(residual_returns, carry_pct_ann)
    rolling = build_rolling_windows(accounting)
    rolling_daily = daily_rolling_last(rolling)
    full_summary = summarize_full(accounting)
    rolling_summary = summarize_rolling(rolling)

    accounting.to_parquet(PCA_DIR / "pc12_residual_after_carry_accounting.parquet", index=False)
    rolling.to_parquet(PCA_DIR / "pc12_residual_after_carry_rolling.parquet", index=False)
    rolling_daily.to_csv(PCA_DIR / "pc12_residual_after_carry_rolling_daily.csv", index=False)
    full_summary.to_csv(PCA_DIR / "pc12_residual_after_carry_full_summary.csv", index=False)
    rolling_summary.to_csv(
        PCA_DIR / "pc12_residual_after_carry_rolling_summary.csv",
        index=False,
    )

    plot_full_cumulative(daily_last(accounting))
    plot_components(accounting)
    plot_rolling_windows(rolling_daily)
    plot_latest_heatmap(rolling_summary)
    write_report(full_summary, rolling_summary)
    print(f"Wrote residual after-carry analysis to {PCA_DIR}", flush=True)


if __name__ == "__main__":
    main()
