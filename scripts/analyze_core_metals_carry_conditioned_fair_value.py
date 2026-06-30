"""Define carry-conditioned fair value from PC1-PC2 residual after-carry levels."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PCA_DIR = REPO_ROOT / "experiments" / "HYP-0042-core-metals-robust-ewma-pca"
RETURNS_DIR = REPO_ROOT / "experiments" / "HYP-0041-core-metals-5m-log-returns"
OUTPUT_DIR = REPO_ROOT / "experiments" / "HYP-0043-core-metals-carry-conditioned-fair-value"

AFTER_CARRY_PATH = INPUT_PCA_DIR / "pc12_residual_after_carry_accounting.parquet"
RETURNS_PATH = RETURNS_DIR / "core_metals_5m_log_returns_wide.parquet"

ROOTS = ["GC", "SI", "HG", "PL", "PA"]
WINDOWS = [
    ("20D", "20D", 160),
    ("60D", "60D", 480),
    ("120D", "120D", 960),
    ("252D", "252D", 2_000),
]
DEFAULT_WINDOW = "120D"
OFF_FAIR_Z = 2.5
EXTREME_Z = 3.5
MAX_EVENT_GAP = pd.Timedelta("36h")
MAD_TO_SIGMA = 1.4826
MIN_SCALE_BP = 5.0
MIN_SCALE_LOG = MIN_SCALE_BP / 10_000.0

COLORS = {
    "GC": "#b68b00",
    "SI": "#7a8591",
    "HG": "#b35c2e",
    "PL": "#3b6ea8",
    "PA": "#5f8f5f",
}


def rolling_mad(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.nan
    median = np.median(finite)
    return float(np.median(np.abs(finite - median)))


def load_actual_cumulative_log_returns() -> pd.DataFrame:
    returns = pd.read_parquet(RETURNS_PATH)
    returns["ts"] = pd.to_datetime(returns["ts"], utc=True)
    returns = returns.sort_values("ts").set_index("ts")[ROOTS].astype("float64")
    cumulative = returns.fillna(0.0).cumsum()
    cumulative.index.name = "ts"
    return cumulative


def load_after_carry_levels() -> pd.DataFrame:
    accounting = pd.read_parquet(AFTER_CARRY_PATH)
    accounting["ts"] = pd.to_datetime(accounting["ts"], utc=True)
    return accounting.sort_values(["root", "ts"])


def build_fair_value_panel() -> pd.DataFrame:
    accounting = load_after_carry_levels()
    actual_cumulative = load_actual_cumulative_log_returns()
    rows = []
    for root, group in accounting.groupby("root", sort=True):
        data = group.sort_values("ts").set_index("ts")
        level = data["cum_after_carry_log_return"].astype("float64")
        actual = actual_cumulative[root].reindex(level.index)
        for label, window, min_obs in WINDOWS:
            lagged_level = level.shift(1)
            fair_level = lagged_level.rolling(window, min_periods=min_obs).median()
            mad = lagged_level.rolling(window, min_periods=min_obs).apply(
                rolling_mad,
                raw=True,
            )
            scale = (mad * MAD_TO_SIGMA).clip(lower=MIN_SCALE_LOG)
            deviation = level - fair_level
            zscore = deviation / scale
            frame = pd.DataFrame(
                {
                    "ts": level.index,
                    "root": root,
                    "window": label,
                    "actual_cum_log_price": actual.to_numpy(dtype="float64"),
                    "after_carry_residual_level": level.to_numpy(dtype="float64"),
                    "fair_after_carry_residual_level": fair_level.to_numpy(dtype="float64"),
                    "fair_cum_log_price": (actual - deviation).to_numpy(dtype="float64"),
                    "fair_deviation_log": deviation.to_numpy(dtype="float64"),
                    "fair_deviation_bp": (deviation * 10_000.0).to_numpy(dtype="float64"),
                    "fair_scale_log": scale.to_numpy(dtype="float64"),
                    "fair_scale_bp": (scale * 10_000.0).to_numpy(dtype="float64"),
                    "fair_zscore": zscore.to_numpy(dtype="float64"),
                    "is_off_fair": zscore.abs().ge(OFF_FAIR_Z).to_numpy(dtype=bool),
                    "is_extreme_off_fair": zscore.abs().ge(EXTREME_Z).to_numpy(dtype=bool),
                }
            )
            rows.append(frame)
    panel = pd.concat(rows, ignore_index=True)
    return panel.sort_values(["window", "root", "ts"]).reset_index(drop=True)


def extract_off_fair_events(panel: pd.DataFrame) -> pd.DataFrame:
    events = []
    for (window, root), group in panel.groupby(["window", "root"], sort=True):
        data = group.dropna(subset=["fair_zscore"]).sort_values("ts")
        current: dict[str, object] | None = None
        previous_ts: pd.Timestamp | None = None
        for row in data.itertuples(index=False):
            off_fair = bool(abs(row.fair_zscore) >= OFF_FAIR_Z)
            side = "rich" if row.fair_zscore > 0 else "cheap"
            gap_break = previous_ts is not None and row.ts - previous_ts > MAX_EVENT_GAP
            side_break = current is not None and current["side"] != side
            if not off_fair or gap_break or side_break:
                if current is not None:
                    events.append(current)
                    current = None
                if not off_fair:
                    previous_ts = row.ts
                    continue

            if current is None:
                current = {
                    "window": window,
                    "root": root,
                    "side": side,
                    "start_ts": row.ts,
                    "end_ts": row.ts,
                    "observations": 1,
                    "max_abs_z": abs(row.fair_zscore),
                    "max_abs_deviation_bp": abs(row.fair_deviation_bp),
                    "latest_z": row.fair_zscore,
                    "latest_deviation_bp": row.fair_deviation_bp,
                }
            else:
                current["end_ts"] = row.ts
                current["observations"] = int(current["observations"]) + 1
                current["max_abs_z"] = max(float(current["max_abs_z"]), abs(row.fair_zscore))
                current["max_abs_deviation_bp"] = max(
                    float(current["max_abs_deviation_bp"]),
                    abs(row.fair_deviation_bp),
                )
                current["latest_z"] = row.fair_zscore
                current["latest_deviation_bp"] = row.fair_deviation_bp
            previous_ts = row.ts
        if current is not None:
            events.append(current)
    if not events:
        return pd.DataFrame()
    output = pd.DataFrame(events)
    output["duration_hours"] = (
        pd.to_datetime(output["end_ts"]) - pd.to_datetime(output["start_ts"])
    ).dt.total_seconds() / 3600.0
    return output.sort_values(["window", "root", "start_ts"]).reset_index(drop=True)


def latest_summary(panel: pd.DataFrame) -> pd.DataFrame:
    latest = (
        panel.dropna(subset=["fair_zscore"])
        .sort_values("ts")
        .groupby(["window", "root"], as_index=False)
        .tail(1)
        .sort_values(["window", "root"])
    )
    latest["state"] = np.where(
        latest["fair_zscore"].ge(OFF_FAIR_Z),
        "rich",
        np.where(latest["fair_zscore"].le(-OFF_FAIR_Z), "cheap", "inside_band"),
    )
    return latest[
        [
            "window",
            "root",
            "ts",
            "actual_cum_log_price",
            "fair_cum_log_price",
            "fair_deviation_bp",
            "fair_scale_bp",
            "fair_zscore",
            "state",
        ]
    ]


def summary_by_window(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (window, root), group in panel.dropna(subset=["fair_zscore"]).groupby(
        ["window", "root"],
        sort=True,
    ):
        event_count = 0
        if not events.empty:
            event_count = len(events[events["window"].eq(window) & events["root"].eq(root)])
        rows.append(
            {
                "window": window,
                "root": root,
                "nobs": len(group),
                "median_abs_deviation_bp": group["fair_deviation_bp"].abs().median(),
                "p90_abs_deviation_bp": group["fair_deviation_bp"].abs().quantile(0.90),
                "p95_abs_deviation_bp": group["fair_deviation_bp"].abs().quantile(0.95),
                "p99_abs_deviation_bp": group["fair_deviation_bp"].abs().quantile(0.99),
                "off_fair_fraction": group["is_off_fair"].mean(),
                "extreme_off_fair_fraction": group["is_extreme_off_fair"].mean(),
                "event_count": event_count,
            }
        )
    return pd.DataFrame(rows)


def daily_last(panel: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for (window, root), group in panel.groupby(["window", "root"], sort=True):
        sampled = (
            group.sort_values("ts")
            .set_index("ts")
            .resample("1D")
            .last()
            .dropna(subset=["fair_zscore"])
            .reset_index()
        )
        sampled["window"] = window
        sampled["root"] = root
        frames.append(sampled)
    return pd.concat(frames, ignore_index=True)


def plot_zscores(panel_daily: pd.DataFrame) -> None:
    data = panel_daily[panel_daily["window"].eq(DEFAULT_WINDOW)]
    fig, axes = plt.subplots(len(ROOTS), 1, figsize=(13, 10), sharex=True)
    for ax, root in zip(axes, ROOTS, strict=True):
        root_data = data[data["root"].eq(root)]
        ax.plot(root_data["ts"], root_data["fair_zscore"], color=COLORS[root], linewidth=1.1)
        ax.axhline(OFF_FAIR_Z, color="#9b1c1c", linewidth=0.8, linestyle="--")
        ax.axhline(-OFF_FAIR_Z, color="#1d4e89", linewidth=0.8, linestyle="--")
        ax.axhline(0.0, color="black", linewidth=0.7)
        ax.set_ylabel(root)
        ax.grid(True, alpha=0.25)
    axes[0].set_title(f"Carry-conditioned fair-value z-scores ({DEFAULT_WINDOW})")
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"fair_value_zscores_{DEFAULT_WINDOW}.png", dpi=170)
    plt.close(fig)


def plot_deviation_bps(panel_daily: pd.DataFrame) -> None:
    data = panel_daily[panel_daily["window"].eq(DEFAULT_WINDOW)]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for root in ROOTS:
        root_data = data[data["root"].eq(root)]
        latest = root_data["fair_deviation_bp"].dropna().iloc[-1]
        ax.plot(
            root_data["ts"],
            root_data["fair_deviation_bp"],
            label=f"{root} ({latest:+.1f} bp)",
            color=COLORS[root],
            linewidth=1.2,
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(f"Carry-conditioned fair-value deviation ({DEFAULT_WINDOW})")
    ax.set_ylabel("Actual minus fair cumulative log price (bp)")
    ax.set_xlabel("Date")
    ax.legend(ncol=len(ROOTS), loc="upper left", frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"fair_value_deviation_bp_{DEFAULT_WINDOW}.png", dpi=170)
    plt.close(fig)


def plot_actual_vs_fair(panel_daily: pd.DataFrame) -> None:
    data = panel_daily[panel_daily["window"].eq(DEFAULT_WINDOW)]
    fig, axes = plt.subplots(len(ROOTS), 1, figsize=(13, 10), sharex=True)
    for ax, root in zip(axes, ROOTS, strict=True):
        root_data = data[data["root"].eq(root)]
        ax.plot(
            root_data["ts"],
            root_data["actual_cum_log_price"],
            label="actual",
            color=COLORS[root],
            linewidth=1.1,
        )
        ax.plot(
            root_data["ts"],
            root_data["fair_cum_log_price"],
            label="fair",
            color="black",
            linewidth=1.0,
            alpha=0.8,
        )
        ax.set_ylabel(root)
        ax.grid(True, alpha=0.25)
    axes[0].legend(ncol=2, loc="upper left", frameon=False)
    axes[0].set_title(
        f"Actual cumulative log price versus carry-conditioned fair price ({DEFAULT_WINDOW})"
    )
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"actual_vs_fair_cum_log_price_{DEFAULT_WINDOW}.png", dpi=170)
    plt.close(fig)


def plot_latest_heatmap(latest: pd.DataFrame) -> None:
    matrix = latest.pivot(index="window", columns="root", values="fair_zscore")
    matrix = matrix.reindex(index=[label for label, _window, _min_obs in WINDOWS], columns=ROOTS)
    values = matrix.to_numpy(dtype=float)
    vmax = max(np.nanpercentile(np.abs(values), 95), OFF_FAIR_Z)
    fig, ax = plt.subplots(figsize=(8, 5.8), constrained_layout=True)
    image = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title("Latest carry-conditioned fair-value z-score")
    ax.set_xticks(np.arange(len(ROOTS)), labels=ROOTS)
    ax.set_yticks(np.arange(len(matrix.index)), labels=matrix.index)
    for i, window in enumerate(matrix.index):
        for j, root in enumerate(ROOTS):
            value = matrix.loc[window, root]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="z-score")
    fig.savefig(OUTPUT_DIR / "latest_fair_value_zscore_heatmap.png", dpi=170)
    plt.close(fig)


def plot_event_counts(events: pd.DataFrame) -> None:
    if events.empty:
        return
    event_data = events.copy()
    event_data["year"] = pd.to_datetime(event_data["start_ts"], utc=True).dt.year
    counts = (
        event_data[event_data["window"].eq(DEFAULT_WINDOW)]
        .groupby(["year", "root"])
        .size()
        .unstack("root")
        .reindex(columns=ROOTS)
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(13, 6.5))
    bottom = np.zeros(len(counts), dtype=float)
    x = np.arange(len(counts))
    for root in ROOTS:
        values = counts[root].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, label=root, color=COLORS[root])
        bottom += values
    ax.set_xticks(x, labels=counts.index.astype(str), rotation=45)
    ax.set_title(f"Off-fair event count by year ({DEFAULT_WINDOW}, |z| >= {OFF_FAIR_Z:g})")
    ax.set_ylabel("Event count")
    ax.legend(ncol=len(ROOTS), loc="upper left", frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"off_fair_event_counts_{DEFAULT_WINDOW}.png", dpi=170)
    plt.close(fig)


def write_report(
    latest: pd.DataFrame,
    summary: pd.DataFrame,
    events: pd.DataFrame,
) -> None:
    active = latest[latest["state"].ne("inside_band")].copy()
    largest_events = (
        events.sort_values("max_abs_z", ascending=False).head(20)
        if not events.empty
        else pd.DataFrame()
    )
    report = [
        "# Core Metals Carry-Conditioned Fair Value",
        "",
        "Fair price definition:",
        "",
        "- Start from PC1-PC2 residual returns converted into log-return units.",
        "- Subtract integrated residual carry cost to get after-carry residual returns.",
        "- Cumulatively sum after-carry residual returns into an after-carry residual level.",
        "- Fair residual level is the lagged rolling median of that level.",
        "- Fair-value deviation is current after-carry residual level minus fair residual level.",
        "- Fair cumulative log price equals actual cumulative log price minus fair-value "
        "deviation.",
        "",
        "Interpretation:",
        "",
        "- Positive deviation/z-score: asset is rich versus PC1/PC2 and carry.",
        "- Negative deviation/z-score: asset is cheap versus PC1/PC2 and carry.",
        f"- Off fair threshold: `|z| >= {OFF_FAIR_Z:g}`.",
        f"- Extreme off fair threshold: `|z| >= {EXTREME_Z:g}`.",
        f"- Robust z-score scale has an economic floor of `{MIN_SCALE_BP:g}` bp.",
        "- Rolling statistics are shifted by one observation to avoid lookahead.",
        "",
        "Caveats:",
        "",
        "- This is a relative/factor fair value, not a metaphysical spot fair value.",
        "- Carry uses the futures-curve proxy from HYP-0036.",
        "- PCA residuals are emitted at diagnostic timestamps, not every raw 5-minute bar.",
        "",
        "## Latest State",
        "",
        latest.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Active Off-Fair States",
        "",
        active.to_markdown(index=False, floatfmt=".4f")
        if not active.empty
        else "No latest states are outside the off-fair threshold.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Largest Historical Off-Fair Events",
        "",
        largest_events.to_markdown(index=False, floatfmt=".4f")
        if not largest_events.empty
        else "No off-fair events.",
        "",
        "## Files",
        "",
        "- `fair_value_panel.parquet`",
        "- `fair_value_daily.csv`",
        "- `latest_fair_value_summary.csv`",
        "- `fair_value_summary_by_window.csv`",
        "- `off_fair_events.csv`",
        f"- `fair_value_zscores_{DEFAULT_WINDOW}.png`",
        f"- `fair_value_deviation_bp_{DEFAULT_WINDOW}.png`",
        f"- `actual_vs_fair_cum_log_price_{DEFAULT_WINDOW}.png`",
        "- `latest_fair_value_zscore_heatmap.png`",
        f"- `off_fair_event_counts_{DEFAULT_WINDOW}.png`",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = build_fair_value_panel()
    events = extract_off_fair_events(panel)
    latest = latest_summary(panel)
    summary = summary_by_window(panel, events)
    daily = daily_last(panel)

    panel.to_parquet(OUTPUT_DIR / "fair_value_panel.parquet", index=False)
    daily.to_csv(OUTPUT_DIR / "fair_value_daily.csv", index=False)
    latest.to_csv(OUTPUT_DIR / "latest_fair_value_summary.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "fair_value_summary_by_window.csv", index=False)
    events.to_csv(OUTPUT_DIR / "off_fair_events.csv", index=False)

    plot_zscores(daily)
    plot_deviation_bps(daily)
    plot_actual_vs_fair(daily)
    plot_latest_heatmap(latest)
    plot_event_counts(events)
    write_report(latest, summary, events)
    print(f"Wrote carry-conditioned fair value outputs to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
