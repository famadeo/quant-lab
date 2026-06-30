"""Slice carry-conditioned fair-value diagnostics into requested calendar periods."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = REPO_ROOT / "experiments" / "HYP-0043-core-metals-carry-conditioned-fair-value"
OUTPUT_DIR = INPUT_DIR / "period_slices"

PANEL_PATH = INPUT_DIR / "fair_value_panel.parquet"

ROOTS = ["GC", "SI", "HG", "PL", "PA"]
WINDOWS = ["20D", "60D", "120D", "252D"]
DEFAULT_WINDOW = "120D"
OFF_FAIR_Z = 2.5
EXTREME_Z = 3.5
MAX_EVENT_GAP = pd.Timedelta("36h")

PERIODS = [
    (
        "2021_2022",
        pd.Timestamp("2021-01-01", tz="UTC"),
        pd.Timestamp("2022-12-31 23:59:59.999999999", tz="UTC"),
    ),
    (
        "2023_2024",
        pd.Timestamp("2023-01-01", tz="UTC"),
        pd.Timestamp("2024-12-31 23:59:59.999999999", tz="UTC"),
    ),
]

COLORS = {
    "GC": "#b68b00",
    "SI": "#7a8591",
    "HG": "#b35c2e",
    "PL": "#3b6ea8",
    "PA": "#5f8f5f",
}


def load_panel() -> pd.DataFrame:
    panel = pd.read_parquet(PANEL_PATH)
    panel["ts"] = pd.to_datetime(panel["ts"], utc=True)
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
                    "end_z": row.fair_zscore,
                    "end_deviation_bp": row.fair_deviation_bp,
                }
            else:
                current["end_ts"] = row.ts
                current["observations"] = int(current["observations"]) + 1
                current["max_abs_z"] = max(float(current["max_abs_z"]), abs(row.fair_zscore))
                current["max_abs_deviation_bp"] = max(
                    float(current["max_abs_deviation_bp"]),
                    abs(row.fair_deviation_bp),
                )
                current["end_z"] = row.fair_zscore
                current["end_deviation_bp"] = row.fair_deviation_bp
            previous_ts = row.ts
        if current is not None:
            events.append(current)

    if not events:
        return pd.DataFrame(
            columns=[
                "window",
                "root",
                "side",
                "start_ts",
                "end_ts",
                "observations",
                "max_abs_z",
                "max_abs_deviation_bp",
                "end_z",
                "end_deviation_bp",
                "duration_hours",
            ]
        )
    output = pd.DataFrame(events)
    output["duration_hours"] = (
        pd.to_datetime(output["end_ts"], utc=True) - pd.to_datetime(output["start_ts"], utc=True)
    ).dt.total_seconds() / 3600.0
    return output.sort_values(["window", "root", "start_ts"]).reset_index(drop=True)


def period_end_summary(panel: pd.DataFrame) -> pd.DataFrame:
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
    valid = panel.dropna(subset=["fair_zscore"])
    for (window, root), group in valid.groupby(["window", "root"], sort=True):
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
    return pd.DataFrame(rows).sort_values(["window", "root"]).reset_index(drop=True)


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


def plot_zscores(panel_daily: pd.DataFrame, output_dir: Path, title_suffix: str) -> None:
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
    axes[0].set_title(f"Carry-conditioned fair-value z-scores ({DEFAULT_WINDOW}, {title_suffix})")
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(output_dir / f"fair_value_zscores_{DEFAULT_WINDOW}.png", dpi=170)
    plt.close(fig)


def plot_deviation_bps(panel_daily: pd.DataFrame, output_dir: Path, title_suffix: str) -> None:
    data = panel_daily[panel_daily["window"].eq(DEFAULT_WINDOW)]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for root in ROOTS:
        root_data = data[data["root"].eq(root)]
        if root_data.empty:
            continue
        latest = root_data["fair_deviation_bp"].dropna().iloc[-1]
        ax.plot(
            root_data["ts"],
            root_data["fair_deviation_bp"],
            label=f"{root} end {latest:+.1f} bp",
            color=COLORS[root],
            linewidth=1.2,
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(f"Carry-conditioned fair-value deviation ({DEFAULT_WINDOW}, {title_suffix})")
    ax.set_ylabel("Actual minus fair cumulative log price (bp)")
    ax.set_xlabel("Date")
    ax.legend(ncol=len(ROOTS), loc="upper left", frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"fair_value_deviation_bp_{DEFAULT_WINDOW}.png", dpi=170)
    plt.close(fig)


def plot_actual_vs_fair(panel_daily: pd.DataFrame, output_dir: Path, title_suffix: str) -> None:
    data = panel_daily[panel_daily["window"].eq(DEFAULT_WINDOW)].copy()
    fig, axes = plt.subplots(len(ROOTS), 1, figsize=(13, 10), sharex=True)
    for ax, root in zip(axes, ROOTS, strict=True):
        root_data = data[data["root"].eq(root)].sort_values("ts").copy()
        if root_data.empty:
            continue
        root_data["actual_rebased"] = (
            root_data["actual_cum_log_price"] - root_data["actual_cum_log_price"].iloc[0]
        )
        root_data["fair_rebased"] = (
            root_data["fair_cum_log_price"] - root_data["fair_cum_log_price"].iloc[0]
        )
        ax.plot(
            root_data["ts"],
            root_data["actual_rebased"],
            label="actual",
            color=COLORS[root],
            linewidth=1.1,
        )
        ax.plot(
            root_data["ts"],
            root_data["fair_rebased"],
            label="fair",
            color="black",
            linewidth=1.0,
            alpha=0.8,
        )
        ax.set_ylabel(root)
        ax.grid(True, alpha=0.25)
    axes[0].legend(ncol=2, loc="upper left", frameon=False)
    axes[0].set_title("Actual versus carry-conditioned fair cumulative log price, "
                      f"rebased ({DEFAULT_WINDOW}, {title_suffix})")
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(output_dir / f"actual_vs_fair_cum_log_price_{DEFAULT_WINDOW}.png", dpi=170)
    plt.close(fig)


def plot_period_end_heatmap(latest: pd.DataFrame, output_dir: Path, title_suffix: str) -> None:
    matrix = latest.pivot(index="window", columns="root", values="fair_zscore")
    matrix = matrix.reindex(index=WINDOWS, columns=ROOTS)
    values = matrix.to_numpy(dtype=float)
    vmax = max(np.nanpercentile(np.abs(values), 95), OFF_FAIR_Z)
    fig, ax = plt.subplots(figsize=(8, 5.8), constrained_layout=True)
    image = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title(f"Period-end fair-value z-score ({title_suffix})")
    ax.set_xticks(np.arange(len(ROOTS)), labels=ROOTS)
    ax.set_yticks(np.arange(len(matrix.index)), labels=matrix.index)
    for i, window in enumerate(matrix.index):
        for j, root in enumerate(ROOTS):
            value = matrix.loc[window, root]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="z-score")
    fig.savefig(output_dir / "period_end_fair_value_zscore_heatmap.png", dpi=170)
    plt.close(fig)


def plot_event_counts(events: pd.DataFrame, output_dir: Path, title_suffix: str) -> None:
    event_data = events[events["window"].eq(DEFAULT_WINDOW)].copy()
    if event_data.empty:
        return
    event_data["year"] = pd.to_datetime(event_data["start_ts"], utc=True).dt.year
    counts = (
        event_data.groupby(["year", "root"])
        .size()
        .unstack("root")
        .reindex(columns=ROOTS)
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    bottom = np.zeros(len(counts), dtype=float)
    x = np.arange(len(counts))
    for root in ROOTS:
        values = counts[root].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, label=root, color=COLORS[root])
        bottom += values
    ax.set_xticks(x, labels=counts.index.astype(str))
    ax.set_title(f"Off-fair event count by year ({DEFAULT_WINDOW}, {title_suffix})")
    ax.set_ylabel("Event count")
    ax.legend(ncol=len(ROOTS), loc="upper left", frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"off_fair_event_counts_{DEFAULT_WINDOW}.png", dpi=170)
    plt.close(fig)


def write_report(
    output_dir: Path,
    period_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
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
    default_summary = summary[summary["window"].eq(DEFAULT_WINDOW)].copy()
    report = [
        f"# Carry-Conditioned Fair Value: {period_name}",
        "",
        f"Period: `{start}` to `{end}`.",
        "",
        "This is a calendar slice of the HYP-0043 fair-value panel. Rolling fair-value",
        "statistics were computed on the full timestamp history with a one-observation",
        "lag, then observations were filtered to this period.",
        "",
        "Interpretation:",
        "",
        "- Positive z-score: asset is rich versus PC1/PC2 residual value and carry.",
        "- Negative z-score: asset is cheap versus PC1/PC2 residual value and carry.",
        f"- Off fair threshold: `|z| >= {OFF_FAIR_Z:g}`.",
        f"- Extreme off fair threshold: `|z| >= {EXTREME_Z:g}`.",
        "",
        "## Period-End State",
        "",
        latest.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Period-End Off-Fair States",
        "",
        active.to_markdown(index=False, floatfmt=".4f")
        if not active.empty
        else "No period-end states are outside the off-fair threshold.",
        "",
        f"## {DEFAULT_WINDOW} Summary",
        "",
        default_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## All-Window Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Largest Off-Fair Events In Period",
        "",
        largest_events.to_markdown(index=False, floatfmt=".4f")
        if not largest_events.empty
        else "No off-fair events in this period.",
        "",
        "## Files",
        "",
        "- `fair_value_panel.csv`",
        "- `fair_value_daily.csv`",
        "- `period_end_fair_value_summary.csv`",
        "- `fair_value_summary_by_window.csv`",
        "- `off_fair_events.csv`",
        f"- `fair_value_zscores_{DEFAULT_WINDOW}.png`",
        f"- `fair_value_deviation_bp_{DEFAULT_WINDOW}.png`",
        f"- `actual_vs_fair_cum_log_price_{DEFAULT_WINDOW}.png`",
        "- `period_end_fair_value_zscore_heatmap.png`",
        f"- `off_fair_event_counts_{DEFAULT_WINDOW}.png`",
    ]
    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")


def write_period_outputs(
    panel: pd.DataFrame,
    name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> None:
    output_dir = OUTPUT_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)

    period_panel = panel[panel["ts"].between(start, end)].copy()
    events = extract_off_fair_events(period_panel)
    latest = period_end_summary(period_panel)
    summary = summary_by_window(period_panel, events)
    daily = daily_last(period_panel)

    period_panel.to_csv(output_dir / "fair_value_panel.csv", index=False)
    daily.to_csv(output_dir / "fair_value_daily.csv", index=False)
    latest.to_csv(output_dir / "period_end_fair_value_summary.csv", index=False)
    summary.to_csv(output_dir / "fair_value_summary_by_window.csv", index=False)
    events.to_csv(output_dir / "off_fair_events.csv", index=False)

    title_suffix = name.replace("_", "-")
    plot_zscores(daily, output_dir, title_suffix)
    plot_deviation_bps(daily, output_dir, title_suffix)
    plot_actual_vs_fair(daily, output_dir, title_suffix)
    plot_period_end_heatmap(latest, output_dir, title_suffix)
    plot_event_counts(events, output_dir, title_suffix)
    write_report(output_dir, name, start, end, latest, summary, events)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    for name, start, end in PERIODS:
        write_period_outputs(panel, name, start, end)
    print(f"Wrote period fair-value slices to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
