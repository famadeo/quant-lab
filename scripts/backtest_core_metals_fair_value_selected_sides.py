"""Out-of-sample selected-side test for fair-value dislocation events."""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import backtest_core_metals_fair_value_dislocations as base

matplotlib.use("Agg")

OUTPUT_DIR = base.OUTPUT_DIR / "selected_sides"
TRAIN_END = pd.Timestamp("2022-12-31 23:59:59.999999999", tz="UTC")
TEST_START = pd.Timestamp("2023-01-01", tz="UTC")
MIN_TRAIN_EVENTS = 5
MIN_TRAIN_HIT_RATE = 0.55
MIN_TRAIN_MEAN_BP = 0.0


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = base.load_inputs()
    variants = event_variants()

    frames = []
    event_frames = []
    selection_frames = []
    for variant in variants:
        raw_targets, events = base.build_event_targets(inputs, variant)
        events = base.attach_event_returns(events, raw_targets, inputs["after_carry_returns"])
        events["strategy"] = variant.name
        selection = select_groups(events)
        if selection.empty:
            continue
        selection["base_strategy"] = variant.name
        selection_frames.append(selection)
        selected_targets = filter_targets(raw_targets, selection)
        selected_name = f"selected_pre2023_{variant.name}"
        selected_events = filter_events(events, selection).copy()
        selected_events["strategy"] = selected_name
        event_frames.append(selected_events)
        strategy = base.simulate_strategy(selected_name, selected_targets, inputs)
        frames.append(strategy)

    if not frames:
        raise RuntimeError("No selected-side strategies were generated.")

    strategies = pd.concat(frames, ignore_index=True)
    events = pd.concat(event_frames, ignore_index=True)
    selections = pd.concat(selection_frames, ignore_index=True)
    metrics = base.build_strategy_metrics(strategies)
    split_metrics = base.build_split_metrics(strategies)
    event_summary = base.summarize_events(events)
    root_contrib = base.build_root_contributions(strategies)

    selections.to_csv(OUTPUT_DIR / "selected_groups.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "selected_strategy_metrics.csv", index=False)
    split_metrics.to_csv(OUTPUT_DIR / "selected_split_metrics.csv", index=False)
    events.to_csv(OUTPUT_DIR / "selected_event_log.csv", index=False)
    event_summary.to_csv(OUTPUT_DIR / "selected_event_summary.csv", index=False)
    root_contrib.to_csv(OUTPUT_DIR / "selected_root_contributions.csv", index=False)
    strategies.to_parquet(OUTPUT_DIR / "selected_strategy_returns.parquet", index=False)

    plot_selected_equity(strategies, split_metrics)
    plot_oos_sharpe(split_metrics)
    write_report(selections, metrics, split_metrics, event_summary)
    print(
        split_metrics[
            split_metrics["split"].isin(["2023_2024", "2025_2026"])
        ].sort_values(["split", "sharpe"], ascending=[True, False]).head(20).to_string(index=False)
    )
    print(f"Wrote {OUTPUT_DIR}")


def event_variants() -> list[base.EventVariant]:
    return [
        base.EventVariant("event_20D_pure", "20D"),
        base.EventVariant("event_60D_pure", "60D"),
        base.EventVariant("event_120D_pure", "120D"),
        base.EventVariant("event_252D_pure", "252D"),
        base.EventVariant("event_120D_carry_tailwind", "120D", carry_tailwind=True),
        base.EventVariant("event_120D_agree_60D", "120D", agree_window="60D"),
        base.EventVariant(
            "event_120D_agree_60D_carry",
            "120D",
            carry_tailwind=True,
            agree_window="60D",
        ),
        base.EventVariant("event_60D_agree_120D", "60D", agree_window="120D", agree_min_abs_z=1.0),
        base.EventVariant(
            "event_20D_agree_60D_carry",
            "20D",
            carry_tailwind=True,
            agree_window="60D",
            agree_min_abs_z=1.0,
        ),
    ]


def select_groups(events: pd.DataFrame) -> pd.DataFrame:
    train = events[events["entry_ts"].le(TRAIN_END)].copy()
    rows = []
    for (root, side), group in train.groupby(["root", "side"], sort=True):
        rows.append(
            {
                "root": root,
                "side": side,
                "train_event_count": len(group),
                "train_mean_trade_return_bp": group["trade_return_bp"].mean(),
                "train_median_trade_return_bp": group["trade_return_bp"].median(),
                "train_hit_rate": group["trade_return_bp"].gt(0).mean(),
                "train_trade_tstat": base.tstat(group["trade_return_log"]),
            }
        )
    selection = pd.DataFrame(rows)
    if selection.empty:
        return selection
    selected = selection[
        selection["train_event_count"].ge(MIN_TRAIN_EVENTS)
        & selection["train_mean_trade_return_bp"].gt(MIN_TRAIN_MEAN_BP)
        & selection["train_hit_rate"].ge(MIN_TRAIN_HIT_RATE)
    ].copy()
    return selected.sort_values("train_mean_trade_return_bp", ascending=False).reset_index(
        drop=True
    )


def filter_targets(raw_targets: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    allowed = set(zip(selection["root"], selection["side"], strict=True))
    filtered = raw_targets.copy() * 0.0
    for root in base.ROOTS:
        long_allowed = (root, "long_cheap") in allowed
        short_allowed = (root, "short_rich") in allowed
        values = raw_targets[root]
        keep = (values.gt(0) & long_allowed) | (values.lt(0) & short_allowed)
        filtered.loc[keep, root] = values.loc[keep]
    return filtered


def filter_events(events: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    allowed = selection[["root", "side"]].drop_duplicates()
    return events.merge(allowed, on=["root", "side"], how="inner")


def plot_selected_equity(strategies: pd.DataFrame, split_metrics: pd.DataFrame) -> None:
    oos = (
        split_metrics[split_metrics["split"].eq("2023_2024")]
        .sort_values("sharpe", ascending=False)
        .head(8)
    )
    net_col = f"net_return_{base.cost_label(base.PRIMARY_COST_BPS)}"
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for strategy in oos["strategy"]:
        data = strategies[strategies["strategy"].eq(strategy)].sort_values("ts")
        data = data[data["ts"].ge(TEST_START)]
        equity = np.exp(data[net_col].cumsum()) - 1.0
        ax.plot(data["ts"], equity, linewidth=1.2, label=strategy)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Selected pre-2023 groups: out-of-sample equity net of 1 bp turnover")
    ax.set_ylabel("Cumulative return")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "selected_oos_equity_net_1bp.png", dpi=170)
    plt.close(fig)


def plot_oos_sharpe(split_metrics: pd.DataFrame) -> None:
    data = split_metrics[split_metrics["split"].isin(["2023_2024", "2025_2026"])].copy()
    top = (
        data[data["split"].eq("2023_2024")]
        .sort_values("sharpe", ascending=False)
        .head(10)["strategy"]
        .tolist()
    )
    matrix = (
        data[data["strategy"].isin(top)]
        .pivot(index="strategy", columns="split", values="sharpe")
        .reindex(index=top, columns=["2023_2024", "2025_2026"])
    )
    values = matrix.to_numpy(dtype=float)
    vmax = max(1.0, np.nanpercentile(np.abs(values), 95))
    fig, ax = plt.subplots(figsize=(8, 6.5), constrained_layout=True)
    image = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title("Selected-side strategy OOS Sharpe, net of 1 bp")
    ax.set_xticks(np.arange(len(matrix.columns)), labels=matrix.columns)
    ax.set_yticks(np.arange(len(matrix.index)), labels=matrix.index)
    for i, strategy in enumerate(matrix.index):
        for j, split in enumerate(matrix.columns):
            value = matrix.loc[strategy, split]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Sharpe")
    fig.savefig(OUTPUT_DIR / "selected_oos_sharpe_heatmap_net_1bp.png", dpi=170)
    plt.close(fig)


def write_report(
    selections: pd.DataFrame,
    metrics: pd.DataFrame,
    split_metrics: pd.DataFrame,
    event_summary: pd.DataFrame,
) -> None:
    primary = metrics[metrics["cost_bps"].eq(base.PRIMARY_COST_BPS)].copy()
    oos_2023 = split_metrics[split_metrics["split"].eq("2023_2024")].copy()
    oos_2025 = split_metrics[split_metrics["split"].eq("2025_2026")].copy()
    report = [
        "# HYP-0044 Selected-Side Out-Of-Sample Test",
        "",
        "Selection rule:",
        "",
        "- Use only events with `entry_ts <= 2022-12-31` for selection.",
        f"- Keep root/side groups with at least `{MIN_TRAIN_EVENTS}` train events.",
        f"- Require train mean return > `{MIN_TRAIN_MEAN_BP:g}` bp.",
        f"- Require train hit rate >= `{MIN_TRAIN_HIT_RATE:.2f}`.",
        "- Apply selected root/side groups unchanged to 2023 onward.",
        "- Portfolio results are gross-normalized and net of 1 bp constituent turnover.",
        "",
        "## Selected Groups",
        "",
        selections.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Full-Sample Selected Strategies",
        "",
        primary.head(15).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 2023-2024 Out-Of-Sample",
        "",
        oos_2023.sort_values("sharpe", ascending=False).head(15).to_markdown(
            index=False,
            floatfmt=".4f",
        ),
        "",
        "## 2025-2026 Out-Of-Sample",
        "",
        oos_2025.sort_values("sharpe", ascending=False).head(15).to_markdown(
            index=False,
            floatfmt=".4f",
        ),
        "",
        "## Event Summary",
        "",
        event_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Files",
        "",
        "- `selected_groups.csv`",
        "- `selected_strategy_metrics.csv`",
        "- `selected_split_metrics.csv`",
        "- `selected_event_log.csv`",
        "- `selected_event_summary.csv`",
        "- `selected_strategy_returns.parquet`",
        "- `selected_oos_equity_net_1bp.png`",
        "- `selected_oos_sharpe_heatmap_net_1bp.png`",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
