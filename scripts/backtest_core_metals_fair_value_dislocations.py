"""Backtest carry-conditioned fair-value dislocation monetization ideas."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_FAIR_DIR = REPO_ROOT / "experiments" / "HYP-0043-core-metals-carry-conditioned-fair-value"
INPUT_PCA_DIR = REPO_ROOT / "experiments" / "HYP-0042-core-metals-robust-ewma-pca"
OUTPUT_DIR = REPO_ROOT / "experiments" / "HYP-0044-fair-value-dislocation-monetization"

FAIR_PANEL_PATH = INPUT_FAIR_DIR / "fair_value_panel.parquet"
AFTER_CARRY_PATH = INPUT_PCA_DIR / "pc12_residual_after_carry_accounting.parquet"
WEIGHTS_PATH = INPUT_PCA_DIR / "pc12_residual_carry_weights.parquet"

ROOTS = ["GC", "SI", "HG", "PL", "PA"]
WINDOWS = ["20D", "60D", "120D", "252D"]

ENTRY_Z = 2.5
EXIT_Z = 0.5
AGREE_Z = 1.5
COST_BPS_GRID = [0.0, 0.25, 0.5, 1.0, 2.0]
PRIMARY_COST_BPS = 1.0
HOURS_PER_YEAR = 365.25 * 24.0
MIN_CROSS_SECTION_ROOTS = 2
MIN_TSTAT_OBS = 2
EPSILON = 1e-12

SPLITS = {
    "full": (None, None),
    "2021_2022": (
        pd.Timestamp("2021-01-01", tz="UTC"),
        pd.Timestamp("2022-12-31 23:59:59.999999999", tz="UTC"),
    ),
    "2023_2024": (
        pd.Timestamp("2023-01-01", tz="UTC"),
        pd.Timestamp("2024-12-31 23:59:59.999999999", tz="UTC"),
    ),
    "2025_2026": (pd.Timestamp("2025-01-01", tz="UTC"), None),
}

COLORS = {
    "GC": "#b68b00",
    "SI": "#7a8591",
    "HG": "#b35c2e",
    "PL": "#3b6ea8",
    "PA": "#5f8f5f",
}


@dataclass(frozen=True)
class EventVariant:
    name: str
    window: str
    entry_z: float = ENTRY_Z
    exit_z: float = EXIT_Z
    carry_tailwind: bool = False
    agree_window: str | None = None
    agree_min_abs_z: float = AGREE_Z


@dataclass(frozen=True)
class CrossSectionVariant:
    name: str
    mode: str
    window: str
    entry_z: float = ENTRY_Z
    spread_z: float = 5.0
    score_windows: tuple[str, ...] = ("60D", "120D", "252D")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs()

    event_variants = [
        EventVariant("event_20D_pure", "20D"),
        EventVariant("event_60D_pure", "60D"),
        EventVariant("event_120D_pure", "120D"),
        EventVariant("event_252D_pure", "252D"),
        EventVariant("event_120D_carry_tailwind", "120D", carry_tailwind=True),
        EventVariant("event_120D_agree_60D", "120D", agree_window="60D"),
        EventVariant(
            "event_120D_agree_60D_carry",
            "120D",
            carry_tailwind=True,
            agree_window="60D",
        ),
        EventVariant("event_60D_agree_120D", "60D", agree_window="120D", agree_min_abs_z=1.0),
        EventVariant(
            "event_20D_agree_60D_carry",
            "20D",
            carry_tailwind=True,
            agree_window="60D",
            agree_min_abs_z=1.0,
        ),
    ]
    xs_variants = [
        CrossSectionVariant("xs_120D_paired_extremes", "paired_extremes", "120D"),
        CrossSectionVariant("xs_120D_paired_spread", "paired_spread", "120D"),
        CrossSectionVariant("xs_120D_demeaned", "demeaned", "120D"),
        CrossSectionVariant("xs_multi_window_demeaned", "multi_demeaned", "120D"),
    ]

    strategy_frames = []
    event_frames = []
    target_store = {}

    for variant in event_variants:
        raw_targets, raw_events = build_event_targets(inputs, variant)
        event_returns = attach_event_returns(raw_events, raw_targets, inputs["after_carry_returns"])
        event_returns["strategy"] = variant.name
        event_frames.append(event_returns)
        target_store[variant.name] = raw_targets
        strategy_frames.append(simulate_strategy(variant.name, raw_targets, inputs))

    for variant in xs_variants:
        raw_targets = build_cross_section_targets(inputs, variant)
        target_store[variant.name] = raw_targets
        strategy_frames.append(simulate_strategy(variant.name, raw_targets, inputs))

    strategies = pd.concat(strategy_frames, ignore_index=True)
    events = pd.concat(event_frames, ignore_index=True)
    metrics = build_strategy_metrics(strategies)
    split_metrics = build_split_metrics(strategies)
    event_summary = summarize_events(events)
    root_contrib = build_root_contributions(strategies)

    metrics.to_csv(OUTPUT_DIR / "strategy_metrics.csv", index=False)
    split_metrics.to_csv(OUTPUT_DIR / "split_metrics.csv", index=False)
    event_summary.to_csv(OUTPUT_DIR / "event_summary.csv", index=False)
    events.to_csv(OUTPUT_DIR / "event_log.csv", index=False)
    root_contrib.to_csv(OUTPUT_DIR / "root_contributions.csv", index=False)
    strategies.to_parquet(OUTPUT_DIR / "strategy_returns.parquet", index=False)

    best_name = select_best_strategy(metrics)
    write_best_strategy_files(best_name, strategies)
    plot_outputs(strategies, metrics, split_metrics, events, root_contrib, best_name)
    write_report(metrics, split_metrics, event_summary, events, root_contrib, best_name)
    write_results_json(inputs, metrics, best_name)

    print(metrics[metrics["cost_bps"].eq(PRIMARY_COST_BPS)].head(15).to_string(index=False))
    print(f"Wrote {OUTPUT_DIR}")


def load_inputs() -> dict[str, Any]:
    fair_panel = pd.read_parquet(FAIR_PANEL_PATH)
    fair_panel["ts"] = pd.to_datetime(fair_panel["ts"], utc=True)
    fair_panel = fair_panel.sort_values(["window", "root", "ts"])

    accounting = pd.read_parquet(AFTER_CARRY_PATH)
    accounting["ts"] = pd.to_datetime(accounting["ts"], utc=True)
    accounting = accounting.sort_values(["root", "ts"])

    returns = (
        accounting.pivot(index="ts", columns="root", values="period_after_carry_log_return")
        .reindex(columns=ROOTS)
        .sort_index()
        .fillna(0.0)
    )
    residual_returns = (
        accounting.pivot(index="ts", columns="root", values="period_residual_log_return")
        .reindex(columns=ROOTS)
        .sort_index()
        .fillna(0.0)
    )
    carry_cost = (
        accounting.pivot(index="ts", columns="root", values="period_carry_cost_log")
        .reindex(columns=ROOTS)
        .sort_index()
        .fillna(0.0)
    )
    carry_ann = (
        accounting.pivot(index="ts", columns="root", values="carry_pct_ann_lagged")
        .reindex(columns=ROOTS)
        .sort_index()
        .ffill()
    )

    index = pd.DatetimeIndex(returns.index).sort_values()
    zscores = {
        window: (
            fair_panel[fair_panel["window"].eq(window)]
            .pivot(index="ts", columns="root", values="fair_zscore")
            .reindex(index=index, columns=ROOTS)
            .sort_index()
        )
        for window in WINDOWS
    }
    deviations = {
        window: (
            fair_panel[fair_panel["window"].eq(window)]
            .pivot(index="ts", columns="root", values="fair_deviation_bp")
            .reindex(index=index, columns=ROOTS)
            .sort_index()
        )
        for window in WINDOWS
    }

    weights = pd.read_parquet(WEIGHTS_PATH)
    weights["ts"] = pd.to_datetime(weights["ts"], utc=True)
    weights = weights.sort_values("ts").set_index("ts").reindex(index).ffill().bfill()

    return {
        "index": index,
        "fair_panel": fair_panel,
        "after_carry_returns": returns.reindex(index),
        "residual_returns": residual_returns.reindex(index),
        "carry_cost": carry_cost.reindex(index),
        "carry_ann": carry_ann.reindex(index),
        "zscores": zscores,
        "deviations": deviations,
        "weights": weights,
    }


def build_event_targets(
    inputs: dict[str, Any],
    variant: EventVariant,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = inputs["index"]
    z = inputs["zscores"][variant.window]
    deviations = inputs["deviations"][variant.window]
    carry_ann = inputs["carry_ann"]
    targets = pd.DataFrame(0.0, index=index, columns=ROOTS)
    active: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []

    for row_num, ts in enumerate(index):
        for root in list(active):
            current_z = z.iat[row_num, ROOTS.index(root)]
            should_exit = np.isfinite(current_z) and abs(current_z) <= variant.exit_z
            if should_exit:
                event = active.pop(root)
                event.update(
                    {
                        "exit_ts": ts,
                        "exit_z": current_z,
                        "exit_deviation_bp": deviations.iat[row_num, ROOTS.index(root)],
                        "exit_reason": "normalized",
                    }
                )
                events.append(event)

        for root_index, root in enumerate(ROOTS):
            if root in active:
                continue
            current_z = z.iat[row_num, root_index]
            if not np.isfinite(current_z) or abs(current_z) < variant.entry_z:
                continue

            position = -float(np.sign(current_z))
            if not entry_filters_pass(inputs, variant, row_num, root_index, position):
                continue

            active[root] = {
                "strategy": variant.name,
                "root": root,
                "window": variant.window,
                "side": "short_rich" if current_z > 0 else "long_cheap",
                "position": position,
                "entry_ts": ts,
                "entry_z": current_z,
                "entry_deviation_bp": deviations.iat[row_num, root_index],
                "entry_carry_pct_ann": carry_ann.iat[row_num, root_index],
                "carry_tailwind_at_entry": -position * carry_ann.iat[row_num, root_index],
                "agree_window": variant.agree_window,
            }

        for root, event in active.items():
            targets.at[ts, root] = float(event["position"])

    if active:
        final_ts = index[-1]
        final_row = len(index) - 1
        for root, event in active.items():
            root_index = ROOTS.index(root)
            current_z = z.iat[final_row, root_index]
            event.update(
                {
                    "exit_ts": final_ts,
                    "exit_z": current_z,
                    "exit_deviation_bp": deviations.iat[final_row, root_index],
                    "exit_reason": "end_of_sample",
                }
            )
            events.append(event)

    event_frame = pd.DataFrame(events)
    if not event_frame.empty:
        event_frame["duration_hours"] = (
            pd.to_datetime(event_frame["exit_ts"], utc=True)
            - pd.to_datetime(event_frame["entry_ts"], utc=True)
        ).dt.total_seconds() / 3600.0
    return targets, event_frame


def entry_filters_pass(
    inputs: dict[str, Any],
    variant: EventVariant,
    row_num: int,
    root_index: int,
    position: float,
) -> bool:
    root = ROOTS[root_index]
    if variant.carry_tailwind:
        carry = inputs["carry_ann"].iat[row_num, root_index]
        if not np.isfinite(carry) or -position * carry <= 0:
            return False
    if variant.agree_window is not None:
        agreement_z = inputs["zscores"][variant.agree_window].iat[row_num, root_index]
        entry_z = inputs["zscores"][variant.window].iat[row_num, root_index]
        if not np.isfinite(agreement_z) or not np.isfinite(entry_z):
            return False
        if np.sign(agreement_z) != np.sign(entry_z):
            return False
        if abs(agreement_z) < variant.agree_min_abs_z:
            return False
    return root in ROOTS


def build_cross_section_targets(
    inputs: dict[str, Any],
    variant: CrossSectionVariant,
) -> pd.DataFrame:
    index = inputs["index"]
    z = inputs["zscores"][variant.window]
    targets = pd.DataFrame(0.0, index=index, columns=ROOTS)

    if variant.mode == "multi_demeaned":
        score = sum(inputs["zscores"][window] for window in variant.score_windows)
        z = score / float(len(variant.score_windows))

    for ts, row in z.iterrows():
        values = row.astype(float).dropna()
        if len(values) < MIN_CROSS_SECTION_ROOTS:
            continue
        max_root = values.idxmax()
        min_root = values.idxmin()
        max_z = values[max_root]
        min_z = values[min_root]

        if variant.mode == "paired_extremes":
            if max_z < variant.entry_z or min_z > -variant.entry_z:
                continue
            targets.at[ts, min_root] = 0.5
            targets.at[ts, max_root] = -0.5
        elif variant.mode == "paired_spread":
            if max_z - min_z < variant.spread_z:
                continue
            targets.at[ts, min_root] = 0.5
            targets.at[ts, max_root] = -0.5
        elif variant.mode in {"demeaned", "multi_demeaned"}:
            if max_z - min_z < variant.spread_z:
                continue
            centered = values - values.mean()
            raw = -centered.reindex(ROOTS).fillna(0.0)
            gross = raw.abs().sum()
            if gross > 0:
                targets.loc[ts, ROOTS] = raw / gross
        else:
            raise ValueError(f"Unknown cross-section mode: {variant.mode}")
    return targets


def attach_event_returns(
    events: pd.DataFrame,
    raw_targets: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    if events.empty:
        return events
    unit_pnl = raw_targets.shift(1).fillna(0.0) * returns
    rows = []
    for event in events.itertuples(index=False):
        mask = (unit_pnl.index > event.entry_ts) & (unit_pnl.index <= event.exit_ts)
        path = unit_pnl.loc[mask, event.root].fillna(0.0)
        cumulative = path.cumsum()
        rows.append(
            {
                **event._asdict(),
                "trade_return_log": path.sum(),
                "trade_return_bp": path.sum() * 10_000.0,
                "mfe_bp": cumulative.max() * 10_000.0 if len(cumulative) else 0.0,
                "mae_bp": cumulative.min() * 10_000.0 if len(cumulative) else 0.0,
                "holding_observations": len(path),
                "normalized": event.exit_reason == "normalized",
            }
        )
    return pd.DataFrame(rows)


def simulate_strategy(
    strategy_name: str,
    raw_targets: pd.DataFrame,
    inputs: dict[str, Any],
) -> pd.DataFrame:
    returns = inputs["after_carry_returns"]
    weights = inputs["weights"]
    targets = normalize_gross(raw_targets)
    exec_positions = targets.shift(1).fillna(0.0)
    gross_return = (exec_positions * returns).sum(axis=1).fillna(0.0)
    root_pnl = (exec_positions * returns).fillna(0.0)
    leg_exposures = build_leg_exposures(targets, weights)
    turnover = leg_exposures.diff().abs().sum(axis=1)
    turnover.iloc[0] = leg_exposures.iloc[0].abs().sum()
    turnover = turnover.fillna(0.0)

    frame = pd.DataFrame(
        {
            "ts": returns.index,
            "strategy": strategy_name,
            "gross_return": gross_return.to_numpy(dtype=float),
            "turnover": turnover.to_numpy(dtype=float),
            "signal_active_count": raw_targets.abs().gt(0).sum(axis=1).to_numpy(dtype=float),
            "signal_gross": raw_targets.abs().sum(axis=1).to_numpy(dtype=float),
            "exec_gross": exec_positions.abs().sum(axis=1).to_numpy(dtype=float),
            "leg_gross": leg_exposures.abs().sum(axis=1).to_numpy(dtype=float),
        }
    )
    for root in ROOTS:
        frame[f"{root}_position"] = exec_positions[root].to_numpy(dtype=float)
        frame[f"{root}_pnl"] = root_pnl[root].to_numpy(dtype=float)
    for cost_bps in COST_BPS_GRID:
        cost = turnover * cost_bps / 10_000.0
        frame[f"net_return_{cost_label(cost_bps)}"] = (
            gross_return - cost
        ).to_numpy(dtype=float)
        frame[f"cost_return_{cost_label(cost_bps)}"] = cost.to_numpy(dtype=float)
    return frame


def normalize_gross(raw_targets: pd.DataFrame) -> pd.DataFrame:
    gross = raw_targets.abs().sum(axis=1)
    normalized = raw_targets.copy().astype("float64")
    active = gross > 0
    normalized.loc[active] = normalized.loc[active].div(gross.loc[active], axis=0)
    normalized.loc[~active] = 0.0
    return normalized


def build_leg_exposures(targets: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    exposures = pd.DataFrame(0.0, index=targets.index, columns=ROOTS)
    for basket_root in ROOTS:
        for asset_root in ROOTS:
            exposures[asset_root] += (
                targets[basket_root].to_numpy(dtype=float)
                * weights[f"{basket_root}_w_{asset_root}"].to_numpy(dtype=float)
            )
    return exposures


def build_strategy_metrics(strategies: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, group in strategies.groupby("strategy", sort=False):
        rows.extend(
            metrics_for_group(strategy, "full", group, cost_bps) for cost_bps in COST_BPS_GRID
        )
    return pd.DataFrame(rows).sort_values(
        ["cost_bps", "sharpe", "cagr"],
        ascending=[True, False, False],
    )


def build_split_metrics(strategies: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, strategy_group in strategies.groupby("strategy", sort=False):
        sorted_group = strategy_group.sort_values("ts")
        for split_name, (start, end) in SPLITS.items():
            split = sorted_group
            if start is not None:
                split = split[split["ts"].ge(start)]
            if end is not None:
                split = split[split["ts"].le(end)]
            if split.empty:
                continue
            rows.append(metrics_for_group(strategy, split_name, split, PRIMARY_COST_BPS))
    return pd.DataFrame(rows).sort_values(["split", "sharpe"], ascending=[True, False])


def metrics_for_group(
    strategy: str,
    split: str,
    group: pd.DataFrame,
    cost_bps: float,
) -> dict[str, Any]:
    group = group.sort_values("ts")
    net_col = f"net_return_{cost_label(cost_bps)}"
    cost_col = f"cost_return_{cost_label(cost_bps)}"
    returns = group[net_col].astype(float)
    gross_returns = group["gross_return"].astype(float)
    elapsed_years = years_between(group["ts"].iloc[0], group["ts"].iloc[-1])
    obs_per_year = len(group) / elapsed_years if elapsed_years > 0 else np.nan
    annual_log_return = returns.sum() / elapsed_years if elapsed_years > 0 else np.nan
    annual_vol = returns.std(ddof=1) * math.sqrt(obs_per_year) if len(group) > 1 else np.nan
    sharpe = annual_log_return / annual_vol if annual_vol and annual_vol > 0 else np.nan
    active = group["exec_gross"].gt(0)
    active_returns = returns[active]
    return {
        "strategy": strategy,
        "split": split,
        "cost_bps": cost_bps,
        "nobs": len(group),
        "years": elapsed_years,
        "active_fraction": active.mean(),
        "cum_log_return": returns.sum(),
        "gross_cum_log_return": gross_returns.sum(),
        "cost_cum_log_return": group[cost_col].sum(),
        "cagr": math.expm1(annual_log_return) if np.isfinite(annual_log_return) else np.nan,
        "annual_log_return": annual_log_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "period_tstat": tstat(returns),
        "max_drawdown": max_drawdown(returns),
        "hit_rate_active": active_returns.gt(0).mean() if len(active_returns) else np.nan,
        "mean_active_return_bp": active_returns.mean() * 10_000.0
        if len(active_returns)
        else np.nan,
        "avg_exec_gross": group["exec_gross"].mean(),
        "avg_leg_gross": group["leg_gross"].mean(),
        "annual_turnover": group["turnover"].sum() / elapsed_years if elapsed_years > 0 else np.nan,
    }


def summarize_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows = []
    keys = ["strategy", "root", "side"]
    for key, group in events.groupby(keys, sort=True):
        rows.append(
            {
                "strategy": key[0],
                "root": key[1],
                "side": key[2],
                "event_count": len(group),
                "normalized_fraction": group["normalized"].mean(),
                "hit_rate": group["trade_return_log"].gt(0).mean(),
                "mean_trade_return_bp": group["trade_return_bp"].mean(),
                "median_trade_return_bp": group["trade_return_bp"].median(),
                "p10_trade_return_bp": group["trade_return_bp"].quantile(0.10),
                "p90_trade_return_bp": group["trade_return_bp"].quantile(0.90),
                "mean_duration_hours": group["duration_hours"].mean(),
                "median_duration_hours": group["duration_hours"].median(),
                "mean_mfe_bp": group["mfe_bp"].mean(),
                "mean_mae_bp": group["mae_bp"].mean(),
                "trade_tstat": tstat(group["trade_return_log"]),
            }
        )
    output = pd.DataFrame(rows)
    return output.sort_values(["strategy", "mean_trade_return_bp"], ascending=[True, False])


def build_root_contributions(strategies: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, group in strategies.groupby("strategy", sort=False):
        total = group["gross_return"].sum()
        for root in ROOTS:
            contribution = group[f"{root}_pnl"].sum()
            rows.append(
                {
                    "strategy": strategy,
                    "root": root,
                    "cum_log_contribution": contribution,
                    "contribution_bp": contribution * 10_000.0,
                    "share_of_gross_pnl": contribution / total if abs(total) > EPSILON else np.nan,
                }
            )
    return pd.DataFrame(rows)


def select_best_strategy(metrics: pd.DataFrame) -> str:
    primary = metrics[metrics["cost_bps"].eq(PRIMARY_COST_BPS)].copy()
    eligible = primary[primary["active_fraction"].ge(0.01)].copy()
    if eligible.empty:
        eligible = primary
    best = eligible.sort_values(["sharpe", "cagr"], ascending=[False, False]).iloc[0]
    return str(best["strategy"])


def write_best_strategy_files(best_name: str, strategies: pd.DataFrame) -> None:
    best = strategies[strategies["strategy"].eq(best_name)].copy()
    best.to_csv(OUTPUT_DIR / "best_strategy_returns.csv", index=False)


def plot_outputs(
    strategies: pd.DataFrame,
    metrics: pd.DataFrame,
    split_metrics: pd.DataFrame,
    events: pd.DataFrame,
    root_contrib: pd.DataFrame,
    best_name: str,
) -> None:
    plot_top_equity(strategies, metrics)
    plot_best_drawdown(strategies, best_name)
    plot_metric_bars(metrics)
    plot_split_heatmap(split_metrics)
    plot_trade_distributions(events, metrics)
    plot_entry_z_scatter(events)
    plot_root_contributions(root_contrib, metrics)


def plot_top_equity(strategies: pd.DataFrame, metrics: pd.DataFrame) -> None:
    primary = metrics[metrics["cost_bps"].eq(PRIMARY_COST_BPS)].head(8)
    fig, ax = plt.subplots(figsize=(13, 6.5))
    net_col = f"net_return_{cost_label(PRIMARY_COST_BPS)}"
    for strategy in primary["strategy"]:
        data = strategies[strategies["strategy"].eq(strategy)].sort_values("ts")
        equity = np.exp(data[net_col].cumsum()) - 1.0
        ax.plot(data["ts"], equity, linewidth=1.2, label=strategy)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(f"Top fair-value dislocation strategies, net of {PRIMARY_COST_BPS:g} bp turnover")
    ax.set_ylabel("Cumulative return")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top_strategy_equity_net_1bp.png", dpi=170)
    plt.close(fig)


def plot_best_drawdown(strategies: pd.DataFrame, best_name: str) -> None:
    data = strategies[strategies["strategy"].eq(best_name)].sort_values("ts").copy()
    net_col = f"net_return_{cost_label(PRIMARY_COST_BPS)}"
    cumulative = data[net_col].cumsum()
    drawdown = np.exp(cumulative - cumulative.cummax()) - 1.0
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    axes[0].plot(data["ts"], np.exp(cumulative) - 1.0, color="#1f4e79", linewidth=1.2)
    axes[0].set_title(f"Best strategy equity: {best_name}")
    axes[0].set_ylabel("Cumulative return")
    axes[0].grid(True, alpha=0.25)
    axes[1].fill_between(data["ts"], drawdown, 0.0, color="#9b1c1c", alpha=0.35)
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "best_strategy_equity_drawdown.png", dpi=170)
    plt.close(fig)


def plot_metric_bars(metrics: pd.DataFrame) -> None:
    primary = metrics[metrics["cost_bps"].eq(PRIMARY_COST_BPS)].head(12).copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharey=True)
    y = np.arange(len(primary))
    axes[0].barh(y, primary["sharpe"], color="#355c7d")
    axes[0].set_yticks(y, labels=primary["strategy"])
    axes[0].invert_yaxis()
    axes[0].set_title("Sharpe")
    axes[0].grid(True, axis="x", alpha=0.25)
    axes[1].barh(y, primary["cagr"] * 100.0, color="#6c8f57")
    axes[1].set_title("CAGR, %")
    axes[1].grid(True, axis="x", alpha=0.25)
    fig.suptitle(f"Top strategies net of {PRIMARY_COST_BPS:g} bp turnover")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top_strategy_metrics_net_1bp.png", dpi=170)
    plt.close(fig)


def plot_split_heatmap(split_metrics: pd.DataFrame) -> None:
    primary = split_metrics[split_metrics["cost_bps"].eq(PRIMARY_COST_BPS)].copy()
    full_rank = (
        primary[primary["split"].eq("full")]
        .sort_values("sharpe", ascending=False)
        .head(10)["strategy"]
        .tolist()
    )
    matrix = (
        primary[primary["strategy"].isin(full_rank)]
        .pivot(index="strategy", columns="split", values="sharpe")
        .reindex(index=full_rank, columns=list(SPLITS))
    )
    values = matrix.to_numpy(dtype=float)
    vmax = max(1.0, np.nanpercentile(np.abs(values), 95))
    fig, ax = plt.subplots(figsize=(9.5, 6.5), constrained_layout=True)
    image = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title("Sharpe by period, net of 1 bp turnover")
    ax.set_xticks(np.arange(len(matrix.columns)), labels=matrix.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), labels=matrix.index)
    for i, strategy in enumerate(matrix.index):
        for j, split in enumerate(matrix.columns):
            value = matrix.loc[strategy, split]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Sharpe")
    fig.savefig(OUTPUT_DIR / "split_sharpe_heatmap_net_1bp.png", dpi=170)
    plt.close(fig)


def plot_trade_distributions(events: pd.DataFrame, metrics: pd.DataFrame) -> None:
    if events.empty:
        return
    primary = metrics[metrics["cost_bps"].eq(PRIMARY_COST_BPS)].head(6)["strategy"].tolist()
    data = events[events["strategy"].isin(primary)].copy()
    if data.empty:
        data = events[events["strategy"].str.startswith("event_120D")].copy()
    strategies = data["strategy"].drop_duplicates().tolist()[:8]
    samples = [
        data[data["strategy"].eq(strategy)]["trade_return_bp"].dropna().to_numpy(dtype=float)
        for strategy in strategies
    ]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.boxplot(samples, tick_labels=strategies, showfliers=False, orientation="vertical")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Unit event return distribution, no execution cost")
    ax.set_ylabel("Trade return (bp)")
    ax.tick_params(axis="x", labelrotation=30)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "event_trade_return_distribution.png", dpi=170)
    plt.close(fig)


def plot_entry_z_scatter(events: pd.DataFrame) -> None:
    data = events[events["strategy"].eq("event_120D_pure")].copy()
    if data.empty:
        return
    data["signed_entry_z"] = data["entry_z"]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for root in ROOTS:
        root_data = data[data["root"].eq(root)]
        ax.scatter(
            root_data["signed_entry_z"],
            root_data["trade_return_bp"],
            s=14,
            alpha=0.45,
            label=root,
            color=COLORS[root],
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_title("Event 120D pure: entry z-score versus unit trade return")
    ax.set_xlabel("Entry fair-value z-score")
    ax.set_ylabel("Trade return to normalization/end (bp)")
    ax.legend(ncol=len(ROOTS), frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "entry_z_vs_trade_return_event_120D.png", dpi=170)
    plt.close(fig)


def plot_root_contributions(root_contrib: pd.DataFrame, metrics: pd.DataFrame) -> None:
    top = metrics[metrics["cost_bps"].eq(PRIMARY_COST_BPS)].head(8)["strategy"].tolist()
    data = root_contrib[root_contrib["strategy"].isin(top)].copy()
    matrix = (
        data.pivot(index="strategy", columns="root", values="contribution_bp")
        .reindex(index=top, columns=ROOTS)
    )
    values = matrix.to_numpy(dtype=float)
    vmax = max(10.0, np.nanpercentile(np.abs(values), 95))
    fig, ax = plt.subplots(figsize=(8.5, 6.2), constrained_layout=True)
    image = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title("Root contribution to no-cost gross PnL")
    ax.set_xticks(np.arange(len(ROOTS)), labels=ROOTS)
    ax.set_yticks(np.arange(len(matrix.index)), labels=matrix.index)
    for i, strategy in enumerate(matrix.index):
        for j, root in enumerate(ROOTS):
            value = matrix.loc[strategy, root]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="bp")
    fig.savefig(OUTPUT_DIR / "root_contribution_heatmap.png", dpi=170)
    plt.close(fig)


def write_report(
    metrics: pd.DataFrame,
    split_metrics: pd.DataFrame,
    event_summary: pd.DataFrame,
    events: pd.DataFrame,
    root_contrib: pd.DataFrame,
    best_name: str,
) -> None:
    primary_metrics = metrics[metrics["cost_bps"].eq(PRIMARY_COST_BPS)].copy()
    no_cost_metrics = metrics[metrics["cost_bps"].eq(0.0)].copy()
    best_primary = primary_metrics[primary_metrics["strategy"].eq(best_name)].iloc[0]
    active_event_summary = (
        event_summary[event_summary["strategy"].eq("event_120D_pure")]
        if not event_summary.empty
        else pd.DataFrame()
    )
    top_events = (
        events[events["strategy"].eq("event_120D_pure")]
        .sort_values("trade_return_bp", ascending=False)
        .head(10)
        if not events.empty
        else pd.DataFrame()
    )
    worst_events = (
        events[events["strategy"].eq("event_120D_pure")]
        .sort_values("trade_return_bp", ascending=True)
        .head(10)
        if not events.empty
        else pd.DataFrame()
    )
    best_roots = root_contrib[root_contrib["strategy"].eq(best_name)].copy()
    report = [
        "# HYP-0044 Fair-Value Dislocation Monetization",
        "",
        "Objective: test whether carry-conditioned actual-versus-fair dislocations can be",
        "monetized as residual mean-reversion or cross-sectional relative value.",
        "",
        "Signal and PnL discipline:",
        "",
        "- Signal is observed at timestamp `t`; position is applied from the next observation.",
        "- Event entries use `|z| >= 2.5`; exits use normalization at `|z| <= 0.5`.",
        "- PnL uses PC1-PC2 residual returns after integrated residual carry cost.",
        "- Portfolio tests gross-normalize active residual basket signals to one unit.",
        "- Execution-cost stress uses constituent basket turnover from residual weights.",
        "- Cost grid is `0`, `0.25`, `0.5`, `1`, and `2` bp per unit constituent turnover.",
        "",
        "Important caveats:",
        "",
        "- This is a residual-basket backtest, not a fully specified futures execution model.",
        "- Cost estimates are stress assumptions, not measured venue-level slippage.",
        "- PC1/PC2 neutrality is built into the residual basket; USD/rates/CL hedging is not",
        "added in this pass because this experiment tests the newer carry-conditioned object",
        "directly.",
        "- Cross-sectional variants rebalance frequently and are intentionally penalized by",
        "turnover costs.",
        "",
        "## Best Strategy Net Of 1 bp Turnover",
        "",
        best_primary.to_frame().T.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Top Strategies, Net Of 1 bp Turnover",
        "",
        primary_metrics.head(15).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Top Strategies, No Cost",
        "",
        no_cost_metrics.head(15).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Split Metrics, Net Of 1 bp Turnover",
        "",
        split_metrics[
            split_metrics["strategy"].isin(primary_metrics.head(8)["strategy"])
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Event 120D Pure Summary",
        "",
        active_event_summary.to_markdown(index=False, floatfmt=".4f")
        if not active_event_summary.empty
        else "No event summary.",
        "",
        "## Best Strategy Root Contributions",
        "",
        best_roots.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Best Event 120D Pure Trades",
        "",
        format_event_table(top_events),
        "",
        "## Worst Event 120D Pure Trades",
        "",
        format_event_table(worst_events),
        "",
        "## Files",
        "",
        "- `strategy_metrics.csv`",
        "- `split_metrics.csv`",
        "- `event_summary.csv`",
        "- `event_log.csv`",
        "- `root_contributions.csv`",
        "- `strategy_returns.parquet`",
        "- `best_strategy_returns.csv`",
        "- `top_strategy_equity_net_1bp.png`",
        "- `best_strategy_equity_drawdown.png`",
        "- `top_strategy_metrics_net_1bp.png`",
        "- `split_sharpe_heatmap_net_1bp.png`",
        "- `event_trade_return_distribution.png`",
        "- `entry_z_vs_trade_return_event_120D.png`",
        "- `root_contribution_heatmap.png`",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(report), encoding="utf-8")


def format_event_table(events: pd.DataFrame) -> str:
    if events.empty:
        return "No events."
    columns = [
        "strategy",
        "root",
        "side",
        "entry_ts",
        "exit_ts",
        "entry_z",
        "exit_z",
        "trade_return_bp",
        "duration_hours",
        "exit_reason",
    ]
    return events[columns].to_markdown(index=False, floatfmt=".4f")


def write_results_json(inputs: dict[str, Any], metrics: pd.DataFrame, best_name: str) -> None:
    summary = {
        "experiment_id": "HYP-0044",
        "completed_at": datetime.now(UTC).isoformat(),
        "data_start": inputs["index"].min().isoformat(),
        "data_end": inputs["index"].max().isoformat(),
        "roots": ROOTS,
        "entry_z": ENTRY_Z,
        "exit_z": EXIT_Z,
        "cost_bps_grid": COST_BPS_GRID,
        "primary_cost_bps": PRIMARY_COST_BPS,
        "best_strategy": best_name,
        "best_metrics_1bp": metrics[
            metrics["cost_bps"].eq(PRIMARY_COST_BPS) & metrics["strategy"].eq(best_name)
        ].iloc[0].to_dict(),
    }
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def years_between(start: pd.Timestamp, end: pd.Timestamp) -> float:
    elapsed_seconds = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds()
    return max(elapsed_seconds / (365.25 * 24.0 * 3600.0), 1.0 / HOURS_PER_YEAR)


def tstat(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < MIN_TSTAT_OBS:
        return np.nan
    std = arr.std(ddof=1)
    if std <= 0:
        return np.nan
    return float(arr.mean() / std * math.sqrt(len(arr)))


def max_drawdown(returns: pd.Series | np.ndarray) -> float:
    arr = pd.Series(np.asarray(returns, dtype=float)).fillna(0.0)
    cumulative = arr.cumsum()
    drawdown = np.exp(cumulative - cumulative.cummax()) - 1.0
    return float(drawdown.min())


def cost_label(cost_bps: float) -> str:
    return f"{cost_bps:g}bp".replace(".", "p")


def json_safe(value: Any) -> Any:  # noqa: PLR0911
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


if __name__ == "__main__":
    main()
