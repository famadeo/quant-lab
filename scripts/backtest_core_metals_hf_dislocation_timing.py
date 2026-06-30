"""High-frequency timing overlays for carry-conditioned fair-value dislocations."""

# ruff: noqa: E402

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import backtest_core_metals_fair_value_dislocations as event_base

matplotlib.use("Agg")

OUTPUT_DIR = REPO_ROOT / "experiments" / "HYP-0045-hf-fair-value-dislocation-timing"

RETURNS_5M_PATH = (
    REPO_ROOT
    / "experiments"
    / "HYP-0041-core-metals-5m-log-returns"
    / "core_metals_5m_log_returns_wide.parquet"
)
WEIGHTS_PATH = (
    REPO_ROOT
    / "experiments"
    / "HYP-0042-core-metals-robust-ewma-pca"
    / "pc12_residual_carry_weights.parquet"
)
CARRY_PATH = (
    REPO_ROOT
    / "experiments"
    / "HYP-0042-core-metals-robust-ewma-pca"
    / "pc12_residual_carry_cost_pct_ann.parquet"
)
FAIR_PANEL_PATH = (
    REPO_ROOT
    / "experiments"
    / "HYP-0043-core-metals-carry-conditioned-fair-value"
    / "fair_value_panel.parquet"
)
SELECTED_GROUPS_PATH = (
    REPO_ROOT
    / "experiments"
    / "HYP-0044-fair-value-dislocation-monetization"
    / "selected_sides"
    / "selected_groups.csv"
)

ROOTS = event_base.ROOTS
WINDOWS = event_base.WINDOWS
ENTRY_Z = 2.5
EXIT_Z = 0.5
SECONDS_PER_YEAR = 365.25 * 24.0 * 60.0 * 60.0
PRIMARY_COST_BPS = 1.0
COST_BPS_GRID = [0.0, 0.25, 0.5, 1.0, 2.0]
EPSILON = 1e-12

BASE_VARIANTS = [
    event_base.EventVariant(
        "event_60D_agree_120D",
        "60D",
        agree_window="120D",
        agree_min_abs_z=1.0,
    ),
    event_base.EventVariant(
        "event_120D_agree_60D_carry",
        "120D",
        carry_tailwind=True,
        agree_window="60D",
    ),
    event_base.EventVariant(
        "event_20D_agree_60D_carry",
        "20D",
        carry_tailwind=True,
        agree_window="60D",
        agree_min_abs_z=1.0,
    ),
    event_base.EventVariant("event_60D_pure", "60D"),
    event_base.EventVariant("event_252D_pure", "252D"),
    event_base.EventVariant("event_120D_carry_tailwind", "120D", carry_tailwind=True),
]

COLORS = {
    "state_hold": "#355c7d",
    "exhaust_30m_flip5_hold_state": "#6c8f57",
    "exhaust_60m_confirm15_hold_state": "#b35c2e",
    "confirm_30m_momo_exit": "#7a5195",
}


@dataclass(frozen=True)
class TimingVariant:
    name: str
    trigger: str
    lookback_bars: int
    confirm_bars: int
    impulse_bps: float
    exit_rule: str
    exit_bars: int
    exit_bps: float = 0.0


TIMING_VARIANTS = [
    TimingVariant("state_hold", "state", 0, 0, 0.0, "state", 0),
    TimingVariant("exhaust_30m_flip5_hold_state", "exhaust", 6, 1, 1.0, "state", 0),
    TimingVariant("exhaust_60m_confirm15_hold_state", "exhaust", 12, 3, 1.5, "state", 0),
    TimingVariant("confirm_30m_momo_exit", "confirm", 6, 6, 1.0, "momentum", 6, 0.0),
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = load_5m_inputs()
    selections = load_selected_groups()

    strategy_frames: dict[str, pd.DataFrame] = {}
    event_frames = []
    metric_rows = []
    split_rows = []
    root_rows = []

    for base_variant in BASE_VARIANTS:
        allowed = selections[selections["base_strategy"].eq(base_variant.name)]
        if allowed.empty:
            continue
        for timing in TIMING_VARIANTS:
            strategy_name = f"hf_{base_variant.name}_{timing.name}"
            targets, events = build_timed_targets(inputs, base_variant, timing, allowed)
            strategy = simulate_strategy(strategy_name, targets, inputs)
            strategy_frames[strategy_name] = strategy
            events["strategy"] = strategy_name
            events["base_strategy"] = base_variant.name
            events["timing"] = timing.name
            event_frames.append(events)
            metric_rows.extend(strategy_metrics(strategy_name, strategy))
            split_rows.extend(split_metrics(strategy_name, strategy))
            root_rows.extend(root_contributions(strategy_name, strategy))

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["cost_bps", "sharpe", "cagr"],
        ascending=[True, False, False],
    )
    splits = pd.DataFrame(split_rows).sort_values(["split", "sharpe"], ascending=[True, False])
    roots = pd.DataFrame(root_rows)
    events_all = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    event_summary = summarize_events(events_all)

    metrics.to_csv(OUTPUT_DIR / "strategy_metrics.csv", index=False)
    splits.to_csv(OUTPUT_DIR / "split_metrics.csv", index=False)
    roots.to_csv(OUTPUT_DIR / "root_contributions.csv", index=False)
    events_all.to_csv(OUTPUT_DIR / "event_log.csv", index=False)
    event_summary.to_csv(OUTPUT_DIR / "event_summary.csv", index=False)

    primary = metrics[metrics["cost_bps"].eq(PRIMARY_COST_BPS)].copy()
    best_strategy = str(primary.iloc[0]["strategy"])
    write_best_returns(strategy_frames[best_strategy], best_strategy)
    write_top_returns(strategy_frames, primary)
    plot_outputs(strategy_frames, metrics, splits, roots, best_strategy)
    write_report(metrics, splits, event_summary, roots, best_strategy, inputs)
    write_results_json(metrics, best_strategy, inputs)

    print(primary.head(20).to_string(index=False))
    print(f"Wrote {OUTPUT_DIR}")


def load_5m_inputs() -> dict[str, Any]:
    returns = pd.read_parquet(RETURNS_5M_PATH)
    returns["ts"] = pd.to_datetime(returns["ts"], utc=True)
    returns = returns.sort_values("ts").set_index("ts")[ROOTS].astype("float64")

    weights = pd.read_parquet(WEIGHTS_PATH)
    weights["ts"] = pd.to_datetime(weights["ts"], utc=True)
    weights = weights.sort_values("ts").set_index("ts")

    carry = pd.read_parquet(CARRY_PATH)
    carry["ts"] = pd.to_datetime(carry["ts"], utc=True)
    carry = carry.sort_values("ts").set_index("ts")
    carry = carry.rename(columns={f"{root}_carry_pct_ann": root for root in ROOTS})[ROOTS]

    fair_panel = pd.read_parquet(FAIR_PANEL_PATH)
    fair_panel["ts"] = pd.to_datetime(fair_panel["ts"], utc=True)
    fair_panel = fair_panel.sort_values(["window", "root", "ts"])

    start = max(weights.index.min(), carry.index.min(), fair_panel["ts"].min())
    returns = returns[returns.index >= start].copy()
    index = returns.index

    weights_5m = weights.reindex(index, method="ffill")
    carry_5m = carry.reindex(index, method="ffill")
    zscores = {
        window: (
            fair_panel[fair_panel["window"].eq(window)]
            .pivot(index="ts", columns="root", values="fair_zscore")
            .reindex(index, method="ffill")
            .reindex(columns=ROOTS)
        )
        for window in WINDOWS
    }

    residual_returns = build_residual_returns(returns, weights_5m)
    elapsed_years = pd.Series(index.to_series().diff().dt.total_seconds() / SECONDS_PER_YEAR)
    elapsed_years = elapsed_years.fillna(0.0).clip(lower=0.0)
    carry_cost = carry_5m.shift(1).div(100.0).mul(elapsed_years, axis=0).fillna(0.0)
    after_carry = residual_returns - carry_cost

    return {
        "index": index,
        "raw_returns": returns,
        "weights": weights_5m,
        "carry_ann": carry_5m,
        "residual_returns": residual_returns,
        "after_carry_returns": after_carry,
        "zscores": zscores,
    }


def build_residual_returns(returns: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    lagged_weights = weights.shift(1)
    output = pd.DataFrame(0.0, index=returns.index, columns=ROOTS)
    for basket_root in ROOTS:
        value = np.zeros(len(returns), dtype=float)
        for asset_root in ROOTS:
            value += (
                lagged_weights[f"{basket_root}_w_{asset_root}"].fillna(0.0).to_numpy(dtype=float)
                * returns[asset_root].to_numpy(dtype=float)
            )
        output[basket_root] = value
    return output


def load_selected_groups() -> pd.DataFrame:
    groups = pd.read_csv(SELECTED_GROUPS_PATH)
    return groups[["base_strategy", "root", "side"]].drop_duplicates().reset_index(drop=True)


def build_timed_targets(
    inputs: dict[str, Any],
    base_variant: event_base.EventVariant,
    timing: TimingVariant,
    allowed_groups: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = inputs["index"]
    returns = inputs["after_carry_returns"]
    z = inputs["zscores"][base_variant.window]
    carry = inputs["carry_ann"]
    targets = pd.DataFrame(0.0, index=index, columns=ROOTS)
    allowed = {(row.root, row.side) for row in allowed_groups.itertuples(index=False)}
    states = precompute_states(inputs, base_variant, allowed)
    triggers = precompute_triggers(inputs, timing, allowed, states)
    exit_momentum = precompute_exit_momentum(inputs, timing)
    entry_recent = {
        root: rolling_sum(returns[root], max(timing.confirm_bars, 1)).to_numpy(dtype=float)
        for root in ROOTS
    }

    active: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []

    for row_num, ts in enumerate(index):
        for root in list(active):
            event = active[root]
            root_index = ROOTS.index(root)
            exit_now, reason = should_exit(
                inputs,
                base_variant,
                timing,
                states,
                exit_momentum,
                row_num,
                root_index,
                event,
            )
            if exit_now:
                active.pop(root)
                event.update(
                    {
                        "exit_ts": ts,
                        "exit_z": z.iat[row_num, root_index],
                        "exit_reason": reason,
                    }
                )
                events.append(event)

        for root_index, root in enumerate(ROOTS):
            if root in active:
                continue
            for side, position in [("long_cheap", 1.0), ("short_rich", -1.0)]:
                if (root, side) not in allowed:
                    continue
                if not triggers[(root, side)][row_num]:
                    continue
                active[root] = {
                    "root": root,
                    "side": side,
                    "position": position,
                    "entry_ts": ts,
                    "entry_z": z.iat[row_num, root_index],
                    "entry_carry_pct_ann": carry.iat[row_num, root_index],
                    "entry_recent_return_bp": (
                        position * entry_recent[root][row_num] * 10_000.0
                    ),
                }
                break

        for root, event in active.items():
            targets.at[ts, root] = float(event["position"])

    if active:
        final_ts = index[-1]
        final_row = len(index) - 1
        for root, event in active.items():
            root_index = ROOTS.index(root)
            event.update(
                {
                    "exit_ts": final_ts,
                    "exit_z": z.iat[final_row, root_index],
                    "exit_reason": "end_of_sample",
                }
            )
            events.append(event)

    event_frame = attach_event_returns(pd.DataFrame(events), targets, inputs["after_carry_returns"])
    return targets, event_frame


def precompute_states(
    inputs: dict[str, Any],
    base_variant: event_base.EventVariant,
    allowed: set[tuple[str, str]],
) -> dict[tuple[str, str], np.ndarray]:
    states: dict[tuple[str, str], np.ndarray] = {}
    for root_index, root in enumerate(ROOTS):
        for side, position in [("long_cheap", 1.0), ("short_rich", -1.0)]:
            key = (root, side)
            if key not in allowed:
                states[key] = np.zeros(len(inputs["index"]), dtype=bool)
                continue
            states[key] = eligible_state(inputs, base_variant, root_index, position).to_numpy(
                dtype=bool
            )
    return states


def precompute_triggers(
    inputs: dict[str, Any],
    timing: TimingVariant,
    allowed: set[tuple[str, str]],
    states: dict[tuple[str, str], np.ndarray],
) -> dict[tuple[str, str], np.ndarray]:
    returns = inputs["after_carry_returns"]
    output: dict[tuple[str, str], np.ndarray] = {}
    for root in ROOTS:
        for side, position in [("long_cheap", 1.0), ("short_rich", -1.0)]:
            key = (root, side)
            if key not in allowed:
                output[key] = np.zeros(len(returns), dtype=bool)
                continue
            state = pd.Series(states[key], index=returns.index)
            if timing.trigger == "state":
                trigger = state
            elif timing.trigger == "exhaust":
                lookback = rolling_sum(returns[root], timing.lookback_bars).shift(1)
                confirm = rolling_sum(returns[root], timing.confirm_bars)
                trigger = (
                    state
                    & (position * lookback <= -(timing.impulse_bps / 10_000.0))
                    & (position * confirm > 0.0)
                )
            elif timing.trigger == "confirm":
                confirm = rolling_sum(returns[root], timing.confirm_bars)
                trigger = state & (position * confirm >= timing.impulse_bps / 10_000.0)
            else:
                raise ValueError(f"Unknown trigger: {timing.trigger}")
            output[key] = trigger.fillna(False).to_numpy(dtype=bool)
    return output


def precompute_exit_momentum(
    inputs: dict[str, Any],
    timing: TimingVariant,
) -> dict[str, np.ndarray]:
    if timing.exit_rule != "momentum":
        return {root: np.zeros(len(inputs["index"]), dtype=float) for root in ROOTS}
    return {
        root: rolling_sum(inputs["after_carry_returns"][root], timing.exit_bars).to_numpy(
            dtype=float
        )
        for root in ROOTS
    }


def eligible_state(
    inputs: dict[str, Any],
    base_variant: event_base.EventVariant,
    root_index: int,
    position: float,
) -> pd.Series:
    z = inputs["zscores"][base_variant.window].iloc[:, root_index]
    state = z.le(-ENTRY_Z) if position > 0 else z.ge(ENTRY_Z)

    if base_variant.carry_tailwind:
        carry = inputs["carry_ann"].iloc[:, root_index]
        state &= (-position * carry).gt(0.0)

    if base_variant.agree_window is not None:
        agree = inputs["zscores"][base_variant.agree_window].iloc[:, root_index]
        state &= np.sign(agree).eq(np.sign(z))
        state &= agree.abs().ge(base_variant.agree_min_abs_z)
    return state.fillna(False)


def should_exit(
    inputs: dict[str, Any],
    base_variant: event_base.EventVariant,
    timing: TimingVariant,
    states: dict[tuple[str, str], np.ndarray],
    exit_momentum: dict[str, np.ndarray],
    row_num: int,
    root_index: int,
    event: dict[str, Any],
) -> tuple[bool, str]:
    root = ROOTS[root_index]
    position = float(event["position"])
    z_value = inputs["zscores"][base_variant.window].iat[row_num, root_index]
    if np.isfinite(z_value) and abs(z_value) <= EXIT_Z:
        return True, "normalized"
    state = states[(root, str(event["side"]))][row_num]
    if not state:
        return True, "state_invalid"
    if timing.exit_rule == "momentum":
        recent = exit_momentum[root][row_num]
        if position * recent <= -(timing.exit_bps / 10_000.0):
            return True, "momentum_failed"
    return False, ""


def rolling_sum(series: pd.Series, bars: int) -> pd.Series:
    if bars <= 1:
        return series
    return series.rolling(bars, min_periods=bars).sum()


def attach_event_returns(
    events: pd.DataFrame,
    raw_targets: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                "root",
                "side",
                "position",
                "entry_ts",
                "exit_ts",
                "entry_z",
                "exit_z",
                "exit_reason",
                "trade_return_log",
                "trade_return_bp",
                "duration_hours",
                "holding_bars",
            ]
        )
    unit_pnl = raw_targets.shift(1).fillna(0.0) * returns
    rows = []
    for event in events.itertuples(index=False):
        mask = (unit_pnl.index > event.entry_ts) & (unit_pnl.index <= event.exit_ts)
        path = unit_pnl.loc[mask, event.root].fillna(0.0)
        rows.append(
            {
                **event._asdict(),
                "trade_return_log": path.sum(),
                "trade_return_bp": path.sum() * 10_000.0,
                "duration_hours": (
                    pd.Timestamp(event.exit_ts) - pd.Timestamp(event.entry_ts)
                ).total_seconds()
                / 3600.0,
                "holding_bars": len(path),
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
    root_pnl = (exec_positions * returns).fillna(0.0)
    gross_return = root_pnl.sum(axis=1)
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
            "exec_gross": exec_positions.abs().sum(axis=1).to_numpy(dtype=float),
            "signal_gross": raw_targets.abs().sum(axis=1).to_numpy(dtype=float),
        }
    )
    for root in ROOTS:
        frame[f"{root}_pnl"] = root_pnl[root].to_numpy(dtype=float)
    for cost_bps in COST_BPS_GRID:
        label = event_base.cost_label(cost_bps)
        cost = turnover * cost_bps / 10_000.0
        frame[f"net_return_{label}"] = (gross_return - cost).to_numpy(dtype=float)
        frame[f"cost_return_{label}"] = cost.to_numpy(dtype=float)
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
        target = targets[basket_root].to_numpy(dtype=float)
        for asset_root in ROOTS:
            exposures[asset_root] += target * weights[f"{basket_root}_w_{asset_root}"].to_numpy(
                dtype=float
            )
    return exposures


def strategy_metrics(strategy_name: str, strategy: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        metrics_for_group(strategy_name, "full", strategy, cost_bps)
        for cost_bps in COST_BPS_GRID
    ]


def split_metrics(strategy_name: str, strategy: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for split_name, (start, end) in event_base.SPLITS.items():
        data = strategy
        if start is not None:
            data = data[data["ts"].ge(start)]
        if end is not None:
            data = data[data["ts"].le(end)]
        if data.empty:
            continue
        rows.append(metrics_for_group(strategy_name, split_name, data, PRIMARY_COST_BPS))
    return rows


def metrics_for_group(
    strategy_name: str,
    split_name: str,
    data: pd.DataFrame,
    cost_bps: float,
) -> dict[str, Any]:
    label = event_base.cost_label(cost_bps)
    net = data[f"net_return_{label}"].astype(float)
    gross = data["gross_return"].astype(float)
    elapsed_years = event_base.years_between(data["ts"].iloc[0], data["ts"].iloc[-1])
    obs_per_year = len(data) / elapsed_years if elapsed_years > 0 else np.nan
    annual_log_return = net.sum() / elapsed_years if elapsed_years > 0 else np.nan
    annual_vol = net.std(ddof=1) * math.sqrt(obs_per_year) if len(data) > 1 else np.nan
    active = data["exec_gross"].gt(0)
    active_returns = net[active]
    return {
        "strategy": strategy_name,
        "split": split_name,
        "cost_bps": cost_bps,
        "nobs": len(data),
        "years": elapsed_years,
        "active_fraction": active.mean(),
        "cum_log_return": net.sum(),
        "gross_cum_log_return": gross.sum(),
        "cost_cum_log_return": data[f"cost_return_{label}"].sum(),
        "cagr": math.expm1(annual_log_return) if np.isfinite(annual_log_return) else np.nan,
        "annual_log_return": annual_log_return,
        "annual_vol": annual_vol,
        "sharpe": annual_log_return / annual_vol if annual_vol and annual_vol > 0 else np.nan,
        "period_tstat": event_base.tstat(net),
        "max_drawdown": event_base.max_drawdown(net),
        "hit_rate_active": active_returns.gt(0).mean() if len(active_returns) else np.nan,
        "mean_active_return_bp": active_returns.mean() * 10_000.0
        if len(active_returns)
        else np.nan,
        "annual_turnover": data["turnover"].sum() / elapsed_years if elapsed_years > 0 else np.nan,
    }


def root_contributions(strategy_name: str, strategy: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    total = strategy["gross_return"].sum()
    for root in ROOTS:
        contribution = strategy[f"{root}_pnl"].sum()
        rows.append(
            {
                "strategy": strategy_name,
                "root": root,
                "cum_log_contribution": contribution,
                "contribution_bp": contribution * 10_000.0,
                "share_of_gross_pnl": contribution / total if abs(total) > EPSILON else np.nan,
            }
        )
    return rows


def summarize_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows = []
    for key, group in events.groupby(["strategy", "root", "side"], sort=True):
        rows.append(
            {
                "strategy": key[0],
                "root": key[1],
                "side": key[2],
                "event_count": len(group),
                "hit_rate": group["trade_return_log"].gt(0).mean(),
                "mean_trade_return_bp": group["trade_return_bp"].mean(),
                "median_trade_return_bp": group["trade_return_bp"].median(),
                "p10_trade_return_bp": group["trade_return_bp"].quantile(0.10),
                "p90_trade_return_bp": group["trade_return_bp"].quantile(0.90),
                "mean_duration_hours": group["duration_hours"].mean(),
                "median_duration_hours": group["duration_hours"].median(),
                "trade_tstat": event_base.tstat(group["trade_return_log"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["strategy", "mean_trade_return_bp"])


def write_best_returns(strategy: pd.DataFrame, best_strategy: str) -> None:
    path = OUTPUT_DIR / "best_strategy_returns.csv"
    strategy.to_csv(path, index=False)
    (OUTPUT_DIR / "best_strategy_name.txt").write_text(best_strategy + "\n", encoding="utf-8")


def write_top_returns(
    strategy_frames: dict[str, pd.DataFrame],
    primary_metrics: pd.DataFrame,
) -> None:
    top_names = primary_metrics.head(8)["strategy"].tolist()
    frames = [strategy_frames[name] for name in top_names]
    pd.concat(frames, ignore_index=True).to_parquet(OUTPUT_DIR / "top_strategy_returns.parquet")


def plot_outputs(
    strategy_frames: dict[str, pd.DataFrame],
    metrics: pd.DataFrame,
    splits: pd.DataFrame,
    roots: pd.DataFrame,
    best_strategy: str,
) -> None:
    plot_top_equity(strategy_frames, metrics)
    plot_best_drawdown(strategy_frames[best_strategy], best_strategy)
    plot_metric_bars(metrics)
    plot_split_heatmap(splits, metrics)
    plot_root_contributions(roots, metrics)


def plot_top_equity(strategy_frames: dict[str, pd.DataFrame], metrics: pd.DataFrame) -> None:
    primary = metrics[metrics["cost_bps"].eq(PRIMARY_COST_BPS)].head(8)
    label = event_base.cost_label(PRIMARY_COST_BPS)
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for strategy_name in primary["strategy"]:
        data = strategy_frames[strategy_name]
        equity = np.exp(data[f"net_return_{label}"].cumsum()) - 1.0
        plot_data = daily_sample(data["ts"], equity, "equity")
        ax.plot(plot_data["ts"], plot_data["equity"], linewidth=1.1, label=strategy_name)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Top 5-minute timed dislocation strategies, net of 1 bp turnover")
    ax.set_ylabel("Cumulative return")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top_strategy_equity_net_1bp.png", dpi=170)
    plt.close(fig)


def plot_best_drawdown(strategy: pd.DataFrame, best_strategy: str) -> None:
    label = event_base.cost_label(PRIMARY_COST_BPS)
    cumulative = strategy[f"net_return_{label}"].cumsum()
    drawdown = np.exp(cumulative - cumulative.cummax()) - 1.0
    equity_plot = daily_sample(strategy["ts"], np.exp(cumulative) - 1.0, "equity")
    drawdown_plot = daily_sample(strategy["ts"], drawdown, "drawdown")
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    axes[0].plot(equity_plot["ts"], equity_plot["equity"], color="#1f4e79", linewidth=1.1)
    axes[0].set_title(f"Best HF timed strategy: {best_strategy}")
    axes[0].set_ylabel("Cumulative return")
    axes[0].grid(True, alpha=0.25)
    axes[1].fill_between(
        drawdown_plot["ts"],
        drawdown_plot["drawdown"],
        0.0,
        color="#9b1c1c",
        alpha=0.35,
    )
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "best_strategy_equity_drawdown.png", dpi=170)
    plt.close(fig)


def daily_sample(ts: pd.Series, values: pd.Series | np.ndarray, column: str) -> pd.DataFrame:
    frame = pd.DataFrame({"ts": pd.to_datetime(ts, utc=True), column: np.asarray(values)})
    sampled = frame.set_index("ts").resample("1D").last().dropna().reset_index()
    sampled["ts"] = sampled["ts"].dt.tz_convert(None)
    return sampled


def plot_metric_bars(metrics: pd.DataFrame) -> None:
    primary = metrics[metrics["cost_bps"].eq(PRIMARY_COST_BPS)].head(12)
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
    fig.suptitle("Top HF timing strategies net of 1 bp turnover")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top_strategy_metrics_net_1bp.png", dpi=170)
    plt.close(fig)


def plot_split_heatmap(splits: pd.DataFrame, metrics: pd.DataFrame) -> None:
    top = metrics[metrics["cost_bps"].eq(PRIMARY_COST_BPS)].head(10)["strategy"].tolist()
    matrix = (
        splits[splits["strategy"].isin(top)]
        .pivot(index="strategy", columns="split", values="sharpe")
        .reindex(index=top, columns=list(event_base.SPLITS))
    )
    values = matrix.to_numpy(dtype=float)
    vmax = max(1.0, np.nanpercentile(np.abs(values), 95))
    fig, ax = plt.subplots(figsize=(9.5, 6.8), constrained_layout=True)
    image = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title("HF timing Sharpe by period, net of 1 bp turnover")
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


def plot_root_contributions(roots: pd.DataFrame, metrics: pd.DataFrame) -> None:
    top = metrics[metrics["cost_bps"].eq(PRIMARY_COST_BPS)].head(8)["strategy"].tolist()
    matrix = (
        roots[roots["strategy"].isin(top)]
        .pivot(index="strategy", columns="root", values="contribution_bp")
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
    splits: pd.DataFrame,
    event_summary: pd.DataFrame,
    roots: pd.DataFrame,
    best_strategy: str,
    inputs: dict[str, Any],
) -> None:
    primary = metrics[metrics["cost_bps"].eq(PRIMARY_COST_BPS)].copy()
    no_cost = metrics[metrics["cost_bps"].eq(0.0)].copy()
    best = primary[primary["strategy"].eq(best_strategy)].iloc[0]
    top_names = primary.head(8)["strategy"].tolist()
    top_splits = splits[splits["strategy"].isin(top_names)].copy()
    best_roots = roots[roots["strategy"].eq(best_strategy)].copy()
    report = [
        "# HYP-0045 HF Fair-Value Dislocation Timing",
        "",
        "Objective: test whether 5-minute residual-return timing improves the selected-side",
        "carry-conditioned fair-value dislocation strategies from HYP-0044.",
        "",
        "Construction:",
        "",
        "- Fair-value z-score state is forward-filled from HYP-0043 hourly diagnostics.",
        "- Residual basket returns are computed every 5 minutes from lagged PC1-PC2-neutral",
        "basket weights and 5-minute metal returns.",
        "- Carry cost is integrated every 5 minutes using lagged annualized residual carry.",
        "- Only pre-2023 selected root/side groups from HYP-0044 are allowed.",
        "- Position is applied one 5-minute bar after the trigger.",
        "- Execution cost is charged on constituent basket turnover.",
        "",
        "Timing overlays:",
        "",
        "- `state_hold`: selected dislocation state, no additional timing.",
        "- `exhaust_30m_flip5_hold_state`: adverse 30-minute residual move, then favorable",
        "5-minute flip; hold while state remains valid.",
        "- `exhaust_60m_confirm15_hold_state`: adverse 60-minute residual move, then favorable",
        "15-minute confirmation; hold while state remains valid.",
        "- `confirm_30m_momo_exit`: favorable 30-minute momentum entry; exit when 30-minute",
        "momentum fails or state invalidates.",
        "",
        "Caveats:",
        "",
        "- This is still a residual-basket proxy, not contract-level execution with "
        "queue/slippage.",
        "- Fair-value state only updates hourly, so the HF layer times entry inside a slower "
        "state.",
        "- Cost assumptions remain stress assumptions, not measured fill costs.",
        "",
        "## Best Strategy Net Of 1 bp Turnover",
        "",
        best.to_frame().T.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Top Strategies Net Of 1 bp Turnover",
        "",
        primary.head(20).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Top Strategies No Cost",
        "",
        no_cost.head(20).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Split Metrics For Top Strategies",
        "",
        top_splits.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Best Strategy Root Contributions",
        "",
        best_roots.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Event Summary For Best Strategy",
        "",
        event_summary[event_summary["strategy"].eq(best_strategy)].to_markdown(
            index=False,
            floatfmt=".4f",
        )
        if not event_summary.empty
        else "No events.",
        "",
        "## Input Span",
        "",
        f"- 5-minute start: `{inputs['index'].min()}`",
        f"- 5-minute end: `{inputs['index'].max()}`",
        f"- rows: `{len(inputs['index'])}`",
        "",
        "## Files",
        "",
        "- `strategy_metrics.csv`",
        "- `split_metrics.csv`",
        "- `event_summary.csv`",
        "- `event_log.csv`",
        "- `root_contributions.csv`",
        "- `best_strategy_returns.csv`",
        "- `top_strategy_returns.parquet`",
        "- `top_strategy_equity_net_1bp.png`",
        "- `best_strategy_equity_drawdown.png`",
        "- `top_strategy_metrics_net_1bp.png`",
        "- `split_sharpe_heatmap_net_1bp.png`",
        "- `root_contribution_heatmap.png`",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(report), encoding="utf-8")


def write_results_json(metrics: pd.DataFrame, best_strategy: str, inputs: dict[str, Any]) -> None:
    summary = {
        "experiment_id": "HYP-0045",
        "completed_at": datetime.now(UTC).isoformat(),
        "data_start": inputs["index"].min().isoformat(),
        "data_end": inputs["index"].max().isoformat(),
        "rows_5m": len(inputs["index"]),
        "entry_z": ENTRY_Z,
        "exit_z": EXIT_Z,
        "primary_cost_bps": PRIMARY_COST_BPS,
        "cost_bps_grid": COST_BPS_GRID,
        "best_strategy": best_strategy,
        "best_metrics_1bp": metrics[
            metrics["cost_bps"].eq(PRIMARY_COST_BPS) & metrics["strategy"].eq(best_strategy)
        ].iloc[0].to_dict(),
    }
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


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
