"""Walk-forward execution test for HF fair-value dislocation strategies."""

# ruff: noqa: E402, PLR0911, PLR0915

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from quantlab.metals_flow.config import CONTRACT_MULTIPLIERS
from scripts import backtest_core_metals_fair_value_dislocations as event_base
from scripts import backtest_core_metals_hf_dislocation_timing as hf_base

OUTPUT_DIR = REPO_ROOT / "experiments" / "HYP-0046-hf-walkforward-execution"
COST_PATH = (
    REPO_ROOT
    / "experiments"
    / "HYP-0014-metals-flow-filtered-residual-reversion-3y"
    / "cost_estimates.csv"
)

ROOTS = event_base.ROOTS
ENTRY_Z = 2.5
EXIT_Z = 0.5
EVAL_START = pd.Timestamp("2021-01-01", tz="UTC")
PRIMARY_COST_MULTIPLIER = 1.0
COST_MULTIPLIERS = [0.0, 0.5, 1.0, 2.0]
SECONDS_PER_YEAR = 365.25 * 24.0 * 60.0 * 60.0
EPSILON = 1e-12


@dataclass(frozen=True)
class StrategySpec:
    base: event_base.EventVariant
    timing: hf_base.TimingVariant
    z_buffer: float

    @property
    def name(self) -> str:
        buffer_label = str(self.z_buffer).replace(".", "p")
        return f"wf_{self.base.name}_{self.timing.name}_zbuf{buffer_label}"


@dataclass(frozen=True)
class SelectionPolicy:
    name: str
    min_events: int
    min_hit_rate: float
    min_mean_bp: float = 0.0
    min_tstat: float | None = None


BASE_VARIANTS = [
    event_base.EventVariant(
        "event_20D_agree_60D_carry",
        "20D",
        carry_tailwind=True,
        agree_window="60D",
        agree_min_abs_z=1.0,
    ),
    event_base.EventVariant(
        "event_60D_agree_120D",
        "60D",
        agree_window="120D",
        agree_min_abs_z=1.0,
    ),
    event_base.EventVariant("event_60D_pure", "60D"),
    event_base.EventVariant("event_252D_pure", "252D"),
]

TIMING_VARIANTS = [
    hf_base.TimingVariant("state_hold", "state", 0, 0, 0.0, "state", 0),
    hf_base.TimingVariant("exhaust_30m_flip5_hold_state", "exhaust", 6, 1, 1.0, "state", 0),
    hf_base.TimingVariant(
        "exhaust_60m_confirm15_hold_state",
        "exhaust",
        12,
        3,
        1.5,
        "state",
        0,
    ),
]

Z_BUFFERS = [0.0, 0.25, 0.5]

SELECTION_POLICIES = [
    SelectionPolicy("expanding_n5_hit52_meanpos", min_events=5, min_hit_rate=0.52),
    SelectionPolicy("expanding_n10_hit55_meanpos", min_events=10, min_hit_rate=0.55),
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = hf_base.load_5m_inputs()
    costs = load_cost_model()

    strategy_frames: dict[str, pd.DataFrame] = {}
    metric_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    root_rows: list[dict[str, Any]] = []
    selected_event_frames: list[pd.DataFrame] = []
    selection_frames: list[pd.DataFrame] = []
    candidate_frames: list[pd.DataFrame] = []

    specs = [
        StrategySpec(base, timing, z_buffer)
        for base in BASE_VARIANTS
        for timing in TIMING_VARIANTS
        for z_buffer in Z_BUFFERS
    ]

    for spec in specs:
        print(f"building candidates: {spec.name}", flush=True)
        candidates = generate_candidate_events(inputs, spec)
        candidates["spec"] = spec.name
        candidates["base_strategy"] = spec.base.name
        candidates["timing"] = spec.timing.name
        candidates["z_buffer"] = spec.z_buffer
        candidate_frames.append(candidates)

        for policy in SELECTION_POLICIES:
            strategy_name = f"{spec.name}_{policy.name}"
            selected_events, selections = select_walk_forward_events(
                candidates,
                inputs["index"],
                spec,
                policy,
            )
            targets = build_targets_from_events(selected_events, inputs["index"])
            strategy = simulate_strategy(strategy_name, targets, inputs, costs)
            strategy_frames[strategy_name] = strategy

            selected_events = selected_events.copy()
            selected_events["strategy"] = strategy_name
            selected_event_frames.append(selected_events)

            selections = selections.copy()
            selections["strategy"] = strategy_name
            selection_frames.append(selections)

            metric_rows.extend(strategy_metrics(strategy_name, strategy))
            split_rows.extend(split_metrics(strategy_name, strategy))
            root_rows.extend(root_contributions(strategy_name, strategy))

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["cost_multiplier", "sharpe", "cagr"],
        ascending=[True, False, False],
    )
    splits = pd.DataFrame(split_rows).sort_values(["split", "sharpe"], ascending=[True, False])
    roots = pd.DataFrame(root_rows)
    selected_events_all = (
        pd.concat(selected_event_frames, ignore_index=True)
        if selected_event_frames
        else empty_event_frame()
    )
    selections_all = (
        pd.concat(selection_frames, ignore_index=True) if selection_frames else pd.DataFrame()
    )
    candidates_all = (
        pd.concat(candidate_frames, ignore_index=True) if candidate_frames else empty_event_frame()
    )
    event_summary = summarize_events(selected_events_all)

    costs.reset_index().rename(columns={"index": "root"}).to_csv(
        OUTPUT_DIR / "cost_model.csv",
        index=False,
    )
    metrics.to_csv(OUTPUT_DIR / "strategy_metrics.csv", index=False)
    splits.to_csv(OUTPUT_DIR / "split_metrics.csv", index=False)
    roots.to_csv(OUTPUT_DIR / "root_contributions.csv", index=False)
    selected_events_all.to_csv(OUTPUT_DIR / "selected_event_log.csv", index=False)
    selections_all.to_csv(OUTPUT_DIR / "walkforward_selections.csv", index=False)
    event_summary.to_csv(OUTPUT_DIR / "event_summary.csv", index=False)
    candidates_all.to_parquet(OUTPUT_DIR / "candidate_event_log.parquet", index=False)

    primary = metrics[metrics["cost_multiplier"].eq(PRIMARY_COST_MULTIPLIER)].copy()
    best_strategy = select_best_strategy(primary)
    write_best_returns(strategy_frames[best_strategy], best_strategy)
    write_top_returns(strategy_frames, primary)
    plot_outputs(strategy_frames, metrics, splits, roots, selections_all, best_strategy)
    write_report(
        metrics,
        splits,
        event_summary,
        roots,
        selections_all,
        costs,
        best_strategy,
        inputs,
    )
    write_results_json(metrics, best_strategy, inputs)

    print(primary.head(20).to_string(index=False))
    print(f"Wrote {OUTPUT_DIR}", flush=True)


def load_cost_model() -> pd.DataFrame:
    if not COST_PATH.exists():
        raise FileNotFoundError(f"Missing cost model: {COST_PATH}")
    costs = pd.read_csv(COST_PATH)
    required = {"root", "per_side_cost_bps"}
    missing = required - set(costs.columns)
    if missing:
        raise ValueError(f"{COST_PATH} missing columns: {sorted(missing)}")
    costs = costs.set_index("root").reindex(ROOTS)
    if costs["per_side_cost_bps"].isna().any():
        missing_roots = costs.index[costs["per_side_cost_bps"].isna()].tolist()
        raise ValueError(f"Missing cost estimates for roots: {missing_roots}")
    costs["contract_multiplier"] = [CONTRACT_MULTIPLIERS[root] for root in costs.index]
    costs["cost_source"] = costs.get("source", "mbp1")
    return costs[["per_side_cost_bps", "contract_multiplier", "cost_source"]]


def generate_candidate_events(inputs: dict[str, Any], spec: StrategySpec) -> pd.DataFrame:
    index: pd.DatetimeIndex = inputs["index"]
    returns = inputs["after_carry_returns"]
    z = inputs["zscores"][spec.base.window]
    carry = inputs["carry_ann"]
    nrows = len(index)

    entry_states = precompute_states(inputs, spec, entry_threshold=ENTRY_Z + spec.z_buffer)
    hold_states = precompute_states(inputs, spec, entry_threshold=ENTRY_Z)
    triggers = precompute_triggers(inputs, spec.timing, entry_states)
    exit_momentum = precompute_exit_momentum(inputs, spec.timing)
    entry_recent = {
        root: hf_base.rolling_sum(returns[root], max(spec.timing.confirm_bars, 1)).to_numpy(
            dtype=float
        )
        for root in ROOTS
    }
    cumulative = {
        root: np.concatenate([[0.0], returns[root].fillna(0.0).to_numpy(dtype=float).cumsum()])
        for root in ROOTS
    }

    events: list[dict[str, Any]] = []
    for root_index, root in enumerate(ROOTS):
        active: dict[str, Any] | None = None
        for row_num, ts in enumerate(index):
            if active is not None:
                should_exit, reason = event_should_exit(
                    inputs,
                    spec,
                    hold_states,
                    exit_momentum,
                    row_num,
                    root_index,
                    active,
                )
                if should_exit:
                    active.update(
                        {
                            "exit_idx": row_num,
                            "exit_ts": ts,
                            "exit_z": z.iat[row_num, root_index],
                            "exit_reason": reason,
                        }
                    )
                    events.append(attach_single_event_return(active, cumulative[root]))
                    active = None

            if active is not None:
                continue

            for side, position in [("long_cheap", 1.0), ("short_rich", -1.0)]:
                if not triggers[(root, side)][row_num]:
                    continue
                active = {
                    "root": root,
                    "side": side,
                    "position": position,
                    "entry_idx": row_num,
                    "entry_ts": ts,
                    "entry_z": z.iat[row_num, root_index],
                    "entry_carry_pct_ann": carry.iat[row_num, root_index],
                    "entry_recent_return_bp": position * entry_recent[root][row_num] * 10_000.0,
                    "window": spec.base.window,
                    "agree_window": spec.base.agree_window,
                    "carry_tailwind": spec.base.carry_tailwind,
                    "entry_threshold": ENTRY_Z + spec.z_buffer,
                    "hold_threshold": ENTRY_Z,
                    "exit_threshold": EXIT_Z,
                }
                break

        if active is not None:
            final_row = nrows - 1
            active.update(
                {
                    "exit_idx": final_row,
                    "exit_ts": index[final_row],
                    "exit_z": z.iat[final_row, root_index],
                    "exit_reason": "end_of_sample",
                }
            )
            events.append(attach_single_event_return(active, cumulative[root]))

    if not events:
        return empty_event_frame()
    output = pd.DataFrame(events)
    output["entry_ts"] = pd.to_datetime(output["entry_ts"], utc=True)
    output["exit_ts"] = pd.to_datetime(output["exit_ts"], utc=True)
    output["duration_hours"] = (
        output["exit_ts"] - output["entry_ts"]
    ).dt.total_seconds() / 3600.0
    output["holding_bars"] = (output["exit_idx"] - output["entry_idx"]).clip(lower=0)
    output["normalized"] = output["exit_reason"].eq("normalized")
    return output.sort_values(["entry_ts", "root", "side"]).reset_index(drop=True)


def precompute_states(
    inputs: dict[str, Any],
    spec: StrategySpec,
    *,
    entry_threshold: float,
) -> dict[tuple[str, str], np.ndarray]:
    states: dict[tuple[str, str], np.ndarray] = {}
    for root_index, root in enumerate(ROOTS):
        for side, position in [("long_cheap", 1.0), ("short_rich", -1.0)]:
            states[(root, side)] = eligible_state(
                inputs,
                spec.base,
                root_index,
                position,
                entry_threshold=entry_threshold,
            ).to_numpy(dtype=bool)
    return states


def eligible_state(
    inputs: dict[str, Any],
    base_variant: event_base.EventVariant,
    root_index: int,
    position: float,
    *,
    entry_threshold: float,
) -> pd.Series:
    z = inputs["zscores"][base_variant.window].iloc[:, root_index]
    state = z.le(-entry_threshold) if position > 0 else z.ge(entry_threshold)

    if base_variant.carry_tailwind:
        carry = inputs["carry_ann"].iloc[:, root_index]
        state &= (-position * carry).gt(0.0)

    if base_variant.agree_window is not None:
        agree = inputs["zscores"][base_variant.agree_window].iloc[:, root_index]
        state &= np.sign(agree).eq(np.sign(z))
        state &= agree.abs().ge(base_variant.agree_min_abs_z)
    return state.fillna(False)


def precompute_triggers(
    inputs: dict[str, Any],
    timing: hf_base.TimingVariant,
    entry_states: dict[tuple[str, str], np.ndarray],
) -> dict[tuple[str, str], np.ndarray]:
    returns = inputs["after_carry_returns"]
    output: dict[tuple[str, str], np.ndarray] = {}
    for root in ROOTS:
        for side, position in [("long_cheap", 1.0), ("short_rich", -1.0)]:
            key = (root, side)
            state = pd.Series(entry_states[key], index=returns.index)
            if timing.trigger == "state":
                trigger = state
            elif timing.trigger == "exhaust":
                lookback = hf_base.rolling_sum(returns[root], timing.lookback_bars).shift(1)
                confirm = hf_base.rolling_sum(returns[root], timing.confirm_bars)
                trigger = (
                    state
                    & (position * lookback <= -(timing.impulse_bps / 10_000.0))
                    & (position * confirm > 0.0)
                )
            elif timing.trigger == "confirm":
                confirm = hf_base.rolling_sum(returns[root], timing.confirm_bars)
                trigger = state & (position * confirm >= timing.impulse_bps / 10_000.0)
            else:
                raise ValueError(f"Unknown trigger: {timing.trigger}")
            output[key] = trigger.fillna(False).to_numpy(dtype=bool)
    return output


def precompute_exit_momentum(
    inputs: dict[str, Any],
    timing: hf_base.TimingVariant,
) -> dict[str, np.ndarray]:
    if timing.exit_rule != "momentum":
        return {root: np.zeros(len(inputs["index"]), dtype=float) for root in ROOTS}
    return {
        root: hf_base.rolling_sum(inputs["after_carry_returns"][root], timing.exit_bars).to_numpy(
            dtype=float
        )
        for root in ROOTS
    }


def event_should_exit(
    inputs: dict[str, Any],
    spec: StrategySpec,
    hold_states: dict[tuple[str, str], np.ndarray],
    exit_momentum: dict[str, np.ndarray],
    row_num: int,
    root_index: int,
    event: dict[str, Any],
) -> tuple[bool, str]:
    root = ROOTS[root_index]
    z_value = inputs["zscores"][spec.base.window].iat[row_num, root_index]
    if np.isfinite(z_value) and abs(z_value) <= EXIT_Z:
        return True, "normalized"
    if not hold_states[(root, str(event["side"]))][row_num]:
        return True, "state_invalid"
    if spec.timing.exit_rule == "momentum":
        recent = exit_momentum[root][row_num]
        if float(event["position"]) * recent <= -(spec.timing.exit_bps / 10_000.0):
            return True, "momentum_failed"
    return False, ""


def attach_single_event_return(event: dict[str, Any], cumulative: np.ndarray) -> dict[str, Any]:
    entry_idx = int(event["entry_idx"])
    exit_idx = int(event["exit_idx"])
    position = float(event["position"])
    if exit_idx <= entry_idx:
        trade_return = 0.0
    else:
        trade_return = position * (cumulative[exit_idx + 1] - cumulative[entry_idx + 1])
    event = event.copy()
    event["trade_return_log"] = trade_return
    event["trade_return_bp"] = trade_return * 10_000.0
    return event


def select_walk_forward_events(
    candidates: pd.DataFrame,
    index: pd.DatetimeIndex,
    spec: StrategySpec,
    policy: SelectionPolicy,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        return empty_event_frame(), pd.DataFrame()

    events = candidates.copy()
    events["entry_ts"] = pd.to_datetime(events["entry_ts"], utc=True)
    events["exit_ts"] = pd.to_datetime(events["exit_ts"], utc=True)
    dates = rebalance_dates(index)
    selected_parts = []
    selection_rows = []

    for rebalance_ts, next_ts in pairwise(dates):
        train = events[events["exit_ts"].lt(rebalance_ts)]
        stats = group_training_stats(train)
        stats["selected"] = selection_mask(stats, policy)
        selected_groups = {
            (row.root, row.side) for row in stats[stats["selected"]].itertuples(index=False)
        }
        period_events = events[
            events["entry_ts"].ge(rebalance_ts) & events["entry_ts"].lt(next_ts)
        ].copy()
        if selected_groups:
            selected_mask = [
                (root, side) in selected_groups
                for root, side in zip(
                    period_events["root"],
                    period_events["side"],
                    strict=True,
                )
            ]
            period_events = period_events[
                selected_mask
            ]
        else:
            period_events = period_events.iloc[0:0]
        if not period_events.empty:
            period_events["rebalance_ts"] = rebalance_ts
            period_events["selection_policy"] = policy.name
            selected_parts.append(period_events)

        if stats.empty:
            selection_rows.append(
                {
                    "spec": spec.name,
                    "base_strategy": spec.base.name,
                    "timing": spec.timing.name,
                    "z_buffer": spec.z_buffer,
                    "selection_policy": policy.name,
                    "rebalance_ts": rebalance_ts,
                    "root": "",
                    "side": "",
                    "train_event_count": 0,
                    "train_hit_rate": np.nan,
                    "train_mean_trade_return_bp": np.nan,
                    "train_tstat": np.nan,
                    "selected": False,
                }
            )
        else:
            selection_rows.extend(
                [
                    {
                        "spec": spec.name,
                        "base_strategy": spec.base.name,
                        "timing": spec.timing.name,
                        "z_buffer": spec.z_buffer,
                        "selection_policy": policy.name,
                        "rebalance_ts": rebalance_ts,
                        "root": row.root,
                        "side": row.side,
                        "train_event_count": int(row.train_event_count),
                        "train_hit_rate": float(row.train_hit_rate),
                        "train_mean_trade_return_bp": float(row.train_mean_trade_return_bp),
                        "train_tstat": float(row.train_tstat)
                        if np.isfinite(row.train_tstat)
                        else np.nan,
                        "selected": bool(row.selected),
                    }
                    for row in stats.itertuples(index=False)
                ]
            )

    selected = (
        pd.concat(selected_parts, ignore_index=True)
        if selected_parts
        else empty_event_frame()
    )
    selections = pd.DataFrame(selection_rows)
    return selected, selections


def rebalance_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    start = max(EVAL_START, index.min())
    end = index.max()
    years = range(start.year, end.year + 2)
    dates = [pd.Timestamp(f"{year}-01-01", tz="UTC") for year in years]
    dates = [date for date in dates if date >= start and date <= end]
    if not dates or dates[0] > start:
        dates.insert(0, start)
    terminal = end + pd.Timedelta(nanoseconds=1)
    if dates[-1] < terminal:
        dates.append(terminal)
    return dates


def group_training_stats(train: pd.DataFrame) -> pd.DataFrame:
    if train.empty:
        return pd.DataFrame(
            columns=[
                "root",
                "side",
                "train_event_count",
                "train_hit_rate",
                "train_mean_trade_return_bp",
                "train_tstat",
            ]
        )
    rows = []
    for key, group in train.groupby(["root", "side"], sort=True):
        rows.append(
            {
                "root": key[0],
                "side": key[1],
                "train_event_count": len(group),
                "train_hit_rate": group["trade_return_log"].gt(0).mean(),
                "train_mean_trade_return_bp": group["trade_return_bp"].mean(),
                "train_tstat": event_base.tstat(group["trade_return_log"]),
            }
        )
    return pd.DataFrame(rows)


def selection_mask(stats: pd.DataFrame, policy: SelectionPolicy) -> pd.Series:
    if stats.empty:
        return pd.Series(dtype=bool)
    mask = (
        stats["train_event_count"].ge(policy.min_events)
        & stats["train_hit_rate"].ge(policy.min_hit_rate)
        & stats["train_mean_trade_return_bp"].gt(policy.min_mean_bp)
    )
    if policy.min_tstat is not None:
        mask &= stats["train_tstat"].ge(policy.min_tstat)
    return mask.fillna(False)


def build_targets_from_events(events: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    target_array = np.zeros((len(index), len(ROOTS)), dtype=float)
    if not events.empty:
        root_to_idx = {root: i for i, root in enumerate(ROOTS)}
        for event in events.itertuples(index=False):
            entry_idx = int(event.entry_idx)
            exit_idx = int(event.exit_idx)
            if exit_idx <= entry_idx:
                continue
            target_array[entry_idx:exit_idx, root_to_idx[event.root]] = float(event.position)
    return pd.DataFrame(target_array, index=index, columns=ROOTS)


def simulate_strategy(
    strategy_name: str,
    raw_targets: pd.DataFrame,
    inputs: dict[str, Any],
    costs: pd.DataFrame,
) -> pd.DataFrame:
    returns = inputs["after_carry_returns"]
    weights = inputs["weights"]
    targets = hf_base.normalize_gross(raw_targets)
    exec_positions = targets.shift(1).fillna(0.0)
    root_pnl = (exec_positions * returns).fillna(0.0)
    gross_return = root_pnl.sum(axis=1)

    leg_exposures = hf_base.build_leg_exposures(targets, weights)
    leg_trade = leg_exposures.diff().abs()
    leg_trade.iloc[0] = leg_exposures.iloc[0].abs()
    leg_trade = leg_trade.fillna(0.0)
    turnover = leg_trade.sum(axis=1)
    base_cost = leg_trade.mul(costs["per_side_cost_bps"], axis=1).sum(axis=1) / 10_000.0

    frame = pd.DataFrame(
        {
            "ts": returns.index,
            "strategy": strategy_name,
            "gross_return": gross_return.to_numpy(dtype=float),
            "turnover": turnover.to_numpy(dtype=float),
            "base_cost_return": base_cost.to_numpy(dtype=float),
            "exec_gross": exec_positions.abs().sum(axis=1).to_numpy(dtype=float),
            "signal_gross": raw_targets.abs().sum(axis=1).to_numpy(dtype=float),
            "leg_gross": leg_exposures.abs().sum(axis=1).to_numpy(dtype=float),
        }
    )
    for root in ROOTS:
        frame[f"{root}_pnl"] = root_pnl[root].to_numpy(dtype=float)
        frame[f"{root}_leg_trade"] = leg_trade[root].to_numpy(dtype=float)
    for multiplier in COST_MULTIPLIERS:
        label = cost_multiplier_label(multiplier)
        cost = base_cost * multiplier
        frame[f"net_return_{label}"] = (gross_return - cost).to_numpy(dtype=float)
        frame[f"cost_return_{label}"] = cost.to_numpy(dtype=float)
    return frame


def strategy_metrics(strategy_name: str, strategy: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        metrics_for_group(strategy_name, "full", strategy, multiplier)
        for multiplier in COST_MULTIPLIERS
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
        rows.append(metrics_for_group(strategy_name, split_name, data, PRIMARY_COST_MULTIPLIER))
    return rows


def metrics_for_group(
    strategy_name: str,
    split_name: str,
    data: pd.DataFrame,
    cost_multiplier: float,
) -> dict[str, Any]:
    data = data.sort_values("ts")
    label = cost_multiplier_label(cost_multiplier)
    net = data[f"net_return_{label}"].astype(float)
    gross = data["gross_return"].astype(float)
    elapsed_years = event_base.years_between(data["ts"].iloc[0], data["ts"].iloc[-1])
    obs_per_year = len(data) / elapsed_years if elapsed_years > 0 else np.nan
    annual_log_return = net.sum() / elapsed_years if elapsed_years > 0 else np.nan
    annual_vol = net.std(ddof=1) * math.sqrt(obs_per_year) if len(data) > 1 else np.nan
    active = data["exec_gross"].gt(0)
    active_returns = net[active]
    cost_sum = data[f"cost_return_{label}"].sum()
    turnover_sum = data["turnover"].sum()
    return {
        "strategy": strategy_name,
        "split": split_name,
        "cost_multiplier": cost_multiplier,
        "nobs": len(data),
        "years": elapsed_years,
        "active_fraction": active.mean(),
        "cum_log_return": net.sum(),
        "gross_cum_log_return": gross.sum(),
        "cost_cum_log_return": cost_sum,
        "realized_avg_cost_bps": cost_sum / turnover_sum * 10_000.0
        if turnover_sum > EPSILON
        else np.nan,
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
        "avg_exec_gross": data["exec_gross"].mean(),
        "avg_leg_gross": data["leg_gross"].mean(),
        "annual_turnover": turnover_sum / elapsed_years if elapsed_years > 0 else np.nan,
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
                "leg_turnover": strategy[f"{root}_leg_trade"].sum(),
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
                "normalized_fraction": group["normalized"].mean(),
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


def select_best_strategy(primary_metrics: pd.DataFrame) -> str:
    eligible = primary_metrics[primary_metrics["active_fraction"].ge(0.005)].copy()
    if eligible.empty:
        eligible = primary_metrics.copy()
    best = eligible.sort_values(["sharpe", "cagr"], ascending=[False, False]).iloc[0]
    return str(best["strategy"])


def write_best_returns(strategy: pd.DataFrame, best_strategy: str) -> None:
    strategy.to_csv(OUTPUT_DIR / "best_strategy_returns.csv", index=False)
    (OUTPUT_DIR / "best_strategy_name.txt").write_text(best_strategy + "\n", encoding="utf-8")


def write_top_returns(
    strategy_frames: dict[str, pd.DataFrame],
    primary_metrics: pd.DataFrame,
) -> None:
    top_names = primary_metrics.head(10)["strategy"].tolist()
    frames = [strategy_frames[name] for name in top_names]
    pd.concat(frames, ignore_index=True).to_parquet(OUTPUT_DIR / "top_strategy_returns.parquet")


def plot_outputs(
    strategy_frames: dict[str, pd.DataFrame],
    metrics: pd.DataFrame,
    splits: pd.DataFrame,
    roots: pd.DataFrame,
    selections: pd.DataFrame,
    best_strategy: str,
) -> None:
    plot_top_equity(strategy_frames, metrics)
    plot_best_drawdown(strategy_frames[best_strategy], best_strategy)
    plot_metric_bars(metrics)
    plot_split_heatmap(splits, metrics)
    plot_root_contributions(roots, metrics)
    plot_selection_counts(selections)
    plot_cost_sensitivity(metrics)


def plot_top_equity(strategy_frames: dict[str, pd.DataFrame], metrics: pd.DataFrame) -> None:
    primary = metrics[metrics["cost_multiplier"].eq(PRIMARY_COST_MULTIPLIER)].head(8)
    label = cost_multiplier_label(PRIMARY_COST_MULTIPLIER)
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for strategy_name in primary["strategy"]:
        data = strategy_frames[strategy_name]
        equity = np.exp(data[f"net_return_{label}"].cumsum()) - 1.0
        plot_data = daily_sample(data["ts"], equity, "equity")
        ax.plot(
            plot_data["ts"],
            plot_data["equity"],
            linewidth=1.1,
            label=compact_strategy_label(strategy_name),
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Top walk-forward HF dislocation strategies, root-specific costs")
    ax.set_ylabel("Cumulative return")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=6.5, frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top_strategy_equity_contract_cost.png", dpi=170)
    plt.close(fig)


def plot_best_drawdown(strategy: pd.DataFrame, best_strategy: str) -> None:
    label = cost_multiplier_label(PRIMARY_COST_MULTIPLIER)
    cumulative = strategy[f"net_return_{label}"].cumsum()
    drawdown = np.exp(cumulative - cumulative.cummax()) - 1.0
    equity_plot = daily_sample(strategy["ts"], np.exp(cumulative) - 1.0, "equity")
    drawdown_plot = daily_sample(strategy["ts"], drawdown, "drawdown")
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
    axes[0].plot(equity_plot["ts"], equity_plot["equity"], color="#1f4e79", linewidth=1.1)
    axes[0].set_title(f"Best walk-forward strategy: {best_strategy}")
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


def plot_metric_bars(metrics: pd.DataFrame) -> None:
    primary = metrics[metrics["cost_multiplier"].eq(PRIMARY_COST_MULTIPLIER)].head(12)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharey=True)
    y = np.arange(len(primary))
    axes[0].barh(y, primary["sharpe"], color="#355c7d")
    axes[0].set_yticks(y, labels=[compact_strategy_label(item) for item in primary["strategy"]])
    axes[0].invert_yaxis()
    axes[0].set_title("Sharpe")
    axes[0].grid(True, axis="x", alpha=0.25)
    axes[1].barh(y, primary["cagr"] * 100.0, color="#6c8f57")
    axes[1].set_title("CAGR, %")
    axes[1].grid(True, axis="x", alpha=0.25)
    fig.suptitle("Top walk-forward strategies, 1x root-specific costs")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top_strategy_metrics_contract_cost.png", dpi=170)
    plt.close(fig)


def plot_split_heatmap(splits: pd.DataFrame, metrics: pd.DataFrame) -> None:
    top = metrics[metrics["cost_multiplier"].eq(PRIMARY_COST_MULTIPLIER)].head(10)[
        "strategy"
    ].tolist()
    matrix = (
        splits[splits["strategy"].isin(top)]
        .pivot(index="strategy", columns="split", values="sharpe")
        .reindex(index=top, columns=list(event_base.SPLITS))
    )
    values = matrix.to_numpy(dtype=float)
    vmax = max(1.0, np.nanpercentile(np.abs(values), 95))
    fig, ax = plt.subplots(figsize=(9.5, 6.8), constrained_layout=True)
    image = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title("Sharpe by period, 1x costs")
    ax.set_xticks(np.arange(len(matrix.columns)), labels=matrix.columns, rotation=30, ha="right")
    ax.set_yticks(
        np.arange(len(matrix.index)),
        labels=[compact_strategy_label(item) for item in matrix.index],
    )
    for i, strategy in enumerate(matrix.index):
        for j, split in enumerate(matrix.columns):
            value = matrix.loc[strategy, split]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Sharpe")
    fig.savefig(OUTPUT_DIR / "split_sharpe_heatmap_contract_cost.png", dpi=170)
    plt.close(fig)


def plot_root_contributions(roots: pd.DataFrame, metrics: pd.DataFrame) -> None:
    top = metrics[metrics["cost_multiplier"].eq(PRIMARY_COST_MULTIPLIER)].head(8)[
        "strategy"
    ].tolist()
    matrix = (
        roots[roots["strategy"].isin(top)]
        .pivot(index="strategy", columns="root", values="contribution_bp")
        .reindex(index=top, columns=ROOTS)
    )
    values = matrix.to_numpy(dtype=float)
    vmax = max(10.0, np.nanpercentile(np.abs(values), 95))
    fig, ax = plt.subplots(figsize=(8.5, 6.2), constrained_layout=True)
    image = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title("Residual basket contribution to gross PnL")
    ax.set_xticks(np.arange(len(ROOTS)), labels=ROOTS)
    ax.set_yticks(
        np.arange(len(matrix.index)),
        labels=[compact_strategy_label(item) for item in matrix.index],
    )
    for i, strategy in enumerate(matrix.index):
        for j, root in enumerate(ROOTS):
            value = matrix.loc[strategy, root]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="bp")
    fig.savefig(OUTPUT_DIR / "root_contribution_heatmap.png", dpi=170)
    plt.close(fig)


def plot_selection_counts(selections: pd.DataFrame) -> None:
    selected = selections[selections["selected"].astype(bool)].copy()
    if selected.empty:
        return
    selected["year"] = pd.to_datetime(selected["rebalance_ts"], utc=True).dt.year
    counts = (
        selected.groupby(["base_strategy", "timing", "z_buffer", "selection_policy", "year"])
        .size()
        .rename("selected_groups")
        .reset_index()
    )
    pivot = counts.pivot_table(
        index=["base_strategy", "timing", "z_buffer", "selection_policy"],
        columns="year",
        values="selected_groups",
        aggfunc="sum",
    ).fillna(0.0)
    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
    image = ax.imshow(pivot.to_numpy(dtype=float), cmap="YlGnBu", aspect="auto")
    ax.set_title("Walk-forward selected root/side group count")
    ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns)
    labels = [
        f"{compact_spec_label(base, timing, buffer)} | {compact_policy_label(policy)}"
        for base, timing, buffer, policy in pivot.index
    ]
    ax.set_yticks(np.arange(len(pivot.index)), labels=labels, fontsize=6)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iat[i, j]
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="groups")
    fig.savefig(OUTPUT_DIR / "walkforward_selection_counts.png", dpi=170)
    plt.close(fig)


def plot_cost_sensitivity(metrics: pd.DataFrame) -> None:
    top = metrics[metrics["cost_multiplier"].eq(PRIMARY_COST_MULTIPLIER)].head(8)[
        "strategy"
    ].tolist()
    data = metrics[metrics["strategy"].isin(top)].copy()
    matrix = data.pivot(index="strategy", columns="cost_multiplier", values="sharpe").reindex(
        index=top,
        columns=COST_MULTIPLIERS,
    )
    values = matrix.to_numpy(dtype=float)
    vmax = max(1.0, np.nanpercentile(np.abs(values), 95))
    fig, ax = plt.subplots(figsize=(8.5, 6.2), constrained_layout=True)
    image = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title("Sharpe sensitivity to root-specific cost multiplier")
    ax.set_xticks(np.arange(len(matrix.columns)), labels=[f"{x:g}x" for x in matrix.columns])
    ax.set_yticks(
        np.arange(len(matrix.index)),
        labels=[compact_strategy_label(item) for item in matrix.index],
    )
    for i, strategy in enumerate(matrix.index):
        for j, multiplier in enumerate(matrix.columns):
            value = matrix.loc[strategy, multiplier]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Sharpe")
    fig.savefig(OUTPUT_DIR / "cost_sensitivity_sharpe_heatmap.png", dpi=170)
    plt.close(fig)


def daily_sample(ts: pd.Series, values: pd.Series | np.ndarray, column: str) -> pd.DataFrame:
    frame = pd.DataFrame({"ts": pd.to_datetime(ts, utc=True), column: np.asarray(values)})
    sampled = frame.set_index("ts").resample("1D").last().dropna().reset_index()
    sampled["ts"] = sampled["ts"].dt.tz_convert(None)
    return sampled


def compact_strategy_label(name: str) -> str:
    label = name.removeprefix("wf_event_").removesuffix("_meanpos")
    replacements = {
        "20D_agree_60D_carry": "20D agree60 carry",
        "60D_agree_120D": "60D agree120",
        "60D_pure": "60D pure",
        "252D_pure": "252D pure",
        "exhaust_30m_flip5_hold_state": "30m exhaust",
        "exhaust_60m_confirm15_hold_state": "60m exhaust",
        "state_hold": "state",
        "zbuf0p0": "z+0.0",
        "zbuf0p25": "z+0.25",
        "zbuf0p5": "z+0.5",
        "expanding_n10_hit55": "n10/h55",
        "expanding_n5_hit52": "n5/h52",
    }
    for old, new in replacements.items():
        label = label.replace(old, new)
    return " | ".join(part for part in label.split("_") if part)


def compact_spec_label(base_strategy: str, timing: str, z_buffer: float) -> str:
    pseudo_name = f"wf_{base_strategy}_{timing}_zbuf{str(z_buffer).replace('.', 'p')}"
    return compact_strategy_label(pseudo_name)


def compact_policy_label(policy: str) -> str:
    return policy.replace("expanding_", "").replace("_meanpos", "").replace("_", "/")


def write_report(
    metrics: pd.DataFrame,
    splits: pd.DataFrame,
    event_summary: pd.DataFrame,
    roots: pd.DataFrame,
    selections: pd.DataFrame,
    costs: pd.DataFrame,
    best_strategy: str,
    inputs: dict[str, Any],
) -> None:
    primary = metrics[metrics["cost_multiplier"].eq(PRIMARY_COST_MULTIPLIER)].copy()
    no_cost = metrics[metrics["cost_multiplier"].eq(0.0)].copy()
    best = primary[primary["strategy"].eq(best_strategy)].iloc[0]
    top_names = primary.head(8)["strategy"].tolist()
    top_splits = splits[splits["strategy"].isin(top_names)].copy()
    best_roots = roots[roots["strategy"].eq(best_strategy)].copy()
    best_events = event_summary[event_summary["strategy"].eq(best_strategy)].copy()
    selected_counts = (
        selections[selections["selected"].astype(bool)]
        .groupby(["strategy", "rebalance_ts"], sort=True)
        .size()
        .rename("selected_group_count")
        .reset_index()
    )
    best_selection_counts = selected_counts[selected_counts["strategy"].eq(best_strategy)]

    report = [
        "# HYP-0046 HF Walk-Forward Execution Test",
        "",
        "Objective: retest the HF fair-value dislocation thesis under tougher assumptions:",
        "walk-forward root/side selection, deeper-entry no-trade buffers, and root-specific",
        "execution costs estimated from MBP1 spreads.",
        "",
        "Construction:",
        "",
        "- Candidate fair-value events are generated on 5-minute PC1-PC2 residual baskets.",
        "- Entry uses hourly fair-value z-scores forward-filled to 5-minute bars.",
        "- A z-buffer is added to the entry threshold; the hold threshold remains 2.5 z.",
        "- Selection is expanding walk-forward by calendar year, using only events closed",
        "before each rebalance date.",
        "- Groups are selected by root and side when prior event count, hit rate, and mean",
        "event return clear the policy hurdle.",
        "- Position is applied one 5-minute bar after the trigger.",
        "- Costs are charged on constituent leg turnover using root-specific empirical",
        "per-side MBP1 spread estimates.",
        "",
        "Cost model:",
        "",
        costs.reset_index().rename(columns={"index": "root"}).to_markdown(
            index=False,
            floatfmt=".4f",
        ),
        "",
        "Caveats:",
        "",
        "- This is a root-specific notional turnover cost proxy, not queue-position or",
        "discrete-contract execution simulation.",
        "- Selection is annual expanding walk-forward; it does not tune continuously inside",
        "a year.",
        "- The fair-value state still updates hourly, so 5-minute logic only times entry",
        "inside a slower signal.",
        "",
        "## Best Strategy At 1x Root-Specific Costs",
        "",
        best.to_frame().T.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Top Strategies At 1x Root-Specific Costs",
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
        "## Best Strategy Selected Group Counts",
        "",
        best_selection_counts.to_markdown(index=False, floatfmt=".4f")
        if not best_selection_counts.empty
        else "No selected groups.",
        "",
        "## Event Summary For Best Strategy",
        "",
        best_events.to_markdown(index=False, floatfmt=".4f")
        if not best_events.empty
        else "No selected events.",
        "",
        "## Input Span",
        "",
        f"- 5-minute start: `{inputs['index'].min()}`",
        f"- 5-minute end: `{inputs['index'].max()}`",
        f"- rows: `{len(inputs['index'])}`",
        f"- evaluation start: `{EVAL_START}`",
        "",
        "## Files",
        "",
        "- `strategy_metrics.csv`",
        "- `split_metrics.csv`",
        "- `event_summary.csv`",
        "- `selected_event_log.csv`",
        "- `candidate_event_log.parquet`",
        "- `walkforward_selections.csv`",
        "- `root_contributions.csv`",
        "- `cost_model.csv`",
        "- `best_strategy_returns.csv`",
        "- `top_strategy_returns.parquet`",
        "- `top_strategy_equity_contract_cost.png`",
        "- `best_strategy_equity_drawdown.png`",
        "- `top_strategy_metrics_contract_cost.png`",
        "- `split_sharpe_heatmap_contract_cost.png`",
        "- `walkforward_selection_counts.png`",
        "- `cost_sensitivity_sharpe_heatmap.png`",
        "- `root_contribution_heatmap.png`",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(report), encoding="utf-8")


def write_results_json(metrics: pd.DataFrame, best_strategy: str, inputs: dict[str, Any]) -> None:
    best_metrics = metrics[
        metrics["cost_multiplier"].eq(PRIMARY_COST_MULTIPLIER)
        & metrics["strategy"].eq(best_strategy)
    ].iloc[0]
    summary = {
        "experiment_id": "HYP-0046",
        "completed_at": datetime.now(UTC).isoformat(),
        "data_start": inputs["index"].min().isoformat(),
        "data_end": inputs["index"].max().isoformat(),
        "rows_5m": len(inputs["index"]),
        "evaluation_start": EVAL_START.isoformat(),
        "entry_z": ENTRY_Z,
        "exit_z": EXIT_Z,
        "z_buffers": Z_BUFFERS,
        "primary_cost_multiplier": PRIMARY_COST_MULTIPLIER,
        "cost_multipliers": COST_MULTIPLIERS,
        "selection_policies": [policy.__dict__ for policy in SELECTION_POLICIES],
        "best_strategy": best_strategy,
        "best_metrics_1x_cost": best_metrics.to_dict(),
    }
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def cost_multiplier_label(multiplier: float) -> str:
    return f"costx{multiplier:g}".replace(".", "p")


def empty_event_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "root",
            "side",
            "position",
            "entry_idx",
            "exit_idx",
            "entry_ts",
            "exit_ts",
            "entry_z",
            "exit_z",
            "exit_reason",
            "trade_return_log",
            "trade_return_bp",
            "duration_hours",
            "holding_bars",
            "normalized",
        ]
    )


def json_safe(value: Any) -> Any:
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
