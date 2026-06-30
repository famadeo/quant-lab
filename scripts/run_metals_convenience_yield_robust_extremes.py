from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_metals_convenience_yield_basis_backtest as base  # noqa: E402

INPUT_PANEL = (
    REPO_ROOT
    / "experiments"
    / "HYP-0031-metals-convenience-yield-basis-5m-sync"
    / "curve_panel.parquet"
)
OUTPUT_DIR = REPO_ROOT / "experiments" / "HYP-0032-metals-convenience-yield-robust-extremes"

RANK_ENTRY_LEVELS = [0.05, 0.10]
MAD_ENTRY_LEVELS = [1.5, 2.0]
MAD_EXIT_LEVEL = 0.25
CURVE_MD_ENTRIES = [2.5, 3.0]
CURVE_MD_EXIT = 1.5
CURVE_COMPONENT_ENTRY = 1.25
CURVE_COMPONENT_EXIT = 0.25
QUANTILE_ENTRY_LEVELS = [0.05, 0.10]
EXIT_LOW_PERCENTILE = 0.40
EXIT_HIGH_PERCENTILE = 0.60
MIN_CURVE_COMPONENTS = 2


@dataclass(frozen=True)
class RobustVariant:
    detector: str
    target_months: int
    min_volume: float
    lookback: int
    side_mode: str
    cost_multiplier: float
    entry_level: float
    exit_level: float = MAD_EXIT_LEVEL
    md_entry: float = np.nan
    md_exit: float = CURVE_MD_EXIT
    component_entry: float = CURVE_COMPONENT_ENTRY
    component_exit: float = CURVE_COMPONENT_EXIT

    @property
    def name(self) -> str:
        min_volume = int(self.min_volume)
        cost = str(self.cost_multiplier).replace(".", "p")
        entry = str(self.entry_level).replace(".", "p")
        if self.detector == "curve_md":
            md_entry = str(self.md_entry).replace(".", "p")
            comp = str(self.component_entry).replace(".", "p")
            return (
                f"{self.detector}_target{self.target_months}m_minv{min_volume}_"
                f"lb{self.lookback}_md{md_entry}_comp{comp}_{self.side_mode}_costx{cost}"
            )
        return (
            f"{self.detector}_target{self.target_months}m_minv{min_volume}_"
            f"lb{self.lookback}_entry{entry}_{self.side_mode}_costx{cost}"
        )


def finite_std(series: pd.Series) -> float:
    return float(series.std(ddof=1)) if len(series) > 1 else np.nan


def periods_per_year(index: pd.Index) -> float:
    return base.periods_per_year(index)


def rolling_univariate_features(frame: pd.DataFrame, lookback: int) -> pd.DataFrame:
    data = frame.sort_values("date").copy()
    values = data["carry_pct_ann"].to_numpy(dtype=float)
    percentiles = np.full(len(data), np.nan)
    medians = np.full(len(data), np.nan)
    mad_scores = np.full(len(data), np.nan)
    quantiles = {
        0.05: np.full(len(data), np.nan),
        0.10: np.full(len(data), np.nan),
        0.40: np.full(len(data), np.nan),
        0.60: np.full(len(data), np.nan),
        0.90: np.full(len(data), np.nan),
        0.95: np.full(len(data), np.nan),
    }

    for idx, value in enumerate(values):
        if not np.isfinite(value):
            continue
        start = max(0, idx - lookback)
        history = values[start:idx]
        history = history[np.isfinite(history)]
        if len(history) < base.MIN_Z_OBSERVATIONS:
            continue
        less = np.sum(history < value)
        equal = np.sum(history == value)
        percentiles[idx] = (less + 0.5 * equal) / len(history)
        median = np.median(history)
        mad = np.median(np.abs(history - median))
        medians[idx] = median
        if mad > 0:
            mad_scores[idx] = (value - median) / (1.4826 * mad)
        for prob, values_for_prob in quantiles.items():
            values_for_prob[idx] = np.quantile(history, prob)

    data["rank_pct"] = percentiles
    data["rolling_median"] = medians
    data["mad_z"] = mad_scores
    for prob, values_for_prob in quantiles.items():
        data[f"q{int(prob * 100):02d}"] = values_for_prob
    return data


def build_univariate_feature_map(
    panel: pd.DataFrame,
) -> dict[tuple[str, float, int, int], pd.DataFrame]:
    features = {}
    grouped = panel.groupby(["root", "min_volume", "target_months"], sort=False)
    for key, group in grouped:
        for lookback in base.LOOKBACKS:
            features[(*key, lookback)] = rolling_univariate_features(group, lookback)
    return features


def build_curve_state_features(panel: pd.DataFrame) -> dict[tuple[str, float, int], pd.DataFrame]:
    out = {}
    for (root, min_volume), group in panel.groupby(["root", "min_volume"], sort=False):
        pivot = (
            group.pivot_table(
                index="date",
                columns="target_months",
                values="carry_pct_ann",
                aggfunc="last",
            )
            .reindex(columns=base.TARGET_MONTHS)
            .sort_index()
        )
        for lookback in base.LOOKBACKS:
            z_frame = pd.DataFrame(index=pivot.index, columns=pivot.columns, dtype=float)
            md_values = pd.Series(index=pivot.index, dtype=float)
            values = pivot.to_numpy(dtype=float)
            for idx in range(len(pivot)):
                start = max(0, idx - lookback)
                history = values[start:idx]
                if history.shape[0] < base.MIN_Z_OBSERVATIONS:
                    continue
                current = values[idx]
                has_history = np.isfinite(history).any(axis=0)
                if has_history.sum() < MIN_CURVE_COMPONENTS:
                    continue
                med = np.full(history.shape[1], np.nan)
                mad = np.full(history.shape[1], np.nan)
                med[has_history] = np.nanmedian(history[:, has_history], axis=0)
                mad[has_history] = np.nanmedian(
                    np.abs(history[:, has_history] - med[has_history]),
                    axis=0,
                )
                scale = 1.4826 * mad
                valid = np.isfinite(current) & np.isfinite(scale) & (scale > 0)
                if valid.sum() < MIN_CURVE_COMPONENTS:
                    continue
                z = np.full_like(current, np.nan, dtype=float)
                z[valid] = (current[valid] - med[valid]) / scale[valid]
                z_frame.iloc[idx] = z
                md_values.iloc[idx] = math.sqrt(float(np.nansum(z[valid] ** 2)))
            rows = []
            for target in base.TARGET_MONTHS:
                target_rows = pd.DataFrame(
                    {
                        "date": z_frame.index,
                        "target_months": target,
                        "curve_component_z": z_frame[target].to_numpy(dtype=float),
                        "curve_md": md_values.to_numpy(dtype=float),
                    }
                )
                rows.append(target_rows)
            out[(root, min_volume, lookback)] = pd.concat(rows, ignore_index=True)
    return out


def allow_sides(side_mode: str) -> tuple[bool, bool]:
    return side_mode in {"both", "backwardation_only"}, side_mode in {"both", "contango_only"}


def position_from_rank(previous: int, rank_pct: float, variant: RobustVariant) -> int:
    allow_backwardation, allow_contango = allow_sides(variant.side_mode)
    next_position = 0
    low_entry = variant.entry_level
    high_entry = 1.0 - variant.entry_level
    if np.isfinite(rank_pct):
        if previous == 0:
            if allow_backwardation and rank_pct <= low_entry:
                next_position = 1
            elif allow_contango and rank_pct >= high_entry:
                next_position = -1
        elif previous > 0:
            if allow_contango and rank_pct >= high_entry:
                next_position = -1
            elif rank_pct < EXIT_LOW_PERCENTILE:
                next_position = 1
        elif allow_backwardation and rank_pct <= low_entry:
            next_position = 1
        elif rank_pct > EXIT_HIGH_PERCENTILE:
            next_position = -1
    return next_position


def position_from_quantile(previous: int, row: pd.Series, variant: RobustVariant) -> int:
    allow_backwardation, allow_contango = allow_sides(variant.side_mode)
    value = row["carry_pct_ann"]
    low = row[f"q{int(variant.entry_level * 100):02d}"]
    high = row[f"q{int((1.0 - variant.entry_level) * 100):02d}"]
    next_position = 0
    if all(np.isfinite([value, low, high, row["q40"], row["q60"]])):
        if previous == 0:
            if allow_backwardation and value <= low:
                next_position = 1
            elif allow_contango and value >= high:
                next_position = -1
        elif previous > 0:
            if allow_contango and value >= high:
                next_position = -1
            elif value < row["q40"]:
                next_position = 1
        elif allow_backwardation and value <= low:
            next_position = 1
        elif value > row["q60"]:
            next_position = -1
    return next_position


def position_from_mad(previous: int, mad_z: float, variant: RobustVariant) -> int:
    return base.directional_position(
        previous,
        mad_z,
        entry_z=variant.entry_level,
        exit_z=variant.exit_level,
        side_mode=variant.side_mode,
    )


def position_from_curve_md(previous: int, row: pd.Series, variant: RobustVariant) -> int:
    allow_backwardation, allow_contango = allow_sides(variant.side_mode)
    component = row["curve_component_z"]
    md_value = row["curve_md"]
    next_position = 0
    if np.isfinite(component) and np.isfinite(md_value):
        long_extreme = (
            md_value >= variant.md_entry
            and allow_backwardation
            and component <= -variant.component_entry
        )
        short_extreme = (
            md_value >= variant.md_entry
            and allow_contango
            and component >= variant.component_entry
        )
        long_normalized = md_value <= variant.md_exit or component >= -variant.component_exit
        short_normalized = md_value <= variant.md_exit or component <= variant.component_exit
        if previous == 0:
            if long_extreme:
                next_position = 1
            elif short_extreme:
                next_position = -1
        elif previous > 0:
            if short_extreme:
                next_position = -1
            elif not long_normalized:
                next_position = 1
        elif long_extreme:
            next_position = 1
        elif not short_normalized:
            next_position = -1
    return next_position


def desired_position(previous: int, row: pd.Series, variant: RobustVariant) -> int:
    if variant.detector == "rank_pct":
        return position_from_rank(previous, float(row["rank_pct"]), variant)
    if variant.detector == "quantile_band":
        return position_from_quantile(previous, row, variant)
    if variant.detector == "mad_z":
        return position_from_mad(previous, float(row["mad_z"]), variant)
    if variant.detector == "curve_md":
        return position_from_curve_md(previous, row, variant)
    raise ValueError(f"Unknown detector: {variant.detector}")


def signal_value(row: pd.Series, variant: RobustVariant) -> float:
    if variant.detector == "rank_pct":
        return float(row["rank_pct"])
    if variant.detector == "quantile_band":
        return float(row["carry_pct_ann"])
    if variant.detector == "mad_z":
        return float(row["mad_z"])
    if variant.detector == "curve_md":
        return float(row["curve_md"])
    return np.nan


def simulate_root(
    frame: pd.DataFrame, variant: RobustVariant, *, root: str, leg_cost_bps: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = frame.sort_values("date").copy()
    rows = []
    events = []
    previous_position = 0
    previous_anchor = ""
    previous_far = ""
    current_event: dict[str, object] | None = None

    for record in data.to_dict("records"):
        row = pd.Series(record)
        new_position = desired_position(previous_position, row, variant)
        if not np.isfinite(row["spread_return"]):
            new_position = 0

        if previous_position not in (0, new_position) and current_event:
            current_event["exit_date"] = row["date"]
            current_event["exit_signal"] = signal_value(row, variant)
            events.append(current_event)
            current_event = None

        if previous_position == 0 and new_position != 0:
            current_event = {
                "variant": variant.name,
                "detector": variant.detector,
                "root": root,
                "target_months": variant.target_months,
                "min_volume": variant.min_volume,
                "cost_multiplier": variant.cost_multiplier,
                "side": "fade_backwardation" if new_position > 0 else "fade_contango",
                "entry_date": row["date"],
                "entry_anchor": row["anchor"],
                "entry_far": row["far"],
                "entry_carry_pct_ann": row["carry_pct_ann"],
                "entry_signal": signal_value(row, variant),
                "gross_spread_return": 0.0,
                "weighted_gross_return": 0.0,
                "weighted_cost_return": 0.0,
                "weighted_net_return": 0.0,
                "holding_days": 0,
                "rolls": 0,
            }

        pair_changed = (
            previous_position != 0
            and new_position != 0
            and (row["anchor"] != previous_anchor or row["far"] != previous_far)
        )
        spread_turnover = abs(new_position - previous_position)
        leg_cost = (
            2.0
            * leg_cost_bps
            * variant.cost_multiplier
            / 10_000.0
            * spread_turnover
            * base.ROOT_WEIGHT
        )
        if pair_changed and new_position == previous_position:
            leg_cost += (
                4.0
                * leg_cost_bps
                * variant.cost_multiplier
                / 10_000.0
                * abs(new_position)
                * base.ROOT_WEIGHT
            )

        gross = new_position * float(row["spread_return"]) * base.ROOT_WEIGHT
        net = gross - leg_cost
        if current_event and new_position != 0:
            current_event["gross_spread_return"] = float(
                current_event["gross_spread_return"]
            ) + new_position * float(row["spread_return"])
            current_event["weighted_gross_return"] = float(
                current_event["weighted_gross_return"]
            ) + gross
            current_event["weighted_cost_return"] = float(
                current_event["weighted_cost_return"]
            ) + leg_cost
            current_event["weighted_net_return"] = float(current_event["weighted_net_return"]) + net
            current_event["holding_days"] = int(current_event["holding_days"]) + 1
            current_event["rolls"] = int(current_event["rolls"]) + int(pair_changed)

        rows.append(
            {
                "date": row["date"],
                "next_date": row["next_date"],
                "variant": variant.name,
                "detector": variant.detector,
                "root": root,
                "target_months": variant.target_months,
                "min_volume": variant.min_volume,
                "cost_multiplier": variant.cost_multiplier,
                "anchor": row["anchor"],
                "far": row["far"],
                "carry_pct_ann": row["carry_pct_ann"],
                "signal_value": signal_value(row, variant),
                "spread_return": row["spread_return"],
                "position": new_position,
                "pair_changed": pair_changed,
                "gross_return": gross,
                "cost_return": leg_cost,
                "net_return": net,
                "spread_exposure": abs(new_position) * base.ROOT_WEIGHT,
                "leg_gross_exposure": abs(new_position) * 2.0 * base.ROOT_WEIGHT,
                "turnover": spread_turnover * base.ROOT_WEIGHT,
            }
        )
        previous_position = new_position
        previous_anchor = row["anchor"]
        previous_far = row["far"]

    if current_event:
        current_event["exit_date"] = data["date"].iloc[-1]
        current_event["exit_signal"] = signal_value(data.iloc[-1], variant)
        current_event["forced_exit"] = True
        events.append(current_event)
    for event in events:
        event.setdefault("forced_exit", False)
        event["duration_days"] = (
            pd.Timestamp(event["exit_date"]) - pd.Timestamp(event["entry_date"])
        ).days
    return pd.DataFrame(rows), pd.DataFrame(events)


def aggregate_variant_returns(root_returns: list[pd.DataFrame]) -> pd.DataFrame:
    return base.aggregate_variant_returns(root_returns)


def summarize_strategy(
    returns: pd.DataFrame, events: pd.DataFrame, variant: RobustVariant
) -> dict[str, object]:
    pp_year = periods_per_year(returns.index)
    net = returns["net_return"].fillna(0.0)
    gross = returns["gross_return"].fillna(0.0)
    costs = returns["cost_return"].fillna(0.0)
    net_std = finite_std(net)
    event_net = events["weighted_net_return"] if not events.empty else pd.Series(dtype=float)
    event_std = finite_std(event_net)
    return {
        "variant": variant.name,
        "detector": variant.detector,
        "target_months": variant.target_months,
        "min_volume": variant.min_volume,
        "lookback": variant.lookback,
        "side_mode": variant.side_mode,
        "cost_multiplier": variant.cost_multiplier,
        "entry_level": variant.entry_level,
        "exit_level": variant.exit_level,
        "md_entry": variant.md_entry,
        "component_entry": variant.component_entry,
        "gross_return": gross.sum(),
        "cost_return": costs.sum(),
        "net_return": net.sum(),
        "ann_return": net.mean() * pp_year,
        "ann_vol": net_std * math.sqrt(pp_year) if np.isfinite(net_std) else np.nan,
        "sharpe": net.mean() / net_std * math.sqrt(pp_year) if net_std > 0 else np.nan,
        "tstat": net.mean() / net_std * math.sqrt(len(net)) if net_std > 0 else np.nan,
        "max_drawdown": base.max_drawdown(net),
        "mean_spread_exposure": returns["spread_exposure"].mean(),
        "mean_leg_gross_exposure": returns["leg_gross_exposure"].mean(),
        "mean_turnover": returns["turnover"].mean(),
        "active_fraction": (returns["spread_exposure"] > 0).mean(),
        "event_count": len(events),
        "event_win_rate": (event_net > 0).mean() if len(event_net) else np.nan,
        "mean_event_net_return": event_net.mean() if len(event_net) else np.nan,
        "event_tstat": event_net.mean() / event_std * math.sqrt(len(event_net))
        if np.isfinite(event_std) and event_std > 0
        else np.nan,
        "bars": len(returns),
        "periods_per_year": pp_year,
    }


def build_variants() -> list[RobustVariant]:
    variants = []
    for target in base.TARGET_MONTHS:
        for min_volume in base.MIN_VOLUME_VARIANTS:
            for lookback in base.LOOKBACKS:
                for side_mode in base.SIDE_MODES:
                    for cost_multiplier in base.COST_MULTIPLIERS:
                        for entry_level in RANK_ENTRY_LEVELS:
                            variants.append(
                                RobustVariant(
                                    "rank_pct",
                                    target,
                                    min_volume,
                                    lookback,
                                    side_mode,
                                    cost_multiplier,
                                    entry_level,
                                )
                            )
                            variants.append(
                                RobustVariant(
                                    "quantile_band",
                                    target,
                                    min_volume,
                                    lookback,
                                    side_mode,
                                    cost_multiplier,
                                    entry_level,
                                )
                            )
                        variants.extend(
                            (
                                RobustVariant(
                                    "mad_z",
                                    target,
                                    min_volume,
                                    lookback,
                                    side_mode,
                                    cost_multiplier,
                                    entry_level,
                                )
                            )
                            for entry_level in MAD_ENTRY_LEVELS
                        )
    for target in base.TARGET_MONTHS:
        for min_volume in base.MIN_VOLUME_VARIANTS:
            for lookback in base.LOOKBACKS:
                for cost_multiplier in base.COST_MULTIPLIERS:
                    variants.extend(
                        (
                            RobustVariant(
                                detector="curve_md",
                                target_months=target,
                                min_volume=min_volume,
                                lookback=lookback,
                                side_mode="both",
                                cost_multiplier=cost_multiplier,
                                entry_level=CURVE_COMPONENT_ENTRY,
                                md_entry=md_entry,
                            )
                        )
                        for md_entry in CURVE_MD_ENTRIES
                    )
    return variants


def run_backtests(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    costs = base.load_costs()
    univariate = build_univariate_feature_map(panel)
    curve_state = build_curve_state_features(panel)
    panel_groups = {
        key: group.copy()
        for key, group in panel.groupby(["root", "min_volume", "target_months"], sort=False)
    }
    metric_rows = []
    all_returns = []
    all_events = []
    variants = build_variants()
    print(f"Running {len(variants)} robust detector variants", flush=True)

    for variant in variants:
        root_returns = []
        root_events = []
        for root in base.ROOTS:
            key = (root, variant.min_volume, variant.target_months)
            root_panel = panel_groups.get(key)
            if root_panel is None or root_panel.empty:
                continue
            if variant.detector == "curve_md":
                curve_key = (root, variant.min_volume, variant.lookback)
                features = curve_state.get(curve_key)
                if features is None or features.empty:
                    continue
                frame = root_panel.merge(
                    features,
                    on=["date", "target_months"],
                    how="left",
                )
            else:
                frame = univariate[(*key, variant.lookback)]
            returns, events = simulate_root(
                frame, variant, root=root, leg_cost_bps=float(costs[root])
            )
            root_returns.append(returns)
            if not events.empty:
                root_events.append(events)
        if not root_returns:
            continue
        variant_returns = aggregate_variant_returns(root_returns)
        events = pd.concat(root_events, ignore_index=True) if root_events else pd.DataFrame()
        metric_rows.append(summarize_strategy(variant_returns, events, variant))
        variant_returns = variant_returns.reset_index()
        variant_returns["variant"] = variant.name
        variant_returns["detector"] = variant.detector
        all_returns.append(variant_returns)
        if not events.empty:
            all_events.append(events)

    metrics = pd.DataFrame(metric_rows).sort_values("net_return", ascending=False)
    returns = pd.concat(all_returns, ignore_index=True)
    events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    return metrics, returns, events


def robustness_tables(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = [
        "detector",
        "target_months",
        "min_volume",
        "cost_multiplier",
        "variant",
        "net_return",
        "cost_return",
        "sharpe",
        "tstat",
        "max_drawdown",
        "event_count",
        "event_tstat",
        "active_fraction",
    ]
    detector_idx = metrics.groupby("detector")["net_return"].idxmax()
    detector_best = metrics.loc[detector_idx, columns].sort_values("net_return", ascending=False)
    volume_cost_idx = metrics.groupby(["detector", "min_volume", "cost_multiplier"])[
        "net_return"
    ].idxmax()
    volume_cost = metrics.loc[volume_cost_idx, columns].sort_values(
        ["detector", "min_volume", "cost_multiplier"]
    )
    one_x = metrics[metrics["cost_multiplier"].eq(1.0)]
    target_volume_idx = one_x.groupby(["detector", "target_months", "min_volume"])[
        "net_return"
    ].idxmax()
    target_volume = one_x.loc[target_volume_idx, columns].sort_values(
        ["detector", "target_months", "min_volume"]
    )
    return detector_best, volume_cost, target_volume


def split_metrics(best_returns: pd.DataFrame, best_events: pd.DataFrame) -> pd.DataFrame:
    return base.split_metrics(best_returns, best_events)


def root_event_summary(events: pd.DataFrame, variant_name: str) -> pd.DataFrame:
    return base.root_event_summary(events, variant_name)


def plot_detector_comparison(detector_best: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    data = detector_best.sort_values("net_return")
    ax.barh(data["detector"], data["net_return"], color="#2f7d8c", alpha=0.85)
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_xlabel("Best net cumulative log return")
    ax.set_title("Best robust extreme detector by family")
    ax.grid(True, axis="x", alpha=0.25)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_report(
    *,
    metrics: pd.DataFrame,
    split: pd.DataFrame,
    root_summary: pd.DataFrame,
    detector_best: pd.DataFrame,
    volume_cost: pd.DataFrame,
    target_volume: pd.DataFrame,
    best_variant: str,
) -> None:
    top_cols = [
        "variant",
        "detector",
        "min_volume",
        "cost_multiplier",
        "net_return",
        "cost_return",
        "sharpe",
        "tstat",
        "max_drawdown",
        "event_count",
        "event_tstat",
        "active_fraction",
    ]
    lines = [
        "# HYP-0032 Metals Convenience-Yield Robust Extreme Detectors",
        "",
        "## Design",
        "",
        "- Uses the HYP-0031 synchronized 5-minute curve panel.",
        "- Compares rolling empirical percentile, rolling quantile bands, median/MAD z-score, "
        "and robust diagonal curve-state MD.",
        "- Position logic, costs, tenors, volume filters, event exits, and root weights match "
        "the prior convenience-yield tests.",
        "",
        "## Best Variant",
        "",
        metrics.head(1)[top_cols].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Best By Detector",
        "",
        detector_best.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Volume And Cost Robustness",
        "",
        volume_cost.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 1x Cost Tenor And Volume Robustness",
        "",
        target_volume.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Split Metrics",
        "",
        split.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Event Summary For Best Variant",
        "",
        root_summary.to_markdown(index=False, floatfmt=".4f")
        if not root_summary.empty
        else "No events.",
        "",
        "## Interpretation",
        "",
        (
            f"The best robust-detector variant is `{best_variant}` with net cumulative log "
            f"return `{metrics.iloc[0]['net_return']:.4f}`, t-stat "
            f"`{metrics.iloc[0]['tstat']:.2f}`, and event t-stat "
            f"`{metrics.iloc[0]['event_tstat']:.2f}`."
        ),
        "",
        "This experiment asks whether the curve-basis edge survives non-Gaussian extreme "
        "definitions. A detector only advances if it remains positive under synchronized marks, "
        "liquid volume filters, and conservative cost multipliers.",
        "",
        "## Files",
        "",
        "- `strategy_metrics.csv`",
        "- `detector_best.csv`",
        "- `volume_cost_robustness.csv`",
        "- `target_volume_robustness_1x.csv`",
        "- `best_strategy_returns.csv`",
        "- `event_log.csv`",
        "- `split_metrics.csv`",
        "- `root_event_summary.csv`",
        "- `best_strategy_equity.png`",
        "- `detector_best.png`",
        "- `top_variant_metrics.png`",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(INPUT_PANEL)
    metrics, returns, events = run_backtests(panel)
    metrics.to_csv(OUTPUT_DIR / "strategy_metrics.csv", index=False)
    returns.to_csv(OUTPUT_DIR / "all_strategy_returns.csv", index=False)
    events.to_csv(OUTPUT_DIR / "event_log.csv", index=False)

    detector_best, volume_cost, target_volume = robustness_tables(metrics)
    detector_best.to_csv(OUTPUT_DIR / "detector_best.csv", index=False)
    volume_cost.to_csv(OUTPUT_DIR / "volume_cost_robustness.csv", index=False)
    target_volume.to_csv(OUTPUT_DIR / "target_volume_robustness_1x.csv", index=False)

    best_variant = str(metrics.iloc[0]["variant"])
    best_returns = returns[returns["variant"] == best_variant].copy()
    best_events = events[events["variant"] == best_variant].copy()
    split = split_metrics(best_returns, best_events)
    root_summary = root_event_summary(events, best_variant)

    best_returns.to_csv(OUTPUT_DIR / "best_strategy_returns.csv", index=False)
    split.to_csv(OUTPUT_DIR / "split_metrics.csv", index=False)
    root_summary.to_csv(OUTPUT_DIR / "root_event_summary.csv", index=False)

    base.plot_best_equity(best_returns, OUTPUT_DIR / "best_strategy_equity.png")
    base.plot_top_variants(metrics, OUTPUT_DIR / "top_variant_metrics.png")
    base.plot_root_events(root_summary, OUTPUT_DIR / "root_event_summary.png")
    plot_detector_comparison(detector_best, OUTPUT_DIR / "detector_best.png")
    write_report(
        metrics=metrics,
        split=split,
        root_summary=root_summary,
        detector_best=detector_best,
        volume_cost=volume_cost,
        target_volume=target_volume,
        best_variant=best_variant,
    )
    print(metrics.head(15).round(4).to_string(index=False))
    print(f"Wrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
