"""Focused negative residual dislocation test for SI, PL, and HG."""

# ruff: noqa: E402

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_metals_macro_residual_dislocation_strategy import (
    METALS,
    PERIODS_PER_YEAR,
    build_factors,
    build_residual_state,
    json_safe,
    load_costs,
    load_returns,
    max_drawdown,
    rolling_macro_residuals,
    rolling_z,
    sharpe,
    tstat,
)
from scripts.run_metals_macro_residual_dislocation_strategy import (
    OUT_DIR as BASE_OUT_DIR,
)

OUT_DIR = REPO_ROOT / "experiments" / "HYP-0029-metals-focused-negative-dislocations"
CARRY_PATH = (
    REPO_ROOT
    / "notebooks"
    / "explorations"
    / "assets"
    / "2026-06-25_metals_carry_curve_review"
    / "daily_liquid_front_carry_history_2023-06-22_2026-06-21.csv"
)

TARGET_ROOTS = ["SI", "PL", "HG"]
ENTRY_Z_LEVELS = [2.0, 2.5, 3.0]
ROOT_Z_LEVELS = [1.5, 2.0]
EXIT_Z_LEVELS = [0.25, 0.50]
MD_ENTRY_LABELS = ["q90", "q95"]
MD_EXIT_LABELS = ["q50"]
POST_2020_ONLY = [False, True]
TOPOLOGY_EXITS = [False, True]

POST_2020_START = pd.Timestamp("2020-01-01", tz="UTC")
TRADE_OVERLAP_START = pd.Timestamp("2023-06-22", tz="UTC")

USD_HEDGE_COST_BPS = 0.25
RATES_HEDGE_COST_BPS = 0.15
CL_HEDGE_COST_BPS = 0.80
MIN_TSTAT_OBS = 2


@dataclass(frozen=True)
class ActiveEvent:
    root: str
    neighbors: tuple[str, ...]
    entry_date: pd.Timestamp
    entry_rel_z: float
    entry_root_z: float
    entry_md: float


@dataclass(frozen=True)
class Variant:
    entry_z: float
    root_z: float
    exit_z: float
    md_entry: str
    md_exit: str
    post_2020_only: bool
    topology_exit: bool

    @property
    def name(self) -> str:
        regime = "post2020" if self.post_2020_only else "all"
        topo = "topo_exit" if self.topology_exit else "topo_hold"
        return (
            f"neg_z{self.entry_z:g}_root{self.root_z:g}_exit{self.exit_z:g}_"
            f"md{self.md_entry}_norm{self.md_exit}_{regime}_{topo}"
        )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    returns = load_returns()
    factors = build_factors(returns)
    residuals, _residual_z, state, betas = load_or_build_base_state(returns, factors)
    costs_bps = load_costs()
    carry = load_carry()

    variants = [
        Variant(entry_z, root_z, exit_z, md_entry, md_exit, post_2020_only, topology_exit)
        for entry_z in ENTRY_Z_LEVELS
        for root_z in ROOT_Z_LEVELS
        for exit_z in EXIT_Z_LEVELS
        for md_entry in MD_ENTRY_LABELS
        for md_exit in MD_EXIT_LABELS
        for post_2020_only in POST_2020_ONLY
        for topology_exit in TOPOLOGY_EXITS
    ]

    strategy_frames: dict[str, pd.DataFrame] = {}
    event_frames = []
    metric_rows = []
    for variant in variants:
        position_signal, events = build_positions_and_events(state, residuals, variant)
        strategy = strategy_returns(
            position_signal,
            returns[METALS],
            residuals,
            betas,
            costs_bps,
            carry,
        )
        strategy["variant"] = variant.name
        events["variant"] = variant.name
        strategy_frames[variant.name] = strategy.reset_index(names="date")
        event_frames.append(events)
        metric_rows.append(metrics_for_strategy(variant.name, strategy, events))

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["net_return_after_all_costs", "sharpe_after_all_costs", "event_count"],
        ascending=[False, False, False],
    )
    events_all = pd.concat(event_frames, ignore_index=True)
    split_metrics = build_split_metrics(strategy_frames, events_all)
    event_stats = event_summary(events_all)

    metrics.to_csv(OUT_DIR / "strategy_metrics.csv", index=False)
    split_metrics.to_csv(OUT_DIR / "split_metrics.csv", index=False)
    events_all.to_csv(OUT_DIR / "event_log.csv", index=False)
    event_stats.to_csv(OUT_DIR / "event_summary.csv", index=False)

    best = metrics.iloc[0]
    best_strategy = strategy_frames[best["variant"]]
    best_strategy.to_csv(OUT_DIR / "best_strategy_returns.csv", index=False)
    plot_equity(
        best_strategy.set_index("date"),
        OUT_DIR / "best_strategy_equity.png",
        f"Focused negative dislocations: {best['variant']}",
    )
    plot_top_variants(metrics, OUT_DIR / "top_variant_metrics.png")
    write_report(metrics, split_metrics, event_stats, best["variant"], returns, carry)

    summary = {
        "experiment_id": "HYP-0029",
        "completed_at": datetime.now(UTC).isoformat(),
        "data_start": returns.index.min().isoformat(),
        "data_end": returns.index.max().isoformat(),
        "target_roots": TARGET_ROOTS,
        "entry_z_levels": ENTRY_Z_LEVELS,
        "root_z_levels": ROOT_Z_LEVELS,
        "macro_hedge_cost_bps": {
            "usd": USD_HEDGE_COST_BPS,
            "rates_price": RATES_HEDGE_COST_BPS,
            "cl": CL_HEDGE_COST_BPS,
            "complex_metals": "root-specific metal costs",
        },
        "carry_start": carry.index.min().isoformat() if not carry.empty else None,
        "carry_end": carry.index.max().isoformat() if not carry.empty else None,
        "best_variant": best.to_dict(),
    }
    (OUT_DIR / "results.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(metrics.head(20).to_string(index=False))
    print(f"Wrote {OUT_DIR}")


def load_or_build_base_state(
    returns: pd.DataFrame, factors: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    residual_path = BASE_OUT_DIR / "macro_residual_returns.parquet"
    z_path = BASE_OUT_DIR / "macro_residual_zscores.parquet"
    state_path = BASE_OUT_DIR / "residual_dislocation_state.parquet"
    beta_path = BASE_OUT_DIR / "rolling_macro_betas.parquet"
    if residual_path.exists() and z_path.exists() and state_path.exists() and beta_path.exists():
        residuals = pd.read_parquet(residual_path)
        residuals.index = pd.to_datetime(residuals.index, utc=True)
        residual_z = pd.read_parquet(z_path)
        residual_z.index = pd.to_datetime(residual_z.index, utc=True)
        state = pd.read_parquet(state_path)
        state["date"] = pd.to_datetime(state["date"], utc=True)
        betas = pd.read_parquet(beta_path)
        betas["date"] = pd.to_datetime(betas["date"], utc=True)
        return residuals, residual_z, state, betas

    residuals, betas = rolling_macro_residuals(returns, factors)
    residual_z = rolling_z(residuals)
    state = build_residual_state(residual_z)
    return residuals, residual_z, state, betas


def load_carry() -> pd.DataFrame:
    if not CARRY_PATH.exists():
        return pd.DataFrame()
    carry = pd.read_csv(CARRY_PATH, parse_dates=["date"])
    carry["date"] = pd.to_datetime(carry["date"], utc=True)
    value_cols = ["carry_3m_pct", "carry_2m_pct", "carry_1m_pct"]
    carry["carry_pct_ann"] = carry[value_cols].bfill(axis=1).iloc[:, 0]
    wide = carry.pivot_table(index="date", columns="root", values="carry_pct_ann", aggfunc="last")
    return wide.reindex(columns=METALS).sort_index()


def build_positions_and_events(  # noqa: PLR0912
    state: pd.DataFrame,
    residuals: pd.DataFrame,
    variant: Variant,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state = state.sort_values("date").reset_index(drop=True)
    dates = pd.DatetimeIndex(state["date"])
    positions = np.zeros((len(state), len(METALS)), dtype=float)
    rel_z = state[[f"{root}_rel_z" for root in METALS]].to_numpy(dtype=float)
    root_z = state[[f"{root}_z" for root in METALS]].to_numpy(dtype=float)
    md = state["md"].to_numpy(dtype=float)
    md_entry = state[f"md_{variant.md_entry}"].to_numpy(dtype=float)
    md_exit = state[f"md_{variant.md_exit}"].to_numpy(dtype=float)
    neighbors = {
        root_index: [
            parse_neighbors(value) for value in state[f"{root}_neighbors"].to_numpy(dtype=object)
        ]
        for root_index, root in enumerate(METALS)
    }
    target_indices = [METALS.index(root) for root in TARGET_ROOTS]

    active: dict[int, ActiveEvent] = {}
    events: list[dict[str, Any]] = []
    for row_index, date in enumerate(dates):
        if variant.post_2020_only and date < POST_2020_START:
            continue
        if not np.isfinite(md_entry[row_index]) or not np.isfinite(md_exit[row_index]):
            continue

        for root_index in list(active):
            event = active[root_index]
            current_rel_z = rel_z[row_index, root_index]
            current_neighbors = neighbors[root_index][row_index]
            exit_event = (
                current_rel_z >= -variant.exit_z
                or md[row_index] <= md_exit[row_index]
                or current_rel_z > 0
                or (variant.topology_exit and current_neighbors != event.neighbors)
            )
            if exit_event:
                events.append(
                    close_event(
                        event,
                        date,
                        current_rel_z,
                        root_z[row_index, root_index],
                        md[row_index],
                        residuals,
                    )
                )
                del active[root_index]

        for root_index in target_indices:
            if root_index in active:
                continue
            current_rel_z = rel_z[row_index, root_index]
            current_root_z = root_z[row_index, root_index]
            if (
                current_rel_z <= -variant.entry_z
                and current_root_z <= -variant.root_z
                and md[row_index] >= md_entry[row_index]
            ):
                root_neighbors = neighbors[root_index][row_index]
                if root_neighbors:
                    active[root_index] = ActiveEvent(
                        root=METALS[root_index],
                        neighbors=root_neighbors,
                        entry_date=date,
                        entry_rel_z=float(current_rel_z),
                        entry_root_z=float(current_root_z),
                        entry_md=float(md[row_index]),
                    )

        position = np.zeros(len(METALS), dtype=float)
        for event in active.values():
            root_index = METALS.index(event.root)
            position[root_index] += 1.0
            neighbor_weight = -1.0 / len(event.neighbors)
            for neighbor in event.neighbors:
                position[METALS.index(neighbor)] += neighbor_weight
        gross = float(np.abs(position).sum())
        if gross > 1.0:
            position /= gross
        positions[row_index] = position

    last_date = dates[-1] if len(dates) else pd.NaT
    for root_index, event in active.items():
        events.append(
            close_event(
                event,
                last_date,
                float(rel_z[-1, root_index]),
                float(root_z[-1, root_index]),
                float(md[-1]),
                residuals,
                forced=True,
            )
        )
    return pd.DataFrame(positions, index=dates, columns=METALS), pd.DataFrame(events)


def parse_neighbors(value: Any) -> tuple[str, ...]:
    if pd.isna(value) or value == "":
        return ()
    return tuple(str(value).split("|"))


def close_event(
    event: ActiveEvent,
    exit_date: pd.Timestamp,
    exit_rel_z: float,
    exit_root_z: float,
    exit_md: float,
    residuals: pd.DataFrame,
    forced: bool = False,
) -> dict[str, Any]:
    holding = residuals.loc[(residuals.index > event.entry_date) & (residuals.index <= exit_date)]
    if holding.empty:
        residual_return = 0.0
    else:
        neighbor_return = holding[list(event.neighbors)].mean(axis=1)
        residual_return = float((holding[event.root] - neighbor_return).sum())
    return {
        "root": event.root,
        "entry_date": event.entry_date,
        "exit_date": exit_date,
        "duration_days": int(max((exit_date - event.entry_date).days, 0)),
        "neighbors": "|".join(event.neighbors),
        "entry_rel_z": event.entry_rel_z,
        "entry_root_z": event.entry_root_z,
        "exit_rel_z": exit_rel_z,
        "exit_root_z": exit_root_z,
        "entry_md": event.entry_md,
        "exit_md": exit_md,
        "event_residual_return": residual_return,
        "forced_exit": forced,
    }


def strategy_returns(
    position_signal: pd.DataFrame,
    raw_returns: pd.DataFrame,
    residual_returns: pd.DataFrame,
    betas: pd.DataFrame,
    metal_costs_bps: pd.Series,
    carry: pd.DataFrame,
) -> pd.DataFrame:
    common = position_signal.index.intersection(raw_returns.index).intersection(
        residual_returns.index
    )
    position_signal = position_signal.reindex(common).fillna(0.0)
    positions = position_signal.shift(1).fillna(0.0)
    raw_returns = raw_returns.reindex(common).fillna(0.0)
    residual_returns = residual_returns.reindex(common).fillna(0.0)

    primary_turnover = positions.diff().abs().fillna(positions.abs())
    primary_cost = primary_turnover.mul(metal_costs_bps / 10_000.0, axis=1).sum(axis=1)

    beta_wide = beta_panel_wide(betas, common)
    hedge_cost, hedge_exposures, complex_positions = macro_hedge_costs(
        positions,
        beta_wide,
        metal_costs_bps,
    )

    gross_residual = (positions * residual_returns).sum(axis=1)
    gross_raw = (positions * raw_returns).sum(axis=1)
    carry_proxy = carry_pnl_proxy(positions.add(complex_positions, fill_value=0.0), carry)

    total_cost = primary_cost + hedge_cost
    frame = pd.DataFrame(
        {
            "gross_residual_return": gross_residual,
            "gross_raw_return": gross_raw,
            "primary_cost_return": primary_cost,
            "macro_hedge_cost_return": hedge_cost,
            "total_cost_return": total_cost,
            "carry_pnl_proxy": carry_proxy.reindex(common).fillna(0.0),
            "net_return_after_all_costs": gross_residual - total_cost,
            "net_return_carry_adjusted_overlap": gross_residual
            - total_cost
            - carry_proxy.reindex(common).fillna(0.0),
            "net_raw_return_after_primary_cost": gross_raw - primary_cost,
            "gross_exposure": positions.abs().sum(axis=1),
            "primary_turnover": primary_turnover.sum(axis=1),
            "hedge_turnover": hedge_exposures.diff()
            .abs()
            .fillna(hedge_exposures.abs())
            .sum(axis=1),
            "complex_hedge_gross": complex_positions.abs().sum(axis=1),
        },
        index=common,
    )
    for root in METALS:
        frame[f"pos_{root}"] = positions[root]
    return frame


def beta_panel_wide(betas: pd.DataFrame, index: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    betas = betas.copy()
    betas["date"] = pd.to_datetime(betas["date"], utc=True)
    result = {}
    for column in ["beta_usd", "beta_rates_price", "beta_cl", "beta_metals_loo"]:
        result[column] = (
            betas.pivot(index="date", columns="root", values=column)
            .reindex(index)
            .ffill()
            .reindex(columns=METALS)
            .fillna(0.0)
        )
    return result


def macro_hedge_costs(
    positions: pd.DataFrame,
    beta_wide: dict[str, pd.DataFrame],
    metal_costs_bps: pd.Series,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    exposures = pd.DataFrame(index=positions.index)
    exposures["usd"] = -(positions * beta_wide["beta_usd"]).sum(axis=1)
    exposures["rates_price"] = -(positions * beta_wide["beta_rates_price"]).sum(axis=1)
    exposures["cl"] = -(positions * beta_wide["beta_cl"]).sum(axis=1)

    complex_positions = pd.DataFrame(0.0, index=positions.index, columns=METALS)
    beta_complex = beta_wide["beta_metals_loo"]
    for root in METALS:
        other_roots = [candidate for candidate in METALS if candidate != root]
        hedge = -positions[root] * beta_complex[root] / len(other_roots)
        for other in other_roots:
            complex_positions[other] += hedge

    factor_turnover = exposures.diff().abs().fillna(exposures.abs())
    factor_cost = (
        factor_turnover["usd"] * USD_HEDGE_COST_BPS
        + factor_turnover["rates_price"] * RATES_HEDGE_COST_BPS
        + factor_turnover["cl"] * CL_HEDGE_COST_BPS
    ) / 10_000.0
    complex_turnover = complex_positions.diff().abs().fillna(complex_positions.abs())
    complex_cost = complex_turnover.mul(metal_costs_bps / 10_000.0, axis=1).sum(axis=1)
    return factor_cost + complex_cost, exposures, complex_positions


def carry_pnl_proxy(positions: pd.DataFrame, carry: pd.DataFrame) -> pd.Series:
    if carry.empty:
        return pd.Series(0.0, index=positions.index)
    aligned = carry.reindex(positions.index).ffill()
    daily_carry = -aligned / 100.0 / PERIODS_PER_YEAR
    return (positions * daily_carry).sum(axis=1)


def metrics_for_strategy(
    variant: str,
    strategy: pd.DataFrame,
    events: pd.DataFrame,
) -> dict[str, Any]:
    net = strategy["net_return_after_all_costs"]
    overlap = strategy[strategy.index >= TRADE_OVERLAP_START]
    event_returns = events["event_residual_return"] if not events.empty else pd.Series(dtype=float)
    overlap_carry_net = (
        overlap["net_return_carry_adjusted_overlap"].sum() if not overlap.empty else np.nan
    )
    return {
        "variant": variant,
        "gross_residual_return": strategy["gross_residual_return"].sum(),
        "primary_cost_return": strategy["primary_cost_return"].sum(),
        "macro_hedge_cost_return": strategy["macro_hedge_cost_return"].sum(),
        "total_cost_return": strategy["total_cost_return"].sum(),
        "net_return_after_all_costs": net.sum(),
        "net_raw_return_after_primary_cost": strategy["net_raw_return_after_primary_cost"].sum(),
        "trade_overlap_net": overlap["net_return_after_all_costs"].sum()
        if not overlap.empty
        else np.nan,
        "trade_overlap_carry_adjusted_net": overlap_carry_net,
        "ann_return": net.mean() * PERIODS_PER_YEAR,
        "ann_vol": net.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR),
        "sharpe_after_all_costs": sharpe(net),
        "tstat_after_all_costs": tstat(net),
        "max_drawdown": max_drawdown(net),
        "mean_gross_exposure": strategy["gross_exposure"].mean(),
        "mean_primary_turnover": strategy["primary_turnover"].mean(),
        "mean_hedge_turnover": strategy["hedge_turnover"].mean(),
        "mean_complex_hedge_gross": strategy["complex_hedge_gross"].mean(),
        "gross_to_total_cost": strategy["gross_residual_return"].sum()
        / strategy["total_cost_return"].sum()
        if strategy["total_cost_return"].sum() > 0
        else np.nan,
        "event_count": len(events),
        "event_win_rate": float((event_returns > 0).mean()) if len(event_returns) else np.nan,
        "mean_event_return": event_returns.mean() if len(event_returns) else np.nan,
        "event_tstat": tstat(event_returns) if len(event_returns) >= MIN_TSTAT_OBS else np.nan,
        "mean_duration_days": events["duration_days"].mean() if not events.empty else np.nan,
        "bars": len(strategy),
    }


def build_split_metrics(
    strategy_frames: dict[str, pd.DataFrame], events: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for variant, strategy_frame in strategy_frames.items():
        frame = strategy_frame.copy()
        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        split_defs = {
            "full": frame,
            "pre_2020": frame[frame["date"] < POST_2020_START],
            "post_2020": frame[frame["date"] >= POST_2020_START],
            "trade_overlap": frame[frame["date"] >= TRADE_OVERLAP_START],
        }
        for split, split_frame in split_defs.items():
            if split_frame.empty:
                continue
            split_events = events[
                (events["variant"] == variant)
                & (pd.to_datetime(events["entry_date"], utc=True).isin(split_frame["date"]))
            ]
            net = split_frame["net_return_after_all_costs"]
            rows.append(
                {
                    "variant": variant,
                    "split": split,
                    "start": split_frame["date"].min(),
                    "end": split_frame["date"].max(),
                    "bars": len(split_frame),
                    "net_return_after_all_costs": net.sum(),
                    "carry_adjusted_net": split_frame["net_return_carry_adjusted_overlap"].sum(),
                    "total_cost_return": split_frame["total_cost_return"].sum(),
                    "sharpe_after_all_costs": sharpe(net),
                    "tstat_after_all_costs": tstat(net),
                    "max_drawdown": max_drawdown(net),
                    "events": len(split_events),
                }
            )
    return pd.DataFrame(rows)


def event_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in events.groupby(["variant", "root"], dropna=False):
        event_returns = group["event_residual_return"]
        rows.append(
            {
                "variant": keys[0],
                "root": keys[1],
                "events": len(group),
                "mean_event_return": event_returns.mean(),
                "median_event_return": event_returns.median(),
                "event_tstat": tstat(event_returns),
                "win_rate": float((event_returns > 0).mean()),
                "mean_duration_days": group["duration_days"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_event_return", "event_tstat"], ascending=False)


def plot_equity(strategy: pd.DataFrame, out_path: Path, title: str) -> None:
    equity = strategy["net_return_after_all_costs"].fillna(0.0).cumsum()
    raw = strategy["net_raw_return_after_primary_cost"].fillna(0.0).cumsum()
    carry_adj = strategy["net_return_carry_adjusted_overlap"].fillna(0.0).cumsum()
    drawdown = equity - equity.cummax()
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(equity.index, equity, label="residual net after all costs", color="#2f6f9f")
    axes[0].plot(raw.index, raw, label="raw metal spread net", color="#8d5a2b")
    axes[0].plot(
        carry_adj.index,
        carry_adj,
        label="residual net less carry proxy",
        color="#5f8f5f",
        alpha=0.8,
    )
    axes[0].set_title(title)
    axes[0].set_ylabel("Cumulative log return")
    axes[0].legend()
    axes[1].fill_between(drawdown.index, drawdown, 0.0, color="#9f3d3d", alpha=0.35)
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_top_variants(metrics: pd.DataFrame, out_path: Path) -> None:
    top = metrics.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(top["variant"], top["net_return_after_all_costs"], color="#2f6f9f")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Focused negative dislocation variants")
    ax.set_xlabel("Net residual return after metal + macro hedge costs")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_report(
    metrics: pd.DataFrame,
    split_metrics: pd.DataFrame,
    event_stats: pd.DataFrame,
    best_variant: str,
    returns: pd.DataFrame,
    carry: pd.DataFrame,
) -> None:
    best = metrics[metrics["variant"] == best_variant]
    best_splits = split_metrics[split_metrics["variant"] == best_variant]
    report = [
        "# HYP-0029 Focused Negative Metals Residual Dislocations",
        "",
        f"Completed at `{datetime.now(UTC).isoformat()}`.",
        "",
        "## Design",
        "",
        "- Targets only `SI`, `PL`, and `HG` negative residual dislocations.",
        "- Entry requires root-vs-MST-neighbor residual z below the threshold, "
        "the root residual z itself below the root threshold, and residual-cloud "
        "MD above its rolling threshold.",
        "- Position is long the dislocated root and short its rolling MST neighbors.",
        "- Exit is event-based: spread normalization, MD normalization, sign reversal, "
        "or optional topology change. No fixed holding time.",
        "- Residual PnL includes a macro hedge cost haircut for USD, rates, CL, and "
        "the leave-one-out metals complex hedge.",
        "- Carry proxy is available only from 2023-06-22 onward and is reported as "
        "an overlap diagnostic, not a full-history adjustment.",
        "",
        "## Coverage",
        "",
        f"- Return/factor span: `{returns.index.min().date()}` to `{returns.index.max().date()}`.",
        f"- Carry proxy span: `{carry.index.min().date()}` to `{carry.index.max().date()}`."
        if not carry.empty
        else "- Carry proxy unavailable.",
        "",
        "## Best Variant",
        "",
        best[
            [
                "variant",
                "net_return_after_all_costs",
                "total_cost_return",
                "macro_hedge_cost_return",
                "trade_overlap_net",
                "trade_overlap_carry_adjusted_net",
                "sharpe_after_all_costs",
                "tstat_after_all_costs",
                "max_drawdown",
                "event_count",
                "event_tstat",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Best Variant Splits",
        "",
        best_splits[
            [
                "split",
                "net_return_after_all_costs",
                "carry_adjusted_net",
                "total_cost_return",
                "sharpe_after_all_costs",
                "tstat_after_all_costs",
                "events",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Top Variants",
        "",
        metrics.head(12)[
            [
                "variant",
                "net_return_after_all_costs",
                "total_cost_return",
                "trade_overlap_net",
                "trade_overlap_carry_adjusted_net",
                "sharpe_after_all_costs",
                "tstat_after_all_costs",
                "event_count",
                "event_tstat",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Event Stats",
        "",
        event_stats[event_stats["variant"].eq(best_variant)]
        .sort_values("mean_event_return", ascending=False)
        .to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Files",
        "",
        "- `strategy_metrics.csv`",
        "- `split_metrics.csv`",
        "- `event_log.csv`",
        "- `event_summary.csv`",
        "- `best_strategy_returns.csv`",
        "- `best_strategy_equity.png`",
        "- `top_variant_metrics.png`",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
