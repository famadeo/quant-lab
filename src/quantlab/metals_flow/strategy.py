# ruff: noqa: PLR2004
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr


@dataclass(frozen=True)
class StrategyMetrics:
    observations: int
    active_bars: int
    gross_return: float
    cost_return: float
    net_return: float
    mean_net_bps: float
    tstat: float
    annualized_sharpe: float
    hit_rate: float
    max_drawdown: float
    gross_to_cost: float
    turnover: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "observations": self.observations,
            "active_bars": self.active_bars,
            "gross_return": self.gross_return,
            "cost_return": self.cost_return,
            "net_return": self.net_return,
            "mean_net_bps": self.mean_net_bps,
            "tstat": self.tstat,
            "annualized_sharpe": self.annualized_sharpe,
            "hit_rate": self.hit_rate,
            "max_drawdown": self.max_drawdown,
            "gross_to_cost": self.gross_to_cost,
            "turnover": self.turnover,
        }


def convergence_state_positions(
    residual_zscores: pd.DataFrame,
    entry_mask: pd.Series,
    root_entry_masks: pd.DataFrame | None = None,
    *,
    entry_z: float = 2.0,
    exit_z: float = 0.25,
    stop_z: float | None = 5.0,
) -> pd.DataFrame:
    zscores = residual_zscores.replace([np.inf, -np.inf], np.nan)
    entry_mask = entry_mask.reindex(zscores.index).fillna(False).astype(bool)
    if root_entry_masks is None:
        root_entry_masks = pd.DataFrame(True, index=zscores.index, columns=zscores.columns)
    else:
        root_entry_masks = root_entry_masks.reindex(index=zscores.index, columns=zscores.columns)
        root_entry_masks = root_entry_masks.fillna(False).astype(bool)

    positions = pd.DataFrame(0.0, index=zscores.index, columns=zscores.columns)
    state = {root: 0.0 for root in zscores.columns}

    for i, (_, row) in enumerate(zscores.iterrows()):
        for root, z_value in row.items():
            if not np.isfinite(z_value):
                state[root] = 0.0
                continue
            current = state[root]
            if (current > 0 and z_value >= -exit_z) or (current < 0 and z_value <= exit_z):
                current = 0.0
            if stop_z is not None and current != 0.0 and abs(z_value) > stop_z:
                current = 0.0
            if (
                current == 0.0
                and bool(entry_mask.iloc[i])
                and bool(root_entry_masks[root].iloc[i])
                and abs(z_value) >= entry_z
            ):
                current = -float(np.sign(z_value))
            state[root] = current
            positions.iat[i, positions.columns.get_loc(root)] = current

    return positions


def residual_momentum_positions(
    residual_zscores: pd.DataFrame,
    entry_mask: pd.Series,
    root_entry_masks: pd.DataFrame | None = None,
    *,
    entry_z: float = 2.0,
    exit_z: float = 0.75,
    max_holding_bars: int = 20,
    stop_z: float | None = 5.0,
    cooldown_bars: int = 0,
) -> pd.DataFrame:
    zscores = residual_zscores.replace([np.inf, -np.inf], np.nan)
    entry_mask = entry_mask.reindex(zscores.index).fillna(False).astype(bool)
    if root_entry_masks is None:
        root_entry_masks = pd.DataFrame(True, index=zscores.index, columns=zscores.columns)
    else:
        root_entry_masks = root_entry_masks.reindex(index=zscores.index, columns=zscores.columns)
        root_entry_masks = root_entry_masks.fillna(False).astype(bool)

    positions = pd.DataFrame(0.0, index=zscores.index, columns=zscores.columns)
    state = {root: 0.0 for root in zscores.columns}
    age = {root: 0 for root in zscores.columns}
    cooldown = {root: 0 for root in zscores.columns}

    for i, (_, row) in enumerate(zscores.iterrows()):
        global_entry = bool(entry_mask.iloc[i])
        for root, z_value in row.items():
            current = state[root]
            if cooldown[root] > 0:
                cooldown[root] -= 1

            if not np.isfinite(z_value):
                current = 0.0
                age[root] = 0
                cooldown[root] = max(cooldown[root], cooldown_bars)
            elif current != 0.0:
                age[root] += 1
                decayed = abs(z_value) <= exit_z
                reversed_sign = float(current) * float(z_value) <= 0.0
                timed_out = max_holding_bars > 0 and age[root] >= max_holding_bars
                stopped = stop_z is not None and abs(z_value) >= stop_z
                if decayed or reversed_sign or timed_out or stopped:
                    current = 0.0
                    age[root] = 0
                    cooldown[root] = max(cooldown[root], cooldown_bars)

            if (
                current == 0.0
                and cooldown[root] == 0
                and np.isfinite(z_value)
                and global_entry
                and bool(root_entry_masks[root].iloc[i])
                and abs(z_value) >= entry_z
            ):
                current = float(np.sign(z_value))
                age[root] = 0

            state[root] = current
            positions.iat[i, positions.columns.get_loc(root)] = current

    return positions


def demean_and_normalize_positions(raw_positions: pd.DataFrame) -> pd.DataFrame:
    positions = raw_positions.astype(float).copy()
    active = positions.abs().sum(axis=1) > 0.0
    positions.loc[active] = positions.loc[active].sub(positions.loc[active].mean(axis=1), axis=0)
    gross = positions.abs().sum(axis=1).replace(0.0, np.nan)
    return positions.div(gross, axis=0).fillna(0.0)


def backtest_positions(
    positions: pd.DataFrame,
    returns: pd.DataFrame,
    cost_bps: pd.Series | dict[str, float] | float,
    *,
    periods_per_year: float,
) -> tuple[pd.DataFrame, StrategyMetrics]:
    positions = positions.reindex(index=returns.index, columns=returns.columns).fillna(0.0)
    applied_positions = positions.shift(1).fillna(0.0)
    gross_by_root = applied_positions * returns.fillna(0.0)
    gross = gross_by_root.sum(axis=1)
    turnover_by_root = positions.diff().abs().fillna(positions.abs())

    if isinstance(cost_bps, int | float):
        cost_by_root = turnover_by_root * float(cost_bps) / 10_000.0
    else:
        cost_series = pd.Series(cost_bps, index=returns.columns, dtype=float).fillna(1.0)
        cost_by_root = turnover_by_root.mul(cost_series, axis=1) / 10_000.0
    cost = cost_by_root.sum(axis=1)
    net = gross - cost
    frame = pd.DataFrame(
        {
            "gross_return": gross,
            "cost_return": cost,
            "net_return": net,
            "turnover": turnover_by_root.sum(axis=1),
            "active": applied_positions.abs().sum(axis=1) > 0.0,
        },
        index=returns.index,
    )
    for root in returns.columns:
        frame[f"{root}_position"] = positions[root]
        frame[f"{root}_gross_return"] = gross_by_root[root]
        frame[f"{root}_cost_return"] = cost_by_root[root]
    return frame, calculate_strategy_metrics(frame, periods_per_year)


def calculate_strategy_metrics(frame: pd.DataFrame, periods_per_year: float) -> StrategyMetrics:
    gross = frame["gross_return"].fillna(0.0)
    cost = frame["cost_return"].fillna(0.0)
    net = frame["net_return"].fillna(0.0)
    observations = len(net)
    active_bars = int(frame["active"].sum()) if "active" in frame else int((net != 0.0).sum())
    mean = float(net.mean()) if observations else 0.0
    std = float(net.std(ddof=1)) if observations > 1 else 0.0
    equity = net.cumsum()
    drawdown = equity - equity.cummax()
    cost_sum = float(cost.sum())
    return StrategyMetrics(
        observations=observations,
        active_bars=active_bars,
        gross_return=float(gross.sum()),
        cost_return=cost_sum,
        net_return=float(net.sum()),
        mean_net_bps=mean * 10_000.0,
        tstat=mean / (std / math.sqrt(observations)) if std > 0 else np.nan,
        annualized_sharpe=mean / std * math.sqrt(periods_per_year) if std > 0 else np.nan,
        hit_rate=float((net > 0.0).mean()) if observations else np.nan,
        max_drawdown=float(drawdown.min()) if observations else np.nan,
        gross_to_cost=float(gross.sum() / cost_sum) if cost_sum > 0 else np.inf,
        turnover=float(frame["turnover"].sum()) if "turnover" in frame else np.nan,
    )


def estimate_mbp1_costs(
    mbp1_dir: Path,
    roots: tuple[str, ...],
    *,
    fallback_bps: float = 1.5,
) -> pd.DataFrame:
    rows = []
    for root in roots:
        files = sorted(str(path) for path in (mbp1_dir / root).glob("*.parquet"))
        if not files:
            rows.append(
                {
                    "root": root,
                    "rows": 0,
                    "median_spread_bps": np.nan,
                    "p75_spread_bps": np.nan,
                    "median_top_depth": np.nan,
                    "per_side_cost_bps": fallback_bps,
                    "source": "fallback",
                }
            )
            continue
        lazy = (
            pl.scan_parquet(files)
            .select("bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00")
            .filter(
                (pl.col("bid_px_00") > 0)
                & (pl.col("ask_px_00") > 0)
                & (pl.col("ask_px_00") >= pl.col("bid_px_00"))
            )
            .with_columns(
                mid=((pl.col("bid_px_00") + pl.col("ask_px_00")) / 2.0),
                spread_bps=(
                    (pl.col("ask_px_00") - pl.col("bid_px_00"))
                    / ((pl.col("bid_px_00") + pl.col("ask_px_00")) / 2.0)
                    * 10_000.0
                ),
                top_depth=(pl.col("bid_sz_00") + pl.col("ask_sz_00")),
            )
        )
        stats = lazy.select(
            pl.len().alias("rows"),
            pl.col("spread_bps").median().alias("median_spread_bps"),
            pl.col("spread_bps").quantile(0.75).alias("p75_spread_bps"),
            pl.col("top_depth").median().alias("median_top_depth"),
        ).collect()
        row = stats.to_dicts()[0]
        median_half_spread = float(row["median_spread_bps"]) / 2.0
        row.update(
            {
                "root": root,
                "per_side_cost_bps": max(median_half_spread, 0.1),
                "source": "mbp1",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).loc[
        :,
        [
            "root",
            "rows",
            "median_spread_bps",
            "p75_spread_bps",
            "median_top_depth",
            "per_side_cost_bps",
            "source",
        ],
    ]


def monthly_information_coefficients(
    features: pd.DataFrame,
    future_returns: pd.DataFrame,
    timestamps: pd.Series,
) -> pd.DataFrame:
    month = pd.to_datetime(timestamps, utc=True).dt.to_period("M").astype(str)
    rows = []
    for feature_name, feature_values in features.items():
        clean_feature = feature_values.replace([np.inf, -np.inf], np.nan)
        for root in future_returns.columns:
            frame = pd.DataFrame(
                {
                    "month": month,
                    "feature": clean_feature,
                    "future_return": future_returns[root],
                }
            ).dropna()
            for month_value, group in frame.groupby("month"):
                if len(group) < 50 or group["feature"].nunique() < 5:
                    continue
                corr, pvalue = spearmanr(group["feature"], group["future_return"])
                rows.append(
                    {
                        "feature": feature_name,
                        "root": root,
                        "month": month_value,
                        "spearman_ic": float(corr),
                        "pvalue": float(pvalue),
                        "observations": len(group),
                    }
                )
    return pd.DataFrame(rows)


def benjamini_hochberg(frame: pd.DataFrame, *, pvalue_col: str = "pvalue") -> pd.DataFrame:
    out = frame.copy()
    valid = out[pvalue_col].replace([np.inf, -np.inf], np.nan).notna()
    pvalues = out.loc[valid, pvalue_col].to_numpy(dtype=float)
    order = np.argsort(pvalues)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(pvalues) + 1)
    qvalues = pvalues * len(pvalues) / ranks
    sorted_q = np.minimum.accumulate(qvalues[order][::-1])[::-1]
    adjusted = np.empty_like(sorted_q)
    adjusted[order] = np.minimum(sorted_q, 1.0)
    out["bh_qvalue"] = np.nan
    out.loc[valid, "bh_qvalue"] = adjusted
    return out


def daily_block_bootstrap(
    returns: pd.DataFrame,
    timestamps: pd.Series,
    *,
    iterations: int = 2_000,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.to_datetime(timestamps, utc=True).dt.date
    daily = (
        returns.assign(day=days)
        .groupby("day", as_index=False)[["gross_return", "cost_return", "net_return"]]
        .sum()
    )
    values = daily["net_return"].to_numpy(dtype=float)
    if len(values) == 0:
        return pd.DataFrame()
    samples = np.empty(iterations)
    for i in range(iterations):
        draw = rng.choice(values, size=len(values), replace=True)
        samples[i] = draw.sum()
    return pd.DataFrame(
        {
            "metric": ["net_return"],
            "observed": [float(values.sum())],
            "bootstrap_mean": [float(samples.mean())],
            "p05": [float(np.quantile(samples, 0.05))],
            "p50": [float(np.quantile(samples, 0.50))],
            "p95": [float(np.quantile(samples, 0.95))],
            "p_positive": [float((samples > 0.0).mean())],
            "days": [len(values)],
            "iterations": [iterations],
        }
    )


def split_strategy_metrics(
    returns: pd.DataFrame,
    timestamps: pd.Series,
    *,
    train_fraction: float = 0.70,
    embargo_bars: int = 50,
    periods_per_year: float,
) -> pd.DataFrame:
    ordered = pd.to_datetime(timestamps, utc=True).reset_index(drop=True)
    split_index = int(len(ordered) * train_fraction)
    split_index = min(max(split_index, 1), len(ordered) - 1)
    train = returns.iloc[:split_index]
    test = returns.iloc[min(split_index + embargo_bars, len(returns)) :]
    rows = []
    for label, frame in (("train", train), ("test_embargoed", test)):
        metrics = calculate_strategy_metrics(frame, periods_per_year)
        rows.append({"split": label, **metrics.to_dict()})
    return pd.DataFrame(rows)
