# ruff: noqa: PLR0911, PLR0912, PLR0915
from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import polars as pl
import yaml

from quantlab.metals_flow.fair_value import ewma_relative_value_residuals

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from numba import njit
except ImportError:  # pragma: no cover - optional acceleration
    njit = None


DEFAULT_CONFIG = Path("experiments/HYP-0024-metals-1m-ewma-event-exit/config.yaml")
DECAY_EXIT = 0
FLIP_EXIT = 1
STOP_EXIT = 2
INVALID_EXIT = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HYP-0024 metals 1m EWMA event-exit test."
    )
    parser.add_argument("config", nargs="?", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    config_path = parse_args().config
    payload = load_yaml(config_path)
    out_dir = Path(payload["outputs"]["directory"])
    out_dir.mkdir(parents=True, exist_ok=True)
    run(payload, out_dir)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return loaded


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True)


def run(payload: dict[str, Any], out_dir: Path) -> None:
    roots = list(payload["universe"]["roots"])
    returns, log_prices, inventory = load_1m_panel(payload, roots)
    costs_bps = load_costs(payload, roots)
    split_index, test_start_index = train_test_indices(payload, returns)
    periods_per_year = infer_periods_per_year(returns.index)
    templates = hedge_templates(len(roots))
    train_allowed = np.arange(len(returns)) < split_index
    test_allowed = np.arange(len(returns)) >= test_start_index
    trade_allowed = train_allowed | test_allowed
    reset_next = np.zeros(len(returns), dtype=np.bool_)
    if split_index > 0:
        reset_next[split_index - 1] = True
    if test_start_index > 0:
        reset_next[test_start_index - 1] = True

    inventory.to_csv(out_dir / "data_inventory.csv", index=False)
    costs_bps.rename("per_side_cost_bps").to_csv(out_dir / "costs_bps.csv")

    rows: list[dict[str, Any]] = []
    selected_artifacts: dict[str, Any] | None = None
    for halflife in payload["research"]["ewma_halflife_bars"]:
        halflife_int = int(halflife)
        min_periods = max(
            20,
            round(halflife_int * float(payload["research"]["min_periods_multiplier"])),
        )
        residuals, zscores = ewma_signal(
            log_prices,
            halflife=halflife_int,
            min_periods=min_periods,
            ridge_alpha=float(payload["research"]["ridge_alpha"]),
        )
        z_values = zscores.to_numpy(dtype=np.float64)
        for entry_z in payload["research"]["entry_z"]:
            for exit_z in payload["research"]["exit_z"]:
                if float(exit_z) >= float(entry_z):
                    continue
                for stop_z in payload["research"]["stop_z"]:
                    if float(stop_z) <= float(entry_z):
                        continue
                    positions, stats = event_positions(
                        z_values,
                        templates,
                        float(entry_z),
                        float(exit_z),
                        float(stop_z),
                        float(payload["research"]["gross_cap"]),
                        trade_allowed,
                        reset_next,
                    )
                    returns_frame = strategy_returns(
                        returns,
                        positions,
                        costs_bps,
                        cost_multiplier=float(payload["research"]["cost_multiplier"]),
                    )
                    metrics = variant_metrics(
                        returns_frame,
                        stats,
                        periods_per_year,
                        split_index,
                        test_start_index,
                    )
                    row = {
                        "variant": variant_name(halflife_int, entry_z, exit_z, stop_z),
                        "halflife_bars": halflife_int,
                        "min_periods": min_periods,
                        "entry_z": float(entry_z),
                        "exit_z": float(exit_z),
                        "stop_z": float(stop_z),
                        **metrics,
                    }
                    rows.append(row)
                    selected_artifacts = maybe_keep_selected(
                        selected_artifacts,
                        row,
                        returns_frame,
                        positions,
                        residuals,
                        zscores,
                        stats,
                    )

    if selected_artifacts is None:
        raise RuntimeError("no variants were evaluated")

    metrics_frame = pd.DataFrame(rows).sort_values(
        ["train_score", "train_net_return", "train_tstat"], ascending=False
    )
    selected_row = select_variant(metrics_frame, payload)
    selected = rerun_selected(
        selected_row,
        payload,
        log_prices,
        returns,
        costs_bps,
        templates,
        trade_allowed,
        reset_next,
        periods_per_year,
        split_index,
        test_start_index,
    )
    decision = make_decision(payload, selected["split_metrics"])

    metrics_frame.to_csv(out_dir / "variant_metrics.csv", index=False)
    selected["returns"].to_csv(out_dir / "selected_strategy_returns.csv")
    selected["positions"].to_csv(out_dir / "selected_strategy_positions.csv")
    selected["residuals"].to_parquet(out_dir / "selected_residuals.parquet")
    selected["zscores"].to_parquet(out_dir / "selected_zscores.parquet")
    selected["event_stats"].to_csv(out_dir / "selected_event_stats.csv", index=False)
    selected["split_metrics"].to_csv(out_dir / "selected_split_metrics.csv", index=False)
    monthly = monthly_returns(selected["returns"])
    monthly.to_csv(out_dir / "selected_monthly_returns.csv", index=False)
    plot_equity(selected["returns"], out_dir / "selected_equity.png")

    summary = {
        "experiment_id": payload["experiment"]["id"],
        "title": payload["experiment"]["title"],
        "completed_at": datetime.now(UTC).isoformat(),
        "decision": decision,
        "data": {
            "roots": roots,
            "bars": len(returns),
            "start": returns.index[0],
            "end": returns.index[-1],
            "split_index": split_index,
            "split_ts": returns.index[split_index],
            "test_start_index": test_start_index,
            "test_start_ts": returns.index[test_start_index],
            "periods_per_year": periods_per_year,
            "note": (
                "CL/rates full 1-minute continuous bars were not present locally; "
                "this is metals-only."
            ),
        },
        "selected_variant": selected_row.to_dict(),
        "selected_split_metrics": selected["split_metrics"].to_dict(orient="records"),
        "selected_event_stats": selected["event_stats"].to_dict(orient="records"),
        "top_train_variants": metrics_frame.head(20).to_dict(orient="records"),
        "top_test_variants": metrics_frame.sort_values(
            ["test_net_return", "test_tstat"], ascending=False
        )
        .head(20)
        .to_dict(orient="records"),
    }
    write_json(out_dir / "results.json", summary)
    write_report(out_dir, summary)
    print(f"wrote {out_dir / 'results.json'}", flush=True)


def load_1m_panel(
    payload: dict[str, Any], roots: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    continuous_dir = Path(payload["data"]["continuous_dir"])
    start = pd.Timestamp(payload["data"]["start"])
    end = pd.Timestamp(payload["data"]["end"])
    parts: list[pd.DataFrame] = []
    inventory_rows: list[dict[str, Any]] = []
    for root in roots:
        path = continuous_dir / f"{root}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing 1m continuous bars for {root}: {path}")
        frame = (
            pl.scan_parquet(path)
            .filter((pl.col("ts") >= start) & (pl.col("ts") < end))
            .select("ts", "cont_logret", "cont_logprice", "cont_close", "volume", "is_roll")
            .collect()
            .to_pandas()
        )
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
        frame = frame.sort_values("ts").replace([np.inf, -np.inf], np.nan)
        inventory_rows.append(
            {
                "root": root,
                "rows": len(frame),
                "first_ts": frame["ts"].min(),
                "last_ts": frame["ts"].max(),
                "roll_bars": int(frame["is_roll"].fillna(False).sum()),
                "mean_volume": safe_float(frame["volume"].mean()),
                "zero_volume_fraction": safe_float((frame["volume"].fillna(0.0) <= 0.0).mean()),
            }
        )
        parts.append(
            frame.set_index("ts")[
                ["cont_logret", "cont_logprice", "cont_close", "volume", "is_roll"]
            ].rename(
                columns={
                    "cont_logret": f"{root}__ret",
                    "cont_logprice": f"{root}__logprice",
                    "cont_close": f"{root}__close",
                    "volume": f"{root}__volume",
                    "is_roll": f"{root}__is_roll",
                }
            )
        )

    panel = pd.concat(parts, axis=1, join="inner").sort_index()
    roll_cols = [f"{root}__is_roll" for root in roots]
    roll_mask = panel[roll_cols].fillna(False).any(axis=1)
    panel = panel.loc[~roll_mask]
    returns = pd.DataFrame({root: panel[f"{root}__ret"] for root in roots}, index=panel.index)
    log_prices = pd.DataFrame(
        {root: panel[f"{root}__logprice"] for root in roots}, index=panel.index
    )
    valid = returns.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    valid &= log_prices.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    returns = returns.loc[valid].astype(float).fillna(0.0)
    log_prices = log_prices.loc[valid].astype(float)
    log_prices = log_prices - log_prices.iloc[0]
    return returns, log_prices, pd.DataFrame(inventory_rows)


def load_costs(payload: dict[str, Any], roots: list[str]) -> pd.Series:
    costs = pd.read_csv(payload["data"]["cost_estimates"])
    costs_bps = costs.set_index("root")["per_side_cost_bps"].reindex(roots).astype(float)
    if costs_bps.isna().any():
        missing = costs_bps[costs_bps.isna()].index.tolist()
        raise ValueError(f"missing cost estimates for {missing}")
    return costs_bps


def train_test_indices(payload: dict[str, Any], returns: pd.DataFrame) -> tuple[int, int]:
    split_index = int(len(returns) * float(payload["research"]["train_fraction"]))
    split_index = min(max(split_index, 1), len(returns) - 2)
    test_start_index = min(split_index + int(payload["research"]["embargo_bars"]), len(returns) - 1)
    return split_index, test_start_index


def infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    elapsed_years = (index[-1] - index[0]).total_seconds() / (365.25 * 24 * 60 * 60)
    return float(len(index) / elapsed_years)


def hedge_templates(n_assets: int) -> np.ndarray:
    templates = np.full((n_assets, n_assets), 1.0 / (n_assets - 1), dtype=np.float64)
    np.fill_diagonal(templates, -1.0)
    return templates


def ewma_signal(
    log_prices: pd.DataFrame,
    *,
    halflife: int,
    min_periods: int,
    ridge_alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    residuals, _unused = ewma_relative_value_residuals(
        log_prices,
        halflife=halflife,
        min_periods=min_periods,
        zscore_window=halflife,
        ridge_alpha=ridge_alpha,
    )
    mean = residuals.ewm(halflife=halflife, min_periods=min_periods, adjust=False).mean().shift(1)
    std = residuals.ewm(halflife=halflife, min_periods=min_periods, adjust=False).std().shift(1)
    zscores = ((residuals - mean) / std.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return residuals, zscores


def event_positions(
    z_values: np.ndarray,
    templates: np.ndarray,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    gross_cap: float,
    trade_allowed: np.ndarray,
    reset_next: np.ndarray,
    entry_gate: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    if entry_gate is None:
        entry_gate = np.ones(z_values.shape, dtype=np.bool_)
    if njit is not None:
        positions, stats = _event_positions_numba(
            z_values,
            templates,
            entry_z,
            exit_z,
            stop_z,
            gross_cap,
            trade_allowed,
            reset_next,
            entry_gate,
        )
    else:
        positions, stats = _event_positions_python(
            z_values,
            templates,
            entry_z,
            exit_z,
            stop_z,
            gross_cap,
            trade_allowed,
            reset_next,
            entry_gate,
        )
    return positions, decode_event_stats(stats)


def decode_event_stats(stats: np.ndarray) -> dict[str, float]:
    closed = float(stats[1])
    mean_hold = float(stats[6] / closed) if closed > 0.0 else np.nan
    variance = float(stats[7] / closed - mean_hold**2) if closed > 1.0 else np.nan
    return {
        "entries": float(stats[0]),
        "closed_trades": closed,
        "decay_exits": float(stats[2]),
        "flip_exits": float(stats[3]),
        "stop_exits": float(stats[4]),
        "invalid_exits": float(stats[5]),
        "mean_hold_bars": mean_hold,
        "std_hold_bars": math.sqrt(max(variance, 0.0)) if math.isfinite(variance) else np.nan,
        "max_hold_bars": float(stats[8]),
        "open_trades_at_end": float(stats[9]),
    }


def _event_positions_python(
    z_values: np.ndarray,
    templates: np.ndarray,
    entry_z: float,
    exit_z: float,
    stop_z: float,
    gross_cap: float,
    trade_allowed: np.ndarray,
    reset_next: np.ndarray,
    entry_gate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_rows, n_assets = z_values.shape
    positions = np.zeros((n_rows, n_assets), dtype=np.float64)
    active = np.zeros(n_assets, dtype=np.bool_)
    entry_sign = np.zeros(n_assets, dtype=np.float64)
    entry_bar = np.zeros(n_assets, dtype=np.int64)
    stats = np.zeros(10, dtype=np.float64)
    for row in range(n_rows - 1):
        if reset_next[row]:
            active[:] = False
        for target in range(n_assets):
            z_value = z_values[row, target]
            if active[target]:
                exit_reason = -1
                if not np.isfinite(z_value):
                    exit_reason = INVALID_EXIT
                elif abs(z_value) <= exit_z:
                    exit_reason = DECAY_EXIT
                elif z_value * entry_sign[target] <= 0.0:
                    exit_reason = FLIP_EXIT
                elif abs(z_value) >= stop_z:
                    exit_reason = STOP_EXIT
                if exit_reason >= 0:
                    hold = row - entry_bar[target]
                    active[target] = False
                    stats[1] += 1.0
                    stats[2 + exit_reason] += 1.0
                    stats[6] += hold
                    stats[7] += hold * hold
                    stats[8] = max(stats[8], hold)
            if (
                not active[target]
                and trade_allowed[row]
                and entry_gate[row, target]
                and np.isfinite(z_value)
                and abs(z_value) >= entry_z
                and abs(z_value) < stop_z
            ):
                active[target] = True
                entry_sign[target] = 1.0 if z_value > 0.0 else -1.0
                entry_bar[target] = row
                stats[0] += 1.0
        for target in range(n_assets):
            if active[target]:
                positions[row + 1] += entry_sign[target] * templates[target]
        gross = np.abs(positions[row + 1]).sum()
        if gross > gross_cap and gross > 0.0:
            positions[row + 1] *= gross_cap / gross
    stats[9] = float(active.sum())
    return positions, stats


if njit is not None:

    @njit(cache=True)
    def _event_positions_numba(
        z_values: np.ndarray,
        templates: np.ndarray,
        entry_z: float,
        exit_z: float,
        stop_z: float,
        gross_cap: float,
        trade_allowed: np.ndarray,
        reset_next: np.ndarray,
        entry_gate: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_rows, n_assets = z_values.shape
        positions = np.zeros((n_rows, n_assets), dtype=np.float64)
        active = np.zeros(n_assets, dtype=np.bool_)
        entry_sign = np.zeros(n_assets, dtype=np.float64)
        entry_bar = np.zeros(n_assets, dtype=np.int64)
        stats = np.zeros(10, dtype=np.float64)
        for row in range(n_rows - 1):
            if reset_next[row]:
                for target in range(n_assets):
                    active[target] = False
            for target in range(n_assets):
                z_value = z_values[row, target]
                if active[target]:
                    exit_reason = -1
                    if not np.isfinite(z_value):
                        exit_reason = INVALID_EXIT
                    elif abs(z_value) <= exit_z:
                        exit_reason = DECAY_EXIT
                    elif z_value * entry_sign[target] <= 0.0:
                        exit_reason = FLIP_EXIT
                    elif abs(z_value) >= stop_z:
                        exit_reason = STOP_EXIT
                    if exit_reason >= 0:
                        hold = row - entry_bar[target]
                        active[target] = False
                        stats[1] += 1.0
                        stats[2 + exit_reason] += 1.0
                        stats[6] += hold
                        stats[7] += hold * hold
                        stats[8] = max(stats[8], hold)
                if (
                    not active[target]
                    and trade_allowed[row]
                    and entry_gate[row, target]
                    and np.isfinite(z_value)
                    and abs(z_value) >= entry_z
                    and abs(z_value) < stop_z
                ):
                    active[target] = True
                    if z_value > 0.0:
                        entry_sign[target] = 1.0
                    else:
                        entry_sign[target] = -1.0
                    entry_bar[target] = row
                    stats[0] += 1.0
            for target in range(n_assets):
                if active[target]:
                    for asset in range(n_assets):
                        positions[row + 1, asset] += entry_sign[target] * templates[target, asset]
            gross = 0.0
            for asset in range(n_assets):
                gross += abs(positions[row + 1, asset])
            if gross > gross_cap and gross > 0.0:
                scale = gross_cap / gross
                for asset in range(n_assets):
                    positions[row + 1, asset] *= scale
        open_count = 0.0
        for target in range(n_assets):
            if active[target]:
                open_count += 1.0
        stats[9] = open_count
        return positions, stats


def strategy_returns(
    returns: pd.DataFrame,
    positions: np.ndarray,
    costs_bps: pd.Series,
    *,
    cost_multiplier: float,
) -> pd.DataFrame:
    pos = pd.DataFrame(positions, index=returns.index, columns=returns.columns)
    gross = (pos * returns).sum(axis=1)
    turnover = pos.diff().abs().fillna(pos.abs())
    costs = (turnover * costs_bps.reindex(returns.columns).to_numpy(dtype=float)).sum(axis=1)
    costs = costs * cost_multiplier / 10_000.0
    out = pd.DataFrame(
        {
            "gross_return": gross,
            "cost_return": costs,
            "net_return": gross - costs,
            "gross_exposure": pos.abs().sum(axis=1),
            "turnover": turnover.sum(axis=1),
        },
        index=returns.index,
    )
    return out


def variant_metrics(
    returns_frame: pd.DataFrame,
    event_stats: dict[str, float],
    periods_per_year: float,
    split_index: int,
    test_start_index: int,
) -> dict[str, float]:
    train = summarize_slice(returns_frame.iloc[:split_index], periods_per_year)
    embargo = summarize_slice(returns_frame.iloc[split_index:test_start_index], periods_per_year)
    test = summarize_slice(returns_frame.iloc[test_start_index:], periods_per_year)
    train_score = score_variant(train)
    return {
        **{f"train_{key}": value for key, value in train.items()},
        **{f"embargo_{key}": value for key, value in embargo.items()},
        **{f"test_{key}": value for key, value in test.items()},
        "train_score": train_score,
        **event_stats,
    }


def summarize_slice(frame: pd.DataFrame, periods_per_year: float) -> dict[str, float]:
    if frame.empty:
        return {
            "gross_return": np.nan,
            "cost_return": np.nan,
            "net_return": np.nan,
            "gross_to_cost": np.nan,
            "tstat": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "mean_gross_exposure": np.nan,
            "turnover": np.nan,
        }
    net = frame["net_return"].astype(float)
    gross = frame["gross_return"].astype(float)
    cost = frame["cost_return"].astype(float)
    std = float(net.std(ddof=1))
    tstat = float(net.mean() / std * math.sqrt(periods_per_year)) if std > 0.0 else np.nan
    equity = net.cumsum()
    drawdown = equity - equity.cummax()
    return {
        "gross_return": float(gross.sum()),
        "cost_return": float(cost.sum()),
        "net_return": float(net.sum()),
        "gross_to_cost": float(gross.sum() / cost.sum()) if cost.sum() > 0.0 else np.nan,
        "tstat": tstat,
        "sharpe": tstat,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else np.nan,
        "mean_gross_exposure": float(frame["gross_exposure"].mean()),
        "turnover": float(frame["turnover"].sum()),
    }


def score_variant(metrics: dict[str, float]) -> float:
    net = metrics["net_return"]
    gross_to_cost = metrics["gross_to_cost"]
    tstat = metrics["tstat"]
    if not all(np.isfinite([net, gross_to_cost, tstat])):
        return -np.inf
    if net <= 0.0 or gross_to_cost <= 1.0:
        return -np.inf
    return float(tstat + 0.25 * np.log1p(max(gross_to_cost, 0.0)) + 10.0 * net)


def maybe_keep_selected(
    selected_artifacts: dict[str, Any] | None,
    row: dict[str, Any],
    returns_frame: pd.DataFrame,
    positions: np.ndarray,
    residuals: pd.DataFrame,
    zscores: pd.DataFrame,
    stats: dict[str, float],
) -> dict[str, Any]:
    if selected_artifacts is None or row["train_score"] > selected_artifacts["row"]["train_score"]:
        return {
            "row": row,
            "returns": returns_frame,
            "positions": positions,
            "residuals": residuals,
            "zscores": zscores,
            "stats": stats,
        }
    return selected_artifacts


def select_variant(metrics: pd.DataFrame, payload: dict[str, Any]) -> pd.Series:
    rules = payload["decision_rules"]
    viable = metrics[
        (metrics["train_net_return"] > 0.0)
        & (metrics["train_gross_to_cost"] >= float(rules["min_train_gross_to_cost"]))
        & np.isfinite(metrics["train_score"])
    ].copy()
    if viable.empty:
        viable = metrics[np.isfinite(metrics["train_score"])].copy()
    if viable.empty:
        viable = metrics.copy()
    viable = viable.sort_values(["train_score", "train_net_return"], ascending=False)
    return viable.iloc[0]


def rerun_selected(
    selected_row: pd.Series,
    payload: dict[str, Any],
    log_prices: pd.DataFrame,
    returns: pd.DataFrame,
    costs_bps: pd.Series,
    templates: np.ndarray,
    trade_allowed: np.ndarray,
    reset_next: np.ndarray,
    periods_per_year: float,
    split_index: int,
    test_start_index: int,
) -> dict[str, Any]:
    residuals, zscores = ewma_signal(
        log_prices,
        halflife=int(selected_row["halflife_bars"]),
        min_periods=int(selected_row["min_periods"]),
        ridge_alpha=float(payload["research"]["ridge_alpha"]),
    )
    positions_array, stats = event_positions(
        zscores.to_numpy(dtype=np.float64),
        templates,
        float(selected_row["entry_z"]),
        float(selected_row["exit_z"]),
        float(selected_row["stop_z"]),
        float(payload["research"]["gross_cap"]),
        trade_allowed,
        reset_next,
    )
    returns_frame = strategy_returns(
        returns,
        positions_array,
        costs_bps,
        cost_multiplier=float(payload["research"]["cost_multiplier"]),
    )
    positions = pd.DataFrame(positions_array, index=returns.index, columns=returns.columns)
    split_metrics = pd.DataFrame(
        [
            {
                "split": "train",
                **summarize_slice(returns_frame.iloc[:split_index], periods_per_year),
            },
            {
                "split": "embargo",
                **summarize_slice(
                    returns_frame.iloc[split_index:test_start_index], periods_per_year
                ),
            },
            {
                "split": "test",
                **summarize_slice(returns_frame.iloc[test_start_index:], periods_per_year),
            },
            {"split": "full", **summarize_slice(returns_frame, periods_per_year)},
        ]
    )
    event_stats = pd.DataFrame([{**selected_row.to_dict(), **stats}])
    return {
        "returns": returns_frame,
        "positions": positions,
        "residuals": residuals,
        "zscores": zscores,
        "split_metrics": split_metrics,
        "event_stats": event_stats,
    }


def make_decision(payload: dict[str, Any], split_metrics: pd.DataFrame) -> str:
    rules = payload["decision_rules"]
    by_split = split_metrics.set_index("split")
    train = by_split.loc["train"]
    test = by_split.loc["test"]
    train_viable = (
        float(train["net_return"]) > 0.0
        and float(train["gross_to_cost"]) >= float(rules["min_train_gross_to_cost"])
    )
    if not train_viable:
        return str(rules["fail_status"])
    passes = (
        float(test["net_return"]) > float(rules["min_test_net_return"])
        and float(test["gross_to_cost"]) >= float(rules["min_test_gross_to_cost"])
        and float(test["tstat"]) >= float(rules["min_test_tstat"])
    )
    if passes:
        return str(rules["pass_status"])
    if float(test["net_return"]) > 0.0 and float(test["gross_to_cost"]) > 1.0:
        return str(rules["revise_status"])
    return str(rules["fail_status"])


def variant_name(halflife: int, entry_z: float, exit_z: float, stop_z: float) -> str:
    return f"ewma{halflife}_entry{entry_z:g}_exit{exit_z:g}_stop{stop_z:g}"


def monthly_returns(returns_frame: pd.DataFrame) -> pd.DataFrame:
    monthly = returns_frame[["gross_return", "cost_return", "net_return"]].resample("ME").sum()
    monthly.index = monthly.index.strftime("%Y-%m")
    return monthly.reset_index(names="month")


def plot_equity(returns_frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    returns_frame["net_return"].cumsum().plot(ax=ax, label="net")
    returns_frame["gross_return"].cumsum().plot(ax=ax, label="gross", alpha=0.7)
    ax.set_title("Selected Strategy Cumulative Return")
    ax.set_xlabel("")
    ax.set_ylabel("log-return units")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(out_dir: Path, summary: dict[str, Any]) -> None:
    selected = summary["selected_variant"]
    split = pd.DataFrame(summary["selected_split_metrics"])
    events = pd.DataFrame(summary["selected_event_stats"])
    lines = [
        "# HYP-0024 Report",
        "",
        f"Decision: **{summary['decision']}**.",
        "",
        "## Selected Variant",
        "",
        f"- Variant: `{selected['variant']}`",
        f"- EWMA half-life: {int(selected['halflife_bars'])} 1-minute bars",
        (
            f"- Entry / exit / stop z: {selected['entry_z']} / {selected['exit_z']} / "
            f"{selected['stop_z']}"
        ),
        "",
        "## Split Metrics",
        "",
        split.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Event Stats",
        "",
        events[
            [
                "entries",
                "closed_trades",
                "decay_exits",
                "flip_exits",
                "stop_exits",
                "invalid_exits",
                "mean_hold_bars",
                "std_hold_bars",
                "max_hold_bars",
                "open_trades_at_end",
            ]
        ].to_markdown(index=False, floatfmt=".2f"),
        "",
        "## Limitation",
        "",
        (
            "This run is metals-only because full 1-minute continuous CL/rates bars "
            "were not present locally."
        ),
        "",
    ]
    (out_dir / "report.qmd").write_text("\n".join(lines), encoding="utf-8")


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


if __name__ == "__main__":
    main()
