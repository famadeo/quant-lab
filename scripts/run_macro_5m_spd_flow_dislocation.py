# ruff: noqa: I001, PLR0911, PLR0915
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import polars as pl
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_CONFIG = Path("experiments/HYP-0023-macro-5m-spd-flow-dislocation/config.yaml")
MIN_GROUP_NEUTRAL_ROOTS = 2
PERIODS_PER_YEAR = 252.0 * 78.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HYP-0023 macro 5m SPD/flow test.")
    parser.add_argument("config", nargs="?", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def safe_float(value: float | np.floating | None) -> float | None:
    if value is None:
        return None
    out = float(value)
    return out if np.isfinite(out) else None


def restrict_time(frame: pd.DataFrame, ts_col: str, start: str, end: str) -> pd.DataFrame:
    timestamps = pd.to_datetime(frame[ts_col], utc=True)
    minutes = timestamps.dt.hour * 60 + timestamps.dt.minute
    start_hour, start_minute = (int(part) for part in start.split(":"))
    end_hour, end_minute = (int(part) for part in end.split(":"))
    lower = start_hour * 60 + start_minute
    upper = end_hour * 60 + end_minute
    return frame[(minutes > lower) & (minutes <= upper)].copy()


def load_5m_panel(
    roots: list[str],
    rates_roots: set[str],
    futures_dir: Path,
    flow_dir: Path,
    rth_start: str,
    rth_end: str,
    flow_feature: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    return_parts: dict[str, pd.DataFrame] = {}
    inventory_rows: list[dict[str, Any]] = []

    for root in roots:
        bar_path = futures_dir / f"{root}.parquet"
        flow_path = flow_dir / f"{root}_flow_5m.parquet"
        if not bar_path.exists():
            raise FileNotFoundError(f"missing 5m futures bars for {root}: {bar_path}")
        if not flow_path.exists():
            raise FileNotFoundError(f"missing 5m flow bars for {root}: {flow_path}")

        bars = (
            pl.read_parquet(bar_path)
            .select("ts", "cont_logret", "cont_logprice", "volume", "is_roll")
            .to_pandas()
        )
        flow = pl.read_parquet(flow_path).select("ts", "tot_vol", flow_feature).to_pandas()
        bars["ts"] = pd.to_datetime(bars["ts"], utc=True)
        flow["ts"] = pd.to_datetime(flow["ts"], utc=True)
        bars = restrict_time(bars, "ts", rth_start, rth_end)
        joined = bars.merge(flow, on="ts", how="inner").sort_values("ts")
        joined = joined.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["cont_logret", "cont_logprice", flow_feature]
        )
        inventory_rows.append(
            {
                "root": root,
                "bar_rows": len(bars),
                "joined_rows": len(joined),
                "first_ts": joined["ts"].min(),
                "last_ts": joined["ts"].max(),
                "roll_bars": int(joined["is_roll"].fillna(False).sum()),
                "mean_volume": safe_float(joined["volume"].mean()),
                "mean_flow_volume": safe_float(joined["tot_vol"].mean()),
            }
        )
        return_parts[root] = joined.set_index("ts")[
            ["cont_logret", "cont_logprice", flow_feature, "tot_vol"]
        ].rename(
            columns={
                "cont_logret": f"{root}__ret",
                "cont_logprice": f"{root}__logprice",
                flow_feature: f"{root}__flow",
                "tot_vol": f"{root}__flow_volume",
            }
        )

    panel = pd.concat(return_parts.values(), axis=1, join="inner").sort_index()
    ret = pd.DataFrame({root: panel[f"{root}__ret"] for root in roots}, index=panel.index)
    logprice = pd.DataFrame({root: panel[f"{root}__logprice"] for root in roots}, index=panel.index)
    flow = pd.DataFrame({root: panel[f"{root}__flow"] for root in roots}, index=panel.index)
    flow_volume = pd.DataFrame(
        {root: panel[f"{root}__flow_volume"] for root in roots}, index=panel.index
    )
    ret = ret.dropna()
    logprice = logprice.reindex(ret.index).ffill().dropna()
    common_index = ret.index.intersection(logprice.index)
    ret = ret.reindex(common_index)
    logprice = logprice.reindex(common_index)
    flow = flow.reindex(common_index).fillna(0.0)
    flow_volume = flow_volume.reindex(common_index).fillna(0.0)
    risk_sign = pd.Series({root: -1.0 if root in rates_roots else 1.0 for root in roots})
    return ret, logprice, flow, flow_volume, risk_sign, pd.DataFrame(inventory_rows)


def matrix_log_correlation(
    returns: np.ndarray,
    shrinkage_alpha: float,
    eigen_floor: float,
) -> tuple[np.ndarray, float, float, float, np.ndarray]:
    corr = np.corrcoef(returns, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = (1.0 - shrinkage_alpha) * corr + shrinkage_alpha * np.eye(corr.shape[0])
    np.fill_diagonal(corr, 1.0)
    corr = (corr + corr.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    eigenvalues = np.maximum(eigenvalues, eigen_floor)
    log_corr = (eigenvectors * np.log(eigenvalues)) @ eigenvectors.T
    weights = eigenvalues / eigenvalues.sum()
    effective_rank = float(np.exp(-(weights * np.log(np.maximum(weights, 1e-12))).sum()))
    top_eigen_share = float(eigenvalues[-1] / eigenvalues.sum())
    n_assets = corr.shape[0]
    avg_corr = float((corr.sum() - n_assets) / (n_assets * (n_assets - 1)))
    return log_corr, top_eigen_share, effective_rank, avg_corr, corr


def mst_edges(corr: np.ndarray, roots: list[str]) -> frozenset[tuple[str, str]]:
    candidates: list[tuple[float, int, int]] = []
    n_assets = corr.shape[0]
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            distance = math.sqrt(max(0.0, 2.0 * (1.0 - float(corr[i, j]))))
            candidates.append((distance, i, j))

    parent = list(range(n_assets))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    selected: list[tuple[str, str]] = []
    for _distance, i, j in sorted(candidates):
        root_i = find(i)
        root_j = find(j)
        if root_i == root_j:
            continue
        parent[root_i] = root_j
        selected.append(tuple(sorted((roots[i], roots[j]))))
        if len(selected) == n_assets - 1:
            break
    return frozenset(selected)


def compute_regime_features(
    risk_returns: pd.DataFrame,
    roots: list[str],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookback = int(config["regime"]["covariance_lookback_bars"])
    q_lookback = int(config["regime"]["quantile_lookback_bars"])
    shrinkage_alpha = float(config["regime"]["shrinkage_alpha"])
    eigen_floor = float(config["regime"]["eigen_floor"])
    rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    previous_log_corr: np.ndarray | None = None
    previous_edges: frozenset[tuple[str, str]] | None = None

    for bar_index, timestamp in enumerate(risk_returns.index):
        if bar_index < lookback:
            rows.append(
                {
                    "ts": timestamp,
                    "spd_velocity": np.nan,
                    "top_eigen_share": np.nan,
                    "effective_rank": np.nan,
                    "average_correlation": np.nan,
                    "edge_persistence": np.nan,
                }
            )
            continue

        window = risk_returns.iloc[bar_index - lookback : bar_index].to_numpy(dtype=float)
        log_corr, top_eigen, effective_rank, avg_corr, corr = matrix_log_correlation(
            window, shrinkage_alpha, eigen_floor
        )
        edges = mst_edges(corr, roots)
        velocity = (
            np.nan
            if previous_log_corr is None
            else float(np.linalg.norm(log_corr - previous_log_corr, ord="fro"))
        )
        edge_persistence = (
            np.nan
            if previous_edges is None
            else float(len(edges & previous_edges) / len(edges | previous_edges))
        )
        rows.append(
            {
                "ts": timestamp,
                "spd_velocity": velocity,
                "top_eigen_share": top_eigen,
                "effective_rank": effective_rank,
                "average_correlation": avg_corr,
                "edge_persistence": edge_persistence,
            }
        )
        for root_a, root_b in edges:
            edge_rows.append({"ts": timestamp, "root_a": root_a, "root_b": root_b})
        previous_log_corr = log_corr
        previous_edges = edges

    features = pd.DataFrame(rows).set_index("ts")
    velocity_q40 = (
        features["spd_velocity"]
        .shift(1)
        .rolling(q_lookback, min_periods=q_lookback // 2)
        .quantile(float(config["regime"]["stable_max_spd_velocity_quantile"]))
    )
    velocity_q80 = (
        features["spd_velocity"]
        .shift(1)
        .rolling(q_lookback, min_periods=q_lookback // 2)
        .quantile(float(config["regime"]["transition_min_spd_velocity_quantile"]))
    )
    edge_q60 = (
        features["edge_persistence"]
        .shift(1)
        .rolling(q_lookback, min_periods=q_lookback // 2)
        .quantile(float(config["regime"]["stable_min_edge_persistence_quantile"]))
    )
    features["stable"] = (features["spd_velocity"] <= velocity_q40) & (
        features["edge_persistence"] >= edge_q60
    )
    features["transition"] = features["spd_velocity"] >= velocity_q80
    features["spd_velocity_q40"] = velocity_q40
    features["spd_velocity_q80"] = velocity_q80
    features["edge_persistence_q60"] = edge_q60
    return features, pd.DataFrame(edge_rows)


def build_factor_panel(risk_prices: pd.DataFrame) -> pd.DataFrame:
    roots = list(risk_prices.columns)
    rates = [root for root in roots if root in {"SR3", "ZQ", "ZT", "ZF", "ZN", "TN", "ZB", "UB"}]
    metals = [root for root in roots if root in {"GC", "SI", "HG", "PL", "PA"}]
    factors = pd.DataFrame(index=risk_prices.index)
    factors["market"] = risk_prices.mean(axis=1)
    factors["rates_level"] = risk_prices[rates].mean(axis=1)
    factors["rates_slope"] = risk_prices[
        [root for root in ["ZN", "TN", "ZB", "UB"] if root in roots]
    ].mean(axis=1) - risk_prices[
        [root for root in ["SR3", "ZQ", "ZT", "ZF"] if root in roots]
    ].mean(axis=1)
    factors["precious"] = risk_prices[["GC", "SI"]].mean(axis=1)
    factors["pgm"] = risk_prices[["PL", "PA"]].mean(axis=1)
    factors["cyclical"] = risk_prices[["HG", "CL"]].mean(axis=1)
    factors["metals_broad"] = risk_prices[metals].mean(axis=1)
    return factors


def rolling_residual_zscores(
    risk_prices: pd.DataFrame,
    roots: list[str],
    lookback: int,
    zscore_window: int,
    ridge: float,
) -> pd.DataFrame:
    factors = build_factor_panel(risk_prices)
    x_all = factors.to_numpy(dtype=float)
    y_all = risk_prices[roots].to_numpy(dtype=float)
    residuals = np.full(y_all.shape, np.nan)

    for bar_index in range(lookback, len(risk_prices)):
        x_window = np.column_stack([np.ones(lookback), x_all[bar_index - lookback : bar_index]])
        x_current = np.column_stack([np.ones(1), x_all[bar_index : bar_index + 1]])
        x_valid = np.isfinite(x_window).all(axis=1)
        if x_valid.sum() < lookback // 2:
            continue
        for root_index in range(len(roots)):
            y_window = y_all[bar_index - lookback : bar_index, root_index]
            valid = x_valid & np.isfinite(y_window)
            if valid.sum() < lookback // 2:
                continue
            x_fit = x_window[valid]
            y_fit = y_window[valid]
            penalty = np.eye(x_fit.shape[1]) * ridge
            penalty[0, 0] = 0.0
            beta = np.linalg.solve(x_fit.T @ x_fit + penalty, x_fit.T @ y_fit)
            residuals[bar_index, root_index] = y_all[bar_index, root_index] - float(
                (x_current @ beta)[0]
            )

    residual_frame = pd.DataFrame(residuals, index=risk_prices.index, columns=roots)
    rolling_mean = residual_frame.rolling(
        zscore_window, min_periods=max(5, zscore_window // 2)
    ).mean()
    rolling_std = residual_frame.rolling(
        zscore_window, min_periods=max(5, zscore_window // 2)
    ).std()
    return (residual_frame - rolling_mean) / rolling_std


def rolling_zscore(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    mean = frame.rolling(window, min_periods=max(5, window // 2)).mean()
    std = frame.rolling(window, min_periods=max(5, window // 2)).std()
    return (frame - mean) / std.replace(0.0, np.nan)


def base_reversion_signal(zscores: pd.DataFrame, entry_z: float, max_abs: float) -> pd.DataFrame:
    values = zscores.to_numpy(dtype=float)
    signal = np.where(
        np.abs(values) >= entry_z,
        -np.sign(values) * np.minimum(np.abs(values) / entry_z, max_abs),
        0.0,
    )
    return pd.DataFrame(signal, index=zscores.index, columns=zscores.columns)


def apply_variant_policy(
    residual_z: pd.DataFrame,
    flow_z: pd.DataFrame,
    regimes: pd.DataFrame,
    variant: str,
    entry_z: float,
    flow_threshold: float,
    max_abs: float,
) -> pd.DataFrame:
    residual_reversion = base_reversion_signal(residual_z, entry_z, max_abs)
    residual_momentum = -residual_reversion
    flow_momentum = pd.DataFrame(
        np.where(np.abs(flow_z) >= flow_threshold, np.sign(flow_z), 0.0),
        index=flow_z.index,
        columns=flow_z.columns,
    )
    flow_reversion = -flow_momentum
    same_direction_flow = (np.sign(flow_z) == np.sign(residual_z)) & (
        np.abs(flow_z) >= flow_threshold
    )
    flow_confirmed_reversion = residual_reversion.where(same_direction_flow, 0.0)
    stable = regimes["stable"].reindex(residual_z.index).fillna(False)
    transition = regimes["transition"].reindex(residual_z.index).fillna(False)

    if variant == "ungated_residual_reversion":
        return residual_reversion
    if variant == "stable_residual_reversion":
        return residual_reversion.where(stable, 0.0)
    if variant == "stable_flow_confirmed_reversion":
        return flow_confirmed_reversion.where(stable, 0.0)
    if variant == "non_transition_flow_confirmed_reversion":
        return flow_confirmed_reversion.where(~transition, 0.0)
    if variant == "transition_flow_momentum":
        return residual_momentum.where(same_direction_flow & transition, 0.0)
    if variant == "flow_only_momentum":
        return flow_momentum
    if variant == "flow_only_reversion":
        return flow_reversion
    if variant == "stable_flow_only_reversion":
        return flow_reversion.where(stable, 0.0)
    if variant == "non_transition_flow_only_reversion":
        return flow_reversion.where(~transition, 0.0)
    raise ValueError(f"unknown variant: {variant}")


def neutralized_risk_positions(signal: pd.DataFrame, groups: dict[str, str]) -> pd.DataFrame:
    positions = signal.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for group in sorted(set(groups.values())):
        roots = [root for root, root_group in groups.items() if root_group == group]
        if len(roots) < MIN_GROUP_NEUTRAL_ROOTS:
            continue
        positions[roots] = positions[roots].sub(positions[roots].mean(axis=1), axis=0)
    positions = positions.sub(positions.mean(axis=1), axis=0)
    gross = positions.abs().sum(axis=1).replace(0.0, np.nan)
    return positions.div(gross, axis=0).fillna(0.0)


def apply_rebalance(positions: pd.DataFrame, rebalance_bars: int) -> pd.DataFrame:
    if rebalance_bars <= 1:
        return positions.fillna(0.0)
    rebalanced = positions.copy() * np.nan
    rebalanced.iloc[::rebalance_bars] = positions.iloc[::rebalance_bars]
    return rebalanced.ffill().fillna(0.0)


def portfolio_returns(
    risk_positions: pd.DataFrame,
    trade_returns: pd.DataFrame,
    risk_sign: pd.Series,
    rebalance_bars: int,
    cost_bps_per_turnover: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trade_positions = risk_positions.multiply(risk_sign, axis=1)
    trade_positions = apply_rebalance(trade_positions, rebalance_bars)
    next_returns = trade_returns.shift(-1)
    gross_by_root = trade_positions * next_returns
    gross_return = gross_by_root.sum(axis=1)
    turnover = trade_positions.diff().abs().sum(axis=1).fillna(trade_positions.abs().sum(axis=1))
    cost_return = turnover * cost_bps_per_turnover / 10_000.0
    returns = pd.DataFrame(
        {
            "gross_return": gross_return,
            "cost_return": cost_return,
            "net_return": gross_return - cost_return,
            "turnover": turnover,
            "gross_exposure": trade_positions.abs().sum(axis=1),
        }
    ).dropna()
    return returns, trade_positions.reindex(returns.index), gross_by_root.reindex(returns.index)


def summarize_returns(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    if frame.empty:
        return {
            "label": label,
            "observations": 0,
            "total_gross_return": 0.0,
            "total_cost_return": 0.0,
            "total_net_return": 0.0,
            "gross_to_cost": None,
            "mean_net_bps": 0.0,
            "tstat": 0.0,
            "annualized_sharpe": 0.0,
            "hit_rate": 0.0,
            "max_drawdown": 0.0,
            "avg_turnover": 0.0,
            "avg_gross_exposure": 0.0,
            "active_fraction": 0.0,
        }
    gross = frame["gross_return"].fillna(0.0)
    cost = frame["cost_return"].fillna(0.0)
    net = frame["net_return"].fillna(0.0)
    std = float(net.std(ddof=1)) if len(net) > 1 else 0.0
    mean = float(net.mean()) if len(net) else 0.0
    equity = net.cumsum()
    drawdown = equity - equity.cummax()
    total_cost = float(cost.sum())
    return {
        "label": label,
        "observations": len(net),
        "total_gross_return": float(gross.sum()),
        "total_cost_return": total_cost,
        "total_net_return": float(net.sum()),
        "gross_to_cost": float(gross.sum() / total_cost) if total_cost > 0 else None,
        "mean_net_bps": mean * 10_000.0,
        "tstat": float(mean / (std / math.sqrt(len(net)))) if std > 0 else 0.0,
        "annualized_sharpe": float(mean / std * math.sqrt(PERIODS_PER_YEAR)) if std > 0 else 0.0,
        "hit_rate": float((net > 0.0).mean()),
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "avg_turnover": float(frame["turnover"].mean()),
        "avg_gross_exposure": float(frame["gross_exposure"].mean()),
        "active_fraction": float((frame["gross_exposure"] > 0.0).mean()),
    }


def split_dates(index: pd.DatetimeIndex, train_fraction: float) -> tuple[set[Any], set[Any]]:
    dates = sorted(pd.Series(index.date).unique())
    split = min(max(int(len(dates) * train_fraction), 1), len(dates) - 1)
    return set(dates[:split]), set(dates[split:])


def selection_sort_key(record: dict[str, Any]) -> tuple[float, float, float]:
    train = record["train"]
    return (
        float(train["tstat"]),
        float(train["total_net_return"]),
        float(train["gross_to_cost"] or -1e9),
    )


def select_variant(metrics: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    min_net = float(config["validation"]["min_train_net_return"])
    min_gc = float(config["validation"]["min_train_gross_to_cost"])
    eligible = [
        record
        for record in metrics
        if record["train"]["total_net_return"] > min_net
        and (record["train"]["gross_to_cost"] or -np.inf) >= min_gc
    ]
    return sorted(eligible or metrics, key=selection_sort_key, reverse=True)[0]


def decide(
    selected: dict[str, Any], control: dict[str, Any] | None, config: dict[str, Any]
) -> dict[str, Any]:
    rules = config["decision_rules"]
    test = selected["test"]
    beats_control = True
    if bool(rules["require_beats_ungated_control"]):
        beats_control = (
            control is not None and test["total_net_return"] > control["test"]["total_net_return"]
        )
    passed = (
        test["total_net_return"] > float(rules["min_test_net_return"])
        and (test["gross_to_cost"] or -np.inf) >= float(rules["min_test_gross_to_cost"])
        and test["tstat"] >= float(rules["min_test_tstat"])
        and beats_control
    )
    return {
        "status": rules["status_if_pass"] if passed else rules["status_if_fail"],
        "passed": bool(passed),
        "beats_ungated_control": bool(beats_control),
    }


def flatten_metrics(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        row = {
            "label": record["label"],
            "variant": record["variant"],
            "lookback_bars": record["lookback_bars"],
            "entry_z": record["entry_z"],
            "flow_z": record["flow_z"],
            "rebalance_bars": record["rebalance_bars"],
        }
        for split in ["train", "test", "full"]:
            for key, value in record[split].items():
                if key == "label":
                    continue
                row[f"{split}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def plot_equity(curves: dict[str, pd.Series], output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 5))
    for label, series in curves.items():
        axis.plot(series.index, series.cumsum() * 100.0, label=label)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_title("HYP-0023 Held-Out 5-Minute Cumulative Net Return")
    axis.set_ylabel("Cumulative net return (%)")
    axis.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run(config_path: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    out_dir = Path(config["outputs"]["directory"])
    out_dir.mkdir(parents=True, exist_ok=True)

    roots = list(config["universe"]["roots"])
    rates_roots = set(config["universe"]["rates_roots"])
    groups = dict(config["universe"]["groups"])
    trade_returns, logprices, flow, flow_volume, risk_sign, inventory = load_5m_panel(
        roots=roots,
        rates_roots=rates_roots,
        futures_dir=Path(config["data"]["futures_5m_dir"]),
        flow_dir=Path(config["data"]["flow_5m_dir"]),
        rth_start=str(config["data"]["rth_start"]),
        rth_end=str(config["data"]["rth_end"]),
        flow_feature=str(config["flow"]["feature"]),
    )
    risk_returns = trade_returns.multiply(risk_sign, axis=1)
    risk_prices = logprices.multiply(risk_sign, axis=1)
    risk_flow = flow.multiply(risk_sign, axis=1)
    regimes, mst_edges_frame = compute_regime_features(risk_returns, roots, config)
    flow_zscores = rolling_zscore(risk_flow, int(config["flow"]["zscore_window_bars"]))
    train_dates, test_dates = split_dates(
        pd.DatetimeIndex(trade_returns.index), float(config["validation"]["train_fraction_dates"])
    )

    inventory.to_csv(out_dir / "data_inventory.csv", index=False)
    regimes.to_csv(out_dir / "regime_features.csv")
    mst_edges_frame.to_csv(out_dir / "mst_edges.csv", index=False)
    flow_volume.sum().rename("total_flow_volume").to_csv(out_dir / "flow_volume_by_root.csv")

    metrics: list[dict[str, Any]] = []
    returns_by_label: dict[str, pd.DataFrame] = {}
    positions_by_label: dict[str, pd.DataFrame] = {}
    gross_by_root_by_label: dict[str, pd.DataFrame] = {}
    zscores_by_lookback: dict[int, pd.DataFrame] = {}

    for lookback_value in config["residual_model"]["lookback_bars"]:
        lookback = int(lookback_value)
        residual_zscores = rolling_residual_zscores(
            risk_prices=risk_prices,
            roots=roots,
            lookback=lookback,
            zscore_window=int(config["residual_model"]["zscore_window_bars"]),
            ridge=float(config["residual_model"]["ridge"]),
        )
        zscores_by_lookback[lookback] = residual_zscores
        for entry_value in config["strategy"]["entry_z"]:
            entry_z = float(entry_value)
            for flow_value in config["flow"]["flow_z"]:
                flow_threshold = float(flow_value)
                for variant in config["strategy"]["variants"]:
                    signal = apply_variant_policy(
                        residual_z=residual_zscores,
                        flow_z=flow_zscores,
                        regimes=regimes,
                        variant=str(variant),
                        entry_z=entry_z,
                        flow_threshold=flow_threshold,
                        max_abs=float(config["strategy"]["max_signal_abs"]),
                    )
                    risk_positions = neutralized_risk_positions(signal, groups)
                    for rebalance_value in config["strategy"]["rebalance_bars"]:
                        rebalance = int(rebalance_value)
                        label = (
                            f"{variant}_lb{lookback}_z{entry_z:g}"
                            f"_fz{flow_threshold:g}_rb{rebalance}"
                        )
                        returns, positions, gross_by_root = portfolio_returns(
                            risk_positions=risk_positions,
                            trade_returns=trade_returns,
                            risk_sign=risk_sign,
                            rebalance_bars=rebalance,
                            cost_bps_per_turnover=float(
                                config["strategy"]["cost_bps_per_unit_turnover"]
                            ),
                        )
                        date_values = pd.DatetimeIndex(returns.index).date
                        train_returns = returns[
                            pd.Series(date_values, index=returns.index).isin(train_dates)
                        ]
                        test_returns = returns[
                            pd.Series(date_values, index=returns.index).isin(test_dates)
                        ]
                        record = {
                            "label": label,
                            "variant": str(variant),
                            "lookback_bars": lookback,
                            "entry_z": entry_z,
                            "flow_z": flow_threshold,
                            "rebalance_bars": rebalance,
                            "train": summarize_returns(train_returns, f"{label}_train"),
                            "test": summarize_returns(test_returns, f"{label}_test"),
                            "full": summarize_returns(returns, f"{label}_full"),
                        }
                        metrics.append(record)
                        returns_by_label[label] = returns
                        positions_by_label[label] = positions
                        gross_by_root_by_label[label] = gross_by_root

    selected = select_variant(metrics, config)
    selected_label = str(selected["label"])
    control_candidates = [
        record
        for record in metrics
        if record["variant"] == "ungated_residual_reversion"
        and record["lookback_bars"] == selected["lookback_bars"]
        and record["entry_z"] == selected["entry_z"]
        and record["flow_z"] == selected["flow_z"]
        and record["rebalance_bars"] == selected["rebalance_bars"]
    ]
    control = control_candidates[0] if control_candidates else None
    decision = decide(selected, control, config)

    variant_metrics = flatten_metrics(metrics).sort_values(
        ["train_tstat", "train_total_net_return"], ascending=False
    )
    variant_metrics.to_csv(out_dir / "variant_metrics.csv", index=False)

    selected_returns = returns_by_label[selected_label]
    selected_positions = positions_by_label[selected_label]
    selected_gross_by_root = gross_by_root_by_label[selected_label]
    selected_returns.to_csv(out_dir / "selected_portfolio_returns.csv")
    selected_positions.to_parquet(out_dir / "selected_positions.parquet")
    selected_gross_by_root.sum().rename("gross_return_contribution").to_csv(
        out_dir / "selected_root_gross_contributions.csv"
    )
    zscores_by_lookback[int(selected["lookback_bars"])].to_parquet(
        out_dir / "selected_zscores.parquet"
    )

    selected_dates = pd.DatetimeIndex(selected_returns.index).date
    selected_test = selected_returns[
        pd.Series(selected_dates, index=selected_returns.index).isin(test_dates)
    ]
    daily_index = pd.DatetimeIndex(selected_test.index).tz_convert(None).to_period("D")
    selected_test["net_return"].groupby(daily_index).sum().to_csv(
        out_dir / "selected_daily_test_returns.csv",
        header=["net_return"],
    )
    curves = {"selected": selected_test["net_return"]}
    if control is not None:
        control_returns = returns_by_label[str(control["label"])]
        control_dates = pd.DatetimeIndex(control_returns.index).date
        curves["ungated_control"] = control_returns[
            pd.Series(control_dates, index=control_returns.index).isin(test_dates)
        ]["net_return"]
    plot_equity(curves, out_dir / "selected_test_equity.png")

    result = {
        "experiment_id": config["experiment"]["id"],
        "title": config["experiment"]["title"],
        "completed_at": datetime.now().astimezone().isoformat(),
        "decision": decision,
        "data": {
            "roots": roots,
            "start": str(trade_returns.index.min()),
            "end": str(trade_returns.index.max()),
            "observations": len(trade_returns),
            "train_dates": len(train_dates),
            "test_dates": len(test_dates),
            "stable_fraction": safe_float(regimes["stable"].mean()),
            "transition_fraction": safe_float(regimes["transition"].mean()),
        },
        "selected": selected,
        "ungated_control": control,
        "artifacts": {
            "data_inventory": str(out_dir / "data_inventory.csv"),
            "flow_volume_by_root": str(out_dir / "flow_volume_by_root.csv"),
            "regime_features": str(out_dir / "regime_features.csv"),
            "mst_edges": str(out_dir / "mst_edges.csv"),
            "variant_metrics": str(out_dir / "variant_metrics.csv"),
            "selected_portfolio_returns": str(out_dir / "selected_portfolio_returns.csv"),
            "selected_positions": str(out_dir / "selected_positions.parquet"),
            "selected_zscores": str(out_dir / "selected_zscores.parquet"),
            "selected_root_gross_contributions": str(
                out_dir / "selected_root_gross_contributions.csv"
            ),
            "selected_daily_test_returns": str(out_dir / "selected_daily_test_returns.csv"),
            "selected_test_equity": str(out_dir / "selected_test_equity.png"),
        },
    }
    write_json(out_dir / "results.json", result)
    return result


def main() -> None:
    args = parse_args()
    result = run(args.config)
    print(json.dumps({"decision": result["decision"], "selected": result["selected"]}, indent=2))


if __name__ == "__main__":
    main()
