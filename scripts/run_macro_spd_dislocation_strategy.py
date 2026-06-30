# ruff: noqa: I001, PLR0915
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
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_CONFIG = Path("experiments/HYP-0022-macro-futures-spd-dislocation/config.yaml")
MIN_GROUP_NEUTRAL_ROOTS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HYP-0022 macro SPD dislocation test.")
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


def load_daily_continuous(
    roots: list[str],
    rates_roots: set[str],
    continuous_dir: Path,
    start: str | None,
    end: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    trade_returns: dict[str, pd.Series] = {}
    log_prices: dict[str, pd.Series] = {}
    inventory_rows: list[dict[str, Any]] = []

    for root in roots:
        path = continuous_dir / f"{root}.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing daily continuous file for {root}: {path}")
        frame = pd.read_csv(path, parse_dates=["date"])
        frame["session_date"] = frame["date"].dt.date
        frame = frame.set_index("session_date").sort_index()
        trade_returns[root] = frame["cont_logret"].astype(float)
        log_prices[root] = frame["cont_logprice"].astype(float)
        inventory_rows.append(
            {
                "root": root,
                "rows": len(frame),
                "first_date": str(frame.index.min()),
                "last_date": str(frame.index.max()),
                "roll_days": int(frame["is_roll"].fillna(False).sum()),
                "active_contracts": int(frame["active"].nunique()),
            }
        )

    trade = pd.DataFrame(trade_returns).replace([np.inf, -np.inf], np.nan)
    prices = pd.DataFrame(log_prices).replace([np.inf, -np.inf], np.nan).ffill()

    if start:
        start_date = pd.Timestamp(start).date()
        trade = trade[trade.index >= start_date]
        prices = prices[prices.index >= start_date]
    if end:
        end_date = pd.Timestamp(end).date()
        trade = trade[trade.index <= end_date]
        prices = prices[prices.index <= end_date]

    trade = trade.dropna(how="any")
    prices = prices.reindex(trade.index).dropna(how="any")
    trade = trade.reindex(prices.index)

    risk_sign = pd.Series({root: -1.0 if root in rates_roots else 1.0 for root in roots})
    risk_returns = trade.multiply(risk_sign, axis=1)
    risk_prices = prices.multiply(risk_sign, axis=1)
    return trade, risk_returns, risk_prices, risk_sign, pd.DataFrame(inventory_rows)


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
    lookback = int(config["regime"]["covariance_lookback_days"])
    q_lookback = int(config["regime"]["quantile_lookback_days"])
    shrinkage_alpha = float(config["regime"]["shrinkage_alpha"])
    eigen_floor = float(config["regime"]["eigen_floor"])

    rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    previous_log_corr: np.ndarray | None = None
    previous_edges: frozenset[tuple[str, str]] | None = None

    for t_index, date in enumerate(risk_returns.index):
        if t_index < lookback:
            rows.append(
                {
                    "date": date,
                    "spd_velocity": np.nan,
                    "top_eigen_share": np.nan,
                    "effective_rank": np.nan,
                    "average_correlation": np.nan,
                    "edge_persistence": np.nan,
                }
            )
            continue

        window = risk_returns.iloc[t_index - lookback : t_index].to_numpy(dtype=float)
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
                "date": date,
                "spd_velocity": velocity,
                "top_eigen_share": top_eigen,
                "effective_rank": effective_rank,
                "average_correlation": avg_corr,
                "edge_persistence": edge_persistence,
            }
        )
        for root_a, root_b in edges:
            edge_rows.append({"date": date, "root_a": root_a, "root_b": root_b})
        previous_log_corr = log_corr
        previous_edges = edges

    features = pd.DataFrame(rows).set_index("date")
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
    factors = pd.DataFrame(index=risk_prices.index)
    rates = ["ZT", "ZF", "ZN", "ZB"]
    metals = ["GC", "SI", "HG", "PL", "PA"]
    factors["market"] = risk_prices.mean(axis=1)
    factors["rates_level"] = risk_prices[rates].mean(axis=1)
    factors["rates_slope"] = risk_prices[["ZN", "ZB"]].mean(axis=1) - risk_prices[
        ["ZT", "ZF"]
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

    for t_index in range(lookback, len(risk_prices)):
        x_window = np.column_stack([np.ones(lookback), x_all[t_index - lookback : t_index]])
        x_current = np.column_stack([np.ones(1), x_all[t_index : t_index + 1]])
        x_valid = np.isfinite(x_window).all(axis=1)
        if x_valid.sum() < lookback // 2:
            continue
        for root_index in range(len(roots)):
            y_window = y_all[t_index - lookback : t_index, root_index]
            valid = x_valid & np.isfinite(y_window)
            if valid.sum() < lookback // 2:
                continue
            x_fit = x_window[valid]
            y_fit = y_window[valid]
            penalty = np.eye(x_fit.shape[1]) * ridge
            penalty[0, 0] = 0.0
            beta = np.linalg.solve(x_fit.T @ x_fit + penalty, x_fit.T @ y_fit)
            fitted = float((x_current @ beta)[0])
            residuals[t_index, root_index] = y_all[t_index, root_index] - fitted

    residual_frame = pd.DataFrame(residuals, index=risk_prices.index, columns=roots)
    rolling_mean = residual_frame.rolling(
        zscore_window, min_periods=max(5, zscore_window // 2)
    ).mean()
    rolling_std = residual_frame.rolling(
        zscore_window, min_periods=max(5, zscore_window // 2)
    ).std()
    return (residual_frame - rolling_mean) / rolling_std


def base_reversion_signal(zscores: pd.DataFrame, entry_z: float, max_abs: float) -> pd.DataFrame:
    values = zscores.to_numpy(dtype=float)
    signal = np.where(
        np.abs(values) >= entry_z,
        -np.sign(values) * np.minimum(np.abs(values) / entry_z, max_abs),
        0.0,
    )
    return pd.DataFrame(signal, index=zscores.index, columns=zscores.columns)


def apply_variant_policy(
    base_signal: pd.DataFrame,
    regimes: pd.DataFrame,
    variant: str,
) -> pd.DataFrame:
    stable = regimes["stable"].reindex(base_signal.index).fillna(False)
    transition = regimes["transition"].reindex(base_signal.index).fillna(False)
    if variant == "ungated_reversion":
        return base_signal
    if variant == "stable_reversion":
        return base_signal.where(stable, 0.0)
    if variant == "non_transition_reversion":
        return base_signal.where(~transition, 0.0)
    if variant == "stable_reversion_transition_momentum":
        out = base_signal.copy() * 0.0
        out = out.where(~stable, base_signal)
        out = out.where(~transition, -base_signal)
        return out
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


def apply_rebalance(positions: pd.DataFrame, rebalance_days: int) -> pd.DataFrame:
    rebalanced = positions.copy() * np.nan
    rebalanced.iloc[::rebalance_days] = positions.iloc[::rebalance_days]
    return rebalanced.ffill().fillna(0.0)


def portfolio_returns(
    risk_positions: pd.DataFrame,
    trade_returns: pd.DataFrame,
    risk_sign: pd.Series,
    rebalance_days: int,
    cost_bps_per_turnover: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trade_positions = risk_positions.multiply(risk_sign, axis=1)
    trade_positions = apply_rebalance(trade_positions, rebalance_days)
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


def summarize_returns(
    frame: pd.DataFrame, label: str, periods_per_year: float = 252.0
) -> dict[str, Any]:
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
        "annualized_sharpe": float(mean / std * math.sqrt(periods_per_year)) if std > 0 else 0.0,
        "hit_rate": float((net > 0.0).mean()),
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "avg_turnover": float(frame["turnover"].mean()),
        "avg_gross_exposure": float(frame["gross_exposure"].mean()),
        "active_fraction": float((frame["gross_exposure"] > 0.0).mean()),
    }


def split_train_test(index: pd.Index, train_fraction: float) -> tuple[pd.Index, pd.Index]:
    split = min(max(int(len(index) * train_fraction), 1), len(index) - 1)
    return index[:split], index[split:]


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
    candidates = eligible or metrics
    return sorted(candidates, key=selection_sort_key, reverse=True)[0]


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


def plot_equity(curves: dict[str, pd.Series], output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 5))
    for label, series in curves.items():
        axis.plot(pd.to_datetime(series.index), series.cumsum() * 100.0, label=label)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_title("HYP-0022 Held-Out Cumulative Net Return")
    axis.set_ylabel("Cumulative net return (%)")
    axis.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def flatten_metrics(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        row = {
            "variant": record["variant"],
            "lookback_days": record["lookback_days"],
            "entry_z": record["entry_z"],
            "rebalance_days": record["rebalance_days"],
        }
        for split in ["train", "test", "full"]:
            for key, value in record[split].items():
                if key == "label":
                    continue
                row[f"{split}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def run(config_path: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    out_dir = Path(config["outputs"]["directory"])
    out_dir.mkdir(parents=True, exist_ok=True)

    roots = list(config["universe"]["roots"])
    rates_roots = set(config["universe"]["rates_roots"])
    groups = dict(config["universe"]["groups"])
    trade_returns, risk_returns, risk_prices, risk_sign, inventory = load_daily_continuous(
        roots=roots,
        rates_roots=rates_roots,
        continuous_dir=Path(config["data"]["continuous_dir"]),
        start=config["data"].get("start"),
        end=config["data"].get("end"),
    )
    regimes, mst_edge_history = compute_regime_features(risk_returns, roots, config)
    train_dates, test_dates = split_train_test(
        trade_returns.index, float(config["validation"]["train_fraction"])
    )

    inventory.to_csv(out_dir / "data_inventory.csv", index=False)
    regimes.to_csv(out_dir / "regime_features.csv")
    mst_edge_history.to_csv(out_dir / "mst_edges.csv", index=False)

    metrics: list[dict[str, Any]] = []
    returns_by_variant: dict[str, pd.DataFrame] = {}
    positions_by_variant: dict[str, pd.DataFrame] = {}
    gross_by_root_by_variant: dict[str, pd.DataFrame] = {}
    zscore_by_lookback: dict[int, pd.DataFrame] = {}

    for lookback in config["residual_model"]["lookback_days"]:
        lookback_days = int(lookback)
        zscores = rolling_residual_zscores(
            risk_prices=risk_prices,
            roots=roots,
            lookback=lookback_days,
            zscore_window=int(config["residual_model"]["zscore_window_days"]),
            ridge=float(config["residual_model"]["ridge"]),
        )
        zscore_by_lookback[lookback_days] = zscores
        for entry_z in config["strategy"]["entry_z"]:
            entry = float(entry_z)
            base_signal = base_reversion_signal(
                zscores,
                entry_z=entry,
                max_abs=float(config["strategy"]["max_signal_abs"]),
            )
            for variant in config["strategy"]["variants"]:
                signal = apply_variant_policy(base_signal, regimes, str(variant))
                risk_positions = neutralized_risk_positions(signal, groups)
                for rebalance_days in config["strategy"]["rebalance_days"]:
                    rebalance = int(rebalance_days)
                    label = f"{variant}_lb{lookback_days}_z{entry:g}_rb{rebalance}"
                    returns, positions, gross_by_root = portfolio_returns(
                        risk_positions=risk_positions,
                        trade_returns=trade_returns,
                        risk_sign=risk_sign,
                        rebalance_days=rebalance,
                        cost_bps_per_turnover=float(
                            config["strategy"]["cost_bps_per_unit_turnover"]
                        ),
                    )
                    train_returns = returns[returns.index.isin(train_dates)]
                    test_returns = returns[returns.index.isin(test_dates)]
                    record = {
                        "label": label,
                        "variant": str(variant),
                        "lookback_days": lookback_days,
                        "entry_z": entry,
                        "rebalance_days": rebalance,
                        "train": summarize_returns(train_returns, f"{label}_train"),
                        "test": summarize_returns(test_returns, f"{label}_test"),
                        "full": summarize_returns(returns, f"{label}_full"),
                    }
                    metrics.append(record)
                    returns_by_variant[label] = returns
                    positions_by_variant[label] = positions
                    gross_by_root_by_variant[label] = gross_by_root

    selected = select_variant(metrics, config)
    selected_label = str(selected["label"])
    control_candidates = [
        record
        for record in metrics
        if record["variant"] == "ungated_reversion"
        and record["lookback_days"] == selected["lookback_days"]
        and record["entry_z"] == selected["entry_z"]
        and record["rebalance_days"] == selected["rebalance_days"]
    ]
    control = control_candidates[0] if control_candidates else None
    decision = decide(selected, control, config)

    variant_metrics = flatten_metrics(metrics).sort_values(
        ["train_tstat", "train_total_net_return"], ascending=False
    )
    variant_metrics.to_csv(out_dir / "variant_metrics.csv", index=False)

    selected_returns = returns_by_variant[selected_label]
    selected_positions = positions_by_variant[selected_label]
    selected_gross_by_root = gross_by_root_by_variant[selected_label]
    selected_returns.to_csv(out_dir / "selected_portfolio_returns.csv")
    selected_positions.to_parquet(out_dir / "selected_positions.parquet")
    selected_gross_by_root.sum().rename("gross_return_contribution").to_csv(
        out_dir / "selected_root_gross_contributions.csv"
    )
    selected_test = selected_returns[selected_returns.index.isin(test_dates)]
    selected_test["net_return"].groupby(
        pd.to_datetime(selected_test.index).to_period("M")
    ).sum().to_csv(
        out_dir / "selected_monthly_test_returns.csv",
        header=["net_return"],
    )
    zscore_by_lookback[int(selected["lookback_days"])].to_parquet(
        out_dir / "selected_zscores.parquet"
    )

    curves = {"selected": selected_test["net_return"]}
    if control is not None:
        control_returns = returns_by_variant[str(control["label"])]
        curves["ungated_control"] = control_returns[control_returns.index.isin(test_dates)][
            "net_return"
        ]
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
            "train_observations": len(train_dates),
            "test_observations": len(test_dates),
            "stable_fraction": safe_float(regimes["stable"].mean()),
            "transition_fraction": safe_float(regimes["transition"].mean()),
        },
        "selected": selected,
        "ungated_control": control,
        "artifacts": {
            "data_inventory": str(out_dir / "data_inventory.csv"),
            "regime_features": str(out_dir / "regime_features.csv"),
            "mst_edges": str(out_dir / "mst_edges.csv"),
            "variant_metrics": str(out_dir / "variant_metrics.csv"),
            "selected_portfolio_returns": str(out_dir / "selected_portfolio_returns.csv"),
            "selected_positions": str(out_dir / "selected_positions.parquet"),
            "selected_zscores": str(out_dir / "selected_zscores.parquet"),
            "selected_root_gross_contributions": str(
                out_dir / "selected_root_gross_contributions.csv"
            ),
            "selected_monthly_test_returns": str(out_dir / "selected_monthly_test_returns.csv"),
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
