# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportReturnType=false
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from quantlab.metals_flow.forward import future_returns


@dataclass(frozen=True)
class DataBundle:
    bars: pd.DataFrame
    returns: pd.DataFrame
    residuals: pd.DataFrame
    features: pd.DataFrame
    anomalies: pd.DataFrame
    size_disagreement: pd.DataFrame
    valid_mask: np.ndarray
    costs_bps: pd.Series
    periods_per_year: float
    split_index: int
    test_start_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen and validate PA residual lead-lag metals patterns."
    )
    parser.add_argument("config", type=Path, help="Experiment config YAML.")
    return parser.parse_args()


def main() -> None:
    config_path = parse_args().config
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    out_dir = Path(payload["outputs"]["directory"])
    out_dir.mkdir(parents=True, exist_ok=True)
    run(payload, out_dir)


def run(payload: dict[str, Any], out_dir: Path) -> None:
    roots = tuple(payload["universe"]["roots"])
    data = load_data(payload, roots)
    signals = build_signal_panel(payload, data, roots)
    screen = screen_candidates(payload, data, signals, roots)
    candidates = select_candidates(payload, screen)
    variants = backtest_candidates(data, signals, candidates, roots)
    selected_returns, selected_positions, selected_metrics = backtest_selected(
        payload, data, signals, roots
    )
    split_metrics = split_selected_metrics(data, selected_returns)
    monthly = monthly_selected_returns(data.bars["end_ts"], selected_returns)
    bootstrap = daily_block_bootstrap(
        data.bars["end_ts"], selected_returns["net_return"].to_numpy(dtype=float)
    )
    event_stats = selected_event_stats(payload, data, signals, roots)
    decision = make_decision(payload, selected_metrics, split_metrics)

    screen.to_csv(out_dir / "candidate_screen.csv", index=False)
    candidates.to_csv(out_dir / "selected_screen_candidates.csv", index=False)
    variants.to_csv(out_dir / "candidate_backtests.csv", index=False)
    selected_returns.to_csv(out_dir / "selected_strategy_returns.csv", index=False)
    selected_positions.to_csv(out_dir / "selected_strategy_positions.csv", index=False)
    split_metrics.to_csv(out_dir / "selected_split_metrics.csv", index=False)
    monthly.to_csv(out_dir / "selected_monthly_returns.csv", index=False)
    bootstrap.to_csv(out_dir / "selected_daily_bootstrap.csv", index=False)
    event_stats.to_csv(out_dir / "selected_event_stats.csv", index=False)

    summary = {
        "experiment_id": payload["experiment"]["id"],
        "title": payload["experiment"]["title"],
        "completed_at": datetime.now(UTC).isoformat(),
        "decision": decision,
        "data": {
            "source_experiment": payload["data"]["source_experiment"],
            "bars": len(data.bars),
            "start": str(data.bars["end_ts"].iloc[0]),
            "end": str(data.bars["end_ts"].iloc[-1]),
            "split_index": int(data.split_index),
            "test_start_index": int(data.test_start_index),
            "test_start_ts": str(data.bars["end_ts"].iloc[data.test_start_index]),
            "valid_price_fraction": float(data.valid_mask.mean()),
        },
        "selected_strategy": payload["selected_strategy"],
        "selected_metrics": selected_metrics,
        "split_metrics": split_metrics.to_dict(orient="records"),
        "daily_bootstrap": bootstrap.to_dict(orient="records"),
        "event_stats": event_stats.to_dict(orient="records"),
        "top_train_screen": candidates.head(20).to_dict(orient="records"),
        "top_costed_candidates": variants.sort_values(
            ["test_net_return", "test_gross_to_cost"], ascending=False
        )
        .head(20)
        .to_dict(orient="records"),
    }
    (out_dir / "results.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(f"wrote {out_dir / 'results.json'}", flush=True)


def load_data(payload: dict[str, Any], roots: tuple[str, ...]) -> DataBundle:
    source = Path(payload["data"]["source_experiment"])
    bars = pd.read_parquet(source / "primary_bars.parquet")
    bars["end_ts"] = pd.to_datetime(bars["end_ts"], utc=True)
    returns = pd.read_parquet(source / "bar_returns.parquet").reindex(columns=roots)
    returns = returns.fillna(0.0)
    residuals = pd.read_parquet(source / "fair_value_zscores.parquet").reindex(columns=roots)
    features = pd.read_parquet(source / "flow_features.parquet")
    anomalies = pd.read_parquet(source / "flow_anomalies.parquet")
    size_disagreement = pd.read_parquet(source / "trade_size_disagreement.parquet")
    valid_mask = (
        pd.read_parquet(source / "price_validity.parquet")["valid_price_mask"]
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )
    costs = pd.read_csv(payload["data"]["cost_estimates"])
    costs_bps = costs.set_index("root")["per_side_cost_bps"].reindex(roots).astype(float)
    elapsed_years = (bars["end_ts"].iloc[-1] - bars["end_ts"].iloc[0]).total_seconds() / (
        365.25 * 24 * 60 * 60
    )
    periods_per_year = len(bars) / elapsed_years
    split_index = int(len(bars) * float(payload["research"]["train_fraction"]))
    test_start_index = min(split_index + int(payload["research"]["embargo_bars"]), len(bars))
    return DataBundle(
        bars=bars,
        returns=returns,
        residuals=residuals,
        features=features,
        anomalies=anomalies,
        size_disagreement=size_disagreement,
        valid_mask=valid_mask,
        costs_bps=costs_bps,
        periods_per_year=periods_per_year,
        split_index=split_index,
        test_start_index=test_start_index,
    )


def build_signal_panel(
    payload: dict[str, Any], data: DataBundle, roots: tuple[str, ...]
) -> dict[str, np.ndarray]:
    research = payload["research"]
    window = int(research["rolling_z_window"])
    min_periods = int(research["rolling_z_min_periods"])
    signals: dict[str, np.ndarray] = {}
    feature_names = (
        "complex_signed_notional_ratio",
        "abs_complex_signed_notional_ratio",
        "entropy_normalized",
        "hhi",
        "distance_from_equal_weight",
        "contribution_velocity_l2",
        "rank_turnover_l1",
        "max_share",
    )
    for name in feature_names:
        if name in data.features:
            signals[f"feat:{name}"] = rolling_z(
                data.features[name], window=window, min_periods=min_periods
            )
    for name in ("large_small_l1_distance", "very_large_small_l1_distance"):
        signals[f"size:{name}"] = rolling_z(
            data.size_disagreement[name], window=window, min_periods=min_periods
        )
    for root in roots:
        signals[f"resid:{root}"] = data.residuals[root].to_numpy(dtype=float)
        for suffix in ("notional_share", "signed_notional_share", "abs_signed_notional_share"):
            name = f"{root}_{suffix}"
            if name in data.features:
                signals[f"feat:{name}"] = rolling_z(
                    data.features[name], window=window, min_periods=min_periods
                )
        name = f"{root}_large_minus_small_share"
        if name in data.size_disagreement:
            signals[f"size:{name}"] = rolling_z(
                data.size_disagreement[name], window=window, min_periods=min_periods
            )
    for name in ("md_rolling", "md_ewma", "md_robust_snapshot"):
        if name in data.anomalies:
            signals[f"anom:{name}"] = rolling_z(
                data.anomalies[name], window=window, min_periods=min_periods
            )
    return signals


def rolling_z(series: pd.Series, *, window: int, min_periods: int) -> np.ndarray:
    values = series.astype(float).replace([np.inf, -np.inf], np.nan)
    mean = values.rolling(window, min_periods=min_periods).mean().shift(1)
    std = values.rolling(window, min_periods=min_periods).std(ddof=1).shift(1)
    return ((values - mean) / std.replace(0.0, np.nan)).to_numpy(dtype=float)


def screen_candidates(
    payload: dict[str, Any],
    data: DataBundle,
    signals: dict[str, np.ndarray],
    roots: tuple[str, ...],
) -> pd.DataFrame:
    horizons = tuple(int(value) for value in payload["research"]["horizons"])
    future = future_returns(data.returns, horizons)
    rows: list[dict[str, Any]] = []
    min_train_events = int(payload["research"]["min_train_events"])
    for signal_name, signal in signals.items():
        sign = np.sign(signal)
        finite = np.isfinite(signal)
        for entry in tuple(float(value) for value in payload["research"]["entry_thresholds"]):
            mask = (np.abs(signal) >= entry) & finite & data.valid_mask
            train_mask = mask[: data.split_index]
            if int(train_mask.sum()) < min_train_events:
                continue
            for target in roots:
                for horizon in horizons:
                    values = (
                        sign
                        * future[(target, horizon)]
                        .replace([np.inf, -np.inf], np.nan)
                        .to_numpy(dtype=float)
                    )[: data.split_index][train_mask]
                    values = values[np.isfinite(values)]
                    if len(values) < min_train_events:
                        continue
                    mean = float(values.mean())
                    std = float(values.std(ddof=1))
                    tstat = mean / (std / math.sqrt(len(values))) if std > 0.0 else np.nan
                    roundtrip_bps = 2.0 * float(data.costs_bps[target])
                    abs_mean_bps = abs(mean) * 10_000.0
                    rows.append(
                        {
                            "signal": signal_name,
                            "entry": entry,
                            "target": target,
                            "horizon": horizon,
                            "events": len(values),
                            "direction": 1.0 if mean >= 0.0 else -1.0,
                            "train_event_mean_bps": abs_mean_bps,
                            "train_event_t": abs(tstat),
                            "roundtrip_bps": roundtrip_bps,
                            "event_gross_to_roundtrip_cost": abs_mean_bps / roundtrip_bps
                            if roundtrip_bps > 0.0
                            else np.inf,
                        }
                    )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["screen_score"] = frame["train_event_t"] * np.minimum(
        frame["event_gross_to_roundtrip_cost"], 5.0
    )
    return frame.sort_values("screen_score", ascending=False).reset_index(drop=True)


def select_candidates(payload: dict[str, Any], screen: pd.DataFrame) -> pd.DataFrame:
    if screen.empty:
        return screen
    filtered = screen.loc[
        screen["train_event_t"] >= float(payload["research"]["min_train_event_t"])
    ].copy()
    return filtered.head(int(payload["research"]["candidate_limit"])).reset_index(drop=True)


def backtest_candidates(
    data: DataBundle,
    signals: dict[str, np.ndarray],
    candidates: pd.DataFrame,
    roots: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    root_index = {root: i for i, root in enumerate(roots)}
    returns = data.returns.to_numpy(dtype=float)
    costs = data.costs_bps.to_numpy(dtype=float)
    for record in candidates.itertuples(index=False):
        positions = event_hold_positions(
            signals[record.signal],
            data.valid_mask,
            roots,
            root_index,
            {str(record.target): 1.0},
            entry=float(record.entry),
            hold_bars=int(record.horizon),
            direction=float(record.direction),
        )
        metrics = {
            "full": metrics_for_slice(positions, returns, costs, data.periods_per_year),
            "train": metrics_for_slice(
                positions, returns, costs, data.periods_per_year, end=data.split_index
            ),
            "test": metrics_for_slice(
                positions,
                returns,
                costs,
                data.periods_per_year,
                start=data.test_start_index,
            ),
        }
        row = record._asdict()
        for prefix, values in metrics.items():
            row.update({f"{prefix}_{key}": value for key, value in values.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def backtest_selected(
    payload: dict[str, Any],
    data: DataBundle,
    signals: dict[str, np.ndarray],
    roots: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    selected = payload["selected_strategy"]
    root_index = {root: i for i, root in enumerate(roots)}
    positions = event_hold_positions(
        signals[selected["signal"]],
        data.valid_mask,
        roots,
        root_index,
        {str(root): float(weight) for root, weight in selected["targets"].items()},
        entry=float(selected["entry"]),
        hold_bars=int(selected["hold_bars"]),
        direction=float(selected["direction"]),
    )
    returns = data.returns.to_numpy(dtype=float)
    costs = data.costs_bps.to_numpy(dtype=float)
    frame = returns_frame(positions, returns, costs)
    position_frame = pd.DataFrame(positions, columns=roots)
    metrics = metrics_for_slice(positions, returns, costs, data.periods_per_year)
    return frame, position_frame, metrics


def event_hold_positions(
    signal: np.ndarray,
    valid_mask: np.ndarray,
    roots: tuple[str, ...],
    root_index: dict[str, int],
    targets: dict[str, float],
    *,
    entry: float,
    hold_bars: int,
    direction: float,
) -> np.ndarray:
    index = np.arange(len(signal))
    event = np.isfinite(signal) & (np.abs(signal) >= entry) & valid_mask
    event_value = np.where(event, direction * np.sign(signal), np.nan)
    event_index = np.where(event, index, np.nan)
    last_value = pd.Series(event_value).ffill().to_numpy(dtype=float)
    last_index = pd.Series(event_index).ffill().to_numpy(dtype=float)
    active = np.isfinite(last_index) & ((index - last_index) < hold_bars) & valid_mask
    side = np.where(active, last_value, 0.0)
    positions = np.zeros((len(signal), len(roots)), dtype=float)
    for root, weight in targets.items():
        positions[:, root_index[root]] = side * weight
    return positions


def returns_frame(
    positions: np.ndarray, returns: np.ndarray, costs_bps: np.ndarray
) -> pd.DataFrame:
    previous = np.vstack([np.zeros(positions.shape[1]), positions[:-1]])
    gross_by_root = previous * returns
    turnover_by_root = np.abs(positions - previous)
    cost_by_root = turnover_by_root * costs_bps / 10_000.0
    gross = gross_by_root.sum(axis=1)
    cost = cost_by_root.sum(axis=1)
    return pd.DataFrame(
        {
            "gross_return": gross,
            "cost_return": cost,
            "net_return": gross - cost,
            "turnover": turnover_by_root.sum(axis=1),
            "active": np.abs(previous).sum(axis=1) > 0.0,
        }
    )


def metrics_for_slice(
    positions: np.ndarray,
    returns: np.ndarray,
    costs_bps: np.ndarray,
    periods_per_year: float,
    *,
    start: int = 0,
    end: int | None = None,
) -> dict[str, Any]:
    if end is None:
        end = len(returns)
    segment_positions = positions[start:end]
    previous = np.vstack(
        [
            positions[start - 1] if start > 0 else np.zeros(positions.shape[1]),
            segment_positions[:-1],
        ]
    )
    segment_returns = returns[start:end]
    gross = (previous * segment_returns).sum(axis=1)
    turnover_by_root = np.abs(segment_positions - previous)
    cost = (turnover_by_root * costs_bps / 10_000.0).sum(axis=1)
    net = gross - cost
    observations = len(net)
    mean = float(net.mean()) if observations else 0.0
    std = float(net.std(ddof=1)) if observations > 1 else 0.0
    equity = np.cumsum(net)
    drawdown = equity - np.maximum.accumulate(equity) if observations else np.array([0.0])
    cost_sum = float(cost.sum())
    gross_sum = float(gross.sum())
    return {
        "observations": observations,
        "active_bars": int((np.abs(previous).sum(axis=1) > 0.0).sum()),
        "gross_return": gross_sum,
        "cost_return": cost_sum,
        "net_return": float(net.sum()),
        "mean_net_bps": mean * 10_000.0,
        "tstat": mean / (std / math.sqrt(observations)) if std > 0.0 else np.nan,
        "annualized_sharpe": mean / std * math.sqrt(periods_per_year) if std > 0.0 else np.nan,
        "hit_rate": float((net > 0.0).mean()) if observations else np.nan,
        "max_drawdown": float(drawdown.min()) if observations else np.nan,
        "gross_to_cost": gross_sum / cost_sum if cost_sum > 0.0 else np.inf,
        "turnover": float(turnover_by_root.sum()),
    }


def split_selected_metrics(data: DataBundle, returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, start, end in (
        ("train", 0, data.split_index),
        ("test_purged", data.test_start_index, len(returns)),
    ):
        frame = returns.iloc[start:end]
        rows.append({"split": label, **metrics_from_returns_frame(frame, data.periods_per_year)})
    return pd.DataFrame(rows)


def metrics_from_returns_frame(frame: pd.DataFrame, periods_per_year: float) -> dict[str, Any]:
    net = frame["net_return"].to_numpy(dtype=float)
    gross = frame["gross_return"].to_numpy(dtype=float)
    cost = frame["cost_return"].to_numpy(dtype=float)
    observations = len(net)
    mean = float(net.mean()) if observations else 0.0
    std = float(net.std(ddof=1)) if observations > 1 else 0.0
    equity = np.cumsum(net)
    drawdown = equity - np.maximum.accumulate(equity) if observations else np.array([0.0])
    gross_sum = float(gross.sum())
    cost_sum = float(cost.sum())
    return {
        "observations": observations,
        "active_bars": int(frame["active"].sum()),
        "gross_return": gross_sum,
        "cost_return": cost_sum,
        "net_return": float(net.sum()),
        "mean_net_bps": mean * 10_000.0,
        "tstat": mean / (std / math.sqrt(observations)) if std > 0.0 else np.nan,
        "annualized_sharpe": mean / std * math.sqrt(periods_per_year) if std > 0.0 else np.nan,
        "hit_rate": float((net > 0.0).mean()) if observations else np.nan,
        "max_drawdown": float(drawdown.min()) if observations else np.nan,
        "gross_to_cost": gross_sum / cost_sum if cost_sum > 0.0 else np.inf,
        "turnover": float(frame["turnover"].sum()),
    }


def monthly_selected_returns(timestamps: pd.Series, returns: pd.DataFrame) -> pd.DataFrame:
    month = pd.to_datetime(timestamps, utc=True).dt.tz_convert(None).dt.to_period("M").astype(str)
    return (
        returns.assign(month=month)
        .groupby("month", as_index=False)[["gross_return", "cost_return", "net_return", "turnover"]]
        .sum()
    )


def daily_block_bootstrap(
    timestamps: pd.Series,
    net_returns: np.ndarray,
    *,
    iterations: int = 2_000,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.to_datetime(timestamps, utc=True).dt.date
    daily = (
        pd.DataFrame({"day": days, "net_return": net_returns}).groupby("day")["net_return"].sum()
    )
    values = daily.to_numpy(dtype=float)
    samples = np.empty(iterations)
    for i in range(iterations):
        samples[i] = rng.choice(values, size=len(values), replace=True).sum()
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


def selected_event_stats(
    payload: dict[str, Any],
    data: DataBundle,
    signals: dict[str, np.ndarray],
    roots: tuple[str, ...],
) -> pd.DataFrame:
    selected = payload["selected_strategy"]
    signal = signals[selected["signal"]]
    event = np.isfinite(signal) & (np.abs(signal) >= float(selected["entry"])) & data.valid_mask
    direction = float(selected["direction"])
    future = future_returns(data.returns, (int(selected["hold_bars"]),))
    rows = []
    signal_sign = pd.Series(
        direction * np.sign(signal),
        index=data.returns.index,
    )
    for root, weight in selected["targets"].items():
        signed = signal_sign * future[(root, int(selected["hold_bars"]))]
        values = signed[event].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        rows.append({"root": root, "weight": float(weight), **return_stats(values)})
    portfolio_values = np.zeros(len(data.returns), dtype=float)
    for root, weight in selected["targets"].items():
        signed = signal_sign * future[(root, int(selected["hold_bars"]))]
        portfolio_values += signed.to_numpy(dtype=float) * float(weight)
    rows.append({"root": "portfolio", "weight": 1.0, **return_stats(portfolio_values[event])})
    return pd.DataFrame(rows)


def return_stats(values: np.ndarray) -> dict[str, Any]:
    values = values[np.isfinite(values)]
    observations = len(values)
    if observations == 0:
        return {
            "events": 0,
            "mean_bps": np.nan,
            "median_bps": np.nan,
            "vol_bps": np.nan,
            "tstat": np.nan,
            "hit_rate": np.nan,
        }
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if observations > 1 else 0.0
    return {
        "events": int(observations),
        "mean_bps": mean * 10_000.0,
        "median_bps": float(np.median(values)) * 10_000.0,
        "vol_bps": std * 10_000.0,
        "tstat": mean / (std / math.sqrt(observations)) if std > 0.0 else np.nan,
        "hit_rate": float((values > 0.0).mean()),
    }


def make_decision(
    payload: dict[str, Any],
    selected_metrics: dict[str, Any],
    split_metrics: pd.DataFrame,
) -> dict[str, str]:
    rules = payload["decision_rules"]
    train = split_metrics.loc[split_metrics["split"] == "train"].iloc[0]
    test = split_metrics.loc[split_metrics["split"] == "test_purged"].iloc[0]
    passes = (
        float(train["net_return"]) > float(rules["min_train_net_return"])
        and float(test["net_return"]) > float(rules["min_test_net_return"])
        and float(test["gross_to_cost"]) >= float(rules["min_test_gross_to_cost"])
        and float(selected_metrics["tstat"]) >= float(rules["min_full_tstat"])
    )
    if passes:
        return {
            "status": str(rules["status_if_pass"]),
            "notes": (
                "Candidate pattern passed train/test net, purged-test gross/cost, "
                "and full-sample t-stat gates. Keep as revise/paper-trade-candidate "
                "because it was discovered through an iterative search and still needs "
                "forward validation."
            ),
        }
    return {
        "status": "reject",
        "notes": "Selected PA residual lead-lag rule failed one or more decision gates.",
    }


def json_safe(value: Any) -> Any:  # noqa: PLR0911
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0.0 else "-inf"
        return value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


if __name__ == "__main__":
    main()
