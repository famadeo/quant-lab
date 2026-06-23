# ruff: noqa: I001, PLR0915, PLR2004
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
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


MONTH_CODES = "FGHJKMNQUVXZ"
ROOT_GROUPS = {
    "6A": "Currencies",
    "6B": "Currencies",
    "6C": "Currencies",
    "6E": "Currencies",
    "6J": "Currencies",
    "CL": "Energy",
    "ES": "Equities",
    "GC": "Metals",
    "HG": "Metals",
    "ALI": "Metals",
    "NQ": "Equities",
    "PA": "Metals",
    "PL": "Metals",
    "SI": "Metals",
    "ZB": "Fixed income",
    "ZF": "Fixed income",
    "ZN": "Fixed income",
    "ZT": "Fixed income",
}


@dataclass(frozen=True)
class ReturnSummary:
    observations: int
    total_gross_return: float
    total_cost_return: float
    total_net_return: float
    mean_net_bps: float
    event_tstat: float
    annualized_sharpe: float
    hit_rate: float
    max_drawdown: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run suggested quant-lab strategy experiments.")
    parser.add_argument(
        "configs",
        nargs="*",
        type=Path,
        help="Experiment config paths. Defaults to the registered suggested strategy configs.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def fit_slope(x_values: pd.Series, y_values: pd.Series) -> dict[str, float]:
    frame = pd.DataFrame({"x": x_values, "y": y_values}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 5 or frame["x"].std(ddof=1) == 0:
        return {"beta": 0.0, "tstat": 0.0, "corr": 0.0, "observations": float(len(frame))}
    x = frame["x"].to_numpy(dtype=float)
    y = frame["y"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residuals = y - design @ beta
    dof = len(x) - 2
    sigma_sq = float(residuals @ residuals / dof)
    cov = sigma_sq * np.linalg.inv(design.T @ design)
    se = math.sqrt(max(float(cov[1, 1]), 0.0))
    corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else 0.0
    return {
        "beta": float(beta[1]),
        "tstat": float(beta[1] / se) if se > 0 else 0.0,
        "corr": corr,
        "observations": float(len(frame)),
    }


def summarize_returns(
    gross: pd.Series,
    cost: pd.Series,
    periods_per_year: float,
    label: str,
) -> dict[str, Any]:
    gross = gross.fillna(0.0).astype(float)
    cost = cost.fillna(0.0).astype(float)
    net = gross - cost
    observations = len(net)
    std = float(net.std(ddof=1)) if observations > 1 else 0.0
    mean = float(net.mean()) if observations else 0.0
    event_tstat = float(mean / (std / math.sqrt(observations))) if std > 0 else 0.0
    sharpe = float(mean / std * math.sqrt(periods_per_year)) if std > 0 else 0.0
    equity = net.cumsum()
    drawdown = equity - equity.cummax()
    max_drawdown = float(drawdown.min()) if observations else 0.0
    return {
        "label": label,
        "observations": int(observations),
        "total_gross_return": float(gross.sum()),
        "total_cost_return": float(cost.sum()),
        "total_net_return": float(net.sum()),
        "mean_net_bps": mean * 10_000.0,
        "event_tstat": event_tstat,
        "annualized_sharpe": sharpe,
        "hit_rate": float((net > 0).mean()) if observations else 0.0,
        "max_drawdown": max_drawdown,
    }


def plot_equity(curves: dict[str, pd.Series], title: str, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 5))
    for label, series in curves.items():
        axis.plot(series.index, series.cumsum() * 100.0, label=label)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_title(title)
    axis.set_ylabel("Cumulative net return (%)")
    axis.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def chronological_split(values: pd.Series, train_fraction: float) -> tuple[set[Any], set[Any]]:
    ordered = sorted(pd.Series(values).dropna().unique())
    train_count = min(max(math.floor(len(ordered) * train_fraction), 1), len(ordered) - 1)
    return set(ordered[:train_count]), set(ordered[train_count:])


def decide_from_thresholds(
    metrics: dict[str, Any],
    config: dict[str, Any],
    extra_checks: dict[str, bool] | None = None,
) -> dict[str, Any]:
    rules = config["decision_rules"]
    passed = metrics["total_net_return"] > float(rules.get("min_net_return", 0.0)) and metrics[
        "event_tstat"
    ] >= float(rules.get("min_event_tstat", 1.65))
    if extra_checks:
        passed = passed and all(extra_checks.values())
    status = str(rules["pass_status"] if passed else rules["fail_status"])
    return {"status": status, "passed": bool(passed)}


def run_hyp_0006(config_path: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    out_dir = config_path.parent
    data = config["data"]
    strategy = config["strategy"]
    roots = set(config["universe"]["roots"])
    releases = set(config["universe"]["releases"])
    frame = pl.read_parquet(data["event_features_path"]).to_pandas()
    frame["ts_utc"] = pd.to_datetime(frame["ts_utc"], utc=True)
    frame = frame[frame["root"].isin(roots) & frame["release"].isin(releases)].copy()
    price_cols = ["p_event", "p_h5", "p_h60"]
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=price_cols)
    frame = frame[(frame[price_cols] > 0).all(axis=1)].copy()
    frame["obs_ret_0_5"] = np.log(frame["p_h5"] / frame["p_event"])
    frame["target_ret_5_60"] = np.log(frame["p_h60"] / frame["p_h5"])
    train_ts, test_ts = chronological_split(frame["ts_utc"], config["validation"]["train_fraction"])

    cost = float(strategy["round_trip_cost_bps"]) / 10_000.0
    root_records = []
    coefficient_records = []
    trade_rows = []

    for root, root_frame in frame.groupby("root", sort=True):
        train = root_frame[root_frame["ts_utc"].isin(train_ts)]
        test = root_frame[root_frame["ts_utc"].isin(test_ts)]
        fit = fit_slope(train["obs_ret_0_5"], train["target_ret_5_60"])
        direction = float(np.sign(fit["beta"]))
        if len(test) < int(config["validation"]["min_test_observations"]):
            direction = 0.0
        test = test.copy()
        test["position"] = direction * np.sign(test["obs_ret_0_5"])
        test["gross_return"] = test["position"] * test["target_ret_5_60"]
        test["cost_return"] = cost * test["position"].abs()
        test["net_return"] = test["gross_return"] - test["cost_return"]
        root_records.append(
            {
                "root": root,
                **summarize_returns(
                    test["gross_return"], test["cost_return"], 252.0, f"{root}_event_trades"
                ),
            }
        )
        coefficient_records.append({"root": root, "direction": direction, **fit})
        trade_rows.append(test)

    trades = pd.concat(trade_rows, ignore_index=True).sort_values(["ts_utc", "root"])
    event_portfolio = trades.groupby("ts_utc", as_index=True)[
        ["gross_return", "cost_return", "net_return"]
    ].mean()
    pooled = summarize_returns(
        event_portfolio["gross_return"],
        event_portfolio["cost_return"],
        252.0,
        "equal_weight_event_portfolio",
    )
    root_metrics = pd.DataFrame(root_records)
    positive_fraction = float((root_metrics["total_net_return"] > 0).mean())
    decision = decide_from_thresholds(
        pooled,
        config,
        {
            "positive_root_fraction": positive_fraction
            >= config["decision_rules"]["min_positive_roots"]
        },
    )
    decision.update(
        {
            "positive_root_fraction": positive_fraction,
            "notes": "Post-5-minute macro-event drift/reversal validation.",
        }
    )

    root_metrics.to_csv(out_dir / "root_metrics.csv", index=False)
    pd.DataFrame(coefficient_records).to_csv(out_dir / "train_coefficients.csv", index=False)
    event_portfolio.to_csv(out_dir / "event_portfolio_returns.csv")
    trades.to_parquet(out_dir / "event_trades.parquet", index=False)
    plot_equity(
        {"macro_event": event_portfolio["net_return"]},
        "HYP-0006 OOS macro event cumulative net return",
        out_dir / "event_portfolio_equity.png",
    )
    result = {
        "experiment_id": config["experiment"]["id"],
        "title": config["experiment"]["title"],
        "completed_at": datetime.now().astimezone().isoformat(),
        "decision": decision,
        "portfolio_metrics": pooled,
        "artifacts": {
            "root_metrics": str(out_dir / "root_metrics.csv"),
            "train_coefficients": str(out_dir / "train_coefficients.csv"),
            "event_portfolio_returns": str(out_dir / "event_portfolio_returns.csv"),
            "event_trades": str(out_dir / "event_trades.parquet"),
            "plot": str(out_dir / "event_portfolio_equity.png"),
        },
    }
    write_json(out_dir / "results.json", result)
    return result


def run_hyp_0007(config_path: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    out_dir = config_path.parent
    bars = pd.read_parquet(config["data"]["volatility_bars_path"])
    bars["ts_end"] = pd.to_datetime(bars["ts_end"], utc=True)
    bars["date"] = pd.to_datetime(bars["ts_end"], utc=True).dt.date
    cost = float(config["strategy"]["round_trip_cost_bps"]) / 10_000.0
    train_fraction = float(config["validation"]["train_fraction_dates"])
    q = float(config["strategy"]["extreme_quantile"])

    root_records = []
    model_records = []
    trade_rows = []

    for root, grouped_root_frame in bars.groupby("root", sort=True):
        root_frame = grouped_root_frame.sort_values("ts_end").copy()
        train_dates, test_dates = chronological_split(root_frame["date"], train_fraction)
        root_frame["next_ret"] = root_frame["ret"].shift(-1)
        train = root_frame[root_frame["date"].isin(train_dates)].copy()
        test = root_frame[root_frame["date"].isin(test_dates)].copy()
        ofi_threshold = float(train["ofi_high"].abs().quantile(q))
        ret_threshold = float(train["ret"].abs().quantile(q))
        train_event = train[
            (train["ofi_high"].abs() >= ofi_threshold)
            & (train["ret"].abs() >= ret_threshold)
            & (np.sign(train["ofi_high"]) == np.sign(train["ret"]))
        ].dropna(subset=["next_ret"])
        signed_next = np.sign(train_event["ret"]) * train_event["next_ret"]
        mean_signed = float(signed_next.mean()) if len(signed_next) else 0.0
        direction = float(np.sign(mean_signed))
        if direction == 0.0:
            direction = -1.0
        test_event = test[
            (test["ofi_high"].abs() >= ofi_threshold)
            & (test["ret"].abs() >= ret_threshold)
            & (np.sign(test["ofi_high"]) == np.sign(test["ret"]))
        ].dropna(subset=["next_ret"])
        test_event = test_event.copy()
        test_event["root"] = root
        test_event["model_direction"] = "continue" if direction > 0 else "fade"
        test_event["position"] = direction * np.sign(test_event["ret"])
        test_event["gross_return"] = test_event["position"] * test_event["next_ret"]
        test_event["cost_return"] = cost * test_event["position"].abs()
        test_event["net_return"] = test_event["gross_return"] - test_event["cost_return"]
        root_records.append(
            {
                "root": root,
                "model_direction": "continue" if direction > 0 else "fade",
                "ofi_high_abs_threshold": ofi_threshold,
                "ret_abs_threshold": ret_threshold,
                "train_events": len(train_event),
                **summarize_returns(
                    test_event["gross_return"],
                    test_event["cost_return"],
                    252.0 * 20,
                    f"{root}_impact_decay",
                ),
            }
        )
        model_records.append(
            {
                "root": root,
                "mean_train_signed_next_return": mean_signed,
                "direction": direction,
                "train_events": len(train_event),
            }
        )
        trade_rows.append(test_event)

    trades = pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame()
    root_metrics = pd.DataFrame(root_records)
    pooled = summarize_returns(
        trades["gross_return"] if not trades.empty else pd.Series(dtype=float),
        trades["cost_return"] if not trades.empty else pd.Series(dtype=float),
        252.0 * 20,
        "pooled_extreme_high_ofi_events",
    )
    positive_fraction = float((root_metrics["total_net_return"] > 0).mean())
    decision = decide_from_thresholds(
        pooled,
        config,
        {
            "positive_root_fraction": positive_fraction
            >= config["decision_rules"]["min_positive_roots"]
        },
    )
    decision.update({"positive_root_fraction": positive_fraction})

    root_metrics.to_csv(out_dir / "root_metrics.csv", index=False)
    pd.DataFrame(model_records).to_csv(out_dir / "train_models.csv", index=False)
    trades.to_parquet(out_dir / "impact_decay_trades.parquet", index=False)
    if not trades.empty:
        plot_equity(
            {"impact_decay": trades.set_index("ts_end")["net_return"]},
            "HYP-0007 impact-decay event cumulative net return",
            out_dir / "impact_decay_equity.png",
        )
    result = {
        "experiment_id": config["experiment"]["id"],
        "title": config["experiment"]["title"],
        "completed_at": datetime.now().astimezone().isoformat(),
        "decision": decision,
        "pooled_metrics": pooled,
        "artifacts": {
            "root_metrics": str(out_dir / "root_metrics.csv"),
            "train_models": str(out_dir / "train_models.csv"),
            "trades": str(out_dir / "impact_decay_trades.parquet"),
            "plot": str(out_dir / "impact_decay_equity.png"),
        },
    }
    write_json(out_dir / "results.json", result)
    return result


def load_daily_continuous_panel(
    roots: list[str], directory: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = []
    returns = []
    for root in roots:
        path = Path(directory) / f"{root}.csv"
        frame = pd.read_csv(path, parse_dates=["date"])
        frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.date
        prices.append(frame.set_index("date")["cont_logprice"].rename(root))
        returns.append(frame.set_index("date")["cont_logret"].rename(root))
    price_panel = pd.concat(prices, axis=1).sort_index()
    return_panel = pd.concat(returns, axis=1).sort_index().fillna(0.0)
    return price_panel, return_panel


def normalized_positions(scores: pd.DataFrame) -> pd.DataFrame:
    clean = scores.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gross = clean.abs().sum(axis=1).replace(0.0, np.nan)
    return clean.div(gross, axis=0).fillna(0.0)


def portfolio_from_positions(
    positions: pd.DataFrame,
    returns: pd.DataFrame,
    cost_bps_per_turnover: float,
) -> pd.DataFrame:
    aligned_returns = returns.reindex(positions.index).shift(-1)
    gross = (positions * aligned_returns).sum(axis=1)
    turnover = positions.diff().abs().sum(axis=1).fillna(positions.abs().sum(axis=1))
    cost = turnover * cost_bps_per_turnover / 10_000.0
    out = pd.DataFrame({"gross_return": gross, "cost_return": cost})
    out["net_return"] = out["gross_return"] - out["cost_return"]
    return out.dropna()


def run_hyp_0008(config_path: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    out_dir = config_path.parent
    roots = list(config["universe"]["roots"])
    price_panel, return_panel = load_daily_continuous_panel(roots, config["data"]["continuous_dir"])
    if config["data"].get("start"):
        start = pd.Timestamp(config["data"]["start"]).date()
        price_panel = price_panel[price_panel.index >= start]
        return_panel = return_panel[return_panel.index >= start]
    if config["data"].get("end"):
        end = pd.Timestamp(config["data"]["end"]).date()
        price_panel = price_panel[price_panel.index <= end]
        return_panel = return_panel[return_panel.index <= end]
    lookback = int(config["strategy"]["lookback_days"])
    entry_z = float(config["strategy"]["entry_z"])
    scores = pd.DataFrame(index=price_panel.index, columns=roots, dtype=float)

    for group in sorted(set(ROOT_GROUPS[root] for root in roots)):
        group_roots = [root for root in roots if ROOT_GROUPS[root] == group]
        if len(group_roots) < 2:
            continue
        group_prices = price_panel[group_roots]
        residuals = group_prices.sub(group_prices.mean(axis=1), axis=0)
        mean = residuals.rolling(lookback, min_periods=lookback // 2).mean()
        std = residuals.rolling(lookback, min_periods=lookback // 2).std()
        zscores = (residuals - mean) / std
        scores[group_roots] = -zscores.clip(-entry_z, entry_z) / entry_z

    positions = normalized_positions(scores)
    returns = portfolio_from_positions(
        positions, return_panel, float(config["strategy"]["cost_bps_per_unit_turnover"])
    )
    _, test_dates = chronological_split(
        pd.Series(returns.index), config["validation"]["train_fraction"]
    )
    test_returns = returns[returns.index.isin(test_dates)]
    metrics = summarize_returns(
        test_returns["gross_return"], test_returns["cost_return"], 252.0, "daily_residual_basket"
    )
    decision = decide_from_thresholds(metrics, config)

    returns.to_csv(out_dir / "portfolio_returns.csv")
    test_returns.to_csv(out_dir / "oos_portfolio_returns.csv")
    positions.to_parquet(out_dir / "positions.parquet")
    by_group = []
    for group in sorted(set(ROOT_GROUPS[root] for root in roots)):
        group_roots = [root for root in roots if ROOT_GROUPS[root] == group]
        group_positions = normalized_positions(scores[group_roots])
        group_returns = portfolio_from_positions(
            group_positions,
            return_panel[group_roots],
            float(config["strategy"]["cost_bps_per_unit_turnover"]),
        )
        group_test = group_returns[group_returns.index.isin(test_dates)]
        by_group.append(
            {
                "group": group,
                **summarize_returns(
                    group_test["gross_return"], group_test["cost_return"], 252.0, group
                ),
            }
        )
    pd.DataFrame(by_group).to_csv(out_dir / "group_metrics.csv", index=False)
    plot_equity(
        {"residual_basket": test_returns["net_return"]},
        f"{config['experiment']['id']} OOS residual-basket cumulative net return",
        out_dir / "residual_basket_equity.png",
    )
    result = {
        "experiment_id": config["experiment"]["id"],
        "title": config["experiment"]["title"],
        "completed_at": datetime.now().astimezone().isoformat(),
        "decision": decision,
        "portfolio_metrics": metrics,
        "artifacts": {
            "portfolio_returns": str(out_dir / "portfolio_returns.csv"),
            "positions": str(out_dir / "positions.parquet"),
            "group_metrics": str(out_dir / "group_metrics.csv"),
            "plot": str(out_dir / "residual_basket_equity.png"),
        },
    }
    write_json(out_dir / "results.json", result)
    return result


def restrict_rth(frame: pd.DataFrame, ts_col: str, start: str, end: str) -> pd.DataFrame:
    ts = pd.to_datetime(frame[ts_col], utc=True)
    minutes = ts.dt.hour * 60 + ts.dt.minute
    start_hour, start_minute = (int(part) for part in start.split(":"))
    end_hour, end_minute = (int(part) for part in end.split(":"))
    lower = start_hour * 60 + start_minute
    upper = end_hour * 60 + end_minute
    return frame[(minutes > lower) & (minutes <= upper)].copy()


def run_hyp_0009(config_path: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    out_dir = config_path.parent
    roots = list(config["universe"]["roots"])
    ret_dir = Path(config["data"]["continuous_5m_dir"])
    flow_dir = Path(config["data"]["flow_5m_dir"])
    cost_rate = float(config["strategy"]["cost_bps_per_unit_turnover"]) / 10_000.0
    lookback = int(config["strategy"]["trend_lookback_bars"])
    train_fraction = float(config["validation"]["train_fraction_dates"])
    root_records = []
    all_rows = []

    for root in roots:
        ret_path = ret_dir / f"{root}.parquet"
        flow_path = flow_dir / f"{root}_flow_5m.parquet"
        if not ret_path.exists() or not flow_path.exists():
            continue
        ret = pl.read_parquet(ret_path).select("ts", "cont_logret").to_pandas()
        flow = pl.read_parquet(flow_path).select("ts", "tot_vol", "ofi_block", "ofi").to_pandas()
        ret["ts"] = pd.to_datetime(ret["ts"], utc=True)
        flow["ts"] = pd.to_datetime(flow["ts"], utc=True)
        ret = restrict_rth(ret, "ts", config["data"]["rth_start"], config["data"]["rth_end"])
        joined = ret.merge(flow, on="ts", how="inner").sort_values("ts")
        if len(joined) < int(config["validation"]["min_observations"]):
            continue
        joined["date"] = joined["ts"].dt.date
        train_dates, test_dates = chronological_split(joined["date"], train_fraction)
        joined["trend_ret"] = joined["cont_logret"].rolling(lookback, min_periods=lookback).sum()
        train = joined[joined["date"].isin(train_dates)]
        trend_threshold = float(
            train["trend_ret"].abs().quantile(config["strategy"]["trend_quantile"])
        )
        volume_threshold = float(train["tot_vol"].quantile(config["strategy"]["volume_quantile"]))
        base_signal = np.where(
            joined["trend_ret"].abs() >= trend_threshold,
            np.sign(joined["trend_ret"]),
            0.0,
        )
        filter_mask = (np.sign(joined["ofi_block"]) == np.sign(base_signal)) & (
            joined["tot_vol"] >= volume_threshold
        )
        signals = {
            "baseline_trend": pd.Series(base_signal, index=joined.index),
            "flow_filtered_trend": pd.Series(
                np.where(filter_mask, base_signal, 0.0), index=joined.index
            ),
        }
        for method, signal in signals.items():
            work = joined.copy()
            work["root"] = root
            work["method"] = method
            work["position"] = signal.shift(1).fillna(0.0)
            work["turnover"] = work["position"].diff().abs().fillna(work["position"].abs())
            work["gross_return"] = work["position"] * work["cont_logret"]
            work["cost_return"] = cost_rate * work["turnover"]
            work["net_return"] = work["gross_return"] - work["cost_return"]
            test = work[work["date"].isin(test_dates)].copy()
            root_records.append(
                {
                    "root": root,
                    "method": method,
                    "trend_threshold": trend_threshold,
                    "volume_threshold": volume_threshold,
                    **summarize_returns(
                        test["gross_return"],
                        test["cost_return"],
                        252.0 * 78.0,
                        f"{root}_{method}",
                    ),
                }
            )
            all_rows.append(test)

    rows = pd.concat(all_rows, ignore_index=True)
    root_metrics = pd.DataFrame(root_records)
    portfolio_records = []
    curves = {}
    for method, method_rows in rows.groupby("method", sort=True):
        portfolio = method_rows.groupby("ts", as_index=True)[["gross_return", "cost_return"]].mean()
        portfolio["net_return"] = portfolio["gross_return"] - portfolio["cost_return"]
        portfolio.to_csv(out_dir / f"{method}_portfolio_returns.csv")
        metrics = summarize_returns(
            portfolio["gross_return"], portfolio["cost_return"], 252.0 * 78.0, method
        )
        portfolio_records.append(metrics)
        curves[method] = portfolio["net_return"]
    portfolio_metrics = pd.DataFrame(portfolio_records)
    primary = portfolio_metrics[
        portfolio_metrics["label"].eq(config["strategy"]["primary_method"])
    ].iloc[0]
    baseline = portfolio_metrics[portfolio_metrics["label"].eq("baseline_trend")].iloc[0]
    decision = decide_from_thresholds(
        primary.to_dict(),
        config,
        {"beats_baseline": primary["total_net_return"] > baseline["total_net_return"]},
    )
    root_metrics.to_csv(out_dir / "root_metrics.csv", index=False)
    portfolio_metrics.to_csv(out_dir / "portfolio_metrics.csv", index=False)
    rows.to_parquet(out_dir / "root_bar_returns.parquet", index=False)
    plot_equity(
        curves, "HYP-0009 OOS trend and flow-filtered trend", out_dir / "trend_flow_equity.png"
    )
    result = {
        "experiment_id": config["experiment"]["id"],
        "title": config["experiment"]["title"],
        "completed_at": datetime.now().astimezone().isoformat(),
        "decision": decision,
        "portfolio_metrics": portfolio_metrics.to_dict(orient="records"),
        "artifacts": {
            "root_metrics": str(out_dir / "root_metrics.csv"),
            "portfolio_metrics": str(out_dir / "portfolio_metrics.csv"),
            "root_bar_returns": str(out_dir / "root_bar_returns.parquet"),
            "plot": str(out_dir / "trend_flow_equity.png"),
        },
    }
    write_json(out_dir / "results.json", result)
    return result


def symbol_pattern(root: str) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(root)}[{MONTH_CODES}]\d{{1,2}}$")


def build_carry_series(root: str, raw_dir: str, continuous_dir: str) -> pd.Series:
    raw = pd.read_csv(Path(raw_dir) / f"{root}.csv", parse_dates=["ts_event"])
    raw["date"] = pd.to_datetime(raw["ts_event"], utc=True).dt.date
    raw["symbol"] = raw["symbol"].astype(str)
    raw = raw[raw["symbol"].str.match(symbol_pattern(root))]
    if raw.empty:
        return pd.Series(dtype=float, name=root)
    expiry_order = (
        raw.groupby("symbol", as_index=False)["date"]
        .max()
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )
    rank = {symbol: idx for idx, symbol in enumerate(expiry_order["symbol"])}
    raw["rank"] = raw["symbol"].map(rank)
    cont = pd.read_csv(Path(continuous_dir) / f"{root}.csv", parse_dates=["date"])
    cont["date"] = pd.to_datetime(cont["date"], utc=True).dt.date
    active = cont.set_index("date")["active"].astype(str)
    values = {}
    for day, day_frame in raw.groupby("date", sort=True):
        active_symbol = active.get(day)
        if not active_symbol or active_symbol not in rank:
            continue
        front = day_frame[day_frame["symbol"].eq(active_symbol)]
        later = day_frame[day_frame["rank"] > rank[active_symbol]].sort_values("rank")
        if front.empty or later.empty:
            continue
        front_close = float(front.iloc[0]["close"])
        second_close = float(later.iloc[0]["close"])
        if front_close > 0 and second_close > 0:
            values[day] = math.log(front_close / second_close)
    return pd.Series(values, name=root).sort_index()


def cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    centered = frame.sub(frame.mean(axis=1), axis=0)
    std = frame.std(axis=1).replace(0.0, np.nan)
    return centered.div(std, axis=0)


def run_hyp_0010(config_path: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    out_dir = config_path.parent
    roots = list(config["universe"]["roots"])
    price_panel, return_panel = load_daily_continuous_panel(roots, config["data"]["continuous_dir"])
    carry_panel = pd.concat(
        [
            build_carry_series(
                root, config["data"]["raw_daily_dir"], config["data"]["continuous_dir"]
            )
            for root in roots
        ],
        axis=1,
    ).reindex(price_panel.index)
    momentum = price_panel - price_panel.shift(int(config["strategy"]["momentum_lookback_days"]))
    carry_z = cross_sectional_zscore(carry_panel)
    momentum_z = cross_sectional_zscore(momentum)
    combined_z = 0.5 * carry_z + 0.5 * momentum_z
    score_map = {"carry": carry_z, "momentum": momentum_z, "combined": combined_z}

    _, test_dates = chronological_split(
        pd.Series(price_panel.index), config["validation"]["train_fraction"]
    )
    records = []
    curves = {}
    for label, scores in score_map.items():
        positions = normalized_positions(scores)
        returns = portfolio_from_positions(
            positions, return_panel, float(config["strategy"]["cost_bps_per_unit_turnover"])
        )
        test_returns = returns[returns.index.isin(test_dates)]
        returns.to_csv(out_dir / f"{label}_portfolio_returns.csv")
        positions.to_parquet(out_dir / f"{label}_positions.parquet")
        records.append(
            summarize_returns(
                test_returns["gross_return"], test_returns["cost_return"], 252.0, label
            )
        )
        curves[label] = test_returns["net_return"]

    metrics = pd.DataFrame(records)
    primary = metrics[metrics["label"].eq(config["strategy"]["primary_method"])].iloc[0]
    decision = decide_from_thresholds(primary.to_dict(), config)
    metrics.to_csv(out_dir / "portfolio_metrics.csv", index=False)
    carry_panel.to_parquet(out_dir / "carry_panel.parquet")
    plot_equity(
        curves,
        "HYP-0010 OOS carry and momentum futures portfolios",
        out_dir / "carry_momentum_equity.png",
    )
    result = {
        "experiment_id": config["experiment"]["id"],
        "title": config["experiment"]["title"],
        "completed_at": datetime.now().astimezone().isoformat(),
        "decision": decision,
        "portfolio_metrics": metrics.to_dict(orient="records"),
        "carry_coverage": {
            "roots": int(carry_panel.notna().any().sum()),
            "first_date": str(carry_panel.dropna(how="all").index.min()),
            "last_date": str(carry_panel.dropna(how="all").index.max()),
        },
        "artifacts": {
            "portfolio_metrics": str(out_dir / "portfolio_metrics.csv"),
            "carry_panel": str(out_dir / "carry_panel.parquet"),
            "plot": str(out_dir / "carry_momentum_equity.png"),
        },
    }
    write_json(out_dir / "results.json", result)
    return result


RUNNERS = {
    "HYP-0006-macro-event-drift": run_hyp_0006,
    "HYP-0007-vol-clock-impact-decay": run_hyp_0007,
    "HYP-0008-cross-asset-residual-baskets": run_hyp_0008,
    "HYP-0009-trend-flow-filter": run_hyp_0009,
    "HYP-0010-futures-carry-momentum": run_hyp_0010,
    "HYP-0011-expanded-metals-residual-basket": run_hyp_0008,
}


def default_configs() -> list[Path]:
    return [
        Path("experiments/HYP-0006-macro-event-drift/config.yaml"),
        Path("experiments/HYP-0007-vol-clock-impact-decay/config.yaml"),
        Path("experiments/HYP-0008-cross-asset-residual-baskets/config.yaml"),
        Path("experiments/HYP-0009-trend-flow-filter/config.yaml"),
        Path("experiments/HYP-0010-futures-carry-momentum/config.yaml"),
        Path("experiments/HYP-0011-expanded-metals-residual-basket/config.yaml"),
    ]


def main() -> None:
    args = parse_args()
    configs = args.configs or default_configs()
    results = []
    for config_path in configs:
        config = load_yaml(config_path)
        experiment_id = config["experiment"]["id"]
        runner = RUNNERS[experiment_id]
        result = runner(config_path)
        results.append(
            {
                "experiment_id": experiment_id,
                "status": result["decision"]["status"],
                "passed": result["decision"]["passed"],
            }
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
