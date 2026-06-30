# ruff: noqa: E501, PLR2004
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
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantlab.metals_flow.forward import future_returns
from quantlab.metals_flow.strategy import (
    backtest_positions,
    calculate_strategy_metrics,
    demean_and_normalize_positions,
    residual_momentum_positions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest residual-state momentum filtered by flow-geometry regimes."
    )
    parser.add_argument("config", type=Path, help="Experiment config YAML.")
    return parser.parse_args()


def main() -> None:
    config_path = parse_args().config
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    out_dir = Path(payload["outputs"]["directory"])
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(exist_ok=True)
    run(payload, out_dir, plot_dir)


def run(payload: dict[str, Any], out_dir: Path, plot_dir: Path) -> None:
    source_dir = Path(payload["data"]["source_experiment"])
    roots = tuple(payload["universe"]["roots"])
    target_roots = tuple(payload["universe"]["target_roots"])
    strategy = payload["strategy"]
    horizons = tuple(int(value) for value in payload["research"]["horizons"])
    primary_horizon = int(payload["research"]["primary_horizon"])

    bars = pd.read_parquet(source_dir / "primary_bars.parquet")
    bars["start_ts"] = pd.to_datetime(bars["start_ts"], utc=True)
    bars["end_ts"] = pd.to_datetime(bars["end_ts"], utc=True)
    returns = pd.read_parquet(source_dir / "bar_returns.parquet").reindex(columns=roots)
    residual_z = pd.read_parquet(source_dir / "fair_value_zscores.parquet").reindex(
        columns=roots
    )
    anomalies = pd.read_parquet(source_dir / "flow_anomalies.parquet")
    price_validity = pd.read_parquet(source_dir / "price_validity.parquet")
    cost_estimates = pd.read_csv(Path(payload["data"]["cost_estimates"]))
    cost_bps = cost_estimates.set_index("root")["per_side_cost_bps"].reindex(roots)

    valid_mask = price_validity["valid_price_mask"].fillna(False).astype(bool)
    valid_mask &= returns.notna().all(axis=1)
    periods_per_year = periods_per_year_from_bars(bars)
    split_index, test_start_index = split_points(
        bars,
        train_fraction=float(strategy["train_fraction"]),
        embargo_bars=int(strategy["embargo_bars"]),
    )
    train_mask = pd.Series(False, index=returns.index)
    train_mask.iloc[:split_index] = True
    test_mask = pd.Series(False, index=returns.index)
    test_mask.iloc[test_start_index:] = True

    md_thresholds = {
        f"md_q{int(q * 100)}": shifted_rolling_quantile(
            anomalies["md_rolling"],
            q,
            int(strategy["threshold_window"]),
            int(strategy["threshold_min_periods"]),
        )
        for q in tuple(float(value) for value in strategy["md_quantiles"])
    }
    gates = {"no_flow": valid_mask}
    gates.update(
        {
            name: (anomalies["md_rolling"] >= threshold) & valid_mask
            for name, threshold in md_thresholds.items()
        }
    )

    future = future_returns(returns, horizons)
    calibration = calibrate_event_edges(
        residual_z=residual_z,
        future=future,
        gates=gates,
        train_mask=train_mask,
        roots=target_roots,
        cost_bps=cost_bps,
        primary_horizon=primary_horizon,
        entry_z=float(strategy["entry_z"]),
        cost_multiple=float(strategy["cost_hurdle_multiple"]),
    )

    variant_specs = build_variant_specs(
        roots=roots,
        target_roots=target_roots,
        gates=gates,
        calibration=calibration,
        strategy=strategy,
        primary_horizon=primary_horizon,
        test_start_index=test_start_index,
    )
    returns_by_variant, positions_by_variant, raw_positions_by_variant, metrics = (
        run_variants(
            residual_z=residual_z,
            returns=returns,
            valid_mask=valid_mask,
            cost_bps=cost_bps,
            periods_per_year=periods_per_year,
            variant_specs=variant_specs,
        )
    )
    split_metrics = run_split_metrics(
        returns_by_variant,
        positions_by_variant,
        split_index=split_index,
        test_start_index=test_start_index,
        periods_per_year=periods_per_year,
    )
    monthly_returns = summarize_monthly_returns(returns_by_variant, bars["end_ts"])
    event_study = residual_momentum_event_study(
        residual_z=residual_z,
        future=future,
        gates=gates,
        valid_mask=valid_mask,
        roots=target_roots,
        horizons=horizons,
        entry_z=float(strategy["entry_z"]),
    )
    portfolio_scatter = portfolio_forward_scatter(
        positions_by_variant=positions_by_variant,
        future=future,
        roots=roots,
        horizons=horizons,
    )
    gate_diagnostics = gate_summary(
        gates=gates,
        valid_mask=valid_mask,
        bars=bars,
        md_thresholds=md_thresholds,
        anomalies=anomalies,
    )

    write_outputs(
        out_dir=out_dir,
        plot_dir=plot_dir,
        payload=payload,
        source_dir=source_dir,
        bars=bars,
        residual_z=residual_z,
        anomalies=anomalies,
        returns_by_variant=returns_by_variant,
        positions_by_variant=positions_by_variant,
        raw_positions_by_variant=raw_positions_by_variant,
        metrics=metrics,
        split_metrics=split_metrics,
        monthly_returns=monthly_returns,
        event_study=event_study,
        portfolio_scatter=portfolio_scatter,
        gate_diagnostics=gate_diagnostics,
        calibration=calibration,
        cost_estimates=cost_estimates,
        roots=roots,
        target_roots=target_roots,
        primary_horizon=primary_horizon,
        split_index=split_index,
        test_start_index=test_start_index,
    )
    print(f"wrote {out_dir / 'results.json'}", flush=True)


def build_variant_specs(
    *,
    roots: tuple[str, ...],
    target_roots: tuple[str, ...],
    gates: dict[str, pd.Series],
    calibration: pd.DataFrame,
    strategy: dict[str, Any],
    primary_horizon: int,
    test_start_index: int,
) -> list[dict[str, Any]]:
    all_roots = tuple(roots)
    q95_name = "md_q95" if "md_q95" in gates else next(iter(gates))
    q99_name = "md_q99" if "md_q99" in gates else q95_name
    cost_allowed = calibration.loc[
        (calibration["gate"] == q99_name)
        & (calibration["horizon"] == primary_horizon)
        & calibration["clears_cost_hurdle"],
        "root",
    ].tolist()

    specs = [
        {
            "variant": "all_resid_momentum_no_flow_h20",
            "gate": "no_flow",
            "target_roots": all_roots,
            "max_holding_bars": 20,
            "entry_z": float(strategy["entry_z"]),
            "exit_z": float(strategy["exit_z"]),
        },
        {
            "variant": "pa_pl_resid_momentum_no_flow_h20",
            "gate": "no_flow",
            "target_roots": target_roots,
            "max_holding_bars": 20,
            "entry_z": float(strategy["entry_z"]),
            "exit_z": float(strategy["exit_z"]),
        },
        {
            "variant": "pa_pl_resid_momentum_md_q95_h20",
            "gate": q95_name,
            "target_roots": target_roots,
            "max_holding_bars": 20,
            "entry_z": float(strategy["entry_z"]),
            "exit_z": float(strategy["exit_z"]),
        },
        {
            "variant": "pa_pl_resid_momentum_md_q99_h20",
            "gate": q99_name,
            "target_roots": target_roots,
            "max_holding_bars": 20,
            "entry_z": float(strategy["entry_z"]),
            "exit_z": float(strategy["exit_z"]),
        },
        {
            "variant": "pa_pl_resid_momentum_md_q99_h50",
            "gate": q99_name,
            "target_roots": target_roots,
            "max_holding_bars": 50,
            "entry_z": float(strategy["entry_z"]),
            "exit_z": float(strategy["exit_z"]),
        },
        {
            "variant": "pa_pl_resid_momentum_md_q99_z2_5_h20",
            "gate": q99_name,
            "target_roots": target_roots,
            "max_holding_bars": 20,
            "entry_z": 2.5,
            "exit_z": float(strategy["exit_z"]),
        },
        {
            "variant": "pa_pl_resid_momentum_md_q99_3x_cost_gate_h20",
            "gate": q99_name,
            "target_roots": tuple(cost_allowed),
            "max_holding_bars": 20,
            "entry_z": float(strategy["entry_z"]),
            "exit_z": float(strategy["exit_z"]),
            "test_only_start": test_start_index,
        },
    ]
    for spec in specs:
        spec["entry_mask"] = gates[str(spec["gate"])]
    return specs


def run_variants(
    *,
    residual_z: pd.DataFrame,
    returns: pd.DataFrame,
    valid_mask: pd.Series,
    cost_bps: pd.Series,
    periods_per_year: float,
    variant_specs: list[dict[str, Any]],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    returns_by_variant: dict[str, pd.DataFrame] = {}
    positions_by_variant: dict[str, pd.DataFrame] = {}
    raw_positions_by_variant: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    zscores = residual_z.where(valid_mask, np.nan)
    for spec in variant_specs:
        name = str(spec["variant"])
        entry_mask = spec["entry_mask"].reindex(residual_z.index).fillna(False).astype(bool)
        if "test_only_start" in spec:
            prefix = pd.Series(False, index=entry_mask.index)
            prefix.iloc[int(spec["test_only_start"]) :] = True
            entry_mask &= prefix
        root_masks = target_root_mask(residual_z.index, residual_z.columns, spec["target_roots"])
        raw_positions = residual_momentum_positions(
            zscores,
            entry_mask,
            root_masks,
            entry_z=float(spec["entry_z"]),
            exit_z=float(spec["exit_z"]),
            max_holding_bars=int(spec["max_holding_bars"]),
            stop_z=float(spec["stop_z"]) if "stop_z" in spec else None,
            cooldown_bars=int(spec.get("cooldown_bars", 0)),
        )
        raw_positions = raw_positions.where(valid_mask, 0.0, axis=0)
        positions = demean_and_normalize_positions(raw_positions)
        frame, metric = backtest_positions(
            positions,
            returns,
            cost_bps,
            periods_per_year=periods_per_year,
        )
        diagnostics = position_diagnostics(raw_positions, positions)
        frame["variant"] = name
        returns_by_variant[name] = frame
        positions_by_variant[name] = positions
        raw_positions_by_variant[name] = raw_positions
        rows.append(
            {
                "variant": name,
                "gate": spec["gate"],
                "target_roots": ",".join(spec["target_roots"]),
                "entry_z": float(spec["entry_z"]),
                "exit_z": float(spec["exit_z"]),
                "max_holding_bars": int(spec["max_holding_bars"]),
                **metric.to_dict(),
                **diagnostics,
            }
        )
    return returns_by_variant, positions_by_variant, raw_positions_by_variant, pd.DataFrame(rows)


def calibrate_event_edges(
    *,
    residual_z: pd.DataFrame,
    future: pd.DataFrame,
    gates: dict[str, pd.Series],
    train_mask: pd.Series,
    roots: tuple[str, ...],
    cost_bps: pd.Series,
    primary_horizon: int,
    entry_z: float,
    cost_multiple: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for gate_name, gate in gates.items():
        for root in roots:
            signal = residual_z[root]
            selected = train_mask & gate & (signal.abs() >= entry_z)
            signed_future = future[(root, primary_horizon)] * np.sign(signal)
            stats = return_stats(signed_future[selected])
            one_way_bps = approximate_single_root_basket_one_way_cost(
                root,
                cost_bps.dropna(),
            )
            round_trip_bps = 2.0 * one_way_bps
            hurdle_bps = cost_multiple * round_trip_bps
            rows.append(
                {
                    "gate": gate_name,
                    "root": root,
                    "horizon": primary_horizon,
                    "entry_z": entry_z,
                    "one_way_cost_bps": one_way_bps,
                    "round_trip_cost_bps": round_trip_bps,
                    "cost_hurdle_multiple": cost_multiple,
                    "hurdle_bps": hurdle_bps,
                    "clears_cost_hurdle": bool(stats["mean_bps"] > hurdle_bps)
                    if np.isfinite(stats["mean_bps"])
                    else False,
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def residual_momentum_event_study(
    *,
    residual_z: pd.DataFrame,
    future: pd.DataFrame,
    gates: dict[str, pd.Series],
    valid_mask: pd.Series,
    roots: tuple[str, ...],
    horizons: tuple[int, ...],
    entry_z: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for gate_name, gate in gates.items():
        for root in roots:
            signal = residual_z[root]
            selected = valid_mask & gate & (signal.abs() >= entry_z)
            for horizon in horizons:
                signed_future = future[(root, horizon)] * np.sign(signal)
                rows.append(
                    {
                        "gate": gate_name,
                        "root": root,
                        "horizon": horizon,
                        **return_stats(signed_future[selected]),
                    }
                )
    return pd.DataFrame(rows)


def portfolio_forward_scatter(
    *,
    positions_by_variant: dict[str, pd.DataFrame],
    future: pd.DataFrame,
    roots: tuple[str, ...],
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant, positions in positions_by_variant.items():
        for horizon in horizons:
            future_by_root = future.xs(horizon, axis=1, level="horizon").reindex(columns=roots)
            forward_return = (positions.reindex(columns=roots) * future_by_root).sum(axis=1)
            signal_strength = positions.abs().sum(axis=1)
            frame = pd.DataFrame(
                {"signal_strength": signal_strength, "forward_return": forward_return}
            )
            frame = frame.loc[frame["signal_strength"] > 0].replace([np.inf, -np.inf], np.nan)
            frame = frame.dropna()
            rows.append(
                {
                    "variant": variant,
                    "horizon": horizon,
                    **correlation_stats(frame["signal_strength"], frame["forward_return"]),
                    **return_stats(frame["forward_return"]),
                }
            )
    return pd.DataFrame(rows)


def run_split_metrics(
    returns_by_variant: dict[str, pd.DataFrame],
    positions_by_variant: dict[str, pd.DataFrame],
    *,
    split_index: int,
    test_start_index: int,
    periods_per_year: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant, frame in returns_by_variant.items():
        positions = positions_by_variant[variant]
        for split, segment in (
            ("train", frame.iloc[:split_index]),
            ("test_purged", frame.iloc[test_start_index:]),
        ):
            metrics = calculate_strategy_metrics(segment, periods_per_year)
            rows.append(
                {
                    "variant": variant,
                    "split": split,
                    "start_index": 0 if split == "train" else test_start_index,
                    "end_index": split_index - 1 if split == "train" else len(frame) - 1,
                    "gross_exposure_mean": float(
                        positions.abs().sum(axis=1).iloc[segment.index].mean()
                    )
                    if len(segment)
                    else 0.0,
                    **metrics.to_dict(),
                }
            )
    return pd.DataFrame(rows)


def summarize_monthly_returns(
    returns_by_variant: dict[str, pd.DataFrame],
    timestamps: pd.Series,
) -> pd.DataFrame:
    rows = []
    month = pd.to_datetime(timestamps, utc=True).dt.to_period("M").astype(str)
    for variant, frame in returns_by_variant.items():
        monthly = (
            frame.assign(month=month)
            .groupby("month", as_index=False)[
                ["gross_return", "cost_return", "net_return", "turnover"]
            ]
            .sum()
        )
        monthly["variant"] = variant
        rows.append(monthly)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def gate_summary(
    *,
    gates: dict[str, pd.Series],
    valid_mask: pd.Series,
    bars: pd.DataFrame,
    md_thresholds: dict[str, pd.Series],
    anomalies: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {"metric": "bars", "value": float(len(bars))},
        {"metric": "valid_mask_fraction", "value": float(valid_mask.mean())},
    ]
    for name, gate in gates.items():
        rows.append({"metric": f"{name}_fraction", "value": float(gate.mean())})
        rows.append({"metric": f"{name}_count", "value": float(gate.sum())})
    for name, threshold in md_thresholds.items():
        rows.append({"metric": f"{name}_threshold_median", "value": float(threshold.median())})
    rows.append({"metric": "md_rolling_median", "value": float(anomalies["md_rolling"].median())})
    return pd.DataFrame(rows)


def write_outputs(
    *,
    out_dir: Path,
    plot_dir: Path,
    payload: dict[str, Any],
    source_dir: Path,
    bars: pd.DataFrame,
    residual_z: pd.DataFrame,
    anomalies: pd.DataFrame,
    returns_by_variant: dict[str, pd.DataFrame],
    positions_by_variant: dict[str, pd.DataFrame],
    raw_positions_by_variant: dict[str, pd.DataFrame],
    metrics: pd.DataFrame,
    split_metrics: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    event_study: pd.DataFrame,
    portfolio_scatter: pd.DataFrame,
    gate_diagnostics: pd.DataFrame,
    calibration: pd.DataFrame,
    cost_estimates: pd.DataFrame,
    roots: tuple[str, ...],
    target_roots: tuple[str, ...],
    primary_horizon: int,
    split_index: int,
    test_start_index: int,
) -> None:
    metrics.to_csv(out_dir / "strategy_variant_metrics.csv", index=False)
    split_metrics.to_csv(out_dir / "split_metrics.csv", index=False)
    monthly_returns.to_csv(out_dir / "monthly_returns.csv", index=False)
    event_study.to_csv(out_dir / "residual_momentum_event_study.csv", index=False)
    portfolio_scatter.to_csv(out_dir / "portfolio_forward_scatter.csv", index=False)
    gate_diagnostics.to_csv(out_dir / "gate_diagnostics.csv", index=False)
    calibration.to_csv(out_dir / "event_edge_calibration.csv", index=False)
    cost_estimates.to_csv(out_dir / "cost_estimates.csv", index=False)

    for variant, frame in returns_by_variant.items():
        frame.to_parquet(out_dir / f"{variant}_returns.parquet", index=False)
    for variant, frame in positions_by_variant.items():
        frame.to_parquet(out_dir / f"{variant}_positions.parquet", index=False)
    for variant, frame in raw_positions_by_variant.items():
        frame.to_parquet(out_dir / f"{variant}_raw_positions.parquet", index=False)

    timestamps = pd.to_datetime(bars["end_ts"], utc=True)
    plot_equity_and_drawdown(plot_dir, timestamps, returns_by_variant)
    plot_metrics(plot_dir, metrics)
    plot_split_metrics(plot_dir, split_metrics)
    plot_monthly_returns(plot_dir, monthly_returns)
    plot_event_study(plot_dir, event_study, primary_horizon)
    plot_portfolio_scatter(
        plot_dir,
        positions_by_variant,
        future_returns(
            pd.read_parquet(source_dir / "bar_returns.parquet").reindex(columns=roots),
            (primary_horizon,),
        ),
        roots,
        primary_horizon,
    )
    plot_residuals_with_flow(plot_dir, timestamps, residual_z, anomalies, target_roots)

    summary = build_summary(
        payload=payload,
        source_dir=source_dir,
        metrics=metrics,
        split_metrics=split_metrics,
        event_study=event_study,
        portfolio_scatter=portfolio_scatter,
        gate_diagnostics=gate_diagnostics,
        calibration=calibration,
        roots=roots,
        target_roots=target_roots,
        primary_horizon=primary_horizon,
        split_index=split_index,
        test_start_index=test_start_index,
    )
    (out_dir / "results.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    write_report(out_dir, summary)


def build_summary(
    *,
    payload: dict[str, Any],
    source_dir: Path,
    metrics: pd.DataFrame,
    split_metrics: pd.DataFrame,
    event_study: pd.DataFrame,
    portfolio_scatter: pd.DataFrame,
    gate_diagnostics: pd.DataFrame,
    calibration: pd.DataFrame,
    roots: tuple[str, ...],
    target_roots: tuple[str, ...],
    primary_horizon: int,
    split_index: int,
    test_start_index: int,
) -> dict[str, Any]:
    trading_metrics = metrics.loc[metrics["entries"] > 0].copy()
    best_net = trading_metrics.sort_values("net_return", ascending=False).head(1)
    best_oos = (
        split_metrics.loc[
            (split_metrics["split"] == "test_purged") & (split_metrics["turnover"] > 0.0)
        ]
        .sort_values("net_return", ascending=False)
        .head(1)
    )
    passing = metrics.loc[
        (metrics["entries"] > 0) & (metrics["net_return"] > 0.0) & (metrics["gross_to_cost"] >= 3.0)
    ].copy()
    calibration_primary = calibration.loc[
        (calibration["horizon"] == primary_horizon) & (calibration["gate"] == "md_q99")
    ].copy()
    return {
        "experiment_id": payload["experiment"]["id"],
        "title": payload["experiment"]["title"],
        "completed_at": datetime.now(UTC).isoformat(),
        "source_experiment": str(source_dir),
        "roots": list(roots),
        "target_roots": list(target_roots),
        "primary_horizon": primary_horizon,
        "split_index": split_index,
        "test_start_index": test_start_index,
        "best_net_variant": best_net.to_dict(orient="records"),
        "best_test_variant": best_oos.to_dict(orient="records"),
        "variants_passing_3x_cost": passing.to_dict(orient="records"),
        "variant_metrics": metrics.to_dict(orient="records"),
        "split_metrics": split_metrics.to_dict(orient="records"),
        "md_q99_calibration": calibration_primary.to_dict(orient="records"),
        "event_study_primary_horizon": event_study.loc[
            event_study["horizon"] == primary_horizon
        ].to_dict(orient="records"),
        "portfolio_scatter_primary_horizon": portfolio_scatter.loc[
            portfolio_scatter["horizon"] == primary_horizon
        ].to_dict(orient="records"),
        "gate_diagnostics": gate_diagnostics.to_dict(orient="records"),
    }


def write_report(out_dir: Path, summary: dict[str, Any]) -> None:
    best = summary["best_net_variant"][0] if summary["best_net_variant"] else {}
    best_test = summary["best_test_variant"][0] if summary["best_test_variant"] else {}
    passing_count = len(summary["variants_passing_3x_cost"])
    calibration = pd.DataFrame(summary["md_q99_calibration"])
    cal_lines = []
    if not calibration.empty:
        cal_lines.extend(
            (
                f"| {row.root} | {row.observations:,} | {row.mean_bps:.2f} | "
                f"{row.round_trip_cost_bps:.2f} | {row.hurdle_bps:.2f} | "
                f"{row.clears_cost_hurdle!s} |"
            )
            for row in calibration.itertuples(index=False)
        )
    metrics = pd.DataFrame(summary["variant_metrics"])
    metric_lines = []
    if not metrics.empty:
        metric_lines.extend(
            (
                f"| `{row.variant}` | {row.entries:,} | {row.gross_return:.4f} | "
                f"{row.cost_return:.4f} | {row.net_return:.4f} | "
                f"{row.gross_to_cost:.2f} | {row.max_drawdown:.4f} |"
            )
            for row in metrics.itertuples(index=False)
        )

    report = f"""# {summary["experiment_id"]} Residual-State Momentum Flow Filter

## Purpose

This experiment tests whether residual-state momentum becomes executable when restricted to extreme cross-sectional flow-geometry regimes.

Source framework: `{summary["source_experiment"]}`

Universe: `{", ".join(summary["roots"])}`

Target roots: `{", ".join(summary["target_roots"])}`

Primary horizon: `{summary["primary_horizon"]}` bars.

## Strategy Definition

The strategy enters in the direction of the residual z-score:

- positive residual z-score: long the displaced root, short the normalized complex hedge;
- negative residual z-score: short the displaced root, long the normalized complex hedge.

Positions are dollar-normalized after cross-sectional demeaning. Exits occur on residual decay, sign reversal, max holding period, stop z-score, invalid price marks, or cooldown rules.

Flow filters use shifted walk-forward Mahalanobis thresholds. The q99 variants therefore do not use the full-sample q99 label from the research framework.

## Cost Hurdle Calibration

The 3x-cost gate is calibrated on the training split only.

| Root | Train events | Mean signed h{summary["primary_horizon"]} bps | Round-trip cost bps | 3x hurdle bps | Clears |
|---|---:|---:|---:|---:|---|
{chr(10).join(cal_lines)}

## Variant Results

| Variant | Entries | Gross | Cost | Net | Gross/Cost | Max DD |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(metric_lines)}

Best nonzero full-sample variant: `{best.get("variant", "n/a")}`, net `{best.get("net_return", float("nan")):.4f}`, gross/cost `{best.get("gross_to_cost", float("nan")):.2f}`.

Best nonzero purged-test variant: `{best_test.get("variant", "n/a")}`, net `{best_test.get("net_return", float("nan")):.4f}`, gross/cost `{best_test.get("gross_to_cost", float("nan")):.2f}`.

Variants passing positive net and 3x gross/cost: `{passing_count}`.

## Conclusion

The direct PA/PL residual-state momentum thesis is rejected as an executable strategy in this implementation. The q99 flow filter reduces turnover and losses, but it does not flip gross returns positive. The 3x-cost gate admits no roots because the training-split q99 expected moves are below the required hurdle.

The remaining research lead is not direct own-residual momentum. If this family is revisited, the next test should be cross-residual predictors, such as `HG` residual state predicting `PL`, with a vectorized state machine and the same train/test and 3x-cost gates.

## Artifacts

- `strategy_variant_metrics.csv`
- `split_metrics.csv`
- `monthly_returns.csv`
- `event_edge_calibration.csv`
- `residual_momentum_event_study.csv`
- `portfolio_forward_scatter.csv`
- `plots/`
"""
    (out_dir / "report.qmd").write_text(report, encoding="utf-8")


def target_root_mask(
    index: pd.Index,
    columns: pd.Index,
    target_roots: tuple[str, ...],
) -> pd.DataFrame:
    mask = pd.DataFrame(False, index=index, columns=columns)
    for root in target_roots:
        if root in mask.columns:
            mask[root] = True
    return mask


def position_diagnostics(
    raw_positions: pd.DataFrame,
    positions: pd.DataFrame,
) -> dict[str, float | int]:
    raw_prev = raw_positions.shift(1).fillna(0.0)
    entries = ((raw_positions != 0.0) & (raw_prev == 0.0)).sum().sum()
    exits = ((raw_positions == 0.0) & (raw_prev != 0.0)).sum().sum()
    active_roots = (raw_positions != 0.0).sum(axis=1)
    return {
        "entries": int(entries),
        "exits": int(exits),
        "root_active_bars": int(active_roots.sum()),
        "max_active_roots": int(active_roots.max()) if len(active_roots) else 0,
        "mean_gross_exposure": float(positions.abs().sum(axis=1).mean()),
    }


def approximate_single_root_basket_one_way_cost(root: str, cost_bps: pd.Series) -> float:
    if root not in cost_bps.index or len(cost_bps) < 2:
        return float(cost_bps.mean()) if len(cost_bps) else np.nan
    hedge_roots = [item for item in cost_bps.index if item != root]
    root_weight = 0.5
    hedge_weight = 0.5 / len(hedge_roots)
    return float(root_weight * cost_bps[root] + hedge_weight * cost_bps[hedge_roots].sum())


def shifted_rolling_quantile(
    series: pd.Series,
    quantile: float,
    window: int,
    min_periods: int,
) -> pd.Series:
    return series.shift(1).rolling(window=window, min_periods=min_periods).quantile(quantile)


def periods_per_year_from_bars(bars: pd.DataFrame) -> float:
    elapsed = (
        pd.to_datetime(bars["end_ts"], utc=True).max()
        - pd.to_datetime(bars["start_ts"], utc=True).min()
    ).total_seconds()
    elapsed_years = elapsed / (365.25 * 86_400.0)
    return len(bars) / elapsed_years if elapsed_years > 0 else float(len(bars))


def split_points(
    bars: pd.DataFrame,
    *,
    train_fraction: float,
    embargo_bars: int,
) -> tuple[int, int]:
    split_index = int(len(bars) * train_fraction)
    split_index = min(max(split_index, 1), len(bars) - 1)
    test_start_index = min(split_index + embargo_bars, len(bars))
    return split_index, test_start_index


def return_stats(values: pd.Series) -> dict[str, float | int]:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    observations = len(clean)
    if observations == 0:
        return {
            "observations": 0,
            "mean_bps": np.nan,
            "median_bps": np.nan,
            "vol_bps": np.nan,
            "tstat": np.nan,
            "hit_rate": np.nan,
        }
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if observations > 1 else 0.0
    return {
        "observations": observations,
        "mean_bps": mean * 10_000.0,
        "median_bps": float(clean.median()) * 10_000.0,
        "vol_bps": std * 10_000.0,
        "tstat": mean / (std / math.sqrt(observations)) if std > 0 else np.nan,
        "hit_rate": float((clean > 0.0).mean()),
    }


def correlation_stats(x_values: pd.Series, y_values: pd.Series) -> dict[str, float | int]:
    frame = pd.DataFrame({"x": x_values, "y": y_values}).replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna()
    if len(frame) < 3 or frame["x"].std(ddof=1) == 0.0 or frame["y"].std(ddof=1) == 0.0:
        return {"scatter_observations": len(frame), "correlation": np.nan, "slope": np.nan}
    x = frame["x"].to_numpy(dtype=float)
    y = frame["y"].to_numpy(dtype=float)
    beta = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
    return {
        "scatter_observations": len(frame),
        "correlation": float(np.corrcoef(x, y)[0, 1]),
        "slope": float(beta[1]),
    }


def plot_equity_and_drawdown(
    plot_dir: Path,
    timestamps: pd.Series,
    returns_by_variant: dict[str, pd.DataFrame],
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for variant, frame in returns_by_variant.items():
        equity = frame["net_return"].cumsum()
        axes[0].plot(timestamps, equity * 100.0, label=variant, linewidth=0.9)
        axes[1].plot(timestamps, (equity - equity.cummax()) * 100.0, label=variant, linewidth=0.9)
    axes[0].set_title("Residual-state momentum variants: net cumulative return")
    axes[0].set_ylabel("Cumulative log return (%)")
    axes[1].set_title("Net drawdown")
    axes[1].set_ylabel("Drawdown (%)")
    axes[0].legend(loc="best", fontsize=7)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(plot_dir / "variant_net_equity_drawdown.png", dpi=160)
    plt.close(fig)


def plot_metrics(plot_dir: Path, metrics: pd.DataFrame) -> None:
    if metrics.empty:
        return
    data = metrics.set_index("variant")[["gross_return", "cost_return", "net_return"]]
    fig, ax = plt.subplots(figsize=(12, 5))
    data.plot.bar(ax=ax)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Strategy variant total returns")
    ax.set_ylabel("Log return")
    fig.tight_layout()
    fig.savefig(plot_dir / "strategy_variant_returns.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    metrics.set_index("variant")["gross_to_cost"].plot.bar(ax=ax)
    ax.axhline(3.0, color="red", linewidth=1.0, linestyle="--")
    ax.set_title("Gross return to cost")
    ax.set_ylabel("Gross/Cost")
    fig.tight_layout()
    fig.savefig(plot_dir / "gross_to_cost.png", dpi=160)
    plt.close(fig)


def plot_split_metrics(plot_dir: Path, split_metrics: pd.DataFrame) -> None:
    if split_metrics.empty:
        return
    pivot = split_metrics.pivot(index="variant", columns="split", values="net_return")
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot.bar(ax=ax)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Train vs purged-test net return")
    ax.set_ylabel("Log return")
    fig.tight_layout()
    fig.savefig(plot_dir / "train_test_net_returns.png", dpi=160)
    plt.close(fig)


def plot_monthly_returns(plot_dir: Path, monthly_returns: pd.DataFrame) -> None:
    if monthly_returns.empty:
        return
    best_variant = (
        monthly_returns.groupby("variant")["net_return"].sum().sort_values(ascending=False).index[0]
    )
    data = monthly_returns.loc[monthly_returns["variant"] == best_variant]
    fig, ax = plt.subplots(figsize=(12, 4))
    data.plot.bar(x="month", y="net_return", ax=ax, legend=False)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(f"Monthly net returns: {best_variant}")
    ax.set_ylabel("Log return")
    fig.tight_layout()
    fig.savefig(plot_dir / "best_variant_monthly_returns.png", dpi=160)
    plt.close(fig)


def plot_event_study(plot_dir: Path, event_study: pd.DataFrame, primary_horizon: int) -> None:
    data = event_study.loc[event_study["horizon"] == primary_horizon]
    if data.empty:
        return
    pivot = data.pivot_table(index="root", columns="gate", values="mean_bps")
    fig, ax = plt.subplots(figsize=(9, 4))
    pivot.plot.bar(ax=ax)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(f"Residual-momentum signed forward returns, h={primary_horizon}")
    ax.set_ylabel("Mean signed forward return (bps)")
    fig.tight_layout()
    fig.savefig(plot_dir / f"event_signed_forward_h{primary_horizon}.png", dpi=160)
    plt.close(fig)


def plot_portfolio_scatter(
    plot_dir: Path,
    positions_by_variant: dict[str, pd.DataFrame],
    future: pd.DataFrame,
    roots: tuple[str, ...],
    primary_horizon: int,
) -> None:
    variants = list(positions_by_variant)
    cols = 2
    rows = math.ceil(len(variants) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), squeeze=False)
    future_by_root = future.xs(primary_horizon, axis=1, level="horizon").reindex(columns=roots)
    for ax, variant in zip(axes.ravel(), variants, strict=False):
        positions = positions_by_variant[variant].reindex(columns=roots)
        signal = positions.abs().sum(axis=1)
        forward_return = (positions * future_by_root).sum(axis=1)
        frame = pd.DataFrame({"signal": signal, "future": forward_return})
        frame = frame.loc[frame["signal"] > 0].replace([np.inf, -np.inf], np.nan).dropna()
        if len(frame) > 20_000:
            frame = frame.sample(n=20_000, random_state=7)
        ax.scatter(frame["signal"], frame["future"] * 10_000.0, s=4, alpha=0.15)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(variant)
        ax.set_xlabel("Gross exposure")
        ax.set_ylabel(f"Forward basket return h={primary_horizon} (bps)")
    for ax in axes.ravel()[len(variants) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(plot_dir / f"portfolio_signal_forward_scatter_h{primary_horizon}.png", dpi=160)
    plt.close(fig)


def plot_residuals_with_flow(
    plot_dir: Path,
    timestamps: pd.Series,
    residual_z: pd.DataFrame,
    anomalies: pd.DataFrame,
    target_roots: tuple[str, ...],
) -> None:
    fig, axes = plt.subplots(len(target_roots) + 1, 1, figsize=(12, 3 * (len(target_roots) + 1)))
    axes = np.ravel(axes)
    axes[0].plot(timestamps, anomalies["md_rolling"], linewidth=0.8)
    axes[0].set_title("Rolling Mahalanobis distance")
    for axis, root in zip(axes[1:], target_roots, strict=True):
        axis.plot(timestamps, residual_z[root], linewidth=0.7)
        axis.axhline(2.0, color="red", linestyle="--", linewidth=0.7)
        axis.axhline(-2.0, color="red", linestyle="--", linewidth=0.7)
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_title(f"{root} fair-value residual z-score")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(plot_dir / "target_residuals_and_mahalanobis.png", dpi=160)
    plt.close(fig)


def json_safe(value: Any) -> Any:  # noqa: PLR0911
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    return value


if __name__ == "__main__":
    main()
