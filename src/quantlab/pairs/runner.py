from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import mlflow
import pandas as pd

from quantlab.config import flatten_model
from quantlab.experiments.runner import _is_loggable_param
from quantlab.pairs.config import PairsExperimentConfig
from quantlab.pairs.data import align_pair, build_intra_asset_class_pairs, load_continuous_5m_roots
from quantlab.pairs.selection import evaluate_pair_selection
from quantlab.pairs.strategy import (
    ReturnMetrics,
    calculate_metrics,
    run_mahalanobis_pair,
    run_zscore_mahalanobis_filter_pair,
    run_zscore_pair,
)


@dataclass(frozen=True)
class PairsExperimentResult:
    experiment_id: str
    method_metrics: dict[str, ReturnMetrics]
    results_path: Path
    pair_selection_path: Path
    pair_metrics_path: Path
    portfolio_returns_path: Path
    pair_returns_path: Path
    mlflow_run_id: str


def run_pairs_experiment(
    config_path: Path,
    tracking_uri: str | None = None,
) -> PairsExperimentResult:
    config = PairsExperimentConfig.from_yaml(config_path)
    output_dir = config.outputs.directory
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = load_continuous_5m_roots(config.data)
    pairs = build_intra_asset_class_pairs(config.data.asset_classes, set(frames))
    pair_selection, pair_returns, pair_metrics = _evaluate_candidate_pairs(config, frames, pairs)
    portfolio_returns = _build_portfolio_returns(pair_returns)
    method_metrics: dict[str, ReturnMetrics] = {
        str(method): calculate_metrics(
            cast(pd.Series, group["portfolio_return"]),
            cast(pd.Series, group["active_pairs"] > 0),
            cast(pd.Series, group["turnover"]),
            config.backtest.periods_per_year,
        )
        for method, group in portfolio_returns.groupby("method")
    }

    results_path = output_dir / config.outputs.results_file
    pair_selection_path = output_dir / config.outputs.pair_selection_file
    pair_metrics_path = output_dir / config.outputs.pair_metrics_file
    portfolio_returns_path = output_dir / config.outputs.portfolio_returns_file
    pair_returns_path = output_dir / config.outputs.pair_returns_file

    pair_selection.to_csv(pair_selection_path, index=False)
    pair_metrics.to_csv(pair_metrics_path, index=False)
    portfolio_returns.to_parquet(portfolio_returns_path, index=False)
    pair_returns.to_parquet(pair_returns_path, index=False)

    mlflow.set_tracking_uri(tracking_uri or config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)
    with mlflow.start_run(run_name=config.experiment_id) as run:
        mlflow.set_tags({"experiment_id": config.experiment_id, **config.tags})
        for key, value in flatten_model(config).items():
            if _is_loggable_param(value):
                mlflow.log_param(key, value)
        for method, metrics in method_metrics.items():
            for key, value in metrics.to_dict().items():
                mlflow.log_metric(f"{method}.{key}", float(value))
        mlflow.log_artifact(str(config_path))
        mlflow.log_artifact(str(pair_selection_path))
        mlflow.log_artifact(str(pair_metrics_path))
        mlflow.log_artifact(str(portfolio_returns_path))
        run_id = run.info.run_id

    payload = {
        "experiment_id": config.experiment_id,
        "title": config.title,
        "completed_at": datetime.now(UTC).isoformat(),
        "decision": config.decision.model_dump(),
        "candidate_pairs": int(pair_selection["pair"].nunique()),
        "selected_pairs": int(pair_metrics["pair"].nunique()),
        "loaded_roots": sorted(frames),
        "method_metrics": {
            method: metrics.to_dict() for method, metrics in sorted(method_metrics.items())
        },
        "top_pairs_by_sharpe": _top_pairs_by_sharpe(pair_metrics),
        "selection_reasons": _selection_reason_counts(pair_selection),
        "artifacts": {
            "pair_selection": str(pair_selection_path),
            "pair_metrics": str(pair_metrics_path),
            "portfolio_returns": str(portfolio_returns_path),
            "pair_returns": str(pair_returns_path),
            "mlflow_run_id": run_id,
        },
    }
    results_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return PairsExperimentResult(
        experiment_id=config.experiment_id,
        method_metrics=method_metrics,
        results_path=results_path,
        pair_selection_path=pair_selection_path,
        pair_metrics_path=pair_metrics_path,
        portfolio_returns_path=portfolio_returns_path,
        pair_returns_path=pair_returns_path,
        mlflow_run_id=run_id,
    )


def _evaluate_candidate_pairs(
    config: PairsExperimentConfig,
    frames: dict[str, pd.DataFrame],
    pairs: list[tuple[str, str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selection_rows: list[dict[str, Any]] = []
    pair_results: list[pd.DataFrame] = []
    pair_metric_rows: list[dict[str, Any]] = []

    for asset_class, root_a, root_b in pairs:
        pair_frame = align_pair(root_a, root_b, frames)
        if len(pair_frame) < config.strategy.min_pair_observations:
            continue
        selection = evaluate_pair_selection(
            pair_frame, asset_class, root_a, root_b, config.selection
        )
        selection_rows.append(selection.to_dict())
        if not selection.selected:
            continue

        for method, runner in (
            ("zscore", run_zscore_pair),
            ("zscore_mahalanobis", run_zscore_mahalanobis_filter_pair),
            ("mahalanobis", run_mahalanobis_pair),
        ):
            result = (
                runner(pair_frame, config.strategy, config.backtest)
                .iloc[selection.test_start_index :]
                .copy()
            )
            result["asset_class"] = asset_class
            result["root_a"] = root_a
            result["root_b"] = root_b
            result["pair"] = f"{root_a}-{root_b}"
            pair_results.append(_select_pair_return_columns(result))
            metrics = calculate_metrics(
                cast(pd.Series, result["pair_return"]),
                cast(pd.Series, result["active"]),
                cast(pd.Series, result["turnover"]),
                config.backtest.periods_per_year,
            )
            pair_metric_rows.append(
                {
                    "method": method,
                    "asset_class": asset_class,
                    "pair": f"{root_a}-{root_b}",
                    "root_a": root_a,
                    "root_b": root_b,
                    "test_start_index": selection.test_start_index,
                    "return_correlation": selection.return_correlation,
                    "price_correlation": selection.price_correlation,
                    "spread_adf_pvalue": selection.spread_adf_pvalue,
                    "half_life_bars": selection.half_life_bars,
                    "average_pair_volume": selection.average_pair_volume,
                    **metrics.to_dict(),
                }
            )

    if not pair_results:
        raise ValueError("no selected pairs produced results")

    return (
        pd.DataFrame(selection_rows),
        pd.concat(pair_results, ignore_index=True),
        pd.DataFrame(pair_metric_rows),
    )


def _select_pair_return_columns(result: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "ts",
        "method",
        "asset_class",
        "pair",
        "root_a",
        "root_b",
        "indicator",
        "filter_indicator",
        "target_a",
        "target_b",
        "position_a",
        "position_b",
        "turnover",
        "cost",
        "pair_return",
        "active",
    ]
    return cast(pd.DataFrame, result.loc[:, columns].copy())


def _build_portfolio_returns(pair_returns: pd.DataFrame) -> pd.DataFrame:
    grouped = pair_returns.groupby(["method", "ts"], as_index=False).agg(
        portfolio_return=("pair_return", "mean"),
        active_pairs=("active", "sum"),
        observed_pairs=("pair", "nunique"),
        turnover=("turnover", "mean"),
    )
    grouped = grouped.set_index(["method", "ts"]).sort_index().reset_index()
    return cast(pd.DataFrame, grouped)


def _top_pairs_by_sharpe(pair_metrics: pd.DataFrame, limit: int = 10) -> list[dict[str, Any]]:
    columns = ["method", "asset_class", "pair", "sharpe_ratio", "total_return", "trades"]
    return (
        pair_metrics.sort_values("sharpe_ratio", ascending=False)
        .head(limit)
        .loc[:, columns]
        .to_dict(orient="records")
    )


def _selection_reason_counts(pair_selection: pd.DataFrame) -> dict[str, int]:
    counts = pair_selection["reason"].value_counts()
    return {str(reason): int(count) for reason, count in counts.items()}
