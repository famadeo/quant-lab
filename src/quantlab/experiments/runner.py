from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import polars as pl

from quantlab.backtests import BacktestMetrics, run_long_only_backtest
from quantlab.config import ExperimentConfig, flatten_model
from quantlab.data import load_bars
from quantlab.signals import moving_average_crossover


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    metrics: BacktestMetrics
    results_path: Path
    equity_curve_path: Path
    mlflow_run_id: str


def run_experiment(config_path: Path, tracking_uri: str | None = None) -> ExperimentResult:
    config = ExperimentConfig.from_yaml(config_path)
    output_dir = config.outputs.directory
    output_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking_uri or config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    bars = load_bars(config.data)
    signal_frame = _build_signal_frame(bars, config)
    backtest_frame, metrics = run_long_only_backtest(signal_frame, config.backtest)

    equity_curve_path = output_dir / config.outputs.equity_curve_file
    results_path = output_dir / config.outputs.results_file
    backtest_frame.write_parquet(equity_curve_path)

    with mlflow.start_run(run_name=config.experiment_id) as run:
        mlflow.set_tags({"experiment_id": config.experiment_id, **config.tags})
        for key, value in flatten_model(config).items():
            if _is_loggable_param(value):
                mlflow.log_param(key, value)
        for key, value in metrics.to_dict().items():
            mlflow.log_metric(key, float(value))
        mlflow.log_artifact(str(config_path))
        mlflow.log_artifact(str(equity_curve_path))
        run_id = run.info.run_id

    payload = {
        "experiment_id": config.experiment_id,
        "title": config.title,
        "completed_at": datetime.now(UTC).isoformat(),
        "decision": config.decision.model_dump(),
        "metrics": metrics.to_dict(),
        "artifacts": {
            "equity_curve": str(equity_curve_path),
            "mlflow_run_id": run_id,
        },
    }
    results_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return ExperimentResult(
        experiment_id=config.experiment_id,
        metrics=metrics,
        results_path=results_path,
        equity_curve_path=equity_curve_path,
        mlflow_run_id=run_id,
    )


def _build_signal_frame(bars: pl.DataFrame, config: ExperimentConfig) -> pl.DataFrame:
    if config.strategy.name == "moving_average_crossover":
        return moving_average_crossover(
            bars,
            fast_window=config.strategy.fast_window,
            slow_window=config.strategy.slow_window,
            price_column=config.data.price_column,
        )
    raise ValueError(f"unsupported strategy: {config.strategy.name}")


def _is_loggable_param(value: Any) -> bool:
    return value is None or isinstance(value, str | int | float | bool)
