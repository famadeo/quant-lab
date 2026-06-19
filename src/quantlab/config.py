from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class DataConfig(BaseModel):
    source: Literal["synthetic", "csv", "parquet"] = "synthetic"
    path: Path | None = None
    symbol: str = "SYNTH"
    start: str = "2020-01-01"
    end: str = "2022-12-31"
    timestamp_column: str = "timestamp"
    price_column: str = "close"
    seed: int = 7

    @model_validator(mode="after")
    def require_path_for_file_sources(self) -> DataConfig:
        if self.source in {"csv", "parquet"} and self.path is None:
            raise ValueError(f"data.path is required when data.source is {self.source}")
        return self


class StrategyConfig(BaseModel):
    name: Literal["moving_average_crossover"] = "moving_average_crossover"
    fast_window: int = Field(default=10, ge=2)
    slow_window: int = Field(default=40, ge=3)

    @model_validator(mode="after")
    def validate_windows(self) -> StrategyConfig:
        if self.fast_window >= self.slow_window:
            raise ValueError("fast_window must be less than slow_window")
        return self


class BacktestConfig(BaseModel):
    initial_capital: float = Field(default=100_000.0, gt=0)
    fee_bps: float = Field(default=0.5, ge=0)
    slippage_bps: float = Field(default=1.0, ge=0)
    signal_lag: int = Field(default=1, ge=1)
    max_leverage: float = Field(default=1.0, gt=0)
    periods_per_year: int = Field(default=252, gt=0)


class OutputConfig(BaseModel):
    directory: Path
    results_file: str = "results.json"
    equity_curve_file: str = "equity_curve.parquet"


class MlflowConfig(BaseModel):
    experiment_name: str = "quant-lab"
    tracking_uri: str = "sqlite:///mlflow.db"


class DecisionConfig(BaseModel):
    status: Literal["reject", "revise", "paper_trade", "archive"] = "revise"
    notes: str = "Smoke run only. No trading claim."


class ExperimentConfig(BaseModel):
    experiment_id: str
    title: str
    hypothesis: str
    data: DataConfig = Field(default_factory=DataConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    outputs: OutputConfig
    mlflow: MlflowConfig = Field(default_factory=MlflowConfig)
    tags: dict[str, str] = Field(default_factory=dict)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentConfig:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a YAML mapping")
        payload = _resolve_relative_paths(payload, path.parent)
        return cls.model_validate(payload)


def _resolve_relative_paths(payload: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Resolve file paths relative to the config file location."""
    data = payload.get("data")
    if isinstance(data, dict) and data.get("path"):
        path = Path(data["path"])
        data["path"] = str(path if path.is_absolute() else (base_dir / path).resolve())

    outputs = payload.get("outputs")
    if isinstance(outputs, dict) and outputs.get("directory"):
        path = Path(outputs["directory"])
        outputs["directory"] = str(path if path.is_absolute() else (base_dir / path).resolve())

    return payload


def flatten_model(model: BaseModel, prefix: str = "") -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, value in model.model_dump().items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            values.update(_flatten_dict(value, full_key))
        else:
            values[full_key] = value
    return values


def _flatten_dict(payload: dict[str, Any], prefix: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, value in payload.items():
        full_key = f"{prefix}.{key}"
        if isinstance(value, dict):
            values.update(_flatten_dict(value, full_key))
        else:
            values[full_key] = value
    return values
