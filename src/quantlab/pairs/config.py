from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class PairsDataConfig(BaseModel):
    root_dir: Path
    roots: list[str]
    asset_classes: dict[str, list[str]]
    start: datetime | None = None
    end: datetime | None = None
    timestamp_column: str = "ts"
    price_column: str = "cont_logprice"
    return_column: str = "cont_logret"

    @model_validator(mode="after")
    def require_roots(self) -> PairsDataConfig:
        if not self.roots:
            raise ValueError("data.roots must not be empty")
        return self


class PairsStrategyConfig(BaseModel):
    pair_scope: Literal["intra_asset_class"] = "intra_asset_class"
    lookback: int = Field(default=96, ge=10)
    min_pair_observations: int = Field(default=500, ge=20)
    z_entry: float = Field(default=2.0, gt=0)
    z_exit: float = Field(default=0.5, ge=0)
    mahalanobis_entry: float = Field(default=2.45, gt=0)
    mahalanobis_exit: float = Field(default=1.0, ge=0)
    covariance_ridge: float = Field(default=1e-6, gt=0)
    signal_lag: int = Field(default=1, ge=1)
    rebalance_every_bars: int = Field(default=1, ge=1)
    min_holding_bars: int = Field(default=0, ge=0)
    cooldown_bars: int = Field(default=0, ge=0)
    min_position_change: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def validate_exits(self) -> PairsStrategyConfig:
        if self.z_exit >= self.z_entry:
            raise ValueError("z_exit must be less than z_entry")
        if self.mahalanobis_exit >= self.mahalanobis_entry:
            raise ValueError("mahalanobis_exit must be less than mahalanobis_entry")
        if self.min_position_change > 1:
            raise ValueError("min_position_change must be less than or equal to 1")
        return self


class PairsSelectionConfig(BaseModel):
    enabled: bool = True
    train_fraction: float = Field(default=0.6, gt=0, lt=1)
    min_train_observations: int = Field(default=500, ge=20)
    min_test_observations: int = Field(default=200, ge=20)
    min_abs_return_correlation: float = Field(default=0.10, ge=0, le=1)
    max_spread_adf_pvalue: float = Field(default=0.10, gt=0, le=1)
    max_half_life_bars: float = Field(default=288.0, gt=0)
    min_average_pair_volume: float = Field(default=10.0, ge=0)
    adf_max_lag: int | None = None


class PairsBacktestConfig(BaseModel):
    fee_bps: float = Field(default=0.5, ge=0)
    slippage_bps: float = Field(default=1.0, ge=0)
    periods_per_year: int = Field(default=69_552, gt=0)


class PairsOutputConfig(BaseModel):
    directory: Path
    results_file: str = "results.json"
    pair_selection_file: str = "pair_selection.csv"
    pair_metrics_file: str = "pair_metrics.csv"
    portfolio_returns_file: str = "portfolio_returns.parquet"
    pair_returns_file: str = "pair_returns.parquet"


class PairsMlflowConfig(BaseModel):
    experiment_name: str = "quant-lab-pairs"
    tracking_uri: str = "sqlite:///mlflow.db"


class PairsDecisionConfig(BaseModel):
    status: Literal["reject", "revise", "paper_trade", "archive"] = "revise"
    notes: str = "Initial comparison only. No trading claim."


class PairsExperimentConfig(BaseModel):
    experiment_id: str
    title: str
    hypothesis: str
    data: PairsDataConfig
    selection: PairsSelectionConfig = Field(default_factory=PairsSelectionConfig)
    strategy: PairsStrategyConfig = Field(default_factory=PairsStrategyConfig)
    backtest: PairsBacktestConfig = Field(default_factory=PairsBacktestConfig)
    outputs: PairsOutputConfig
    mlflow: PairsMlflowConfig = Field(default_factory=PairsMlflowConfig)
    tags: dict[str, str] = Field(default_factory=dict)
    decision: PairsDecisionConfig = Field(default_factory=PairsDecisionConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> PairsExperimentConfig:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a YAML mapping")
        payload = _resolve_relative_paths(payload, path.parent)
        return cls.model_validate(payload)


def _resolve_relative_paths(payload: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    data = payload.get("data")
    if isinstance(data, dict) and data.get("root_dir"):
        root_dir = Path(data["root_dir"])
        data["root_dir"] = str(
            root_dir if root_dir.is_absolute() else (base_dir / root_dir).resolve()
        )

    outputs = payload.get("outputs")
    if isinstance(outputs, dict) and outputs.get("directory"):
        output_dir = Path(outputs["directory"])
        outputs["directory"] = str(
            output_dir if output_dir.is_absolute() else (base_dir / output_dir).resolve()
        )

    return payload
