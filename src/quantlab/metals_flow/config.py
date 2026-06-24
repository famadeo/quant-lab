from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_ROOTS = ("GC", "SI", "HG", "PL", "PA", "ALI")
DEFAULT_THRESHOLDS = (100_000_000.0, 250_000_000.0, 500_000_000.0, 1_000_000_000.0)
DEFAULT_HORIZONS = (1, 2, 5, 10, 20, 50)

CONTRACT_MULTIPLIERS = {
    "GC": 100.0,
    "SI": 5_000.0,
    "HG": 25_000.0,
    "PL": 50.0,
    "PA": 100.0,
    "ALI": 25.0,
}


@dataclass(frozen=True)
class MetalsFlowConfig:
    experiment_id: str = "HYP-0012-metals-flow-anomaly-framework"
    title: str = "Metals cross-sectional flow anomaly framework"
    roots: tuple[str, ...] = DEFAULT_ROOTS
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS
    primary_threshold: float = 250_000_000.0
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    start: str = "2026-05-24T00:00:00Z"
    end: str = "2026-06-22T00:00:00Z"
    rolling_window: int = 500
    min_periods: int = 100
    ewma_halflife: int = 250
    fair_value_lookback: int = 500
    fair_value_min_periods: int = 250
    trade_dir: Path = Path(
        "/home/famadeo/research/databento-asset-browser/data/metals_trades_12m/outright"
    )
    mbp1_dir: Path = Path(
        "/home/famadeo/research/databento-asset-browser/data/metals_mbp1_30d/outright_chunks"
    )
    cache_dir: Path = Path("data/features/metals_cross_sectional_dollar_bars")
    output_dir: Path = Path("experiments/HYP-0012-metals-flow-anomaly-framework")

    @classmethod
    def from_yaml(cls, path: Path) -> MetalsFlowConfig:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a YAML mapping")
        return cls.from_mapping(payload, path.parent)

    @classmethod
    def from_mapping(
        cls,
        payload: dict[str, Any],
        base_dir: Path | None = None,
    ) -> MetalsFlowConfig:
        data = payload.get("data", {})
        universe = payload.get("universe", {})
        research = payload.get("research", {})
        outputs = payload.get("outputs", {})
        experiment = payload.get("experiment", {})

        def path_value(value: str | Path | None, default: Path) -> Path:
            if value is None:
                return default
            path = Path(value)
            if not path.is_absolute() and base_dir is not None:
                path = (base_dir / path).resolve()
            return path

        roots = tuple(str(root) for root in universe.get("roots", DEFAULT_ROOTS))
        thresholds = tuple(float(value) for value in research.get("thresholds", DEFAULT_THRESHOLDS))
        horizons = tuple(int(value) for value in research.get("horizons", DEFAULT_HORIZONS))
        primary_threshold = float(research.get("primary_threshold", 250_000_000.0))

        return cls(
            experiment_id=str(experiment.get("id", cls.experiment_id)),
            title=str(experiment.get("title", cls.title)),
            roots=roots,
            thresholds=thresholds,
            primary_threshold=primary_threshold,
            horizons=horizons,
            start=str(data.get("start", cls.start)),
            end=str(data.get("end", cls.end)),
            rolling_window=int(research.get("rolling_window", cls.rolling_window)),
            min_periods=int(research.get("min_periods", cls.min_periods)),
            ewma_halflife=int(research.get("ewma_halflife", cls.ewma_halflife)),
            fair_value_lookback=int(research.get("fair_value_lookback", cls.fair_value_lookback)),
            fair_value_min_periods=int(
                research.get("fair_value_min_periods", cls.fair_value_min_periods)
            ),
            trade_dir=path_value(data.get("trade_dir"), cls.trade_dir),
            mbp1_dir=path_value(data.get("mbp1_dir"), cls.mbp1_dir),
            cache_dir=path_value(data.get("cache_dir"), cls.cache_dir),
            output_dir=path_value(outputs.get("directory"), cls.output_dir),
        )

    @property
    def date_tag(self) -> str:
        start = self.start[:10].replace("-", "")
        end = self.end[:10].replace("-", "")
        return f"{start}_{end}"

    @property
    def multipliers(self) -> dict[str, float]:
        missing = sorted(set(self.roots) - set(CONTRACT_MULTIPLIERS))
        if missing:
            raise ValueError(f"missing contract multipliers for roots: {missing}")
        return {root: CONTRACT_MULTIPLIERS[root] for root in self.roots}
