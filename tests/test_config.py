from __future__ import annotations

from pathlib import Path

import pytest

from quantlab.config import ExperimentConfig, StrategyConfig


def test_experiment_config_resolves_output_directory() -> None:
    config = ExperimentConfig.from_yaml(Path("experiments/HYP-0000-smoke/config.yaml"))

    assert config.experiment_id == "HYP-0000-smoke"
    assert config.outputs.directory.is_absolute()
    assert config.strategy.fast_window < config.strategy.slow_window


def test_strategy_config_rejects_inverted_windows() -> None:
    with pytest.raises(ValueError, match="fast_window"):
        StrategyConfig(fast_window=50, slow_window=10)
