from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import yaml

from quantlab.pairs.config import PairsBacktestConfig, PairsSelectionConfig, PairsStrategyConfig
from quantlab.pairs.runner import run_pairs_experiment
from quantlab.pairs.selection import evaluate_pair_selection
from quantlab.pairs.strategy import (
    run_mahalanobis_pair,
    run_zscore_mahalanobis_filter_pair,
    run_zscore_pair,
)

SELECTION_TEST_ROWS = 220
SELECTION_TRAIN_ROWS = 132
SELECTION_TEST_OBSERVATIONS = 88
SELECTION_AVERAGE_PAIR_VOLUME = 100.0


def _synthetic_pair_frame(rows: int = 160) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC")
    steps = np.linspace(0, 8 * np.pi, rows)
    x = 0.001 * np.arange(rows) + 0.01 * np.sin(steps)
    y = x + 0.006 * np.sin(steps * 1.7)
    y[80:95] += np.linspace(0.0, 0.05, 15)
    y[95:110] += np.linspace(0.05, 0.0, 15)
    return pd.DataFrame(
        {
            "ts": timestamps,
            "log_price_a": x,
            "log_price_b": y,
            "log_return_a": pd.Series(x).diff().fillna(0.0),
            "log_return_b": pd.Series(y).diff().fillna(0.0),
        }
    )


def test_pair_selection_uses_train_window_statistics() -> None:
    frame = _synthetic_pair_frame(rows=SELECTION_TEST_ROWS)
    frame["volume_a"] = SELECTION_AVERAGE_PAIR_VOLUME
    frame["volume_b"] = 120.0

    selection = evaluate_pair_selection(
        frame,
        asset_class="Test",
        root_a="AAA",
        root_b="BBB",
        config=PairsSelectionConfig(
            train_fraction=0.6,
            min_train_observations=80,
            min_test_observations=50,
            min_abs_return_correlation=0.0,
            max_spread_adf_pvalue=1.0,
            max_half_life_bars=10_000.0,
            min_average_pair_volume=1.0,
        ),
    )

    assert selection.selected
    assert selection.train_observations == SELECTION_TRAIN_ROWS
    assert selection.test_observations == SELECTION_TEST_OBSERVATIONS
    assert selection.test_start_index == SELECTION_TRAIN_ROWS
    assert selection.average_pair_volume == SELECTION_AVERAGE_PAIR_VOLUME


def test_zscore_pair_positions_are_lagged() -> None:
    strategy = PairsStrategyConfig(
        lookback=20,
        min_pair_observations=50,
        z_entry=1.0,
        z_exit=0.25,
        signal_lag=1,
    )
    result = run_zscore_pair(
        _synthetic_pair_frame(),
        strategy,
        PairsBacktestConfig(fee_bps=0.0, slippage_bps=0.0),
    )

    np.testing.assert_allclose(
        result["position_a"].iloc[1:].to_numpy(),
        result["target_a"].iloc[:-1].to_numpy(),
    )
    np.testing.assert_allclose(
        result["position_b"].iloc[1:].to_numpy(),
        result["target_b"].iloc[:-1].to_numpy(),
    )


def test_mahalanobis_pair_creates_bounded_positions() -> None:
    strategy = PairsStrategyConfig(
        lookback=20,
        min_pair_observations=50,
        mahalanobis_entry=1.5,
        mahalanobis_exit=0.75,
        signal_lag=1,
    )
    result = run_mahalanobis_pair(
        _synthetic_pair_frame(),
        strategy,
        PairsBacktestConfig(fee_bps=0.0, slippage_bps=0.0),
    )

    gross_target = result["target_a"].abs() + result["target_b"].abs()
    assert gross_target.max() <= 1.0 + 1e-12
    assert result["indicator"].notna().sum() > 0


def test_zscore_mahalanobis_filter_blocks_unconfirmed_entries() -> None:
    strategy = PairsStrategyConfig(
        lookback=20,
        min_pair_observations=50,
        z_entry=1.0,
        z_exit=0.25,
        mahalanobis_entry=999.0,
        signal_lag=1,
    )
    result = run_zscore_mahalanobis_filter_pair(
        _synthetic_pair_frame(),
        strategy,
        PairsBacktestConfig(fee_bps=0.0, slippage_bps=0.0),
    )

    assert int(result["active"].sum()) == 0
    assert int(result["filter_indicator"].notna().sum()) > 0


def test_zscore_mahalanobis_filter_creates_lagged_positions() -> None:
    strategy = PairsStrategyConfig(
        lookback=20,
        min_pair_observations=50,
        z_entry=1.0,
        z_exit=0.25,
        mahalanobis_entry=0.1,
        mahalanobis_exit=0.05,
        signal_lag=1,
    )
    result = run_zscore_mahalanobis_filter_pair(
        _synthetic_pair_frame(),
        strategy,
        PairsBacktestConfig(fee_bps=0.0, slippage_bps=0.0),
    )

    gross_target = result["target_a"].abs() + result["target_b"].abs()
    assert gross_target.max() <= 1.0 + 1e-12
    np.testing.assert_allclose(
        result["position_a"].iloc[1:].to_numpy(),
        result["target_a"].iloc[:-1].to_numpy(),
    )
    np.testing.assert_allclose(
        result["position_b"].iloc[1:].to_numpy(),
        result["target_b"].iloc[:-1].to_numpy(),
    )


def test_turnover_controls_reduce_zscore_turnover() -> None:
    frame = _synthetic_pair_frame(rows=300)
    base_strategy = PairsStrategyConfig(
        lookback=20,
        min_pair_observations=50,
        z_entry=1.0,
        z_exit=0.25,
        signal_lag=1,
    )
    conservative_strategy = PairsStrategyConfig(
        lookback=20,
        min_pair_observations=50,
        z_entry=1.0,
        z_exit=0.25,
        signal_lag=1,
        rebalance_every_bars=6,
        min_holding_bars=6,
        cooldown_bars=6,
        min_position_change=0.2,
    )

    base = run_zscore_pair(frame, base_strategy, PairsBacktestConfig(fee_bps=0.0, slippage_bps=0.0))
    conservative = run_zscore_pair(
        frame, conservative_strategy, PairsBacktestConfig(fee_bps=0.0, slippage_bps=0.0)
    )

    assert conservative["turnover"].sum() < base["turnover"].sum()


def test_turnover_controls_reduce_mahalanobis_turnover() -> None:
    frame = _synthetic_pair_frame(rows=300)
    base_strategy = PairsStrategyConfig(
        lookback=20,
        min_pair_observations=50,
        mahalanobis_entry=1.5,
        mahalanobis_exit=0.75,
        signal_lag=1,
    )
    conservative_strategy = PairsStrategyConfig(
        lookback=20,
        min_pair_observations=50,
        mahalanobis_entry=1.5,
        mahalanobis_exit=0.75,
        signal_lag=1,
        rebalance_every_bars=6,
        min_holding_bars=6,
        cooldown_bars=6,
        min_position_change=0.2,
    )

    base = run_mahalanobis_pair(
        frame, base_strategy, PairsBacktestConfig(fee_bps=0.0, slippage_bps=0.0)
    )
    conservative = run_mahalanobis_pair(
        frame, conservative_strategy, PairsBacktestConfig(fee_bps=0.0, slippage_bps=0.0)
    )

    assert conservative["turnover"].sum() < base["turnover"].sum()


def test_run_pairs_experiment_writes_comparison_artifacts(tmp_path: Path) -> None:
    data_dir = tmp_path / "futures_5m"
    output_dir = tmp_path / "experiment"
    data_dir.mkdir()

    timestamps = pd.date_range("2026-01-01", periods=180, freq="5min", tz="UTC")
    base = np.cumsum(np.sin(np.linspace(0, 10, len(timestamps))) * 0.001)
    for idx, root in enumerate(["AAA", "BBB", "CCC"]):
        log_price = base + idx * 0.002 + np.sin(np.linspace(0, 6, len(timestamps)) + idx) * 0.003
        frame = pl.DataFrame(
            {
                "ts": timestamps,
                "cont_logprice": log_price,
                "cont_logret": pd.Series(log_price).diff().fillna(0.0),
                "volume": 1000.0 + idx,
            }
        )
        frame.write_parquet(data_dir / f"{root}.parquet")

    config_path = tmp_path / "config.yaml"
    payload = {
        "experiment_id": "HYP-9998-pairs-test",
        "title": "Pairs test",
        "hypothesis": "Synthetic pairs test.",
        "data": {
            "root_dir": str(data_dir),
            "roots": ["AAA", "BBB", "CCC"],
            "asset_classes": {"Test": ["AAA", "BBB", "CCC"]},
        },
        "strategy": {
            "lookback": 20,
            "min_pair_observations": 60,
            "z_entry": 1.0,
            "z_exit": 0.25,
            "mahalanobis_entry": 1.5,
            "mahalanobis_exit": 0.75,
            "signal_lag": 1,
            "rebalance_every_bars": 6,
            "min_holding_bars": 6,
            "cooldown_bars": 6,
            "min_position_change": 0.2,
        },
        "selection": {
            "enabled": True,
            "train_fraction": 0.6,
            "min_train_observations": 60,
            "min_test_observations": 40,
            "min_abs_return_correlation": 0.0,
            "max_spread_adf_pvalue": 1.0,
            "max_half_life_bars": 10000.0,
            "min_average_pair_volume": 1.0,
        },
        "backtest": {
            "fee_bps": 0.0,
            "slippage_bps": 0.0,
            "periods_per_year": 1000,
        },
        "outputs": {
            "directory": str(output_dir),
            "results_file": "results.json",
            "pair_metrics_file": "pair_metrics.csv",
            "portfolio_returns_file": "portfolio_returns.parquet",
            "pair_returns_file": "pair_returns.parquet",
        },
        "mlflow": {
            "experiment_name": "quant-lab-pairs-tests",
            "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
        },
        "decision": {"status": "revise", "notes": "Test only."},
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    result = run_pairs_experiment(config_path)

    assert result.results_path.exists()
    assert result.pair_selection_path.exists()
    assert result.pair_metrics_path.exists()
    assert result.portfolio_returns_path.exists()
    assert result.pair_returns_path.exists()
    assert set(result.method_metrics) == {"mahalanobis", "zscore", "zscore_mahalanobis"}
