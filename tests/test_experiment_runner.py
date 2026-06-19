from __future__ import annotations

from pathlib import Path

import yaml

from quantlab.experiments import run_experiment


def test_run_experiment_writes_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "experiment"
    tracking_db = tmp_path / "mlflow.db"
    config_path = tmp_path / "config.yaml"
    payload = {
        "experiment_id": "HYP-9999-test",
        "title": "Test experiment",
        "hypothesis": "Synthetic smoke test.",
        "data": {
            "source": "synthetic",
            "symbol": "SYNTH",
            "start": "2021-01-01",
            "end": "2021-06-30",
            "seed": 3,
        },
        "strategy": {
            "name": "moving_average_crossover",
            "fast_window": 5,
            "slow_window": 20,
        },
        "backtest": {
            "initial_capital": 100000,
            "fee_bps": 0.5,
            "slippage_bps": 1.0,
            "signal_lag": 1,
            "max_leverage": 1.0,
            "periods_per_year": 252,
        },
        "outputs": {
            "directory": str(output_dir),
            "results_file": "results.json",
            "equity_curve_file": "equity_curve.parquet",
        },
        "mlflow": {
            "experiment_name": "quant-lab-tests",
            "tracking_uri": f"sqlite:///{tracking_db}",
        },
        "decision": {
            "status": "revise",
            "notes": "Test only.",
        },
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    result = run_experiment(config_path)

    assert result.results_path.exists()
    assert result.equity_curve_path.exists()
    assert result.metrics.observations > 0
    assert result.mlflow_run_id
