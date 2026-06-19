from __future__ import annotations

from pathlib import Path

from quantlab.experiments import run_experiment


def run_experiment_flow(config_path: str) -> object:
    """Run an experiment as a Prefect flow when Prefect is installed."""
    try:
        from prefect import flow, task
    except ImportError as exc:
        raise RuntimeError("Install workflow extras with `uv sync --extra workflow`.") from exc

    @task
    def execute(path: str) -> str:
        result = run_experiment(Path(path))
        return str(result.results_path)

    @flow(name="quantlab-run-experiment")
    def _flow(path: str) -> str:
        return execute(path)

    return _flow(config_path)
