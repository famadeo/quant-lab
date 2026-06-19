---
name: quant-experiment-run
description: Run or debug a Quant Lab experiment from config.yaml, verify tests, log MLflow artifacts, and report metrics with reproducibility notes.
---

# Quant Experiment Run

Use this skill when asked to run, debug, or reproduce an experiment.

## Workflow

1. Identify the experiment config path.
2. Read `AGENTS.md`, the experiment `hypothesis.md`, and `config.yaml`.
3. Run:

   ```bash
   uv run ruff format --check .
   uv run ruff check .
   uv run pyright
   uv run pytest
   uv run quantlab run-experiment <config-path>
   ```

4. If a deterministic failure occurs, inspect the relevant source and tests.
5. Fix narrow implementation issues only. Do not change raw data or weaken tests.
6. Re-run the failing command and then the full check.

## Output

Return:

- commands run,
- result status,
- metrics from `results.json`,
- artifact paths,
- MLflow run id,
- any remaining test or environment gaps.

Do not infer economic edge from a smoke run or single in-sample backtest.
