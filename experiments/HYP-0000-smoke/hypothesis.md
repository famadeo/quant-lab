# HYP-0000-smoke: Synthetic moving-average smoke test

## Hypothesis

This is an environment smoke test. It does not assert a real market inefficiency.

## Prediction

The configured strategy should run deterministically on synthetic bars, produce an
equity curve, save `results.json`, and log an MLflow run.

## Falsification Criteria

- Reject the environment setup if the experiment cannot run from a clean `uv`
  environment.
- Reject the implementation if positions are not lagged by at least one bar.
- Reject the implementation if costs are absent from the reported return stream.

## Data Assumptions

- Universe: one synthetic symbol.
- Date range: 2020-01-01 through 2022-12-31.
- Corporate actions: none.
- Survivorship handling: not applicable.
- Timestamp alignment: daily synthetic bars with next-bar execution.

## Decision

Status: `revise`

Rationale: Useful for validating the stack only.
