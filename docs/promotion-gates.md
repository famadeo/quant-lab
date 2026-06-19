# Strategy Promotion Gates

## Research To Review

- Configured experiment folder exists.
- Tests pass.
- Results are logged to MLflow.
- Costs and one-bar signal lag are included.

## Review To Paper Trading

- Out-of-sample result survives realistic costs.
- Sensitivity checks are documented.
- Drawdown, turnover, liquidity, and capacity are acceptable.
- Reviewer notes are closed.
- Paper-trading kill criteria are written.

## Paper Trading To Live

This repository should not place live orders. Live promotion requires a separate
execution repository, credential isolation, broker risk controls, and manual
approval.
