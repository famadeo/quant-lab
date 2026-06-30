# Strategy Promotion Gates

## Research To Review

- Configured experiment folder exists.
- Tests pass.
- Results are logged to MLflow.
- Costs and one-bar signal lag are included.
- Statistical power is sufficient: the count of independent out-of-sample
  observations (events, days, or non-overlapping bars) is large enough to detect the
  hypothesized effect. A test that cannot reject the null because the sample is tiny is
  recorded as `inconclusive`, not `reject`. State the minimum sample in the hypothesis.
- Headline Sharpe ratios carry a significance qualifier (t-stat, probabilistic Sharpe,
  or deflated Sharpe when many configurations were screened), not just a point estimate.
- Traded returns come from tradable marks: roll-adjusted continuous or active-contract
  prices, never mixed outright-contract endpoints (see the metals data-quality audit).

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
