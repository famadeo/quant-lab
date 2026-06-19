# HYP-0001: Mahalanobis Pairs Trading Versus Z-Score Pairs Trading

## Hypothesis

For intra-sector futures pairs, a Mahalanobis-distance mean-reversion signal can
identify joint deviations more effectively than a scalar spread z-score because
it accounts for pair covariance and direction in the two-asset state space.

## Prediction

Using the same Databento 5-minute continuous futures data, entry/exit logic, lag,
cost model, and intra-sector pair universe, the Mahalanobis method should produce
better after-cost portfolio-level risk-adjusted returns than the z-score method.

## Falsification Criteria

- Reject if Mahalanobis after-cost Sharpe is not higher than the z-score variant.
- Reject if any apparent improvement comes from a small number of pairs or trades.
- Reject if performance is dominated by thin or sparse markets.
- Reject if results are unstable under reasonable threshold, lookback, or cost changes.

## Data Assumptions

- Universe: 53 GLBX-active futures roots grouped by macro sector.
- Data: Databento 5-minute continuous futures Parquet files under
  `/home/famadeo/research/databento-asset-browser/data/futures_5m`.
- Date range: whatever is present in the local files, currently approximately
  May 17, 2026 through June 16, 2026 for liquid roots.
- Execution: signals are shifted by one 5-minute bar before returns are applied.
- Turnover control: entries/exits are checked hourly, positions must be held for
  at least one hour, exited pairs have a one-hour cooldown, and small target
  changes are suppressed.
- Pair selection: filters are estimated on the first 60% of each pair's aligned
  observations. Performance is evaluated only on the remaining out-of-sample
  observations.
- Costs: 0.5 bps fees plus 1.0 bps slippage per unit of one-way gross turnover.

## Risk And Bias Review

- Lookahead risk: mitigated by one-bar signal lag, but still requires code review.
- Survivorship risk: root universe was supplied before this experiment, but the
  local download may already reflect available/active instruments.
- Multiple-testing risk: high, because many intra-sector pairs are evaluated.
- Liquidity/capacity risk: not yet modeled beyond turnover costs.
- Regime sensitivity: severe, because the available 5-minute sample is short.

## Decision

Status: `revise`

Rationale: Initial method-comparison experiment with train/test pair selection.
No edge claim.
