# {{EXPERIMENT_ID}}: {{TITLE}}

## Hypothesis

State a falsifiable market inefficiency in one or two sentences.

## Prediction

Describe the measurable effect you expect to observe.

## Statistical Power

- Minimum independent out-of-sample observations required (events / days / non-overlapping bars):
- If the realized sample falls below this, record `inconclusive`, not `reject`.

## Falsification Criteria

- Reject if out-of-sample Sharpe is not materially above zero after costs AND the test
  is adequately powered (otherwise mark `inconclusive`).
- Reject if performance depends on a narrow date range or a small number of trades.
- Reject if the effect disappears under reasonable fee, slippage, or delay assumptions.

## Data Assumptions

- Universe:
- Date range:
- Corporate actions:
- Survivorship handling:
- Timestamp alignment:

## Risk And Bias Review

- Lookahead risk:
- Survivorship risk:
- Multiple-testing risk:
- Statistical power / minimum sample:
- Price-construction integrity (roll-adjusted/active marks, no mixed-contract returns):
- Liquidity/capacity risk:
- Regime sensitivity:

## Decision

Status: `revise`

Rationale:
