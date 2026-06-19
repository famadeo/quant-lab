# Scientific Method For Strategy Research

Every strategy begins as a falsifiable hypothesis and ends with a decision.

## Hypothesis

Write the market inefficiency before running the backtest. Include why the edge
should exist and what observation would disconfirm it.

## Experiment Design

Define:

- Universe and inclusion rules.
- Date range and validation windows.
- Data source and point-in-time assumptions.
- Features, signal, sizing, and rebalance cadence.
- Fees, slippage, spread, latency, and borrow/funding costs.
- Primary metric and promotion threshold.

## Bias Checklist

- Lookahead bias.
- Survivorship bias.
- Selection bias.
- Multiple testing and researcher degrees of freedom.
- Data revisions and timestamp alignment.
- Liquidity, capacity, borrow, and market impact.
- Regime concentration and event sensitivity.

## Promotion Gates

1. Research backtest passes deterministic tests.
2. Out-of-sample evidence survives costs and signal delay.
3. Sensitivity analysis does not depend on one brittle parameter.
4. Reviewer notes are resolved.
5. Paper-trading plan exists with kill criteria.
