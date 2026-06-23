# HYP-0011: Expanded Metals Residual Mean-Reversion Basket

## Hypothesis

Adding `PL`, `PA`, and `ALI` to the original `GC/SI/HG` metals residual basket
improves robustness by broadening the metals complex while preserving the same
daily relative-value mean-reversion mechanism.

## Data

- Dataset: Databento `GLBX.MDP3`.
- Raw schema: per-contract `ohlcv-1d`.
- Roots: `GC`, `SI`, `HG`, `PL`, `PA`, `ALI`.
- Continuous construction: same monotonic volume-roll, within-contract return
  splice used by the existing daily futures pipeline.
- Start date: `2014-05-06`, because `ALI` history starts there in the pull.

## Method

At each daily close:

1. Compute each root's continuous log price.
2. Subtract the cross-sectional metals mean to form residual log price.
3. Convert residuals to rolling 126-day z-scores.
4. Fade residual z-scores clipped at `+/-2`.
5. Normalize positions to one unit gross exposure.
6. Hold for the next day and rebalance daily.

Costs are `1.5` bps per unit of turnover.

## Decision Rule

Reject unless the out-of-sample portfolio has positive net return and a daily
event t-statistic of at least `1.65`.
