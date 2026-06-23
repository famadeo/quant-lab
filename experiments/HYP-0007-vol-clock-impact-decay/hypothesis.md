# HYP-0007: Volatility-Clock Impact Decay After Extreme High-Size OFI

## Hypothesis

Extreme high-size OFI combined with a large same-bar price move contains
next-volatility-bar information. The training set decides whether each root is a
continuation or fade market.

## Method

Use completed volatility bars from `HYP-0005`. An event requires both absolute
`ofi_high` and absolute same-bar return to exceed their training 80th percentile,
with OFI and return signs aligned. The test trades the next volatility bar.

## Decision Rule

Reject unless pooled net return is positive, event t-statistic is at least 1.65,
and at least 60% of roots are positive after costs.
