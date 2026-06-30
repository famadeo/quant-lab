# HYP-0014: Three-year core metals flow-filtered residual reversion

## Hypothesis

The HYP-0013 flow-filtered residual mean-reversion result remains positive over
a longer three-year core-metals sample when `ALI` is excluded and the traded
universe is restricted to `GC`, `SI`, `HG`, `PL`, and `PA`.

## Test

Use the same strategy parameters as HYP-0013, but replace the 12-month trade
sample with a three-year core-only tick-trade sample from `2023-06-22` through
`2026-06-22`.

## Decision Standard

The longer sample should remain positive after base MBP-1 spread costs, clear at
least three times transaction costs under stress, and avoid concentrating all
profit in the original 12-month validation window.
