# Metals Flow Data Quality Audit

Last updated: 2026-06-24

## What Was Wrong

The HYP-0013 and HYP-0014 metals flow strategy runs were invalid because root-level traded returns were computed from mixed outright-contract endpoint prices. That created artificial jumps around active-contract switches and made the strategy appear much stronger than it was.

The corrected rule is:

- use all outright trades for flow features;
- use active or roll-adjusted continuous marks for traded returns and fair-value residuals;
- mask stale prices and bars adjacent to continuous-contract switches;
- preserve endpoint contract symbols as diagnostics, not as a return source.

## Implemented Controls

Reusable checks now live in `src/quantlab/metals_flow/data_quality.py`.

Trade checks:

- required fields: `ts_event`, `symbol`, `price`, `size`, `side`;
- positive price and size;
- aggressor side restricted to `A`, `B`, or `N`;
- duplicate trade-row counts;
- side-level notional shares.

Continuous mark checks:

- required fields: `ts`, `active`, `cont_close`, `cont_logprice`, `is_roll`;
- positive continuous close;
- active-contract switch counts;
- duplicate timestamp counts;
- roll-row counts;
- stale mark and roll-adjacent masking when aligned to bars.

Regression tests cover invalid trade columns, invalid aggressor sides, duplicate trade summaries, continuous switch diagnostics, stale mark masking, and roll-adjacent return masking.

## Corrected Evidence

HYP-0015 rebuilt the strategy using all-outright flow and roll-adjusted 1-minute continuous marks. The corrected base-cost result was:

- gross return: 0.1826;
- cost return: 0.3911;
- net return: -0.2085;
- gross/cost: 0.47x;
- valid price fraction: 99.81%.

HYP-0016 then tested whether flow features add value beyond residual dislocation. Every strategy variant remained net negative, and the best gross/cost ratio was still below 0.70x.

## Research Interpretation

The current data does show weak conditional information in flow variables, especially around PA and PL residual reversion. That information is not strong enough to survive current turnover and spread assumptions.

This means the current result is not "no information in flow." It is "no executable edge in this bar-level residual-reversion implementation after corrected price construction and realistic costs."

## Required Standard Going Forward

Any future metals-flow backtest must include:

- explicit trade-data inventory by root and contract;
- continuous-price alignment diagnostics;
- stale-price and roll-transition masks;
- endpoint contract symbol diagnostics;
- cost sensitivity and gross/cost ratio;
- artifact-level reproducibility under `experiments/`;
- registry status that distinguishes `reject` from `invalid_data_construction`.
