# HYP-0010: Futures Carry Plus Momentum Baseline

## Hypothesis

A cross-sectional futures portfolio combining term-structure carry and 12-month
momentum has positive out-of-sample net return without needing fast execution.

## Method

Use daily per-contract futures data from 2010-2024. Carry is approximated as the
log front/next-contract close ratio using the active contract from the
roll-adjusted continuous series. Momentum is the 252-day continuous log-price
change. Each factor is cross-sectionally standardized; the combined strategy is
the average of carry and momentum scores.

## Decision Rule

Reject unless the combined out-of-sample portfolio has positive net return and a
daily event t-statistic of at least 1.65 after turnover costs.
