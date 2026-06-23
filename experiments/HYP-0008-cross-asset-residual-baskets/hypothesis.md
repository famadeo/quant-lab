# HYP-0008: Cross-Asset Residual Mean-Reversion Baskets

## Hypothesis

Within related futures groups, normalized price residuals mean-revert. A daily
close signal that fades residual z-scores should produce positive net return out
of sample.

## Method

Use daily roll-adjusted continuous futures from 2010-2024. For each asset group,
compute each root's residual log price versus the group mean, convert it to a
rolling z-score, and fade it with one-day execution lag. Positions are normalized
to one unit of gross exposure.

## Decision Rule

Reject unless the out-of-sample portfolio net return is positive and the daily
event t-statistic is at least 1.65 after turnover costs.
