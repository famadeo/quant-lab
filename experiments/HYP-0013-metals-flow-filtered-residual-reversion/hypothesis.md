# HYP-0013 Metals Flow-Filtered Residual Reversion

## Hypothesis

Liquid metals residual dislocations mean-revert more reliably when cross-sectional
flow geometry is abnormal and large-trade flow disagrees with small-trade flow.

## Universe

Core traded roots: `GC`, `SI`, `HG`, `PL`, `PA`.

Isolated diagnostic root: `ALI`, excluded from the first tradable sleeve because
it is much smaller and lacks matching MBP-1 coverage in the current pull.

## Test

Use 12 months of tick trades to build synchronized cross-sectional dollar bars.
Estimate dynamic residual z-scores using no-lookahead EWMA conditional fair value.
Enter residual mean-reversion positions only when EWMA Mahalanobis flow distance
and large-minus-small flow disagreement are both elevated. Exit when residuals
converge toward zero.

## Caveat

The full-sample strategy layer uses EWMA Mahalanobis and EWMA conditional fair
value for runtime stability. The 30-day HYP-0012 notebook remains the heavier
visual diagnostic harness for rolling/expanding/robust variants.
