# HYP-0006: Post-Macro-Event Futures Drift/Reversal

## Hypothesis

After scheduled US macro releases, the first five minutes of price reaction can
be used to select either continuation or reversal for the next 55 minutes.

## Method

For each root, fit the relationship between the event-to-5-minute return and the
5-to-60-minute forward return on the training events only. In the test set, trade
the sign implied by the training slope. Costs are 3 bps round trip per event leg.

## Decision Rule

Reject unless the equal-weight event portfolio has positive net return, event
t-statistic of at least 1.65, and at least 60% positive roots out of sample.
