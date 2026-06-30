# Metals Residual Reversion: Next Steps

Last updated: 2026-06-24

This note follows the [metals flow data-quality audit](metals_flow_data_quality_audit.md)
and records the one direction in the metals program that still has a real, if small,
signal — and why the next step is an execution study, not another bar-level backtest.

## What the corrected evidence actually shows

Across the corrected runs (HYP-0015 strategy, HYP-0016 incremental validation):

- The **residual-reversion signal itself is genuinely predictive** at short horizons,
  concentrated in PA and PL (HAC regression t-stats: PA h1 ~11, PL h1 ~6, correctly
  signed — reversion pays).
- It is **not monetizable as implemented**: gross return is ~0 even before costs at the
  bar frequency, and turnover x spread turns every variant net negative
  (gross/cost <= 0.70x; selected variant gross 0.18 vs cost 0.39).
- The cross-sectional **flow gates add only marginal conditional information** (12 terms
  survive FDR, but coefficients are fractions of a bp per 1-SD).

Conclusion: this is a low-turnover-execution problem, not a signal-discovery problem.
Running the same entry-z=2 reversion rule on finer bars will keep losing to spread.

## The next experiment (do this instead of re-tuning the bar strategy)

1. **Reframe as passive/queue execution, not aggressive crossing.** The "edge" is a few
   bps of mean reversion; it can only survive if entries earn (or at least do not pay)
   the spread. Model resting limit orders with realistic fill probability, not
   marketable orders charged the half-spread.
2. **Cut turnover hard.** Hold longer, widen entry/exit bands, and net positions across
   the basket before trading. Target turnover an order of magnitude below the current
   ~1,700 (3y).
3. **Restrict to where the signal lives.** Focus on PA/PL and the q99 flow states the
   research framework flags, rather than trading all five roots every qualifying bar.
4. **Power and integrity guardrails** (now in the promotion gates and hypothesis
   template): preregister a minimum independent-sample count; keep returns on
   roll-adjusted/active marks; report a deflated Sharpe, not just a point estimate.

## What would falsify the lead

If, under passive-fill assumptions and >=10x lower turnover, PA/PL residual reversion
still cannot clear measured spreads out-of-sample, retire the bar-level relative-value
idea for this complex and treat the flow features as context for a different signal.
