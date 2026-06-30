# HYP-0053-metals-convenience-yield-mechanism: Metals convenience-yield mechanism study

> This page is part of Francisco Amadeo's proprietary quantitative research record. Public access is provided for review and documentation only; no license is granted to copy, reuse, redistribute, commercialize, or implement the research, strategy logic, or derived conclusions.

- Status: `research`
- Owner: `famadeo`
- Decision notes: Mechanism-focused convenience-yield study linking futures curve carry, inventories, crowding, and executable basis trades.

## Hypothesis

> **Type:** Diagnostic / mechanism study, **not** a backtest. The goal is conviction in
> *why* the HYP-0030/0031 curve-carry reversion makes money before any capital decision.
> No new variant search, no promotion claim. Output is a verdict memo.

## Hypothesis

The profit in the metals convenience-yield basis strategy (HYP-0030/0031) comes from
**genuine economic compression of the metal-specific convenience-yield/storage basis** —
i.e. when a metal's curve carry is anomalously far from its own recent level, the *spread
itself* reverts and that move is captured. It is **not** primarily an artifact of (a) a
rolling z-score band re-centering by construction, (b) a common USD financing-rate
component that gold also carries, or (c) the front-contract roll/expiry cycle.

This is falsifiable: each of (a), (b), (c) is a concrete null that the diagnostics below
can confirm instead.

## Prediction

If the hypothesis holds, we expect:

1. **Money, not mirror.** A large majority of each event's z-reversion is driven by the
   carry (spread) actually moving toward its mean, and the realized `gross_spread_return`
   is positive and accounts for the bulk of P&L. "Mirror" exits (z closes while the spread
   did not move) are a small minority.
2. **Convenience-yield, not rates.** The reversion P&L survives — and is concentrated in —
   the metal-specific basis `X_carry − GC_carry` (financing stripped out). Stripping the
   common gold/financing component does not destroy the edge; gold itself contributes
   little once it is used as the financing benchmark.
3. **Economically-located edge.** Genuine (spread-move) P&L is concentrated in the
   industrial / supply-constrained metals (HG, PA, PL) where convenience yield is
   economically meaningful, more than in GC.
4. **Coherent asymmetry.** Contango and backwardation reversions behave in a way
   consistent with storage/scarcity economics, and any asymmetry is stable across roots.
5. **Persistent process.** The structural mean-reversion half-life of the raw
   convenience-yield level roughly agrees with realized event durations (the strategy is
   not just exiting on noise), and is stable across the sample.
6. **Not a roll artifact.** Reversion P&L is not concentrated at front-contract
   expiry/roll, and `rolls=0` events carry the edge on their own.

## Statistical Power

- Sample: ~3 years (2023-06-22 → 2026-06-21), daily curve panel, 5 roots, ~100–240
  events per analyzed variant. This is **adequate for characterizing a mechanism**
  (effect sizes, shares, half-lives) but **not** for a powered promote/reject decision.
- Event-level t-statistics are reported as **descriptive only** — events overlap and are
  cross-sectionally clustered, so t-stats overstate significance (a documented lab-wide
  issue). Conclusions lead with effect sizes, P&L shares, and half-lives, not t-stats.
- If a diagnostic is dominated by one root, one side, or one quarter, that is recorded as
  a limitation, not smoothed over.

## Diagnostic Methods

All analysis replicates the **exact** signal from
`scripts/run_metals_convenience_yield_basis_backtest.py`:

```
carry = carry_pct_ann = log(far_close / anchor_close) / (months_from_anchor / 12) * 100
mu, sigma = carry.shift(1).rolling(lookback, min_periods=MIN_Z_OBS).mean(), .std(ddof=0)
z = (carry - mu) / sigma
entry: |z| >= 1.5 ;  exit: |z| <= 0.25
z <= -1.5 -> fade_backwardation (long deferred / short front)
z >= +1.5 -> fade_contango      (short deferred / long front)
```

**Inputs (all in-repo, no external data):**
`curve_panel.parquet` and `event_log.csv` from **HYP-0031 (sync marks, primary)** and
**HYP-0030 (daily marks, robustness contrast)**.

**Fixed variant set (pre-declared, to avoid re-mining):** the documented best
`target3m_minv10` **and** the liquid `target1m_minv500` and `target1m_minv1000`. We
characterize the mechanism in both the illiquid-strong and liquid-weak regimes.

### D1 — Mirror or money? (the spine)

For each event, attribute the entry→exit change in `z` to three sources:

- **carry move:** `ΔC / σ_entry` — the only component that produces `spread_return`.
- **mean drift:** `−Δμ / σ` — the rolling band re-centering (untradable).
- **vol drift:** residual from `σ` changing.

Use entry-held counterfactuals: `z_price_only = (C_exit − μ_entry)/σ_entry`,
`z_band_only = (C_entry − μ_exit)/σ_exit`, and report each component's share of the total
`Δz`. Cross-check against realized `gross_spread_return`: classify an exit as **"mirror"**
when `z` closed but `|gross_spread_return|` is below the event's round-trip execution cost
(the spread move could not even cover trading it). Report per root×side: carry-move share,
band-drift share, % mirror exits, realized gross P&L. Plot Δlog_spread vs Δz.
**Verdict:** real basis compression vs re-centering band. *(Run on both 0030 and 0031
panels — the mechanism should survive the mark change.)*

### D2 — Per-root / per-side attribution

Recompute, cleanly, by root×side: genuine (carry-move) P&L, win rate, event count, mean
duration, and the D1 mirror share. Rank roots by genuine P&L.
**Verdict:** where the real money actually lives.

### D3 — Financing vs convenience yield (the gold test)

Gold has ~zero convenience yield, so `GC_carry ≈ USD financing rate` (panel shows ~5.8%
in 2023). Define, at matched `(date, target_months)`, the **convenience-yield basis**
`cy_X = X_carry − GC_carry`. Re-derive the z-signal on `cy_X` (same rolling construction)
and measure whether reversion events/P&L on the stripped basis match those on raw carry.
Separately, regress/attribute each metal's carry-reversion P&L onto its common
(GC/financing) component vs its idiosyncratic (`cy_X`) component. Sanity check (qualitative,
no new data dependency): confirm the `GC_carry` level tracks the known 2023–2026 US
policy-rate path; an in-repo rates-futures (e.g. ZQ) level cross-check is optional only if a
clean series is readily at hand.
**Verdict:** convenience yield vs disguised rates.

### D4 — Contango vs backwardation asymmetry

Split by side: reversion strength, half-life, win rate, and genuine P&L for
`fade_contango` (carry too high → glut) vs `fade_backwardation` (carry too low → scarcity).
Test whether the asymmetry is stable across roots.
**Verdict:** symmetric edge vs one-sided, and whether it matches storage/scarcity economics.

### D5 — Reversion half-life

Two independent estimates: (i) realized event-duration distribution per root×side;
(ii) structural OU/AR(1) half-life `ln(2)/λ` fit on the **raw convenience-yield level**
`cy_X` (deliberately *not* on `z`, which mean-reverts by construction — see D1). Compare
the two and check stability across the three annual sub-samples.
**Verdict:** persistent economic process vs noise-timing.

### D6 — Roll-cycle confound

Using anchor/far contract codes, `anchor_months_out`, the event_log `rolls` column, and
days-to-front-expiry at entry: test whether genuine P&L concentrates near front-contract
expiry/roll, and compare `rolls=0` vs `rolls>0` events.
**Verdict:** tradable basis reversion vs mechanical expiry/liquidity-migration artifact.

## Data Assumptions

- **Universe:** GC, SI, HG, PL, PA (the five roots already in the curve panels).
- **Date range:** 2023-06-22 → 2026-06-21 (3y, existing panels). No history rebuild here.
- **Corporate actions:** N/A (outright futures contracts; spreads of same-root contracts).
- **Survivorship handling:** N/A — full contract chain per root, no instrument selection.
- **Timestamp alignment:** primary analysis on HYP-0031 synchronized 5-minute marks
  (last exact shared 5m timestamp for near/far); HYP-0030 daily-close marks used only as a
  robustness contrast for D1.
- **Point-in-time:** z-score uses `carry.shift(1)` rolling stats; confirmed no lookahead in
  the source signal. The diagnostics add no forward-looking information.

## Risk And Bias Review

- **Lookahead risk:** Low. Signal is point-in-time (lagged rolling stats); diagnostics are
  descriptive decompositions of already-realized events.
- **Survivorship risk:** None (no instrument or pair selection beyond the fixed variant set).
- **Multiple-testing risk:** Controlled by design — **no new variant scan**. The analyzed
  variants are pre-declared (best + two liquid). Event t-stats are descriptive only.
- **Statistical power / minimum sample:** ~3y, ~100–240 events/variant; sufficient for
  mechanism characterization, insufficient for a powered Sharpe verdict (so none is made).
- **Price-construction integrity:** Spreads are between two contracts of the **same root**
  on the same date; `spread_return` is computed on fixed per-date contracts (no mixed-root
  or stitched-outright returns — the HYP-0013/0014 failure mode does not apply here, and
  D6 explicitly tests the roll dimension).
- **Liquidity/capacity risk:** Out of scope for this diagnostic (it is the reason for the
  fixed liquid-variant contrast), but the min_volume=10 vs 500/1000 contrast is carried
  through so the mechanism's liquidity-dependence is visible.
- **Regime sensitivity:** Explicitly tested in D5 (half-life stability) and via the per-year
  splits already present in the source experiments.

## Deliverable

`experiments/HYP-0053-metals-convenience-yield-mechanism/`:

- `report.md` — conviction memo: top-line verdict first, then D1–D6 with economic
  interpretation and a per-question verdict.
- Diagnostic CSVs: `reversion_decomposition.csv`, `root_side_attribution.csv`,
  `financing_vs_convenience.csv`, `asymmetry_summary.csv`, `halflife.csv`, `roll_cycle.csv`.
- Plots: Δlog_spread-vs-Δz (mirror-or-money), convenience-yield-vs-GC-carry time series,
  half-life bars, roll-proximity P&L.
- Analysis code: `scripts/analyze_metals_convenience_yield_mechanism.py`, run under the
  project `.venv`.

## Out Of Scope (fast-follows if the mechanism survives)

10-year history rebuild; exchange-inventory fundamentals; strategy tuning;
spread-instrument execution-cost modeling.

## Decision

Status: `research`

Rationale: Diagnostic mechanism study. It does not promote or reject a strategy; it
produces a verdict on whether the HYP-0030/0031 edge is economically real and, if so, for
which roots/sides and via which mechanism. That verdict gates whether a 10-year /
liquid-execution validation is worth building.

## Conceptual Description

> **Type:** Diagnostic / mechanism study, **not** a backtest. The goal is conviction in > *why* the HYP-0030/0031 curve-carry reversion makes money before any capital decision. > No new variant search, no promotion claim. Output is a verdict memo.

## Experiment Design

- Roots: `n/a`
- Asset groups: `n/a`
- Pair scope: `n/a`
- Lookback: `n/a` bars
- Signal lag: `n/a` bars
- Rebalance interval: `n/a` bars
- Selection enabled: `n/a`
- Train fraction: `n/a`
- Fee bps: `n/a`
- Slippage bps: `n/a`

## Results

No result artifact was available when the wiki was built.

## Publication Notes

- Proprietary work by Francisco Amadeo. All rights reserved.
- Public access does not grant permission to copy, reuse, redistribute, commercialize, or implement this research.
- Local data roots and artifact paths are intentionally omitted.
- Raw data, parquet outputs, MLflow state, and credentials are not published.
- This page is a summary; the experiment folder remains the source of truth.
