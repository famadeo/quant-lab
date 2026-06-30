# Core Metals Drawdown-Control Overlay Study

Last updated: 2026-06-30

This note summarizes HYP-0052, a drawdown-control study for the core metals
portfolio research line.

## Scope

The tested portfolio is the HYP-0050/HYP-0051 long-only minimum-variance core
metals benchmark:

- Universe: `GC`, `SI`, `HG`, `PL`, `PA`.
- Base portfolio: `MINVAR_30D_SHRINK25`.
- Return frequency: 5-minute log returns.
- Costs: root-specific turnover costs from the HYP-0046 MBP1 spread model.
- Control objective: reduce realized drawdowns without introducing a new
  return-forecasting alpha model.

The overlay families were:

- No-leverage volatility targeting using lagged 30-calendar-day realized
  portfolio volatility.
- Trailing trend risk-off using lagged 60-calendar-day portfolio log return.
- Realized drawdown throttles using only previous strategy wealth and previous
  peak.
- Daily variants that update exposure once per UTC day.

All control signals are lagged.

## Key Variants

The most useful variants were:

| Variant | Definition |
|---|---|
| `BASE_MINVAR` | Original long-only min-var benchmark with no overlay. |
| `VOL14_TSMOM60_HALF_DAILY` | 14% no-leverage vol target; if 60-day trailing return is negative, cut exposure to 50%; update daily. |
| `VOL12_TSMOM60_DD_SOFT_DAILY` | 12% vol target; 60-day trend half-risk-off; soft drawdown throttle to 75% below -8% drawdown and 50% below -15%; update daily. |
| `VOL12_TSMOM60_DD_HARD_DAILY` | 12% vol target; 60-day trend half-risk-off; hard drawdown throttle to 75% below -5%, 50% below -10%, and 25% below -15%; update daily. |

## Full Sample Result

Full available HYP-0052 sample:

- Start: 2021-01-03 23:00:00 UTC.
- End: 2026-06-19 21:55:00 UTC.

| Variant | CAGR | Annual Vol | Sharpe | Max Drawdown |
|---|---:|---:|---:|---:|
| `BASE_MINVAR` | 16.76% | 16.34% | 0.95 | -24.24% |
| `VOL14_TSMOM60_HALF_DAILY` | 13.42% | 11.88% | 1.06 | -16.42% |
| `VOL12_TSMOM60_DD_SOFT_DAILY` | 12.19% | 10.52% | 1.09 | -15.13% |
| `VOL12_TSMOM60_DD_HARD_DAILY` | 11.59% | 10.17% | 1.08 | -15.02% |

The full-sample result says drawdown control is effective but not free. The best
risk-adjusted candidate is `VOL12_TSMOM60_DD_SOFT_DAILY`; it cuts max drawdown
by roughly 9 percentage points and improves Sharpe, while giving up about 4.6
percentage points of CAGR versus the unconstrained benchmark.

## 2021-2024 Rerun

Restricted sample:

- Start: 2021-01-03 23:00:00 UTC.
- End: 2024-12-31 21:55:00 UTC.

| Variant | Total Return | CAGR | Annual Vol | Sharpe | Max Drawdown |
|---|---:|---:|---:|---:|---:|
| `BASE_MINVAR` | 26.26% | 6.02% | 13.78% | 0.42 | -24.24% |
| `VOL12_TSMOM60_DD_SOFT_DAILY` | 19.82% | 4.63% | 9.63% | 0.47 | -15.12% |
| `VOL12_TSMOM60_DD_HARD_DAILY` | 16.37% | 3.87% | 9.06% | 0.42 | -14.93% |
| `VOL14_TSMOM60_HALF_DAILY` | 20.48% | 4.78% | 10.80% | 0.43 | -16.43% |

The drawdown controls still help before the large 2025 upside, but they mostly
convert the portfolio into a lower-volatility, lower-return profile.

## 2021-2022 Rerun

Restricted sample:

- Start: 2021-01-03 23:00:00 UTC.
- End: 2022-12-30 21:55:00 UTC.

| Variant | Total Return | CAGR | Annual Vol | Sharpe | Max Drawdown |
|---|---:|---:|---:|---:|---:|
| `BASE_MINVAR` | -0.95% | -0.48% | 14.44% | -0.03 | -24.24% |
| `VOL12_TSMOM60_DD_SOFT_DAILY` | 0.84% | 0.42% | 9.38% | 0.04 | -15.11% |
| `VOL10_DD_HARD_DAILY` | 0.83% | 0.41% | 9.43% | 0.04 | -15.26% |
| `VOL12_TSMOM60_DD_HARD_DAILY` | 0.21% | 0.11% | 9.00% | 0.01 | -14.91% |

The 2021-2022 result should not be read as strong alpha evidence. It is mainly
evidence that the overlay preserves capital and reduces drawdown in a weak
portfolio regime.

## Research Judgment

The drawdown-control overlay is a risk-control layer, not a return engine.

`VOL12_TSMOM60_DD_SOFT_DAILY` is the preferred candidate because it gives most of
the drawdown reduction of the hard throttle while retaining more upside and a
better Sharpe profile. The hard throttle only marginally reduces max drawdown
relative to the soft version and gives up additional return.

The next research step is parameter stability: test nearby volatility targets,
trend windows, and drawdown thresholds in a walk-forward grid, then select a
parameter region rather than a single best point estimate.

## Local Artifacts

The source commit includes:

- `scripts/run_core_metals_drawdown_control_overlays.py`
- `experiments/HYP-0052-core-metals-drawdown-control/report.md`
- `experiments/HYP-0052-core-metals-drawdown-control-2021-2024/report.md`
- `experiments/HYP-0052-core-metals-drawdown-control-2021-2022/report.md`

Large parquet caches are intentionally excluded from Git and the public Wiki.
