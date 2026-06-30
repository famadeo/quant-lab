# HYP-0027 Metals USD-Beta-Hedged Vol-Adjusted Returns

Completed at `2026-06-26T01:11:28.713418+00:00`.

## Method

- USD factor: negative equal-weight daily log return of `6E`, `6B`, `6J`, `6A`, `6C`.
- Hedge: `metal_logret - lagged_rolling_252d_usd_beta * usd_factor`.
- Vol adjustment: hedged log return scaled to 10% annualized volatility using lagged 63d realized vol.
- Coverage follows the daily continuous factor store, which ends at `2024-11-29`.

## Trade-Era Overlap

| root   |   nobs |   cum_vol_adjusted_logret |   sharpe_like |   mean_usd_beta |   last_usd_beta |
|:-------|-------:|--------------------------:|--------------:|----------------:|----------------:|
| GC     |    449 |                    0.3835 |        1.8736 |         -1.1832 |         -1.4250 |
| HG     |    449 |                    0.1431 |        0.7686 |         -1.4945 |         -2.0431 |
| PA     |    449 |                   -0.0895 |       -0.4765 |         -2.7188 |         -2.9841 |
| PL     |    449 |                    0.0340 |        0.1868 |         -2.0735 |         -1.9915 |
| SI     |    449 |                    0.2056 |        1.1023 |         -2.4062 |         -2.6354 |

## Full Available History

| root   |   nobs |   cum_vol_adjusted_logret |   sharpe_like |   mean_usd_beta |   last_usd_beta |
|:-------|-------:|--------------------------:|--------------:|----------------:|----------------:|
| GC     |   3589 |                    0.8185 |        0.5282 |         -1.1086 |         -1.4250 |
| HG     |   3589 |                    0.3075 |        0.2080 |         -1.2969 |         -2.0431 |
| PA     |   3528 |                    0.2912 |        0.1936 |         -1.6233 |         -2.9841 |
| PL     |   3589 |                   -0.1825 |       -0.1225 |         -1.7520 |         -1.9915 |
| SI     |   3589 |                    0.2294 |        0.1506 |         -2.0183 |         -2.6354 |

## Files

- `usd_beta_hedged_vol_adjusted_returns.csv`
- `summary.csv`
- `usd_beta_hedged_vol_adjusted_cumulative.png`
