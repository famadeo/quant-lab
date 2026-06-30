# Core Metals PC1-PC2 Residual Cost of Carry

Definition:

- For each asset residual, build the PC1-PC2-neutral projection row `e_i' (I - V V')` in standardized-return space.
- Convert standardized residual weights to raw-return weights by dividing by each asset's lagged EWMA volatility.
- Gross-normalize raw weights to one.
- Apply `3M` annualized curve-implied funding from HYP-0036.
- Positive carry means the long residual basket pays carry; negative means it earns carry under the funding sign convention.

Caveats:

- This is a futures-curve proxy, not true cash/spot carry.
- The residual basket is a PCA/risk-space proxy, not an executable calendar-spread trade definition.
- Funding is aligned by backward as-of match with a 14-day maximum tolerance.
- Plots use daily medians of hourly PCA diagnostic timestamps.

## Summary

| root   |   nobs |   mean_carry_pct_ann |   median_carry_pct_ann |   p10_carry_pct_ann |   p90_carry_pct_ann |   latest_carry_pct_ann |   max_funding_age_hours |   median_max_funding_age_hours |
|:-------|-------:|---------------------:|-----------------------:|--------------------:|--------------------:|-----------------------:|------------------------:|-------------------------------:|
| GC     |  57513 |               0.9237 |                 0.6824 |              0.0596 |              2.3035 |                 0.0926 |                335.7500 |                         3.0000 |
| SI     |  57513 |              -0.4848 |                -0.2629 |             -1.5357 |              0.2165 |                -2.3077 |                335.7500 |                         3.0000 |
| HG     |  57513 |               0.2526 |                 0.4102 |             -1.5558 |              1.8399 |                 1.6670 |                335.7500 |                         3.0000 |
| PL     |  57513 |              -0.4176 |                -0.3089 |             -1.4333 |              0.3750 |                -0.1168 |                335.7500 |                         3.0000 |
| PA     |  57513 |              -0.2258 |                -0.1527 |             -1.8401 |              1.3000 |                 1.3998 |                335.7500 |                         3.0000 |

## Files

- `pc12_residual_carry_cost_pct_ann.parquet`
- `pc12_residual_carry_cost_daily.csv`
- `pc12_residual_carry_weights.parquet`
- `pc12_residual_carry_funding_age_hours.csv`
- `pc12_residual_carry_summary.csv`
- `pc12_residual_carry_cost_overlay.png`
- `pc12_residual_carry_cost_panels.png`