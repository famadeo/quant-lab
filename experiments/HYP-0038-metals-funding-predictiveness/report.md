# HYP-0038 Metals Funding Predictiveness

## Question

Does the current funding state forecast future realized return after realized funding paid?

The dependent variable is:

`forward_excess = forward_log_return - realized_path_funding_paid`

## Signals

- `funding_pct_ann`: current annualized curve-implied funding level.
- `funding_z_126d`: current funding versus its own 126-day trailing history.
- `funding_change_24h_pct_ann`: 24-hour change in annualized funding.
- `funding_z_change_24h`: 24-hour change in standardized funding.
- `abs_funding_z_126d`: absolute funding extremeness.

## Method

- Universe: `GC`, `SI`, `HG`, `PL`, `PA`.
- Tenors: `1M`, `3M`, `6M` front-to-deferred funding targets.
- Horizons: `1h`, `4h`, `1d`, `3d`, `1w`, `1m`.
- Regressions use standardized signals and HAC standard errors, capped at 168 lags.
- Decile spreads compare top 10% signal observations versus bottom 10%.

## Top Absolute HAC t-Statistics

| root   |   target_months | horizon   | signal               |     n |   beta_bp_per_1sd_signal |    hac_t |   pearson_corr |   spearman_corr |   low_decile_excess_bp |   high_decile_excess_bp |   high_minus_low_excess_bp |   high_positive_fraction |   low_positive_fraction |
|:-------|----------------:|:----------|:---------------------|------:|-------------------------:|---------:|---------------:|----------------:|-----------------------:|------------------------:|---------------------------:|-------------------------:|------------------------:|
| PL     |               3 | 4h        | funding_z_change_24h | 58152 |                  -5.3801 | -10.8567 |        -0.0656 |         -0.0779 |                 9.8138 |                -10.6453 |                   -20.4592 |                   0.4477 |                  0.5520 |
| HG     |               6 | 4h        | funding_z_change_24h | 54567 |                  -6.1476 |  -9.7083 |        -0.0960 |         -0.1127 |                13.6626 |                -11.8079 |                   -25.4705 |                   0.4372 |                  0.5781 |
| PL     |               3 | 1h        | funding_z_change_24h | 58173 |                  -1.4626 |  -9.3823 |        -0.0349 |         -0.0444 |                 2.6281 |                 -2.7358 |                    -5.3639 |                   0.4595 |                  0.5161 |
| HG     |               6 | 1h        | funding_z_change_24h | 55668 |                  -1.7380 |  -9.0540 |        -0.0537 |         -0.0557 |                 3.4072 |                 -3.1085 |                    -6.5157 |                   0.4570 |                  0.5194 |
| HG     |               1 | 4h        | funding_z_change_24h | 56386 |                  -4.7649 |  -8.5747 |        -0.0745 |         -0.0686 |                 9.4378 |                 -7.1024 |                   -16.5402 |                   0.4660 |                  0.5475 |
| HG     |               1 | 1h        | funding_z_change_24h | 57160 |                  -1.2880 |  -7.5608 |        -0.0399 |         -0.0350 |                 2.5670 |                 -1.8289 |                    -4.3959 |                   0.4747 |                  0.5052 |
| SI     |               6 | 4h        | funding_z_change_24h | 45659 |                  -5.7146 |  -7.4261 |        -0.0646 |         -0.0676 |                12.3352 |                 -7.7120 |                   -20.0472 |                   0.4721 |                  0.5408 |
| PL     |               3 | 1d        | funding_z_change_24h | 58012 |                 -16.9714 |  -6.8995 |        -0.0850 |         -0.0999 |                31.7133 |                -33.1692 |                   -64.8824 |                   0.4235 |                  0.5698 |
| SI     |               6 | 1h        | funding_z_change_24h | 47459 |                  -1.6286 |  -6.0041 |        -0.0365 |         -0.0334 |                 3.2802 |                 -2.2609 |                    -5.5411 |                   0.4693 |                  0.5097 |
| PL     |               6 | 4h        | funding_z_change_24h | 35121 |                  -5.2548 |  -5.7724 |        -0.0578 |         -0.0866 |                10.2321 |                 -8.5311 |                   -18.7632 |                   0.4706 |                  0.5407 |
| HG     |               3 | 4h        | funding_z_change_24h | 56286 |                  -3.3127 |  -5.5855 |        -0.0522 |         -0.0865 |                 5.4251 |                 -5.3571 |                   -10.7821 |                   0.4603 |                  0.5317 |
| HG     |               3 | 1h        | funding_z_change_24h | 57111 |                  -0.9787 |  -5.4644 |        -0.0304 |         -0.0441 |                 1.6703 |                 -1.8103 |                    -3.4806 |                   0.4636 |                  0.4940 |
| PA     |               3 | 4h        | funding_z_change_24h | 47923 |                  -5.3221 |  -5.4104 |        -0.0489 |         -0.0522 |                 8.4668 |                 -9.6111 |                   -18.0779 |                   0.4831 |                  0.5418 |
| PL     |               6 | 1h        | funding_z_change_24h | 36267 |                  -1.4752 |  -5.3785 |        -0.0319 |         -0.0462 |                 2.6427 |                 -2.4991 |                    -5.1418 |                   0.4682 |                  0.5212 |
| PA     |               1 | 4h        | funding_z_change_24h | 47867 |                  -4.3427 |  -4.6744 |        -0.0399 |         -0.0494 |                 7.5632 |                 -8.6586 |                   -16.2219 |                   0.4829 |                  0.5356 |
| HG     |               6 | 1d        | funding_z_change_24h | 47812 |                 -13.5931 |  -4.6635 |        -0.0845 |         -0.1312 |                34.5115 |                -24.9447 |                   -59.4562 |                   0.4385 |                  0.5947 |
| HG     |               1 | 4h        | funding_z_126d       | 56410 |                  -2.4612 |  -4.2933 |        -0.0385 |         -0.0358 |                 6.1131 |                 -3.9385 |                   -10.0517 |                   0.4800 |                  0.5335 |
| HG     |               1 | 1h        | funding_z_126d       | 57184 |                  -0.7091 |  -4.1358 |        -0.0220 |         -0.0195 |                 1.6161 |                 -1.1211 |                    -2.7372 |                   0.4764 |                  0.5093 |
| GC     |               1 | 4h        | funding_z_126d       | 57796 |                  -1.2444 |  -4.0354 |        -0.0296 |         -0.0489 |                 1.5702 |                 -3.0382 |                    -4.6084 |                   0.4699 |                  0.5248 |
| HG     |               1 | 1d        | funding_z_change_24h | 51473 |                 -11.1800 |  -4.0335 |        -0.0701 |         -0.0778 |                25.3654 |                -24.7155 |                   -50.0809 |                   0.4409 |                  0.5573 |

## Funding z-score, 3M target, 1D horizon

| root   |   target_months | horizon   | signal         |     n |   beta_bp_per_1sd_signal |   hac_t |   pearson_corr |   spearman_corr |   low_decile_excess_bp |   high_decile_excess_bp |   high_minus_low_excess_bp |   high_positive_fraction |   low_positive_fraction |
|:-------|----------------:|:----------|:---------------|------:|-------------------------:|--------:|---------------:|----------------:|-----------------------:|------------------------:|---------------------------:|-------------------------:|------------------------:|
| GC     |               3 | 1d        | funding_z_126d | 56912 |                  -3.6608 | -1.9782 |        -0.0346 |         -0.0497 |                17.5085 |                 -7.8442 |                   -25.3527 |                   0.4825 |                  0.5653 |
| HG     |               3 | 1d        | funding_z_126d | 51148 |                  -1.0183 | -0.3731 |        -0.0065 |         -0.0093 |                -1.9160 |                 -5.4891 |                    -3.5732 |                   0.4782 |                  0.4790 |
| PA     |               3 | 1d        | funding_z_126d | 42097 |                  -8.7284 | -1.5903 |        -0.0336 |         -0.0304 |                28.6636 |                  2.5409 |                   -26.1227 |                   0.5467 |                  0.5711 |
| PL     |               3 | 1d        | funding_z_126d | 58036 |                  -2.6498 | -0.5975 |        -0.0133 |         -0.0312 |                 9.9312 |                 -1.8879 |                   -11.8191 |                   0.4818 |                  0.5337 |
| SI     |               3 | 1d        | funding_z_126d | 56310 |                  -2.9278 | -0.7395 |        -0.0137 |         -0.0507 |                 2.8225 |                -15.0060 |                   -17.8285 |                   0.4577 |                  0.5355 |

## Curve State, 3M target, 1D horizon

| root   |   target_months | horizon   |     n |   contango_n |   backwardation_n |   contango_mean_excess_bp |   backwardation_mean_excess_bp |   backwardation_minus_contango_bp |   contango_positive_fraction |   backwardation_positive_fraction |
|:-------|----------------:|:----------|------:|-------------:|------------------:|--------------------------:|-------------------------------:|----------------------------------:|-----------------------------:|----------------------------------:|
| GC     |               3 | 1d        | 57507 |        57265 |               240 |                    2.3454 |                        12.3718 |                           10.0265 |                       0.5175 |                            0.5125 |
| HG     |               3 | 1d        | 51703 |        44273 |              7225 |                    1.5246 |                         3.5313 |                            2.0067 |                       0.5087 |                            0.4918 |
| PA     |               3 | 1d        | 42652 |        24720 |             17711 |                   -4.5571 |                        13.6913 |                           18.2484 |                       0.4964 |                            0.5600 |
| PL     |               3 | 1d        | 58695 |        54871 |              3643 |                   -0.5147 |                         6.6881 |                            7.2028 |                       0.5075 |                            0.4875 |
| SI     |               3 | 1d        | 56927 |        56239 |               621 |                    2.9970 |                       -70.2245 |                          -73.2215 |                       0.5092 |                            0.4348 |

## Files

- `signal_predictiveness_summary.csv`
- `signal_decile_summary.csv`
- `curve_state_summary.csv`
- Heatmaps for 3M funding z-score, 24h funding change, and curve state.
- Decile profile plots for 3M/1D funding z-score and 24h funding change.

## Caveats

- This is a broad multiple-testing screen; isolated t-stats should be treated as hypotheses, not evidence of tradable alpha.
- Funding is still measured from the futures curve, not true spot/cash.
- Overlapping horizons and regime clustering make the effective sample size much smaller than the raw row count.