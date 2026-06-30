# SPD, Sphere, Graph, and Dislocation Stat-Arb Research Note

Last updated: 2026-06-24

## Summary

The proposed stack is coherent, but it should be tested as a regime-conditioned
equity stat-arb framework rather than as another metals screen. The metals
complex has too few instruments for MST/PMFG structure to add much. The local
S&P 500 archive already has the right scale, adjusted daily prices, sector maps,
and a weak graph/PCA residual baseline that a better geometric model must beat.

The clean decomposition is:

1. **SPD trajectory for regimes.** Rolling covariance or correlation matrices
   live on the symmetric positive definite manifold. Use this layer to decide
   whether residual dislocations should mean-revert, continue, or be ignored.
2. **Standardized sphere plus MST/PMFG for structure.** Standardized return
   histories are vectors on a sphere. Correlation distance then gives a
   defensible market map. MST gives the sparse backbone; PMFG adds triangles
   and 4-cliques without falling back to a dense, noisy graph.
3. **R^N dislocation cloud for trades.** At each date, the cross-section of
   residuals is a point in R^N. Trades should be selected from cloud outliers,
   not from fixed pairs, then constrained by graph clusters, beta/sector
   neutrality, liquidity, and turnover.

## Literature Anchor

- Mantegna's correlation-tree work starts from a correlation matrix and extracts
  a market taxonomy from the resulting graph:
  https://arxiv.org/abs/cond-mat/9802256
- Tumminello, Aste, Di Matteo, and Mantegna introduce PMFG as a planar extension
  of the MST that preserves the hierarchy while retaining richer internal
  structure, including triangles and 4-cliques:
  https://arxiv.org/abs/cond-mat/0501335
- Pennec, Fillard, and Ayache justify treating SPD matrices as a manifold with
  affine-invariant geometry:
  https://link.springer.com/article/10.1007/s11263-005-3222-z
- Arsigny, Fillard, Pennec, and Ayache give the log-Euclidean alternative, which
  is computationally attractive because operations move to matrix-log space:
  https://www-sop.inria.fr/asclepios/Publications/Vincent.Arsigny/Arsigny_SIAM_tensors_07.pdf
- Avellaneda and Lee provide the baseline residual stat-arb template: PCA or ETF
  residuals, model idiosyncratic returns as mean-reverting, and evaluate after
  transaction costs:
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1153505
- Cartea, Cucuringu, and Jin are directly adjacent: cluster market-residual
  correlation matrices, then build mean-reverting stat-arb portfolios inside
  clusters:
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4560455

## Local Baseline

The closest existing local artifact is:

- `/home/famadeo/archive/SP500_statarb/scripts/backtest_graph_pca_residual.py`
- `/home/famadeo/archive/SP500_statarb/data/backtests/graph_pca_residual/metrics.json`

That run used adjusted daily closes from 2021-05-24 to 2026-05-20 across 503
symbols. It is a useful baseline because it already combines PCA residuals,
correlation-neighbor spreads, and a sector-dispersion gate.

The results are weak:

- `graph_pca_gated`: net return -8.15%, gross return before costs +1.15%,
  total cost 9.64%, Sharpe -0.32.
- `graph_pca_ungated`: net return -3.57%, gross return before costs +6.82%,
  total cost 10.23%, Sharpe -0.26.
- `pca_only_gated`: net return +0.73%, gross return before costs +10.57%,
  total cost 9.33%, Sharpe +0.06.

Conclusion: the current graph/PCA residual edge is not monetizable as built.
The next research question is whether the sign and monetization of residual
dislocations depends on covariance-geometry regime and graph stability.

## Geometry Stack

### 1. SPD Trajectory

For each rebalance date `t`:

1. Use only data available through `t`.
2. Estimate a rolling return covariance or correlation matrix `C_t`.
3. Apply shrinkage and an eigenvalue floor so `C_t` is SPD.
4. Map to matrix-log coordinates: `L_t = log(C_t)`.
5. Build regime features:
   - `spd_velocity = ||L_t - L_{t-1}||_F`
   - `spd_acceleration = ||L_t - 2L_{t-1} + L_{t-2}||_F`
   - `distance_to_calm_centroid`
   - `distance_to_stress_centroid`
   - top eigenvalue share / market-mode concentration
   - effective rank
   - correlation dispersion
   - graph rewiring rate from the next layer

Regime labels should be learned walk-forward, not on the full sample. A simple
first pass is quantile states:

- **Stable/calm:** low SPD velocity, high graph edge persistence.
- **Crowded/stressed:** high top-eigenvalue share, high average correlation.
- **Transition:** high SPD velocity or high graph rewiring.

Expected trade behavior:

- Stable/calm: residual cloud outliers should mean-revert.
- Crowded/stressed: size down or require faster exits.
- Transition: test residual continuation separately; do not assume reversion.

### 2. Standardized Sphere and MST/PMFG

For each rolling window, standardize each name's return history to zero mean and
unit variance. Each standardized name vector can then be normalized to the unit
sphere. The correlation distance is:

```text
d_ij = sqrt(2 * (1 - rho_ij))
```

This distance is the chord distance between standardized spherical vectors. Use
it to build:

- **MST:** `N - 1` edges, robust but very sparse.
- **PMFG:** up to `3 * (N - 2)` edges, richer but still sparse and planar.

Graph features:

- persistent neighbors and edge survival rates
- MST branch centrality / periphery
- PMFG 3-cliques and 4-cliques
- communities from the filtered graph
- cluster centroid vectors on the standardized sphere
- angular distance of each asset from its cluster centroid

This layer should define who can hedge whom. It should not by itself define the
trade direction.

### 3. R^N Dislocation Cloud

At date `t`, build a cross-sectional residual vector `e_t`:

```text
r_t = B_t f_t + e_t
```

`B_t f_t` can be estimated from market, sector, PCA, and graph-neighbor factors.
Then scale residuals robustly:

```text
z_t = D_t^-1 e_t
```

where `D_t` is a rolling robust residual-vol estimate or a shrinkage covariance
whitener. The point `z_t` is the R^N dislocation cloud state.

Candidate outlier scores:

- univariate residual z-score by name
- robust Mahalanobis distance of `z_t`
- local density / nearest historical analog score
- signed projection onto graph-cluster centroids
- clique barycentric deviation for PMFG cliques

The trade is a sparse portfolio, not a pair:

```text
minimize   - expected_alpha(w) + lambda_cost * turnover + lambda_risk * w' Sigma w
subject to sum(abs(w)) <= gross_cap
           sum(w) = 0
           beta' w = 0
           sector_exposure(w) approx 0
           cluster_exposure(w) bounded
           liquidity and borrow constraints
```

## Candidate Strategies

### A. Stable-Regime Graph-Cluster Reversion

Only trade when SPD velocity is low and graph edge persistence is high. Within
each persistent PMFG/MST cluster, long the negative residual-cloud tail and short
the positive tail. Rebalance weekly or every 10 trading days to keep costs below
the existing graph-PCA baseline.

This is the first strategy to test.

### B. Transition-Regime Residual Continuation

When SPD velocity and graph rewiring are high, do not fade dislocations by
default. Test whether residual outliers continue for 1-5 trading days as crowded
positions unwind. If continuation only works in transition regimes, it explains
why unconditional residual reversion is weak.

### C. PMFG Clique Barycentric Reversion

For each persistent 3- or 4-clique, compute the clique barycenter on the
standardized sphere and each member's residual deviation from it. Trade only
within cliques that persist across several windows. This is a more local version
of graph-cluster reversion.

### D. Cloud Analog Trades

For each date, find historical dislocation-cloud states with similar SPD regime
and graph topology. Trade only when historical analogs have positive forward
cluster-neutral returns after costs. This is slower to implement, but it is a
good guard against assuming every large residual should revert.

### E. Cross-Cluster Centroid Relative Value

Construct cluster centroid residuals first, trade dislocated clusters against
other clusters, and then hedge intra-cluster with the name-level cloud. This
should reduce single-name noise but may dilute alpha.

## First Backtest Specification

Start with strategy A plus an explicit test of strategy B as a sign flip in
transition regimes.

Data:

- Source: `/home/famadeo/archive/SP500_statarb`
- Prices: adjusted daily closes.
- Universe: S&P 500 symbols with point-in-time membership approximation already
  present in the archive.
- Date range: 2021-05-24 to 2026-05-20 for comparability with graph/PCA baseline.

Parameters:

- covariance lookback: 252 trading days
- signal lookback: 20 or 40 trading days
- rebalance: 5 or 10 trading days
- costs: 5 bps per unit turnover primary, 1 bp sensitivity
- shrinkage: Ledoit-Wolf/OAS if available, otherwise convex blend with diagonal
- graph: MST required; PMFG if implementation cost is acceptable, otherwise TMFG
  or MST plus top-k-neighbor approximation as a first pass
- regimes: rolling quantile states, then k-means on matrix-log features only if
  the quantile state passes

Decision gates:

- out-of-sample net return above zero after 5 bps costs
- gross/cost above 3x
- daily t-stat above 2.0 or bootstrap p-value below 5%
- OOS Sharpe above 1.0
- drawdown below 15%
- no single sector contributes more than 35% of gross alpha
- result beats archived `pca_only_gated` and `graph_pca_ungated`

## Failure Modes

- **SPD overfitting:** too many regime labels on a short daily sample.
- **Graph churn:** unstable MST/PMFG edges create turnover without signal.
- **Cost illusion:** gross edge exists but turnover consumes it, as in the
  existing graph/PCA baseline.
- **Survivorship leakage:** archive universe must be checked before any claim
  stronger than a research prototype.
- **Reversion assumption:** transition regimes may require continuation or no
  trade; unconditional mean reversion is already weak locally.

## Implementation Sketch

Create reusable geometry modules in `src/quantlab/geometry/`:

- `spd.py`: eigenvalue floor, matrix log/exp, log-Euclidean distance, rolling SPD
  features.
- `filtered_graph.py`: correlation distance, MST, PMFG/TMFG or top-k fallback,
  edge persistence, clique extraction.
- `dislocation.py`: residual-cloud construction, robust z-scores, cloud outlier
  scores, sparse neutral portfolio builder.

Then add:

- `scripts/run_spd_graph_dislocation_statarb.py`
- `experiments/HYP-0021-spd-graph-dislocation-statarb/`

Tests should cover no-lookahead alignment, SPD eigenvalue floors, MST edge count,
PMFG edge bound, and portfolio neutrality.
