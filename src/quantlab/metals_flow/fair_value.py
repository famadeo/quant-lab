# ruff: noqa: PLR2004
from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

try:
    from numba import njit
except ImportError:  # pragma: no cover - optional acceleration
    njit = None


def rolling_relative_value_residuals(
    log_prices: pd.DataFrame,
    *,
    lookback: int = 500,
    min_periods: int = 250,
    ridge_alpha: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = log_prices.replace([np.inf, -np.inf], np.nan).ffill()
    values = clean.to_numpy(dtype=float)
    if njit is not None:
        residual_values = _rolling_ridge_residuals_numba(
            values,
            lookback,
            min_periods,
            ridge_alpha,
        )
        residuals = pd.DataFrame(residual_values, index=clean.index, columns=clean.columns)
        zscores = _rolling_zscore(residuals, lookback, min_periods)
        return residuals, zscores

    residuals = pd.DataFrame(index=clean.index, columns=clean.columns, dtype=float)

    for target_index, target in enumerate(clean.columns):
        predictor_index = [j for j in range(values.shape[1]) if j != target_index]
        design = np.column_stack([np.ones(len(values)), values[:, predictor_index]])
        y = values[:, target_index]
        valid = np.isfinite(design).all(axis=1) & np.isfinite(y)
        xx_prefix, xy_prefix, count_prefix = _rolling_regression_prefixes(design, y, valid)
        penalty = np.eye(design.shape[1]) * ridge_alpha
        penalty[0, 0] = 0.0

        indices = np.arange(len(clean))
        starts = np.maximum(0, indices - lookback)
        counts = count_prefix[indices] - count_prefix[starts]
        usable = valid & (counts >= min_periods)
        usable_indices = indices[usable]
        if len(usable_indices) == 0:
            continue
        usable_starts = starts[usable]
        xx = xx_prefix[usable_indices] - xx_prefix[usable_starts]
        xy = xy_prefix[usable_indices] - xy_prefix[usable_starts]
        beta = np.linalg.solve(xx + penalty, xy[..., None]).squeeze(axis=-1)
        predictions = np.einsum("ij,ij->i", design[usable_indices], beta)
        residuals.iloc[usable_indices, residuals.columns.get_loc(target)] = (
            y[usable_indices] - predictions
        )

    zscores = _rolling_zscore(residuals, lookback, min_periods)
    return residuals, zscores


def ewma_relative_value_residuals(
    log_prices: pd.DataFrame,
    *,
    halflife: int = 500,
    min_periods: int = 500,
    zscore_window: int = 2_000,
    ridge_alpha: float = 1e-5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = log_prices.replace([np.inf, -np.inf], np.nan).ffill()
    values = clean.to_numpy(dtype=float)
    if njit is not None:
        residual_values = _ewma_conditional_residuals_numba(
            values,
            halflife,
            min_periods,
            ridge_alpha,
        )
    else:
        residual_values = _ewma_conditional_residuals_python(
            values,
            halflife=halflife,
            min_periods=min_periods,
            ridge_alpha=ridge_alpha,
        )
    residuals = pd.DataFrame(residual_values, index=clean.index, columns=clean.columns)
    zscores = _rolling_zscore(residuals, zscore_window, min_periods)
    return residuals, zscores


def pairwise_cointegration(log_prices: pd.DataFrame, *, min_obs: int = 500) -> pd.DataFrame:
    rows = []
    clean = log_prices.replace([np.inf, -np.inf], np.nan).ffill().dropna(how="all")
    for root_a, root_b in combinations(clean.columns, 2):
        pair = clean[[root_a, root_b]].dropna()
        if len(pair) < min_obs:
            continue
        score, pvalue, _ = coint(pair[root_a], pair[root_b])
        beta = _ols_beta(pair[root_b], pair[root_a])
        spread = pair[root_a] - beta * pair[root_b]
        adf = adfuller(spread.dropna(), autolag="AIC")
        rows.append(
            {
                "root_a": root_a,
                "root_b": root_b,
                "observations": len(pair),
                "coint_tstat": float(score),
                "coint_pvalue": float(pvalue),
                "hedge_beta_a_on_b": beta,
                "spread_adf_tstat": float(adf[0]),
                "spread_adf_pvalue": float(adf[1]),
                "half_life_bars": half_life(spread),
            }
        )
    return pd.DataFrame(rows)


def half_life(series: pd.Series) -> float:
    values = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 20:
        return np.nan
    lag = values.shift(1).dropna()
    delta = values.diff().dropna()
    aligned = pd.DataFrame({"lag": lag, "delta": delta}).dropna()
    if len(aligned) < 20 or aligned["lag"].std(ddof=1) == 0:
        return np.nan
    x = aligned["lag"].to_numpy(dtype=float)
    y = aligned["delta"].to_numpy(dtype=float)
    beta = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0][1]
    if beta >= 0:
        return np.inf
    return float(-math.log(2.0) / beta)


def _rolling_zscore(frame: pd.DataFrame, lookback: int, min_periods: int) -> pd.DataFrame:
    mean = frame.rolling(lookback, min_periods=min_periods).mean()
    std = frame.rolling(lookback, min_periods=min_periods).std(ddof=1)
    return ((frame - mean) / std.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _ols_beta(x_values: pd.Series, y_values: pd.Series) -> float:
    x = x_values.to_numpy(dtype=float)
    y = y_values.to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    return float(np.linalg.lstsq(design, y, rcond=None)[0][1])


def _rolling_regression_prefixes(
    design: np.ndarray,
    y_values: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean_design = np.where(valid[:, None], design, 0.0)
    clean_y = np.where(valid, y_values, 0.0)
    xx = np.einsum("ni,nj->nij", clean_design, clean_design)
    xy = clean_design * clean_y[:, None]
    xx_prefix = np.concatenate(
        [np.zeros((1, design.shape[1], design.shape[1])), np.cumsum(xx, axis=0)]
    )
    xy_prefix = np.vstack([np.zeros((1, design.shape[1])), np.cumsum(xy, axis=0)])
    count_prefix = np.r_[0, np.cumsum(valid.astype(int))]
    return xx_prefix, xy_prefix, count_prefix


def _ewma_conditional_residuals_python(
    values: np.ndarray,
    *,
    halflife: int,
    min_periods: int,
    ridge_alpha: float,
) -> np.ndarray:
    n_rows, n_cols = values.shape
    residuals = np.full((n_rows, n_cols), np.nan)
    alpha = 1.0 - np.exp(np.log(0.5) / max(float(halflife), 1.0))
    mean = np.zeros(n_cols)
    cov = np.eye(n_cols) * ridge_alpha
    seen = 0
    initialized = False
    for i, row in enumerate(values):
        if not np.isfinite(row).all():
            continue
        if initialized and seen >= min_periods:
            for target in range(n_cols):
                predictors = [col for col in range(n_cols) if col != target]
                cov_xx = cov[np.ix_(predictors, predictors)] + np.eye(n_cols - 1) * ridge_alpha
                cov_yx = cov[target, predictors]
                diff = row[predictors] - mean[predictors]
                beta = np.linalg.solve(cov_xx, cov_yx)
                prediction = mean[target] + float(diff @ beta)
                residuals[i, target] = row[target] - prediction
        if not initialized:
            mean = row.copy()
            initialized = True
        else:
            diff = row - mean
            mean = (1.0 - alpha) * mean + alpha * row
            cov = (1.0 - alpha) * cov + alpha * np.outer(diff, diff)
        seen += 1
    return residuals


if njit is not None:

    @njit(cache=True)
    def _rolling_ridge_residuals_numba(  # noqa: PLR0912, PLR0915
        values: np.ndarray,
        lookback: int,
        min_periods: int,
        ridge_alpha: float,
    ) -> np.ndarray:
        n_rows, n_cols = values.shape
        residuals = np.empty((n_rows, n_cols), dtype=np.float64)
        residuals[:, :] = np.nan
        for target in range(n_cols):
            m = n_cols
            xx_prefix = np.zeros((n_rows + 1, m, m), dtype=np.float64)
            xy_prefix = np.zeros((n_rows + 1, m), dtype=np.float64)
            count_prefix = np.zeros(n_rows + 1, dtype=np.int64)

            for i in range(n_rows):
                for a in range(m):
                    xy_prefix[i + 1, a] = xy_prefix[i, a]
                    for b in range(m):
                        xx_prefix[i + 1, a, b] = xx_prefix[i, a, b]
                count_prefix[i + 1] = count_prefix[i]

                valid = True
                if not np.isfinite(values[i, target]):
                    valid = False
                for col in range(n_cols):
                    if col != target and not np.isfinite(values[i, col]):
                        valid = False
                if not valid:
                    continue

                design = np.empty(m, dtype=np.float64)
                design[0] = 1.0
                idx = 1
                for col in range(n_cols):
                    if col != target:
                        design[idx] = values[i, col]
                        idx += 1
                y_value = values[i, target]
                for a in range(m):
                    xy_prefix[i + 1, a] += design[a] * y_value
                    for b in range(m):
                        xx_prefix[i + 1, a, b] += design[a] * design[b]
                count_prefix[i + 1] += 1

            for i in range(n_rows):
                valid_current = True
                if not np.isfinite(values[i, target]):
                    valid_current = False
                for col in range(n_cols):
                    if col != target and not np.isfinite(values[i, col]):
                        valid_current = False
                if not valid_current:
                    continue

                start = max(i - lookback, 0)
                count = count_prefix[i] - count_prefix[start]
                if count < min_periods:
                    continue

                xx = np.empty((m, m), dtype=np.float64)
                xy = np.empty(m, dtype=np.float64)
                for a in range(m):
                    xy[a] = xy_prefix[i, a] - xy_prefix[start, a]
                    for b in range(m):
                        xx[a, b] = xx_prefix[i, a, b] - xx_prefix[start, a, b]
                for a in range(m):
                    if a > 0:
                        xx[a, a] += ridge_alpha

                beta = np.linalg.solve(xx, xy)
                prediction = beta[0]
                idx = 1
                for col in range(n_cols):
                    if col != target:
                        prediction += beta[idx] * values[i, col]
                        idx += 1
                residuals[i, target] = values[i, target] - prediction
        return residuals

    @njit(cache=True)
    def _ewma_conditional_residuals_numba(  # noqa: PLR0912
        values: np.ndarray,
        halflife: int,
        min_periods: int,
        ridge_alpha: float,
    ) -> np.ndarray:
        n_rows, n_cols = values.shape
        residuals = np.empty((n_rows, n_cols), dtype=np.float64)
        residuals[:, :] = np.nan
        alpha = 1.0 - np.exp(np.log(0.5) / max(float(halflife), 1.0))
        mean = np.zeros(n_cols, dtype=np.float64)
        cov = np.eye(n_cols, dtype=np.float64) * ridge_alpha
        seen = 0
        initialized = False

        for i in range(n_rows):
            row_valid = True
            for col in range(n_cols):
                if not np.isfinite(values[i, col]):
                    row_valid = False
            if not row_valid:
                continue

            if initialized and seen >= min_periods:
                for target in range(n_cols):
                    m = n_cols - 1
                    cov_xx = np.empty((m, m), dtype=np.float64)
                    cov_yx = np.empty(m, dtype=np.float64)
                    diff = np.empty(m, dtype=np.float64)
                    a = 0
                    for col_a in range(n_cols):
                        if col_a == target:
                            continue
                        b = 0
                        for col_b in range(n_cols):
                            if col_b == target:
                                continue
                            cov_xx[a, b] = cov[col_a, col_b]
                            if a == b:
                                cov_xx[a, b] += ridge_alpha
                            b += 1
                        cov_yx[a] = cov[target, col_a]
                        diff[a] = values[i, col_a] - mean[col_a]
                        a += 1
                    beta = np.linalg.solve(cov_xx, cov_yx)
                    prediction = mean[target]
                    for j in range(m):
                        prediction += diff[j] * beta[j]
                    residuals[i, target] = values[i, target] - prediction

            if not initialized:
                for col in range(n_cols):
                    mean[col] = values[i, col]
                initialized = True
            else:
                diff_full = np.empty(n_cols, dtype=np.float64)
                for col in range(n_cols):
                    diff_full[col] = values[i, col] - mean[col]
                    mean[col] = (1.0 - alpha) * mean[col] + alpha * values[i, col]
                for row_idx in range(n_cols):
                    for col_idx in range(n_cols):
                        cov[row_idx, col_idx] = (1.0 - alpha) * cov[
                            row_idx, col_idx
                        ] + alpha * diff_full[row_idx] * diff_full[col_idx]
            seen += 1
        return residuals
