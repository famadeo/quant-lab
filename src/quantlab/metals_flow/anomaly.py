from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet


def mahalanobis_distances(
    shares: pd.DataFrame,
    *,
    method: str = "rolling",
    window: int = 500,
    min_periods: int = 100,
    ridge: float = 1e-6,
    ewma_halflife: int = 250,
) -> pd.Series:
    if method not in {"rolling", "expanding", "ewma"}:
        raise ValueError("method must be one of: rolling, expanding, ewma")
    values = shares.to_numpy(dtype=float)
    if method == "ewma":
        distances = _ewma_mahalanobis(values, min_periods, ridge, ewma_halflife)
    else:
        distances = _windowed_mahalanobis(values, method, window, min_periods, ridge)
    return pd.Series(distances, index=shares.index, name=f"md_{method}")


def robust_mahalanobis_snapshot(
    shares: pd.DataFrame,
    *,
    min_periods: int = 100,
    ridge: float = 1e-6,
    random_state: int = 7,
) -> pd.Series:
    values = shares.to_numpy(dtype=float)
    distances = np.full(len(values), np.nan)
    valid = np.isfinite(values).all(axis=1)
    if valid.sum() < min_periods:
        return pd.Series(distances, index=shares.index, name="md_robust_snapshot")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        estimator = MinCovDet(random_state=random_state, support_fraction=0.90).fit(values[valid])
    center = estimator.location_
    cov = _regularize_covariance(estimator.covariance_, ridge)
    inverse = np.linalg.pinv(cov)
    diffs = values - center
    distances[valid] = np.sqrt(np.einsum("ij,jk,ik->i", diffs[valid], inverse, diffs[valid]))
    return pd.Series(distances, index=shares.index, name="md_robust_snapshot")


def anomaly_flags(
    distances: pd.Series,
    *,
    quantiles: tuple[float, ...] = (0.90, 0.95, 0.99),
) -> pd.DataFrame:
    out = pd.DataFrame(index=distances.index)
    valid = distances.dropna()
    for quantile in quantiles:
        threshold = float(valid.quantile(quantile)) if not valid.empty else np.nan
        label = f"md_q{int(quantile * 100):02d}"
        out[label] = distances >= threshold
        out[f"{label}_threshold"] = threshold
    return out


def build_anomaly_frame(
    shares: pd.DataFrame,
    *,
    window: int = 500,
    min_periods: int = 100,
    ewma_halflife: int = 250,
) -> pd.DataFrame:
    rolling = mahalanobis_distances(
        shares, method="rolling", window=window, min_periods=min_periods
    )
    expanding = mahalanobis_distances(
        shares, method="expanding", window=window, min_periods=min_periods
    )
    ewma = mahalanobis_distances(
        shares,
        method="ewma",
        window=window,
        min_periods=min_periods,
        ewma_halflife=ewma_halflife,
    )
    robust = robust_mahalanobis_snapshot(shares, min_periods=min_periods)
    flags = anomaly_flags(rolling)
    return pd.concat([rolling, expanding, ewma, robust, flags], axis=1)


def _windowed_mahalanobis(
    values: np.ndarray,
    method: str,
    window: int,
    min_periods: int,
    ridge: float,
) -> np.ndarray:
    distances = np.full(len(values), np.nan)
    for i in range(len(values)):
        start = 0 if method == "expanding" else max(0, i - window)
        history = values[start:i]
        history = history[np.isfinite(history).all(axis=1)]
        if len(history) < min_periods or not np.isfinite(values[i]).all():
            continue
        center = history.mean(axis=0)
        cov = np.cov(history, rowvar=False)
        inverse = np.linalg.pinv(_regularize_covariance(cov, ridge))
        diff = values[i] - center
        distances[i] = float(np.sqrt(diff @ inverse @ diff.T))
    return distances


def _ewma_mahalanobis(
    values: np.ndarray,
    min_periods: int,
    ridge: float,
    halflife: int,
) -> np.ndarray:
    alpha = 1.0 - np.exp(np.log(0.5) / max(float(halflife), 1.0))
    distances = np.full(len(values), np.nan)
    mean: np.ndarray | None = None
    cov: np.ndarray | None = None
    seen = 0

    for i, row in enumerate(values):
        if not np.isfinite(row).all():
            continue
        if mean is not None and cov is not None and seen >= min_periods:
            diff = row - mean
            inverse = np.linalg.pinv(_regularize_covariance(cov, ridge))
            distances[i] = float(np.sqrt(diff @ inverse @ diff.T))
        if mean is None:
            mean = row.copy()
            cov = np.eye(values.shape[1]) * ridge
        else:
            diff = row - mean
            mean = (1.0 - alpha) * mean + alpha * row
            cov = (1.0 - alpha) * cov + alpha * np.outer(diff, diff)
        seen += 1
    return distances


def _regularize_covariance(cov: np.ndarray, ridge: float) -> np.ndarray:
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    scale = float(np.trace(cov) / cov.shape[0]) if cov.shape[0] else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return cov + np.eye(cov.shape[0]) * ridge * scale
