from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.metals_flow.anomaly import mahalanobis_distances


def test_expanding_mahalanobis_matches_bruteforce_history() -> None:
    min_periods = 3
    shares = pd.DataFrame(
        {
            "GC": [0.50, 0.45, 0.40, 0.35, 0.30, 0.25],
            "SI": [0.30, 0.35, 0.30, 0.25, 0.40, 0.45],
            "HG": [0.20, 0.20, 0.30, 0.40, 0.30, 0.30],
        }
    )

    actual = mahalanobis_distances(
        shares,
        method="expanding",
        min_periods=min_periods,
        ridge=1e-6,
    )

    expected = np.full(len(shares), np.nan)
    values = shares.to_numpy(dtype=float)
    for i in range(len(values)):
        history = values[:i]
        if len(history) < min_periods:
            continue
        center = history.mean(axis=0)
        cov = np.cov(history, rowvar=False)
        scale = np.trace(cov) / cov.shape[0]
        regularized = cov + np.eye(cov.shape[0]) * 1e-6 * scale
        diff = values[i] - center
        expected[i] = np.sqrt(diff @ np.linalg.pinv(regularized) @ diff.T)

    np.testing.assert_allclose(actual.to_numpy(), expected, equal_nan=True)
