from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from quantlab.pairs.config import PairsSelectionConfig

MIN_STATIONARITY_OBSERVATIONS = 20


@dataclass(frozen=True)
class PairSelectionResult:
    asset_class: str
    root_a: str
    root_b: str
    pair: str
    selected: bool
    reason: str
    observations: int
    train_observations: int
    test_observations: int
    test_start_index: int
    return_correlation: float
    price_correlation: float
    hedge_ratio: float
    spread_adf_pvalue: float
    half_life_bars: float
    average_pair_volume: float

    def to_dict(self) -> dict[str, str | bool | int | float]:
        return asdict(self)


def evaluate_pair_selection(
    pair_frame: pd.DataFrame,
    asset_class: str,
    root_a: str,
    root_b: str,
    config: PairsSelectionConfig,
) -> PairSelectionResult:
    observations = len(pair_frame)
    train_observations = int(observations * config.train_fraction)
    test_observations = observations - train_observations
    pair = f"{root_a}-{root_b}"

    if not config.enabled:
        return PairSelectionResult(
            asset_class=asset_class,
            root_a=root_a,
            root_b=root_b,
            pair=pair,
            selected=True,
            reason="selection_disabled",
            observations=observations,
            train_observations=train_observations,
            test_observations=test_observations,
            test_start_index=train_observations,
            return_correlation=np.nan,
            price_correlation=np.nan,
            hedge_ratio=np.nan,
            spread_adf_pvalue=np.nan,
            half_life_bars=np.nan,
            average_pair_volume=np.nan,
        )

    if train_observations < config.min_train_observations:
        return _reject(
            asset_class,
            root_a,
            root_b,
            observations,
            train_observations,
            test_observations,
            "insufficient_train_observations",
        )
    if test_observations < config.min_test_observations:
        return _reject(
            asset_class,
            root_a,
            root_b,
            observations,
            train_observations,
            test_observations,
            "insufficient_test_observations",
        )

    train = pair_frame.iloc[:train_observations].copy()
    return_correlation = float(train["log_return_a"].corr(train["log_return_b"]))
    price_correlation = float(train["log_price_a"].corr(train["log_price_b"]))
    average_pair_volume = float(train[["volume_a", "volume_b"]].min(axis=1).mean())
    hedge_ratio = _hedge_ratio(train)
    spread = train["log_price_b"] - hedge_ratio * train["log_price_a"]
    spread_adf_pvalue = _adf_pvalue(spread, config.adf_max_lag)
    half_life_bars = _half_life_bars(spread)

    reason = _selection_failure_reason(
        config=config,
        return_correlation=return_correlation,
        hedge_ratio=hedge_ratio,
        spread_adf_pvalue=spread_adf_pvalue,
        half_life_bars=half_life_bars,
        average_pair_volume=average_pair_volume,
    )

    return PairSelectionResult(
        asset_class=asset_class,
        root_a=root_a,
        root_b=root_b,
        pair=pair,
        selected=reason == "selected",
        reason=reason,
        observations=observations,
        train_observations=train_observations,
        test_observations=test_observations,
        test_start_index=train_observations,
        return_correlation=return_correlation,
        price_correlation=price_correlation,
        hedge_ratio=hedge_ratio,
        spread_adf_pvalue=spread_adf_pvalue,
        half_life_bars=half_life_bars,
        average_pair_volume=average_pair_volume,
    )


def _reject(
    asset_class: str,
    root_a: str,
    root_b: str,
    observations: int,
    train_observations: int,
    test_observations: int,
    reason: str,
) -> PairSelectionResult:
    return PairSelectionResult(
        asset_class=asset_class,
        root_a=root_a,
        root_b=root_b,
        pair=f"{root_a}-{root_b}",
        selected=False,
        reason=reason,
        observations=observations,
        train_observations=train_observations,
        test_observations=test_observations,
        test_start_index=train_observations,
        return_correlation=np.nan,
        price_correlation=np.nan,
        hedge_ratio=np.nan,
        spread_adf_pvalue=np.nan,
        half_life_bars=np.nan,
        average_pair_volume=np.nan,
    )


def _hedge_ratio(frame: pd.DataFrame) -> float:
    x = frame["log_price_a"].to_numpy(dtype=float)
    y = frame["log_price_b"].to_numpy(dtype=float)
    variance = float(np.var(x, ddof=1))
    if variance <= 0 or not np.isfinite(variance):
        return np.nan
    return float(np.cov(y, x, ddof=1)[0, 1] / variance)


def _adf_pvalue(spread: pd.Series, max_lag: int | None) -> float:
    values = spread.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(values) < MIN_STATIONARITY_OBSERVATIONS or float(np.std(values)) == 0:
        return np.nan
    try:
        if max_lag is None:
            result = adfuller(values, autolag="AIC")
        else:
            result = adfuller(values, maxlag=max_lag)
    except (ValueError, np.linalg.LinAlgError):
        return np.nan
    return float(result[1])


def _half_life_bars(spread: pd.Series) -> float:
    values = spread.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(values) < MIN_STATIONARITY_OBSERVATIONS:
        return np.nan
    lagged = values[:-1]
    delta = np.diff(values)
    variance = float(np.var(lagged, ddof=1))
    if variance <= 0 or not np.isfinite(variance):
        return np.nan
    slope = float(np.cov(delta, lagged, ddof=1)[0, 1] / variance)
    if slope >= 0 or not np.isfinite(slope):
        return np.inf
    return float(-np.log(2) / slope)


def _selection_failure_reason(
    *,
    config: PairsSelectionConfig,
    return_correlation: float,
    hedge_ratio: float,
    spread_adf_pvalue: float,
    half_life_bars: float,
    average_pair_volume: float,
) -> str:
    checks = [
        (not np.isfinite(return_correlation), "invalid_return_correlation"),
        (abs(return_correlation) < config.min_abs_return_correlation, "low_return_correlation"),
        (not np.isfinite(hedge_ratio), "invalid_hedge_ratio"),
        (not np.isfinite(spread_adf_pvalue), "invalid_spread_adf"),
        (spread_adf_pvalue > config.max_spread_adf_pvalue, "spread_not_stationary"),
        (not np.isfinite(half_life_bars), "invalid_half_life"),
        (half_life_bars > config.max_half_life_bars, "half_life_too_slow"),
        (average_pair_volume < config.min_average_pair_volume, "low_average_pair_volume"),
    ]
    for failed, reason in checks:
        if failed:
            return reason
    return "selected"
