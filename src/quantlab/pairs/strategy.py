from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import cast

import numpy as np
import pandas as pd

from quantlab.pairs.config import PairsBacktestConfig, PairsStrategyConfig


@dataclass(frozen=True)
class ReturnMetrics:
    observations: int
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    active_fraction: float
    total_turnover: float
    trades: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def run_zscore_pair(
    pair_frame: pd.DataFrame,
    strategy: PairsStrategyConfig,
    backtest: PairsBacktestConfig,
) -> pd.DataFrame:
    beta, zscore = _rolling_spread_zscore(pair_frame, strategy.lookback)
    target_a = np.zeros(len(pair_frame), dtype=float)
    target_b = np.zeros(len(pair_frame), dtype=float)
    state = 0
    holding_bars = 0
    cooldown_remaining = 0
    current_target_a = 0.0
    current_target_b = 0.0
    for idx, value in enumerate(zscore.to_numpy(dtype=float)):
        hedge_ratio = float(beta.iloc[idx]) if not np.isnan(beta.iloc[idx]) else np.nan
        previous_state = state
        can_rebalance = idx % strategy.rebalance_every_bars == 0
        can_exit = holding_bars >= strategy.min_holding_bars
        can_enter = cooldown_remaining == 0

        if can_rebalance:
            state = _next_zscore_state(
                state=state,
                value=value,
                hedge_ratio=hedge_ratio,
                strategy=strategy,
                can_enter=can_enter,
                can_exit=can_exit,
            )

        if state == 0:
            current_target_a = 0.0
            current_target_b = 0.0
        elif can_rebalance and np.isfinite(hedge_ratio):
            raw_a = -state * hedge_ratio
            raw_b = float(state)
            gross = abs(raw_a) + abs(raw_b)
            if gross > 0:
                current_target_a = raw_a / gross
                current_target_b = raw_b / gross
        target_a[idx] = current_target_a
        target_b[idx] = current_target_b
        holding_bars, cooldown_remaining = _advance_trade_timers(
            previous_state, state, holding_bars, cooldown_remaining, strategy.cooldown_bars
        )

    result = cast(pd.DataFrame, pair_frame.loc[:, ["ts", "log_return_a", "log_return_b"]].copy())
    result["method"] = "zscore"
    result["indicator"] = zscore.to_numpy(dtype=float)
    result["filter_indicator"] = np.nan
    result["target_a"] = target_a
    result["target_b"] = target_b
    return _apply_lagged_returns(result, strategy, backtest)


def _rolling_spread_zscore(pair_frame: pd.DataFrame, lookback: int) -> tuple[pd.Series, pd.Series]:
    x = pair_frame["log_price_a"].to_numpy(dtype=float)
    y = pair_frame["log_price_b"].to_numpy(dtype=float)
    x_series = pd.Series(x)
    y_series = pd.Series(y)
    beta = y_series.rolling(lookback).cov(x_series) / x_series.rolling(lookback).var()
    beta = beta.replace([np.inf, -np.inf], np.nan)
    spread = y_series - beta * x_series
    spread_mean = spread.rolling(lookback).mean()
    spread_std = spread.rolling(lookback).std()
    zscore = ((spread - spread_mean) / spread_std).replace([np.inf, -np.inf], np.nan)
    return beta, zscore


def _next_zscore_state(
    *,
    state: int,
    value: float,
    hedge_ratio: float,
    strategy: PairsStrategyConfig,
    can_enter: bool,
    can_exit: bool,
) -> int:
    next_state = state
    if not np.isfinite(value) or not np.isfinite(hedge_ratio):
        next_state = 0 if can_exit else state
    elif state == 0 and can_enter and value > strategy.z_entry:
        next_state = -1
    elif state == 0 and can_enter and value < -strategy.z_entry:
        next_state = 1
    elif state != 0 and can_exit and abs(value) < strategy.z_exit:
        next_state = 0
    elif state == 1 and can_exit and value > strategy.z_entry:
        next_state = -1
    elif state == -1 and can_exit and value < -strategy.z_entry:
        next_state = 1
    return next_state


def run_zscore_mahalanobis_filter_pair(
    pair_frame: pd.DataFrame,
    strategy: PairsStrategyConfig,
    backtest: PairsBacktestConfig,
) -> pd.DataFrame:
    beta, zscore = _rolling_spread_zscore(pair_frame, strategy.lookback)
    mahalanobis = _rolling_mahalanobis_distances(pair_frame, strategy)
    target_a = np.zeros(len(pair_frame), dtype=float)
    target_b = np.zeros(len(pair_frame), dtype=float)
    state = 0
    holding_bars = 0
    cooldown_remaining = 0
    current_target_a = 0.0
    current_target_b = 0.0

    for idx, value in enumerate(zscore.to_numpy(dtype=float)):
        hedge_ratio = float(beta.iloc[idx]) if not np.isnan(beta.iloc[idx]) else np.nan
        mahalanobis_distance = float(mahalanobis[idx])
        previous_state = state
        can_rebalance = idx % strategy.rebalance_every_bars == 0
        can_exit = holding_bars >= strategy.min_holding_bars
        can_enter = cooldown_remaining == 0

        if can_rebalance:
            state = _next_zscore_mahalanobis_filter_state(
                state=state,
                value=value,
                hedge_ratio=hedge_ratio,
                mahalanobis_distance=mahalanobis_distance,
                strategy=strategy,
                can_enter=can_enter,
                can_exit=can_exit,
            )

        if state == 0:
            current_target_a = 0.0
            current_target_b = 0.0
        elif can_rebalance and np.isfinite(hedge_ratio):
            raw_a = -state * hedge_ratio
            raw_b = float(state)
            gross = abs(raw_a) + abs(raw_b)
            if gross > 0:
                current_target_a = raw_a / gross
                current_target_b = raw_b / gross
        target_a[idx] = current_target_a
        target_b[idx] = current_target_b
        holding_bars, cooldown_remaining = _advance_trade_timers(
            previous_state, state, holding_bars, cooldown_remaining, strategy.cooldown_bars
        )

    result = cast(pd.DataFrame, pair_frame.loc[:, ["ts", "log_return_a", "log_return_b"]].copy())
    result["method"] = "zscore_mahalanobis"
    result["indicator"] = zscore.to_numpy(dtype=float)
    result["filter_indicator"] = mahalanobis
    result["target_a"] = target_a
    result["target_b"] = target_b
    return _apply_lagged_returns(result, strategy, backtest)


def _next_zscore_mahalanobis_filter_state(
    *,
    state: int,
    value: float,
    hedge_ratio: float,
    mahalanobis_distance: float,
    strategy: PairsStrategyConfig,
    can_enter: bool,
    can_exit: bool,
) -> int:
    next_state = state
    if not np.isfinite(value) or not np.isfinite(hedge_ratio):
        next_state = 0 if can_exit else state
    else:
        entry_confirmed = bool(
            np.isfinite(mahalanobis_distance) and mahalanobis_distance > strategy.mahalanobis_entry
        )
        if state == 0 and can_enter and entry_confirmed and value > strategy.z_entry:
            next_state = -1
        elif state == 0 and can_enter and entry_confirmed and value < -strategy.z_entry:
            next_state = 1
        elif state != 0 and can_exit and abs(value) < strategy.z_exit:
            next_state = 0
        elif state == 1 and can_exit and value > strategy.z_entry:
            next_state = -1 if can_enter and entry_confirmed else 0
        elif state == -1 and can_exit and value < -strategy.z_entry:
            next_state = 1 if can_enter and entry_confirmed else 0
    return next_state


def _rolling_mahalanobis_distances(
    pair_frame: pd.DataFrame,
    strategy: PairsStrategyConfig,
) -> np.ndarray:
    values = pair_frame[["log_price_a", "log_price_b"]].to_numpy(dtype=float)
    distances = np.full(len(pair_frame), np.nan, dtype=float)

    for idx in range(strategy.lookback - 1, len(pair_frame)):
        window = values[idx - strategy.lookback + 1 : idx + 1]
        if not np.isfinite(window).all():
            continue

        mean = window.mean(axis=0)
        covariance = np.cov(window, rowvar=False) + np.eye(2) * strategy.covariance_ridge
        precision = np.linalg.pinv(covariance)
        diff = values[idx] - mean
        squared_distance = float(diff.T @ precision @ diff)
        if squared_distance >= 0 and np.isfinite(squared_distance):
            distances[idx] = float(np.sqrt(squared_distance))

    return distances


def run_mahalanobis_pair(
    pair_frame: pd.DataFrame,
    strategy: PairsStrategyConfig,
    backtest: PairsBacktestConfig,
) -> pd.DataFrame:
    values = pair_frame[["log_price_a", "log_price_b"]].to_numpy(dtype=float)
    distances = np.full(len(pair_frame), np.nan, dtype=float)
    target_a = np.zeros(len(pair_frame), dtype=float)
    target_b = np.zeros(len(pair_frame), dtype=float)
    active = False
    holding_bars = 0
    cooldown_remaining = 0
    current_target_a = 0.0
    current_target_b = 0.0

    for idx in range(strategy.lookback - 1, len(pair_frame)):
        previous_state = active
        can_rebalance = idx % strategy.rebalance_every_bars == 0
        can_exit = holding_bars >= strategy.min_holding_bars
        can_enter = cooldown_remaining == 0
        window = values[idx - strategy.lookback + 1 : idx + 1]
        if not np.isfinite(window).all():
            active = False if can_exit else active
            continue

        mean = window.mean(axis=0)
        covariance = np.cov(window, rowvar=False)
        covariance = covariance + np.eye(2) * strategy.covariance_ridge
        precision = np.linalg.pinv(covariance)
        diff = values[idx] - mean
        squared_distance = float(diff.T @ precision @ diff)
        if squared_distance < 0 or not np.isfinite(squared_distance):
            active = False if can_exit else active
            continue

        distance = float(np.sqrt(squared_distance))
        distances[idx] = distance
        if active and can_rebalance and can_exit and distance < strategy.mahalanobis_exit:
            active = False
        elif not active and can_rebalance and can_enter and distance > strategy.mahalanobis_entry:
            active = True

        if not active:
            current_target_a = 0.0
            current_target_b = 0.0
        elif can_rebalance:
            raw = -(precision @ diff)
            gross = float(np.abs(raw).sum())
            if gross > 0 and np.isfinite(gross):
                current_target_a = float(raw[0] / gross)
                current_target_b = float(raw[1] / gross)
        target_a[idx] = current_target_a
        target_b[idx] = current_target_b
        holding_bars, cooldown_remaining = _advance_trade_timers(
            int(previous_state),
            int(active),
            holding_bars,
            cooldown_remaining,
            strategy.cooldown_bars,
        )

    result = cast(pd.DataFrame, pair_frame.loc[:, ["ts", "log_return_a", "log_return_b"]].copy())
    result["method"] = "mahalanobis"
    result["indicator"] = distances
    result["filter_indicator"] = np.nan
    result["target_a"] = target_a
    result["target_b"] = target_b
    return _apply_lagged_returns(result, strategy, backtest)


def _advance_trade_timers(
    previous_state: int,
    current_state: int,
    holding_bars: int,
    cooldown_remaining: int,
    cooldown_bars: int,
) -> tuple[int, int]:
    if current_state != 0:
        return holding_bars + 1 if previous_state == current_state else 1, 0
    if previous_state != 0 and current_state == 0:
        return 0, cooldown_bars
    return 0, max(cooldown_remaining - 1, 0)


def calculate_metrics(
    returns: pd.Series,
    active: pd.Series,
    turnover: pd.Series,
    periods_per_year: int,
) -> ReturnMetrics:
    cleaned_returns = returns.fillna(0.0).to_numpy(dtype=float)
    equity = np.cumprod(1.0 + cleaned_returns)
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    volatility = float(np.std(cleaned_returns, ddof=1)) if len(cleaned_returns) > 1 else 0.0
    mean_return = float(np.mean(cleaned_returns)) if len(cleaned_returns) else 0.0
    sharpe = mean_return / volatility * np.sqrt(periods_per_year) if volatility > 0 else 0.0
    annualized_return = (1.0 + mean_return) ** periods_per_year - 1.0

    return ReturnMetrics(
        observations=len(cleaned_returns),
        total_return=float(equity[-1] - 1.0) if len(equity) else 0.0,
        annualized_return=float(annualized_return),
        annualized_volatility=float(volatility * np.sqrt(periods_per_year)),
        sharpe_ratio=float(sharpe),
        max_drawdown=float(np.min(drawdown)) if len(drawdown) else 0.0,
        active_fraction=float(active.fillna(False).mean()) if len(active) else 0.0,
        total_turnover=float(turnover.fillna(0.0).sum()),
        trades=int((turnover.fillna(0.0) > 0).sum()),
    )


def _apply_lagged_returns(
    frame: pd.DataFrame,
    strategy: PairsStrategyConfig,
    backtest: PairsBacktestConfig,
) -> pd.DataFrame:
    result = frame.copy()
    result = _suppress_small_target_changes(result, strategy.min_position_change)
    result["position_a"] = result["target_a"].shift(strategy.signal_lag).fillna(0.0)
    result["position_b"] = result["target_b"].shift(strategy.signal_lag).fillna(0.0)
    result["turnover"] = (
        result["position_a"].diff().fillna(result["position_a"]).abs()
        + result["position_b"].diff().fillna(result["position_b"]).abs()
    )
    cost_rate = (backtest.fee_bps + backtest.slippage_bps) / 10_000.0
    gross_log_return = (
        result["position_a"] * result["log_return_a"]
        + result["position_b"] * result["log_return_b"]
    )
    result["cost"] = result["turnover"] * cost_rate
    result["pair_return"] = np.expm1(gross_log_return) - result["cost"]
    result["active"] = (result["position_a"].abs() + result["position_b"].abs()) > 0
    return result


def _suppress_small_target_changes(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if threshold <= 0:
        return frame
    result = frame.copy()
    last_a = 0.0
    last_b = 0.0
    target_a_values = result["target_a"].to_numpy(dtype=float)
    target_b_values = result["target_b"].to_numpy(dtype=float)
    for idx, target_a in enumerate(target_a_values):
        target_b = float(target_b_values[idx])
        change = abs(target_a - last_a) + abs(target_b - last_b)
        if change < threshold:
            target_a_values[idx] = last_a
            target_b_values[idx] = last_b
        else:
            last_a = float(target_a)
            last_b = target_b
    result["target_a"] = target_a_values
    result["target_b"] = target_b_values
    return result
