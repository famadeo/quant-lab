from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import polars as pl

from quantlab.config import BacktestConfig
from quantlab.features import add_returns


@dataclass(frozen=True)
class BacktestMetrics:
    observations: int
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    average_exposure: float
    total_turnover: float
    trades: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def run_long_only_backtest(
    signal_frame: pl.DataFrame,
    config: BacktestConfig,
) -> tuple[pl.DataFrame, BacktestMetrics]:
    required = {"timestamp", "symbol", "close", "raw_signal"}
    missing = required.difference(signal_frame.columns)
    if missing:
        raise ValueError(f"signal frame missing columns: {sorted(missing)}")

    cost_rate = (config.fee_bps + config.slippage_bps) / 10_000.0
    result = (
        add_returns(signal_frame)
        .with_columns(
            pl.col("raw_signal")
            .shift(config.signal_lag)
            .over("symbol")
            .fill_null(0.0)
            .clip(lower_bound=0.0, upper_bound=config.max_leverage)
            .alias("position")
        )
        .with_columns(
            pl.col("position")
            .diff()
            .over("symbol")
            .fill_null(pl.col("position").abs())
            .abs()
            .alias("turnover")
        )
        .with_columns(
            (pl.col("turnover") * cost_rate).alias("cost"),
            (pl.col("position") * pl.col("asset_return") - pl.col("turnover") * cost_rate).alias(
                "strategy_return"
            ),
        )
        .with_columns(
            (
                config.initial_capital * (1.0 + pl.col("strategy_return")).cum_prod().over("symbol")
            ).alias("equity")
        )
        .with_columns(
            (pl.col("equity") / pl.col("equity").cum_max().over("symbol") - 1.0).alias("drawdown")
        )
        .sort(["symbol", "timestamp"])
    )

    metrics = _calculate_metrics(result, config)
    return result, metrics


def _calculate_metrics(result: pl.DataFrame, config: BacktestConfig) -> BacktestMetrics:
    if result.is_empty():
        raise ValueError("backtest result is empty")

    returns = result["strategy_return"].to_numpy()
    equity = result["equity"].to_numpy()
    drawdown = result["drawdown"].to_numpy()
    exposure = result["position"].to_numpy()
    turnover = result["turnover"].to_numpy()

    total_return = float(equity[-1] / config.initial_capital - 1.0)
    mean_return = float(np.mean(returns))
    volatility = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    annualized_return = float((1.0 + mean_return) ** config.periods_per_year - 1.0)
    annualized_volatility = volatility * float(np.sqrt(config.periods_per_year))
    sharpe_ratio = (
        mean_return / volatility * float(np.sqrt(config.periods_per_year))
        if volatility > 0
        else 0.0
    )

    return BacktestMetrics(
        observations=len(result),
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe_ratio=float(sharpe_ratio),
        max_drawdown=float(np.min(drawdown)),
        average_exposure=float(np.mean(exposure)),
        total_turnover=float(np.sum(turnover)),
        trades=int(np.sum(turnover > 0)),
    )
