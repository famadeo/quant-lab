"""Test drawdown-control overlays for the best core metals min-var portfolio."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = (
    REPO_ROOT
    / "experiments"
    / "HYP-0051-core-metals-long-short-overlays"
    / "core_metals_5m_long_short_overlay_benchmark.parquet"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "HYP-0052-core-metals-drawdown-control"

ROOTS = ["GC", "SI", "HG", "PL", "PA"]
BASE_VARIANT = "MINVAR_30D_SHRINK25"
VOL_LOOKBACK = "30D"
MIN_VOL_OBS = 1_000
EPSILON = 1e-12
PLOT_VARIANT_COUNT = 10

# Per-side root-specific cost estimates from HYP-0046 MBP1 spread model.
COST_BPS = pd.Series(
    {
        "GC": 0.5508,
        "SI": 1.8695,
        "HG": 0.8004,
        "PL": 2.5632,
        "PA": 5.5939,
    },
    dtype=float,
)


@dataclass(frozen=True)
class OverlaySpec:
    name: str
    vol_target: float | None = None
    momentum_window: str | None = None
    momentum_risk_off_scale: float = 1.0
    dd_rules: tuple[tuple[float, float], ...] = ()
    rebalance: str = "bar"


OVERLAY_SPECS = [
    OverlaySpec("BASE_MINVAR"),
    OverlaySpec("VOL_TARGET_16", vol_target=0.16),
    OverlaySpec("VOL_TARGET_14", vol_target=0.14),
    OverlaySpec("VOL_TARGET_12", vol_target=0.12),
    OverlaySpec("VOL_TARGET_10", vol_target=0.10),
    OverlaySpec("TSMOM_20D_HALF", momentum_window="20D", momentum_risk_off_scale=0.50),
    OverlaySpec("TSMOM_60D_HALF", momentum_window="60D", momentum_risk_off_scale=0.50),
    OverlaySpec("TSMOM_60D_FLAT", momentum_window="60D", momentum_risk_off_scale=0.00),
    OverlaySpec("DD_SOFT", dd_rules=((-0.15, 0.50), (-0.08, 0.75))),
    OverlaySpec("DD_HARD", dd_rules=((-0.15, 0.25), (-0.10, 0.50), (-0.05, 0.75))),
    OverlaySpec(
        "VOL12_DD_SOFT",
        vol_target=0.12,
        dd_rules=((-0.15, 0.50), (-0.08, 0.75)),
    ),
    OverlaySpec(
        "VOL12_DD_HARD",
        vol_target=0.12,
        dd_rules=((-0.15, 0.25), (-0.10, 0.50), (-0.05, 0.75)),
    ),
    OverlaySpec(
        "VOL10_DD_HARD",
        vol_target=0.10,
        dd_rules=((-0.15, 0.25), (-0.10, 0.50), (-0.05, 0.75)),
    ),
    OverlaySpec(
        "VOL14_TSMOM60_HALF",
        vol_target=0.14,
        momentum_window="60D",
        momentum_risk_off_scale=0.50,
    ),
    OverlaySpec(
        "VOL12_TSMOM60_HALF",
        vol_target=0.12,
        momentum_window="60D",
        momentum_risk_off_scale=0.50,
    ),
    OverlaySpec(
        "VOL12_TSMOM60_DD_SOFT",
        vol_target=0.12,
        momentum_window="60D",
        momentum_risk_off_scale=0.50,
        dd_rules=((-0.15, 0.50), (-0.08, 0.75)),
    ),
    OverlaySpec(
        "VOL12_TSMOM60_DD_HARD",
        vol_target=0.12,
        momentum_window="60D",
        momentum_risk_off_scale=0.50,
        dd_rules=((-0.15, 0.25), (-0.10, 0.50), (-0.05, 0.75)),
    ),
    OverlaySpec("VOL_TARGET_12_DAILY", vol_target=0.12, rebalance="1D"),
    OverlaySpec(
        "VOL14_TSMOM60_HALF_DAILY",
        vol_target=0.14,
        momentum_window="60D",
        momentum_risk_off_scale=0.50,
        rebalance="1D",
    ),
    OverlaySpec(
        "VOL12_TSMOM60_DD_SOFT_DAILY",
        vol_target=0.12,
        momentum_window="60D",
        momentum_risk_off_scale=0.50,
        dd_rules=((-0.15, 0.50), (-0.08, 0.75)),
        rebalance="1D",
    ),
    OverlaySpec(
        "VOL12_TSMOM60_DD_HARD_DAILY",
        vol_target=0.12,
        momentum_window="60D",
        momentum_risk_off_scale=0.50,
        dd_rules=((-0.15, 0.25), (-0.10, 0.50), (-0.05, 0.75)),
        rebalance="1D",
    ),
    OverlaySpec(
        "VOL10_DD_HARD_DAILY",
        vol_target=0.10,
        dd_rules=((-0.15, 0.25), (-0.10, 0.50), (-0.05, 0.75)),
        rebalance="1D",
    ),
]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    root_returns, base_weights = load_inputs(args.start, args.end)
    obs_per_year = infer_obs_per_year(root_returns.index)
    base_gross_returns = compute_gross_returns(root_returns, base_weights)
    base_signal_frame = build_base_signal_frame(base_gross_returns, obs_per_year)

    portfolio_parts = []
    scale_parts = []
    for spec in OVERLAY_SPECS:
        result = run_overlay(spec, root_returns, base_weights, base_signal_frame)
        portfolio_parts.append(result.portfolio)
        scale_parts.append(result.scales)

    portfolio = pd.concat(portfolio_parts, axis=1)
    scales = pd.concat(scale_parts, axis=1)
    metrics = build_metrics(portfolio, scales)
    split_metrics = build_split_metrics(portfolio, scales)
    scale_summary = build_scale_summary(scales)
    turnover_summary = build_turnover_summary(portfolio)
    daily = daily_sample(portfolio, scales)

    portfolio.to_parquet(output_dir / "core_metals_5m_drawdown_control.parquet")
    daily.to_csv(output_dir / "drawdown_control_daily.csv", index=False)
    metrics.to_csv(output_dir / "drawdown_control_metrics.csv", index=False)
    split_metrics.to_csv(output_dir / "drawdown_control_split_metrics.csv", index=False)
    scale_summary.to_csv(output_dir / "drawdown_control_scale_summary.csv", index=False)
    turnover_summary.to_csv(output_dir / "drawdown_control_turnover_summary.csv", index=False)

    best_calmar = select_best_calmar(metrics)
    best_drawdown = select_best_drawdown(metrics)
    plot_cumulative_returns(daily, metrics, output_dir)
    plot_drawdowns(portfolio, metrics, output_dir)
    plot_metric_bars(metrics, output_dir)
    plot_scales(daily, best_calmar, best_drawdown, output_dir)
    write_report(
        metrics,
        split_metrics,
        scale_summary,
        turnover_summary,
        best_calmar,
        best_drawdown,
        portfolio,
        output_dir,
    )
    write_results_json(
        metrics,
        split_metrics,
        scale_summary,
        best_calmar,
        best_drawdown,
        portfolio,
        output_dir,
    )

    print(metrics.sort_values(["calmar", "sharpe_0rf"], ascending=False).to_string(index=False))
    print(f"Best Calmar variant: {best_calmar}")
    print(f"Best drawdown variant: {best_drawdown}")
    print(f"Wrote {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test drawdown-control overlays for the core metals min-var portfolio.",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Inclusive UTC sample start timestamp, for example 2021-01-01.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Inclusive UTC sample end timestamp, for example 2024-12-31 23:59:59.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for output artifacts.",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class OverlayResult:
    portfolio: pd.DataFrame
    scales: pd.DataFrame


def load_inputs(start: str | None, end: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)
    frame = pd.read_parquet(INPUT_PATH)
    frame = filter_frame_by_time(frame, start, end)
    root_return_cols = [f"{root}_log_return_5m" for root in ROOTS]
    weight_cols = [f"{BASE_VARIANT}_{root}_weight" for root in ROOTS]
    missing = [col for col in root_return_cols + weight_cols if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    root_returns = frame[root_return_cols].copy()
    root_returns.columns = ROOTS
    base_weights = frame[weight_cols].copy()
    base_weights.columns = ROOTS
    return root_returns.astype(float), base_weights.astype(float)


def filter_frame_by_time(frame: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start is not None:
        start_ts = pd.Timestamp(start)
        start_ts = (
            start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts.tz_convert("UTC")
        )
        frame = frame.loc[frame.index >= start_ts]
    if end is not None:
        end_ts = pd.Timestamp(end)
        end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
        frame = frame.loc[frame.index <= end_ts]
    if frame.empty:
        raise ValueError("No rows remain after applying start/end filters.")
    return frame


def infer_obs_per_year(index: pd.DatetimeIndex) -> float:
    years = years_between(index[0], index[-1])
    return len(index) / years


def compute_gross_returns(root_returns: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    simple_returns = np.expm1(root_returns[ROOTS])
    portfolio_simple = (weights[ROOTS] * simple_returns).sum(axis=1)
    return np.log1p(portfolio_simple)


def build_base_signal_frame(base_gross_returns: pd.Series, obs_per_year: float) -> pd.DataFrame:
    signals = pd.DataFrame(index=base_gross_returns.index)
    rolling_vol = (
        base_gross_returns.rolling(VOL_LOOKBACK, min_periods=MIN_VOL_OBS).std()
        * math.sqrt(obs_per_year)
    )
    signals["ann_vol_30d"] = rolling_vol.shift(1)
    for window in sorted({spec.momentum_window for spec in OVERLAY_SPECS if spec.momentum_window}):
        signals[f"momentum_{window}"] = base_gross_returns.rolling(window).sum().shift(1)
    return signals


def run_overlay(
    spec: OverlaySpec,
    root_returns: pd.DataFrame,
    base_weights: pd.DataFrame,
    signals: pd.DataFrame,
) -> OverlayResult:
    signal_scale = build_signal_scale(spec, signals)
    root_simple_returns = np.expm1(root_returns[ROOTS]).to_numpy(dtype=float)
    base_weight_values = base_weights[ROOTS].to_numpy(dtype=float)
    cost_per_unit = COST_BPS.reindex(ROOTS).to_numpy(dtype=float) / 10_000.0
    signal_scale_values = signal_scale.to_numpy(dtype=float)
    rebalance_flags = rebalance_flags_for_index(root_returns.index, spec.rebalance)

    gross_returns = np.zeros(len(root_returns), dtype=float)
    cost_returns = np.zeros(len(root_returns), dtype=float)
    net_returns = np.zeros(len(root_returns), dtype=float)
    scales = np.zeros(len(root_returns), dtype=float)
    turnovers = np.zeros(len(root_returns), dtype=float)
    gross_exposure = np.zeros(len(root_returns), dtype=float)
    dd_before = np.zeros(len(root_returns), dtype=float)

    previous_weights: np.ndarray | None = None
    held_scale = 1.0
    wealth = 1.0
    peak = 1.0
    for idx in range(len(root_returns)):
        previous_drawdown = wealth / peak - 1.0
        dd_before[idx] = previous_drawdown
        if rebalance_flags[idx]:
            dd_scale = drawdown_scale(spec, previous_drawdown)
            held_scale = min(signal_scale_values[idx], dd_scale)
        scale = held_scale
        weights = base_weight_values[idx] * scale
        if previous_weights is None:
            root_turnover = np.zeros(len(ROOTS), dtype=float)
        else:
            root_turnover = np.abs(weights - previous_weights)
        portfolio_simple = float(np.dot(weights, root_simple_returns[idx]))
        gross_return = math.log1p(portfolio_simple)
        cost_return = float(np.dot(root_turnover, cost_per_unit))
        net_return = gross_return - cost_return
        wealth *= math.exp(net_return)
        peak = max(peak, wealth)

        gross_returns[idx] = gross_return
        cost_returns[idx] = cost_return
        net_returns[idx] = net_return
        scales[idx] = scale
        turnovers[idx] = float(root_turnover.sum())
        gross_exposure[idx] = float(np.abs(weights).sum())
        previous_weights = weights

    index = root_returns.index
    portfolio = pd.DataFrame(
        {
            f"{spec.name}_gross_log_return_5m": gross_returns,
            f"{spec.name}_cost_log_return_5m": cost_returns,
            f"{spec.name}_net_log_return_5m": net_returns,
            f"{spec.name}_cum_log_return": np.cumsum(net_returns),
            f"{spec.name}_gross_cum_log_return": np.cumsum(gross_returns),
            f"{spec.name}_cost_cum_log_return": np.cumsum(cost_returns),
            f"{spec.name}_turnover": turnovers,
            f"{spec.name}_gross_exposure": gross_exposure,
            f"{spec.name}_drawdown_before_scale": dd_before,
        },
        index=index,
    )
    scale_frame = pd.DataFrame({spec.name: scales}, index=index)
    return OverlayResult(portfolio=portfolio, scales=scale_frame)


def build_signal_scale(spec: OverlaySpec, signals: pd.DataFrame) -> pd.Series:
    scale = pd.Series(1.0, index=signals.index)
    if spec.vol_target is not None:
        vol_scale = spec.vol_target / signals["ann_vol_30d"]
        vol_scale = vol_scale.clip(lower=0.0, upper=1.0).replace([np.inf, -np.inf], np.nan)
        scale = np.minimum(scale, vol_scale.fillna(1.0))
    if spec.momentum_window is not None:
        momentum = signals[f"momentum_{spec.momentum_window}"].fillna(0.0)
        momentum_scale = pd.Series(1.0, index=signals.index)
        momentum_scale.loc[momentum.lt(0.0)] = spec.momentum_risk_off_scale
        scale = np.minimum(scale, momentum_scale)
    return pd.Series(scale, index=signals.index).clip(lower=0.0, upper=1.0).fillna(1.0)


def drawdown_scale(spec: OverlaySpec, drawdown: float) -> float:
    if not spec.dd_rules:
        return 1.0
    for threshold, scale in sorted(spec.dd_rules, key=lambda item: item[0]):
        if drawdown <= threshold:
            return scale
    return 1.0


def rebalance_flags_for_index(index: pd.DatetimeIndex, rebalance: str) -> np.ndarray:
    if rebalance == "bar":
        return np.ones(len(index), dtype=bool)
    buckets = index.floor(rebalance)
    flags = np.ones(len(index), dtype=bool)
    if len(index) > 1:
        flags[1:] = buckets[1:] != buckets[:-1]
    return flags


def build_metrics(portfolio: pd.DataFrame, scales: pd.DataFrame) -> pd.DataFrame:
    rows = [
        metrics_for_variant(portfolio, scales, variant, "full")
        for variant in variants(portfolio)
    ]
    return pd.DataFrame(rows)


def build_split_metrics(portfolio: pd.DataFrame, scales: pd.DataFrame) -> pd.DataFrame:
    splits = {
        "2021_2022": (
            pd.Timestamp("2021-01-01", tz="UTC"),
            pd.Timestamp("2022-12-31 23:59:59", tz="UTC"),
        ),
        "2023_2024": (
            pd.Timestamp("2023-01-01", tz="UTC"),
            pd.Timestamp("2024-12-31 23:59:59", tz="UTC"),
        ),
        "2025_2026": (pd.Timestamp("2025-01-01", tz="UTC"), portfolio.index.max()),
    }
    rows = []
    for split, (start, end) in splits.items():
        part = portfolio[(portfolio.index >= start) & (portfolio.index <= end)]
        scale_part = scales[(scales.index >= start) & (scales.index <= end)]
        if part.empty:
            continue
        rows.extend(
            metrics_for_variant(part, scale_part, variant, split) for variant in variants(part)
        )
    return pd.DataFrame(rows)


def metrics_for_variant(
    portfolio: pd.DataFrame,
    scales: pd.DataFrame,
    variant: str,
    split: str,
) -> dict[str, Any]:
    returns = portfolio[f"{variant}_net_log_return_5m"].astype(float)
    gross_returns = portfolio[f"{variant}_gross_log_return_5m"].astype(float)
    cost_returns = portfolio[f"{variant}_cost_log_return_5m"].astype(float)
    years = years_between(portfolio.index[0], portfolio.index[-1])
    obs_per_year = len(returns) / years
    cumulative = returns.cumsum()
    max_dd = max_drawdown_from_cum_log(cumulative)
    annual_log_return = returns.sum() / years
    annual_vol = returns.std(ddof=1) * math.sqrt(obs_per_year)
    cagr = math.expm1(annual_log_return)
    variant_scale = scales[variant].reindex(portfolio.index).astype(float)
    return {
        "variant": variant,
        "split": split,
        "start_ts": portfolio.index[0],
        "end_ts": portfolio.index[-1],
        "nobs_5m": len(portfolio),
        "years": years,
        "cum_log_return": returns.sum(),
        "gross_cum_log_return": gross_returns.sum(),
        "cost_cum_log_return": cost_returns.sum(),
        "total_return_pct": math.expm1(returns.sum()) * 100.0,
        "cagr": cagr,
        "annual_log_return": annual_log_return,
        "annual_vol": annual_vol,
        "sharpe_0rf": annual_log_return / annual_vol if annual_vol > 0 else np.nan,
        "max_drawdown": max_dd,
        "calmar": cagr / abs(max_dd) if max_dd < 0 else np.nan,
        "avg_scale": variant_scale.mean(),
        "p10_scale": variant_scale.quantile(0.10),
        "p50_scale": variant_scale.quantile(0.50),
        "p90_scale": variant_scale.quantile(0.90),
        "risk_off_fraction": variant_scale.lt(0.999).mean(),
        "annual_turnover": portfolio[f"{variant}_turnover"].sum() / years,
    }


def build_scale_summary(scales: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in scales.columns:
        scale = scales[variant].astype(float)
        rows.append(
            {
                "variant": variant,
                "avg_scale": scale.mean(),
                "min_scale": scale.min(),
                "p10_scale": scale.quantile(0.10),
                "p50_scale": scale.quantile(0.50),
                "p90_scale": scale.quantile(0.90),
                "max_scale": scale.max(),
                "risk_off_fraction": scale.lt(0.999).mean(),
            }
        )
    return pd.DataFrame(rows)


def build_turnover_summary(portfolio: pd.DataFrame) -> pd.DataFrame:
    years = years_between(portfolio.index[0], portfolio.index[-1])
    rows = []
    for variant in variants(portfolio):
        turnover = portfolio[f"{variant}_turnover"].astype(float)
        rows.append(
            {
                "variant": variant,
                "cum_turnover": turnover.sum(),
                "annual_turnover": turnover.sum() / years,
                "mean_5m_turnover": turnover.mean(),
                "p95_5m_turnover": turnover.quantile(0.95),
                "max_5m_turnover": turnover.max(),
            }
        )
    return pd.DataFrame(rows)


def variants(portfolio: pd.DataFrame) -> list[str]:
    suffix = "_net_log_return_5m"
    return [col[: -len(suffix)] for col in portfolio.columns if col.endswith(suffix)]


def daily_sample(portfolio: pd.DataFrame, scales: pd.DataFrame) -> pd.DataFrame:
    cumulative_cols = [col for col in portfolio.columns if col.endswith("_cum_log_return")]
    exposure_cols = [col for col in portfolio.columns if col.endswith("_gross_exposure")]
    sampled = (
        pd.concat(
            [
                portfolio[cumulative_cols + exposure_cols],
                scales.add_suffix("_scale"),
            ],
            axis=1,
        )
        .resample("1D")
        .last()
        .dropna(how="all")
        .reset_index()
    )
    sampled["ts"] = sampled["ts"].dt.tz_convert(None)
    return sampled


def select_best_calmar(metrics: pd.DataFrame) -> str:
    candidates = metrics[metrics["cagr"].gt(0.0)].copy()
    ranked = candidates.sort_values(["calmar", "sharpe_0rf"], ascending=False)
    return str(ranked.iloc[0]["variant"])


def select_best_drawdown(metrics: pd.DataFrame) -> str:
    candidates = metrics[metrics["cagr"].gt(0.0)].copy()
    ranked = candidates.sort_values(["max_drawdown", "calmar"], ascending=False)
    return str(ranked.iloc[0]["variant"])


def years_between(start: pd.Timestamp, end: pd.Timestamp) -> float:
    seconds = (end - start).total_seconds()
    return seconds / (365.25 * 24 * 60 * 60)


def max_drawdown_from_cum_log(cumulative: pd.Series) -> float:
    wealth = np.exp(cumulative)
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())


def display_name(variant: str) -> str:
    return variant.replace("_", " ").replace("VOL", "Vol").replace("TSMOM", "Trend")


def variants_for_plots(metrics: pd.DataFrame) -> list[str]:
    priority = [
        "BASE_MINVAR",
        "VOL_TARGET_14",
        "VOL_TARGET_12",
        "VOL_TARGET_10",
        "DD_HARD",
        "VOL12_DD_HARD",
        "VOL10_DD_HARD",
        "VOL12_TSMOM60_DD_HARD",
    ]
    ranked = metrics.sort_values(["calmar", "sharpe_0rf"], ascending=False)["variant"].tolist()
    selected = []
    for variant in priority + ranked:
        if variant not in selected:
            selected.append(variant)
        if len(selected) >= PLOT_VARIANT_COUNT:
            break
    return selected


def plot_cumulative_returns(
    daily: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
) -> None:
    selected = variants_for_plots(metrics)
    fig, ax = plt.subplots(figsize=(15, 7))
    for variant in selected:
        col = f"{variant}_cum_log_return"
        if col in daily:
            ax.plot(daily["ts"], daily[col], label=display_name(variant), linewidth=1.5)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Core metals min-var drawdown-control overlays")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net cumulative log return")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "drawdown_control_cum_log_returns_2021.png", dpi=170)
    plt.close(fig)


def plot_drawdowns(portfolio: pd.DataFrame, metrics: pd.DataFrame, output_dir: Path) -> None:
    selected = variants_for_plots(metrics)
    fig, ax = plt.subplots(figsize=(15, 7))
    for variant in selected:
        returns = portfolio[f"{variant}_net_log_return_5m"].astype(float)
        wealth = np.exp(returns.cumsum())
        drawdown = wealth / wealth.cummax() - 1.0
        sampled = drawdown.resample("1D").last().dropna()
        ax.plot(sampled.index.tz_convert(None), sampled, label=display_name(variant), linewidth=1.4)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Core metals min-var drawdown-control drawdowns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "drawdown_control_drawdowns_2021.png", dpi=170)
    plt.close(fig)


def plot_metric_bars(metrics: pd.DataFrame, output_dir: Path) -> None:
    selected = metrics.sort_values(["calmar", "sharpe_0rf"], ascending=False).head(12).copy()
    labels = [display_name(variant) for variant in selected["variant"]]
    fig, axes = plt.subplots(1, 4, figsize=(17, 7), sharey=True)
    plot_specs = [
        ("calmar", "Calmar", 1.0),
        ("sharpe_0rf", "Sharpe", 1.0),
        ("cagr", "CAGR, %", 100.0),
        ("max_drawdown", "Max DD, %", 100.0),
    ]
    for ax, (col, title, scale) in zip(axes, plot_specs, strict=True):
        ax.barh(labels, selected[col] * scale)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        ax.invert_yaxis()
    fig.suptitle("Drawdown-control overlay metrics")
    fig.tight_layout()
    fig.savefig(output_dir / "drawdown_control_metric_bars.png", dpi=170)
    plt.close(fig)


def plot_scales(
    daily: pd.DataFrame,
    best_calmar: str,
    best_drawdown: str,
    output_dir: Path,
) -> None:
    selected = []
    for variant in ["BASE_MINVAR", best_calmar, best_drawdown, "VOL_TARGET_12", "VOL12_DD_HARD"]:
        if variant not in selected:
            selected.append(variant)
    fig, ax = plt.subplots(figsize=(15, 5.5))
    for variant in selected:
        col = f"{variant}_scale"
        if col in daily:
            ax.plot(daily["ts"], daily[col], label=display_name(variant), linewidth=1.3)
    ax.set_title("Exposure scales for selected drawdown controls")
    ax.set_xlabel("Date")
    ax.set_ylabel("Exposure scale")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "drawdown_control_scales_2021.png", dpi=170)
    plt.close(fig)


def write_report(
    metrics: pd.DataFrame,
    split_metrics: pd.DataFrame,
    scale_summary: pd.DataFrame,
    turnover_summary: pd.DataFrame,
    best_calmar: str,
    best_drawdown: str,
    portfolio: pd.DataFrame,
    output_dir: Path,
) -> None:
    sorted_metrics = metrics.sort_values(["calmar", "sharpe_0rf"], ascending=False)
    dd_sorted = metrics[metrics["cagr"].gt(0.0)].sort_values(
        ["max_drawdown", "calmar"],
        ascending=False,
    )
    top_variants = sorted(
        set(sorted_metrics.head(8)["variant"]).union(dd_sorted.head(5)["variant"]).union({"BASE_MINVAR"})
    )
    top_splits = split_metrics[split_metrics["variant"].isin(top_variants)].sort_values(
        ["split", "calmar"],
        ascending=[True, False],
    )
    lines = [
        "# Core Metals Drawdown-Control Overlays",
        "",
        "Objective: reduce drawdowns of the best core metals portfolio from HYP-0051,",
        "`MINVAR_30D_SHRINK25`, without introducing a new return-forecasting model.",
        "",
        "Construction:",
        "",
        "- Source: HYP-0051 root returns and daily min-var weights.",
        "- Base strategy: `MINVAR_30D_SHRINK25`.",
        "- Overlay families: no-leverage volatility targets, trailing trend risk-off,",
        "  realized strategy drawdown throttles, and combined controls.",
        "- Variants suffixed `DAILY` only change the overlay exposure once per UTC day.",
        "- Volatility target uses lagged 30-calendar-day realized portfolio volatility.",
        "- Trend filters use lagged trailing portfolio log return.",
        "- Drawdown throttle uses only previous strategy wealth and previous peak.",
        "- Metrics are net of root-specific turnover costs from HYP-0046 MBP1 estimates.",
        "",
        f"Best Calmar variant: `{best_calmar}`.",
        f"Lowest positive-CAGR drawdown variant: `{best_drawdown}`.",
        "",
        "## Metrics Sorted By Calmar",
        "",
        sorted_metrics.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Metrics Sorted By Lowest Drawdown",
        "",
        dd_sorted.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Split Metrics For Top Variants",
        "",
        top_splits.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Scale Summary",
        "",
        scale_summary.sort_values("variant").to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Turnover Summary",
        "",
        turnover_summary.sort_values("variant").to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Input Span",
        "",
        f"- start: `{portfolio.index.min()}`",
        f"- end: `{portfolio.index.max()}`",
        f"- rows: `{len(portfolio)}`",
        f"- base variant: `{BASE_VARIANT}`",
        "",
        "## Files",
        "",
        "- `drawdown_control_cum_log_returns_2021.png`",
        "- `drawdown_control_drawdowns_2021.png`",
        "- `drawdown_control_metric_bars.png`",
        "- `drawdown_control_scales_2021.png`",
        "- `core_metals_5m_drawdown_control.parquet`",
        "- `drawdown_control_daily.csv`",
        "- `drawdown_control_metrics.csv`",
        "- `drawdown_control_split_metrics.csv`",
        "- `drawdown_control_scale_summary.csv`",
        "- `drawdown_control_turnover_summary.csv`",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_results_json(
    metrics: pd.DataFrame,
    split_metrics: pd.DataFrame,
    scale_summary: pd.DataFrame,
    best_calmar: str,
    best_drawdown: str,
    portfolio: pd.DataFrame,
    output_dir: Path,
) -> None:
    payload = {
        "best_calmar_variant": best_calmar,
        "best_drawdown_variant": best_drawdown,
        "start_ts": str(portfolio.index.min()),
        "end_ts": str(portfolio.index.max()),
        "rows": len(portfolio),
        "metrics": json.loads(metrics.to_json(orient="records", date_format="iso")),
        "split_metrics": json.loads(split_metrics.to_json(orient="records", date_format="iso")),
        "scale_summary": json.loads(scale_summary.to_json(orient="records")),
        "cost_bps": COST_BPS.to_dict(),
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
