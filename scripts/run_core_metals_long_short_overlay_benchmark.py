"""Test constrained long/short overlays for the core metals portfolio."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
RETURNS_PATH = (
    REPO_ROOT
    / "experiments"
    / "HYP-0041-core-metals-5m-log-returns"
    / "core_metals_5m_log_returns_wide.parquet"
)
FAIR_VALUE_PATH = (
    REPO_ROOT
    / "experiments"
    / "HYP-0043-core-metals-carry-conditioned-fair-value"
    / "fair_value_panel.parquet"
)
OUTPUT_DIR = REPO_ROOT / "experiments" / "HYP-0051-core-metals-long-short-overlays"

ROOTS = ["GC", "SI", "HG", "PL", "PA"]
START_TS = pd.Timestamp("2021-01-01", tz="UTC")
LOOKBACK = pd.Timedelta(days=30)
MIN_OBS = 1_000
RIDGE = 1e-12
EPSILON = 1e-12
FEASIBILITY_TOLERANCE = 1e-8
PLOT_VARIANT_COUNT = 8

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
class CovarianceVariant:
    name: str
    method: str
    shrinkage: float = 0.25
    lower: float = 0.0
    upper: float = 1.0
    gross_cap: float | None = None


@dataclass(frozen=True)
class AlphaSpec:
    name: str
    window: str
    threshold: float | None
    mode: str


COVARIANCE_VARIANTS = [
    CovarianceVariant("EW_1N", "equal", 0.25),
    CovarianceVariant("INV_VOL_30D_DAILY", "inverse_vol", 0.25),
    CovarianceVariant("MINVAR_30D_SHRINK25", "minvar", 0.25, 0.0, 1.0, None),
    CovarianceVariant("MINVAR_CAP40_SHRINK25", "minvar", 0.25, 0.0, 0.40, None),
    CovarianceVariant("MINVAR_CAP50_SHRINK25", "minvar", 0.25, 0.0, 0.50, None),
    CovarianceVariant("MINVAR_CAP60_SHRINK25", "minvar", 0.25, 0.0, 0.60, None),
    CovarianceVariant("LS_MINVAR_G125_B25", "minvar", 0.25, -0.25, 0.75, 1.25),
    CovarianceVariant("LS_MINVAR_G150_B25", "minvar", 0.25, -0.25, 0.75, 1.50),
    CovarianceVariant("LS_MINVAR_G200_B25", "minvar", 0.25, -0.25, 0.75, 2.00),
]

ALPHA_SPECS = [
    AlphaSpec("FVD_20D_T1P5_DN", "20D", 1.5, "threshold"),
    AlphaSpec("FVD_20D_T2P5_DN", "20D", 2.5, "threshold"),
    AlphaSpec("FVD_60D_T2P5_DN", "60D", 2.5, "threshold"),
    AlphaSpec("FVD_120D_T2P5_DN", "120D", 2.5, "threshold"),
    AlphaSpec("FVD_20D_RANK_DN", "20D", None, "rank"),
    AlphaSpec("FVD_60D_RANK_DN", "60D", None, "rank"),
]

BLEND_SPECS = [
    ("BLEND_MINVAR_FVD20_T2P5_10", "MINVAR_30D_SHRINK25", "FVD_20D_T2P5_DN", 0.10),
    ("BLEND_MINVAR_FVD20_T2P5_20", "MINVAR_30D_SHRINK25", "FVD_20D_T2P5_DN", 0.20),
    ("BLEND_MINVAR_FVD60_T2P5_10", "MINVAR_30D_SHRINK25", "FVD_60D_T2P5_DN", 0.10),
    ("BLEND_MINVAR_FVD60_T2P5_20", "MINVAR_30D_SHRINK25", "FVD_60D_T2P5_DN", 0.20),
    ("BLEND_MINVAR_FVD20_RANK_10", "MINVAR_30D_SHRINK25", "FVD_20D_RANK_DN", 0.10),
    ("BLEND_MINVAR_FVD20_RANK_20", "MINVAR_30D_SHRINK25", "FVD_20D_RANK_DN", 0.20),
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    returns_full = load_returns()
    eval_returns = prepare_eval_returns(returns_full)
    rebalance_dates = build_rebalance_dates(eval_returns.index)

    daily_weight_frames, diagnostics = build_covariance_weights(returns_full, rebalance_dates)
    daily_weights = pd.concat(daily_weight_frames, names=["variant", "ts"])
    diagnostics_frame = pd.DataFrame(diagnostics)

    covariance_weights = align_daily_weights(daily_weights, eval_returns.index)
    alpha_weights = build_alpha_weights(eval_returns.index)
    all_weights = build_all_weights(covariance_weights, alpha_weights)

    portfolio = build_portfolio(eval_returns, all_weights)
    metrics = build_metrics(portfolio)
    split_metrics = build_split_metrics(portfolio)
    weight_summary = build_weight_summary(all_weights)
    turnover_summary = build_turnover_summary(portfolio)
    daily = daily_sample(portfolio, all_weights)

    daily_weights.reset_index().to_csv(OUTPUT_DIR / "daily_covariance_weights.csv", index=False)
    diagnostics_frame.to_csv(OUTPUT_DIR / "rebalance_diagnostics.csv", index=False)
    portfolio.to_parquet(OUTPUT_DIR / "core_metals_5m_long_short_overlay_benchmark.parquet")
    daily.to_csv(OUTPUT_DIR / "long_short_overlay_daily.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "benchmark_metrics.csv", index=False)
    split_metrics.to_csv(OUTPUT_DIR / "split_metrics.csv", index=False)
    weight_summary.to_csv(OUTPUT_DIR / "weights_summary.csv", index=False)
    turnover_summary.to_csv(OUTPUT_DIR / "turnover_summary.csv", index=False)

    best_variant = select_best_variant(metrics)
    plot_cumulative_returns(daily, metrics)
    plot_drawdowns(portfolio, metrics)
    plot_metric_bars(metrics)
    plot_best_weights(daily, best_variant)
    write_report(
        metrics,
        split_metrics,
        weight_summary,
        turnover_summary,
        diagnostics_frame,
        best_variant,
        portfolio,
    )
    write_results_json(metrics, split_metrics, turnover_summary, best_variant, portfolio)

    print(metrics.sort_values(["sharpe_0rf", "cagr"], ascending=False).to_string(index=False))
    print(f"Best variant: {best_variant}")
    print(f"Wrote {OUTPUT_DIR}", flush=True)


def load_returns() -> pd.DataFrame:
    if not RETURNS_PATH.exists():
        raise FileNotFoundError(RETURNS_PATH)
    frame = pd.read_parquet(RETURNS_PATH)
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    returns = frame.sort_values("ts").set_index("ts")[ROOTS].astype("float64")
    return returns.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)


def prepare_eval_returns(returns_full: pd.DataFrame) -> pd.DataFrame:
    returns = returns_full[returns_full.index >= START_TS].copy()
    if returns.empty:
        raise ValueError(f"No rows at or after {START_TS}")
    returns.iloc[0] = 0.0
    return returns


def build_rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    first = index.min().floor("1D")
    last = index.max().floor("1D")
    return pd.date_range(first, last, freq="1D", tz="UTC")


def build_covariance_weights(
    returns_full: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    previous = {
        variant.name: initial_weights_for_variant(variant) for variant in COVARIANCE_VARIANTS
    }
    frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []

    for rebalance_ts in rebalance_dates:
        window = returns_full[
            (returns_full.index < rebalance_ts) & (returns_full.index >= rebalance_ts - LOOKBACK)
        ]
        covariance, diag, cov_diag = estimate_covariance(window)

        rows = []
        for variant in COVARIANCE_VARIANTS:
            shrunk = shrink_covariance(covariance, cov_diag, variant.shrinkage)
            weights, success, objective = weights_for_covariance_variant(
                variant,
                shrunk,
                diag,
                previous[variant.name],
            )
            previous[variant.name] = weights
            rows.append(pd.Series(weights, index=ROOTS, name=variant.name))
            diagnostics.append(
                {
                    "rebalance_ts": rebalance_ts,
                    "variant": variant.name,
                    "method": variant.method,
                    "shrinkage": variant.shrinkage,
                    "lower": variant.lower,
                    "upper": variant.upper,
                    "gross_cap": variant.gross_cap,
                    "success": success,
                    "objective": objective,
                    "window_obs": len(window),
                    "avg_pairwise_corr": avg_pairwise_corr(shrunk),
                    "condition_number": np.linalg.cond(shrunk),
                }
            )

        frame = pd.DataFrame(rows)
        frame.index.name = "variant"
        frame["ts"] = rebalance_ts
        frame = frame.set_index("ts", append=True)
        frames.append(frame.reorder_levels(["variant", "ts"]).sort_index())

    return frames, diagnostics


def initial_weights_for_variant(variant: CovarianceVariant) -> np.ndarray:
    if variant.method == "equal":
        return np.full(len(ROOTS), 1.0 / len(ROOTS), dtype=float)
    if variant.method == "inverse_vol":
        return np.full(len(ROOTS), 1.0 / len(ROOTS), dtype=float)
    if variant.lower < 0:
        return np.full(len(ROOTS), 1.0 / len(ROOTS), dtype=float)
    return np.full(len(ROOTS), min(1.0 / len(ROOTS), variant.upper), dtype=float)


def estimate_covariance(window: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(window) < MIN_OBS:
        covariance = np.eye(len(ROOTS), dtype=float)
    else:
        covariance = window[ROOTS].cov().to_numpy(dtype=float)
    covariance = regularize_covariance(covariance)
    diag = np.clip(np.diag(covariance), RIDGE, None)
    cov_diag = np.diag(diag)
    return covariance, diag, cov_diag


def regularize_covariance(covariance: np.ndarray) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    covariance = np.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
    covariance = 0.5 * (covariance + covariance.T)
    covariance += np.eye(covariance.shape[0]) * RIDGE
    return covariance


def shrink_covariance(covariance: np.ndarray, cov_diag: np.ndarray, shrinkage: float) -> np.ndarray:
    shrunk = (1.0 - shrinkage) * covariance + shrinkage * cov_diag
    return regularize_covariance(shrunk)


def scale_covariance(covariance: np.ndarray) -> np.ndarray:
    covariance = regularize_covariance(covariance)
    scale = float(np.mean(np.diag(covariance)))
    if scale > EPSILON:
        return covariance / scale
    return covariance


def weights_for_covariance_variant(
    variant: CovarianceVariant,
    covariance: np.ndarray,
    diag: np.ndarray,
    previous: np.ndarray,
) -> tuple[np.ndarray, bool, float]:
    if variant.method == "equal":
        weights = np.full(len(ROOTS), 1.0 / len(ROOTS), dtype=float)
        return weights, True, 0.0
    if variant.method == "inverse_vol":
        weights = inverse_vol_weights_from_diag(diag)
        return weights, True, 0.0
    if variant.method == "minvar":
        return optimize_minvar(covariance, previous, variant)
    raise ValueError(f"Unknown method: {variant.method}")


def inverse_vol_weights_from_diag(diag: np.ndarray) -> np.ndarray:
    vol = np.sqrt(np.clip(diag, RIDGE, None))
    inv = 1.0 / vol
    return inv / inv.sum()


def optimize_minvar(
    covariance: np.ndarray,
    previous: np.ndarray,
    variant: CovarianceVariant,
) -> tuple[np.ndarray, bool, float]:
    covariance = scale_covariance(covariance)

    def objective(weights: np.ndarray) -> float:
        return float(weights @ covariance @ weights)

    result = minimize_constrained(objective, previous, variant)
    weights = normalize_net_weights(result.x if result.success else previous)
    return weights, bool(result.success), objective(weights)


def minimize_constrained(objective: Any, start: np.ndarray, variant: CovarianceVariant) -> Any:
    constraints: list[dict[str, Any]] = [
        {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
    ]
    if variant.gross_cap is not None:
        constraints.append(
            {"type": "ineq", "fun": lambda weights: variant.gross_cap - np.sum(np.abs(weights))}
        )
    bounds = [(variant.lower, variant.upper)] * len(start)
    feasible_start = project_start(start, variant)
    return minimize(
        objective,
        feasible_start,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-12, "disp": False},
    )


def project_start(weights: np.ndarray, variant: CovarianceVariant) -> np.ndarray:
    clean = np.clip(np.asarray(weights, dtype=float), variant.lower, variant.upper)
    clean += (1.0 - clean.sum()) / len(clean)
    clean = np.clip(clean, variant.lower, variant.upper)
    if abs(clean.sum() - 1.0) > FEASIBILITY_TOLERANCE:
        clean = np.full(len(clean), 1.0 / len(clean), dtype=float)
    if variant.gross_cap is not None and np.sum(np.abs(clean)) > variant.gross_cap:
        clean = np.full(len(clean), 1.0 / len(clean), dtype=float)
    return clean


def normalize_net_weights(weights: np.ndarray) -> np.ndarray:
    clean = np.asarray(weights, dtype=float)
    total = clean.sum()
    if abs(total) <= EPSILON:
        return np.full(len(clean), 1.0 / len(clean), dtype=float)
    return clean / total


def avg_pairwise_corr(covariance: np.ndarray) -> float:
    diag = np.sqrt(np.clip(np.diag(covariance), RIDGE, None))
    corr = covariance / np.outer(diag, diag)
    mask = ~np.eye(corr.shape[0], dtype=bool)
    return float(np.nanmean(corr[mask]))


def align_daily_weights(
    daily_weights: pd.DataFrame,
    index: pd.DatetimeIndex,
) -> dict[str, pd.DataFrame]:
    output = {}
    for variant in daily_weights.index.get_level_values("variant").unique():
        weights = daily_weights.xs(variant, level="variant")[ROOTS].sort_index()
        aligned = weights.reindex(index, method="ffill").fillna(1.0 / len(ROOTS))
        output[str(variant)] = aligned
    return output


def build_alpha_weights(index: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    zscores_by_window = load_fair_value_zscores(index)
    output = {}
    for spec in ALPHA_SPECS:
        zscores = zscores_by_window[spec.window]
        if spec.mode == "threshold":
            weights = threshold_alpha_weights(zscores, spec.threshold or 0.0)
        elif spec.mode == "rank":
            weights = rank_alpha_weights(zscores)
        else:
            raise ValueError(f"Unknown alpha mode: {spec.mode}")
        output[spec.name] = weights.shift(1).fillna(0.0)
    return output


def load_fair_value_zscores(index: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    if not FAIR_VALUE_PATH.exists():
        raise FileNotFoundError(FAIR_VALUE_PATH)
    frame = pd.read_parquet(
        FAIR_VALUE_PATH,
        columns=["ts", "root", "window", "fair_zscore"],
    )
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    frame = frame[frame["root"].isin(ROOTS)]
    output = {}
    for window in sorted({spec.window for spec in ALPHA_SPECS}):
        wide = (
            frame[frame["window"].eq(window)]
            .pivot(index="ts", columns="root", values="fair_zscore")
            .sort_index()
            .reindex(columns=ROOTS)
            .replace([np.inf, -np.inf], np.nan)
        )
        aligned = wide.reindex(index, method="ffill").fillna(0.0)
        output[window] = aligned
    return output


def threshold_alpha_weights(zscores: pd.DataFrame, threshold: float) -> pd.DataFrame:
    raw = -zscores.where(zscores.abs().ge(threshold), 0.0)
    centered = raw.sub(raw.mean(axis=1), axis=0)
    centered = centered.where(raw.abs().sum(axis=1).gt(0.0), 0.0)
    return normalize_gross(centered, gross=1.0)


def rank_alpha_weights(zscores: pd.DataFrame) -> pd.DataFrame:
    raw = -zscores
    centered = raw.sub(raw.mean(axis=1), axis=0)
    return normalize_gross(centered, gross=1.0)


def normalize_gross(weights: pd.DataFrame, gross: float) -> pd.DataFrame:
    denom = weights.abs().sum(axis=1).replace(0.0, np.nan)
    normalized = weights.div(denom, axis=0).fillna(0.0) * gross
    return normalized.reindex(columns=ROOTS).astype(float)


def build_all_weights(
    covariance_weights: dict[str, pd.DataFrame],
    alpha_weights: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    output = dict(covariance_weights)
    output.update(alpha_weights)
    for blend_name, core_name, alpha_name, overlay_scale in BLEND_SPECS:
        output[blend_name] = (
            covariance_weights[core_name] + overlay_scale * alpha_weights[alpha_name]
        )
    return output


def build_portfolio(
    eval_returns: pd.DataFrame,
    all_weights: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    simple_returns = np.expm1(eval_returns[ROOTS])
    output: dict[str, Any] = {}
    cost_per_unit = COST_BPS.reindex(ROOTS).to_numpy(dtype=float) / 10_000.0

    for root in ROOTS:
        root_returns = eval_returns[root].astype(float)
        output[f"{root}_log_return_5m"] = root_returns
        output[f"{root}_cum_log_return"] = root_returns.cumsum()

    for variant, variant_weights in all_weights.items():
        weights = variant_weights.reindex(eval_returns.index).ffill().fillna(0.0)
        portfolio_simple = (weights[ROOTS] * simple_returns).sum(axis=1)
        gross_log_return = np.log1p(portfolio_simple)
        root_turnover = weights[ROOTS].diff().abs().fillna(0.0)
        turnover = root_turnover.sum(axis=1)
        cost_log_return = root_turnover.to_numpy() @ cost_per_unit
        net_log_return = gross_log_return - cost_log_return
        gross_exposure = weights[ROOTS].abs().sum(axis=1)
        short_exposure = weights[ROOTS].clip(upper=0.0).abs().sum(axis=1)

        output[f"{variant}_gross_log_return_5m"] = gross_log_return
        output[f"{variant}_cost_log_return_5m"] = cost_log_return
        output[f"{variant}_net_log_return_5m"] = net_log_return
        output[f"{variant}_cum_log_return"] = net_log_return.cumsum()
        output[f"{variant}_gross_cum_log_return"] = gross_log_return.cumsum()
        output[f"{variant}_cost_cum_log_return"] = cost_log_return.cumsum()
        output[f"{variant}_turnover"] = turnover
        output[f"{variant}_gross_exposure"] = gross_exposure
        output[f"{variant}_short_exposure"] = short_exposure
        for root in ROOTS:
            output[f"{variant}_{root}_weight"] = weights[root]
    return pd.DataFrame(output, index=eval_returns.index)


def build_metrics(portfolio: pd.DataFrame) -> pd.DataFrame:
    rows = [metrics_for_variant(portfolio, variant, "full") for variant in variants(portfolio)]
    return pd.DataFrame(rows)


def build_split_metrics(portfolio: pd.DataFrame) -> pd.DataFrame:
    rows = []
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
    for split, (start, end) in splits.items():
        part = portfolio[(portfolio.index >= start) & (portfolio.index <= end)]
        if part.empty:
            continue
        rows.extend(metrics_for_variant(part, variant, split) for variant in variants(portfolio))
    return pd.DataFrame(rows)


def metrics_for_variant(portfolio: pd.DataFrame, variant: str, split: str) -> dict[str, Any]:
    returns = portfolio[f"{variant}_net_log_return_5m"].astype(float)
    gross_returns = portfolio[f"{variant}_gross_log_return_5m"].astype(float)
    cost_returns = portfolio[f"{variant}_cost_log_return_5m"].astype(float)
    years = years_between(portfolio.index[0], portfolio.index[-1])
    obs_per_year = len(returns) / years if years > 0 else np.nan
    cumulative = returns.cumsum()
    annual_log_return = returns.sum() / years
    annual_vol = returns.std(ddof=1) * math.sqrt(obs_per_year)
    gross_exposure = portfolio[f"{variant}_gross_exposure"].astype(float)
    short_exposure = portfolio[f"{variant}_short_exposure"].astype(float)
    turnover = portfolio[f"{variant}_turnover"].astype(float)
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
        "cagr": math.expm1(annual_log_return),
        "annual_log_return": annual_log_return,
        "annual_vol": annual_vol,
        "sharpe_0rf": annual_log_return / annual_vol if annual_vol > 0 else np.nan,
        "max_drawdown": max_drawdown_from_cum_log(cumulative),
        "avg_gross_exposure": gross_exposure.mean(),
        "p95_gross_exposure": gross_exposure.quantile(0.95),
        "avg_short_exposure": short_exposure.mean(),
        "active_short_fraction": short_exposure.gt(1e-8).mean(),
        "annual_turnover": turnover.sum() / years,
    }


def build_weight_summary(all_weights: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for variant, weights in all_weights.items():
        for root in ROOTS:
            root_weights = weights[root].astype(float)
            rows.append(
                {
                    "variant": variant,
                    "root": root,
                    "mean_weight": root_weights.mean(),
                    "median_weight": root_weights.median(),
                    "min_weight": root_weights.min(),
                    "p10_weight": root_weights.quantile(0.10),
                    "p90_weight": root_weights.quantile(0.90),
                    "max_weight": root_weights.max(),
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


def daily_sample(portfolio: pd.DataFrame, all_weights: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cumulative_cols = [col for col in portfolio.columns if col.endswith("_cum_log_return")]
    exposure_cols = [
        col
        for col in portfolio.columns
        if col.endswith("_gross_exposure") or col.endswith("_short_exposure")
    ]
    weight_cols = []
    for variant in all_weights:
        weight_cols.extend([f"{variant}_{root}_weight" for root in ROOTS])
    sampled = (
        portfolio[cumulative_cols + exposure_cols + weight_cols]
        .resample("1D")
        .last()
        .dropna(how="all")
        .reset_index()
    )
    sampled["ts"] = sampled["ts"].dt.tz_convert(None)
    return sampled


def select_best_variant(metrics: pd.DataFrame) -> str:
    ranked = metrics.sort_values(["sharpe_0rf", "cagr"], ascending=False)
    return str(ranked.iloc[0]["variant"])


def years_between(start: pd.Timestamp, end: pd.Timestamp) -> float:
    seconds = (end - start).total_seconds()
    return seconds / (365.25 * 24 * 60 * 60)


def max_drawdown_from_cum_log(cumulative: pd.Series) -> float:
    wealth = np.exp(cumulative)
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())


def display_name(variant: str) -> str:
    return (
        variant.replace("_30D_SHRINK25", "")
        .replace("_SHRINK25", "")
        .replace("_DN", "")
        .replace("_", " ")
        .replace("MINVAR", "MinVar")
        .replace("INV VOL", "InvVol")
        .replace("FVD", "FVD")
    )


def top_variants_for_plots(metrics: pd.DataFrame) -> list[str]:
    priority = [
        "EW_1N",
        "INV_VOL_30D_DAILY",
        "MINVAR_30D_SHRINK25",
        "MINVAR_CAP50_SHRINK25",
        "LS_MINVAR_G150_B25",
        "FVD_20D_T2P5_DN",
    ]
    ranked = metrics.sort_values(["sharpe_0rf", "cagr"], ascending=False)["variant"].tolist()
    selected = []
    for variant in ranked + priority:
        if variant not in selected:
            selected.append(variant)
        if len(selected) >= PLOT_VARIANT_COUNT:
            break
    return selected


def plot_cumulative_returns(daily: pd.DataFrame, metrics: pd.DataFrame) -> None:
    selected = top_variants_for_plots(metrics)
    fig, ax = plt.subplots(figsize=(15, 7))
    for variant in selected:
        col = f"{variant}_cum_log_return"
        if col in daily:
            ax.plot(daily["ts"], daily[col], label=display_name(variant), linewidth=1.6)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Core metals long/short overlays versus long-only baselines")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net cumulative log return")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "long_short_overlay_cum_log_returns_2021.png", dpi=170)
    plt.close(fig)


def plot_drawdowns(portfolio: pd.DataFrame, metrics: pd.DataFrame) -> None:
    selected = top_variants_for_plots(metrics)
    fig, ax = plt.subplots(figsize=(15, 7))
    for variant in selected:
        returns = portfolio[f"{variant}_net_log_return_5m"].astype(float)
        wealth = np.exp(returns.cumsum())
        drawdown = wealth / wealth.cummax() - 1.0
        sampled = drawdown.resample("1D").last().dropna()
        ax.plot(sampled.index.tz_convert(None), sampled, label=display_name(variant), linewidth=1.4)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Core metals long/short overlay drawdowns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "long_short_overlay_drawdowns_2021.png", dpi=170)
    plt.close(fig)


def plot_metric_bars(metrics: pd.DataFrame) -> None:
    selected = metrics.sort_values(["sharpe_0rf", "cagr"], ascending=False).head(12).copy()
    labels = [display_name(variant) for variant in selected["variant"]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True)
    plot_specs = [
        ("sharpe_0rf", "Sharpe", 1.0),
        ("cagr", "CAGR, %", 100.0),
        ("max_drawdown", "Max DD, %", 100.0),
    ]
    for ax, (col, title, scale) in zip(axes, plot_specs, strict=True):
        ax.barh(labels, selected[col] * scale)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        ax.invert_yaxis()
    fig.suptitle("Long/short overlay benchmark metrics")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "long_short_overlay_metric_bars.png", dpi=170)
    plt.close(fig)


def plot_best_weights(daily: pd.DataFrame, best_variant: str) -> None:
    cols = [f"{best_variant}_{root}_weight" for root in ROOTS]
    if not all(col in daily for col in cols):
        return
    fig, ax = plt.subplots(figsize=(15, 6))
    for root, col in zip(ROOTS, cols, strict=True):
        ax.plot(daily["ts"], daily[col], label=root, linewidth=1.0)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(f"{display_name(best_variant)} weights")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", ncol=len(ROOTS))
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "best_long_short_overlay_weights_2021.png", dpi=170)
    plt.close(fig)


def write_report(
    metrics: pd.DataFrame,
    split_metrics: pd.DataFrame,
    weight_summary: pd.DataFrame,
    turnover_summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    best_variant: str,
    portfolio: pd.DataFrame,
) -> None:
    sorted_metrics = metrics.sort_values(["sharpe_0rf", "cagr"], ascending=False)
    best_weights = weight_summary[weight_summary["variant"].eq(best_variant)]
    best_turnover = turnover_summary[turnover_summary["variant"].eq(best_variant)]
    diagnostics_summary = (
        diagnostics.groupby("variant", as_index=False)
        .agg(
            success_rate=("success", "mean"),
            median_avg_pairwise_corr=("avg_pairwise_corr", "median"),
            median_condition_number=("condition_number", "median"),
            mean_window_obs=("window_obs", "mean"),
        )
        .sort_values("variant")
    )
    top_split = split_metrics[
        split_metrics["variant"].isin(sorted_metrics.head(8)["variant"])
    ].sort_values(["split", "sharpe_0rf"], ascending=[True, False])

    lines = [
        "# Core Metals Long/Short Overlay Benchmark",
        "",
        "Objective: test whether allowing short exposure improves the 2021 onward core",
        "metals portfolio versus the 1/N, inverse-vol, and min-var long-only baselines.",
        "",
        "Construction:",
        "",
        "- Source returns: HYP-0041 raw 5-minute continuous futures close-to-close log returns.",
        "- Evaluation starts at the first available bar after `2021-01-01`.",
        "- Covariance portfolios rebalance daily from prior 30 calendar days of 5-minute returns.",
        "- Long-only capped min-var variants constrain max asset weight to 40%, 50%, or 60%.",
        "- Net-long long/short min-var variants keep `sum(weights)=1`, use bounds `[-25%, 75%]`,",
        "  and gross caps of 1.25x, 1.50x, and 2.00x.",
        "- Fair-value dislocation overlays use HYP-0043 carry-conditioned fair-value z-scores,",
        "  shifted by one 5-minute bar, dollar-neutral, and gross-normalized to 1x before scaling.",
        "- Blend variants add a 10% or 20% dollar-neutral overlay to the uncapped min-var core.",
        "- Metrics below are net of root-specific turnover costs from the HYP-0046 MBP1",
        "  spread model.",
        "",
        "## Metrics",
        "",
        sorted_metrics.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"## Best Variant: `{best_variant}`",
        "",
        "### Best Weight Summary",
        "",
        best_weights.to_markdown(index=False, floatfmt=".4f"),
        "",
        "### Best Turnover Summary",
        "",
        best_turnover.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Split Metrics For Top Variants",
        "",
        top_split.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Optimization Diagnostics",
        "",
        diagnostics_summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Input Span",
        "",
        f"- start: `{portfolio.index.min()}`",
        f"- end: `{portfolio.index.max()}`",
        f"- rows: `{len(portfolio)}`",
        "- covariance lookback: `30D`",
        "- fair-value signal source: `HYP-0043 fair_value_panel.parquet`",
        "",
        "## Files",
        "",
        "- `long_short_overlay_cum_log_returns_2021.png`",
        "- `long_short_overlay_drawdowns_2021.png`",
        "- `long_short_overlay_metric_bars.png`",
        "- `best_long_short_overlay_weights_2021.png`",
        "- `core_metals_5m_long_short_overlay_benchmark.parquet`",
        "- `long_short_overlay_daily.csv`",
        "- `daily_covariance_weights.csv`",
        "- `benchmark_metrics.csv`",
        "- `split_metrics.csv`",
        "- `weights_summary.csv`",
        "- `turnover_summary.csv`",
        "- `rebalance_diagnostics.csv`",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_results_json(
    metrics: pd.DataFrame,
    split_metrics: pd.DataFrame,
    turnover_summary: pd.DataFrame,
    best_variant: str,
    portfolio: pd.DataFrame,
) -> None:
    payload = {
        "best_variant": best_variant,
        "start_ts": str(portfolio.index.min()),
        "end_ts": str(portfolio.index.max()),
        "rows": len(portfolio),
        "metrics": json.loads(metrics.to_json(orient="records", date_format="iso")),
        "split_metrics": json.loads(split_metrics.to_json(orient="records", date_format="iso")),
        "turnover_summary": json.loads(turnover_summary.to_json(orient="records")),
        "cost_bps": COST_BPS.to_dict(),
    }
    (OUTPUT_DIR / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
