"""Test correlation-aware core metals portfolio benchmarks."""

# ruff: noqa: PLR0911

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = (
    REPO_ROOT
    / "experiments"
    / "HYP-0041-core-metals-5m-log-returns"
    / "core_metals_5m_log_returns_wide.parquet"
)
OUTPUT_DIR = REPO_ROOT / "experiments" / "HYP-0050-core-metals-correlation-aware-benchmark"

ROOTS = ["GC", "SI", "HG", "PL", "PA"]
START_TS = pd.Timestamp("2021-01-01", tz="UTC")
LOOKBACK = pd.Timedelta(days=30)
MIN_OBS = 1_000
RIDGE = 1e-12
EPSILON = 1e-12


@dataclass(frozen=True)
class Variant:
    name: str
    method: str
    shrinkage: float


VARIANTS = [
    Variant("ERC_30D_SHRINK0", "erc", 0.0),
    Variant("ERC_30D_SHRINK25", "erc", 0.25),
    Variant("ERC_30D_SHRINK50", "erc", 0.50),
    Variant("MINVAR_30D_SHRINK25", "minvar", 0.25),
    Variant("MINVAR_30D_SHRINK50", "minvar", 0.50),
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    returns_full = load_returns()
    eval_returns = prepare_eval_returns(returns_full)
    rebalance_dates = build_rebalance_dates(eval_returns.index)

    weight_frames = []
    diagnostics = []
    previous_weights = {
        variant.name: np.full(len(ROOTS), 1.0 / len(ROOTS), dtype=float) for variant in VARIANTS
    }
    previous_weights["EW_1N"] = np.full(len(ROOTS), 1.0 / len(ROOTS), dtype=float)
    previous_weights["INV_VOL_30D_DAILY"] = np.full(len(ROOTS), 1.0 / len(ROOTS), dtype=float)

    for rebalance_ts in rebalance_dates:
        window = returns_full[
            (returns_full.index < rebalance_ts) & (returns_full.index >= rebalance_ts - LOOKBACK)
        ]
        covariance, diag, cov_diag = estimate_covariance(window)

        rows = [
            weights_row(rebalance_ts, "EW_1N", previous_weights["EW_1N"], "equal", 1.0),
        ]

        inv_vol = inverse_vol_weights_from_diag(diag)
        previous_weights["INV_VOL_30D_DAILY"] = inv_vol
        rows.append(
            weights_row(
                rebalance_ts,
                "INV_VOL_30D_DAILY",
                inv_vol,
                "inverse_vol",
                1.0,
            )
        )

        for variant in VARIANTS:
            shrunk = shrink_covariance(covariance, cov_diag, variant.shrinkage)
            weights, success, objective = optimize_weights(
                shrunk,
                variant,
                previous_weights[variant.name],
            )
            previous_weights[variant.name] = weights
            rows.append(weights_row(rebalance_ts, variant.name, weights, variant.method, objective))
            diagnostics.append(
                {
                    "rebalance_ts": rebalance_ts,
                    "variant": variant.name,
                    "method": variant.method,
                    "shrinkage": variant.shrinkage,
                    "success": success,
                    "objective": objective,
                    "window_obs": len(window),
                    "avg_pairwise_corr": avg_pairwise_corr(shrunk),
                    "condition_number": np.linalg.cond(shrunk),
                }
            )

        weight_frames.extend(rows)

    weights = pd.DataFrame(weight_frames)
    diagnostics_frame = pd.DataFrame(diagnostics)
    portfolio = build_portfolio(eval_returns, weights)
    daily = daily_sample(portfolio)
    metrics = build_metrics(portfolio)
    weight_summary = build_weight_summary(portfolio)
    turnover_summary = build_turnover_summary(portfolio)

    weights.to_csv(OUTPUT_DIR / "daily_rebalance_weights.csv", index=False)
    diagnostics_frame.to_csv(OUTPUT_DIR / "rebalance_diagnostics.csv", index=False)
    portfolio.to_parquet(OUTPUT_DIR / "core_metals_5m_correlation_aware_benchmark.parquet")
    daily.to_csv(OUTPUT_DIR / "core_metals_correlation_aware_benchmark_daily.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "benchmark_metrics.csv", index=False)
    weight_summary.to_csv(OUTPUT_DIR / "weights_summary.csv", index=False)
    turnover_summary.to_csv(OUTPUT_DIR / "turnover_summary.csv", index=False)

    best_variant = select_best_variant(metrics)
    plot_cumulative_returns(daily, metrics)
    plot_drawdowns(portfolio, metrics)
    plot_metric_bars(metrics)
    plot_best_weights(daily, best_variant)
    write_report(
        metrics,
        weight_summary,
        turnover_summary,
        diagnostics_frame,
        best_variant,
        portfolio,
    )
    write_results_json(metrics, weight_summary, turnover_summary, best_variant, portfolio)

    print(metrics.sort_values(["sharpe_0rf", "cagr"], ascending=False).to_string(index=False))
    print(f"Best variant: {best_variant}")
    print(f"Wrote {OUTPUT_DIR}", flush=True)


def load_returns() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)
    frame = pd.read_parquet(INPUT_PATH)
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


def estimate_covariance(window: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(window) < MIN_OBS:
        variances = np.ones(len(ROOTS), dtype=float)
        covariance = np.diag(variances)
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


def inverse_vol_weights_from_diag(diag: np.ndarray) -> np.ndarray:
    vol = np.sqrt(np.clip(diag, RIDGE, None))
    inv = 1.0 / vol
    return inv / inv.sum()


def optimize_weights(
    covariance: np.ndarray,
    variant: Variant,
    previous: np.ndarray,
) -> tuple[np.ndarray, bool, float]:
    covariance = regularize_covariance(covariance)
    scale = float(np.mean(np.diag(covariance)))
    if scale > EPSILON:
        covariance = covariance / scale
    if variant.method == "erc":
        return optimize_erc(covariance, previous)
    if variant.method == "minvar":
        return optimize_minvar(covariance, previous)
    raise ValueError(f"Unknown method: {variant.method}")


def optimize_erc(covariance: np.ndarray, previous: np.ndarray) -> tuple[np.ndarray, bool, float]:
    n_assets = covariance.shape[0]
    target = np.full(n_assets, 1.0 / n_assets)

    def objective(weights: np.ndarray) -> float:
        portfolio_var = float(weights @ covariance @ weights)
        if portfolio_var <= EPSILON:
            return 1e6
        marginal = covariance @ weights
        contributions = weights * marginal / portfolio_var
        return float(np.sum((contributions - target) ** 2))

    result = minimize_simplex(objective, previous)
    weights = normalize_weights(result.x if result.success else previous)
    return weights, bool(result.success), objective(weights)


def optimize_minvar(covariance: np.ndarray, previous: np.ndarray) -> tuple[np.ndarray, bool, float]:
    def objective(weights: np.ndarray) -> float:
        return float(weights @ covariance @ weights)

    result = minimize_simplex(objective, previous)
    weights = normalize_weights(result.x if result.success else previous)
    return weights, bool(result.success), objective(weights)


def minimize_simplex(objective: Any, start: np.ndarray) -> Any:
    n_assets = len(start)
    constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}]
    bounds = [(0.0, 1.0)] * n_assets
    return minimize(
        objective,
        normalize_weights(start),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-12, "disp": False},
    )


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    clean = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    total = clean.sum()
    if total <= EPSILON:
        return np.full(len(clean), 1.0 / len(clean), dtype=float)
    return clean / total


def weights_row(
    rebalance_ts: pd.Timestamp,
    variant: str,
    weights: np.ndarray,
    method: str,
    objective: float,
) -> dict[str, Any]:
    row = {
        "rebalance_ts": rebalance_ts,
        "variant": variant,
        "method": method,
        "objective": objective,
    }
    for root, weight in zip(ROOTS, weights, strict=True):
        row[f"{root}_weight"] = weight
    return row


def avg_pairwise_corr(covariance: np.ndarray) -> float:
    vol = np.sqrt(np.clip(np.diag(covariance), RIDGE, None))
    corr = covariance / np.outer(vol, vol)
    upper = corr[np.triu_indices_from(corr, k=1)]
    return float(np.nanmean(upper))


def build_portfolio(eval_returns: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    output = eval_returns.add_suffix("_log_return_5m").copy()
    simple_returns = np.expm1(eval_returns)
    for root in ROOTS:
        output[f"{root}_cum_log_return"] = eval_returns[root].cumsum()

    variant_names = weights["variant"].drop_duplicates().tolist()
    daily_weights_by_variant = {
        variant: (
            weights[weights["variant"].eq(variant)]
            .set_index("rebalance_ts")[[f"{root}_weight" for root in ROOTS]]
            .rename(columns={f"{root}_weight": root for root in ROOTS})
            .sort_index()
        )
        for variant in variant_names
    }

    for variant in variant_names:
        variant_weights = daily_weights_by_variant[variant]
        aligned_weights = variant_weights.reindex(eval_returns.index, method="ffill")
        aligned_weights = aligned_weights.fillna(1.0 / len(ROOTS))
        portfolio_simple = (aligned_weights * simple_returns).sum(axis=1)
        portfolio_log = np.log1p(portfolio_simple)
        output[f"{variant}_log_return_5m"] = portfolio_log
        output[f"{variant}_cum_log_return"] = portfolio_log.cumsum()
        output[f"{variant}_turnover"] = aligned_weights.diff().abs().sum(axis=1).fillna(0.0)
        for root in ROOTS:
            output[f"{variant}_{root}_weight"] = aligned_weights[root]
    return output


def daily_sample(portfolio: pd.DataFrame) -> pd.DataFrame:
    cumulative_cols = [col for col in portfolio.columns if col.endswith("_cum_log_return")]
    weight_cols = [col for col in portfolio.columns if col.endswith("_weight")]
    sampled = portfolio[cumulative_cols + weight_cols].resample("1D").last().dropna().reset_index()
    sampled["ts"] = sampled["ts"].dt.tz_convert(None)
    return sampled


def build_metrics(portfolio: pd.DataFrame) -> pd.DataFrame:
    rows = []
    years = years_between(portfolio.index[0], portfolio.index[-1])
    obs_per_year = len(portfolio) / years
    variants = ["EW_1N", "INV_VOL_30D_DAILY"] + [variant.name for variant in VARIANTS]
    for variant in variants:
        returns = portfolio[f"{variant}_log_return_5m"].astype(float)
        cumulative = returns.cumsum()
        annual_log_return = returns.sum() / years
        annual_vol = returns.std(ddof=1) * math.sqrt(obs_per_year)
        rows.append(
            {
                "variant": variant,
                "start_ts": portfolio.index[0],
                "end_ts": portfolio.index[-1],
                "nobs_5m": len(portfolio),
                "years": years,
                "cum_log_return": returns.sum(),
                "total_return_pct": math.expm1(returns.sum()) * 100.0,
                "cagr": math.expm1(annual_log_return),
                "annual_log_return": annual_log_return,
                "annual_vol": annual_vol,
                "sharpe_0rf": annual_log_return / annual_vol if annual_vol > 0 else np.nan,
                "max_drawdown": max_drawdown_from_cum_log(cumulative),
            }
        )
    return pd.DataFrame(rows)


def build_weight_summary(portfolio: pd.DataFrame) -> pd.DataFrame:
    rows = []
    variants = ["EW_1N", "INV_VOL_30D_DAILY"] + [variant.name for variant in VARIANTS]
    for variant in variants:
        for root in ROOTS:
            weights = portfolio[f"{variant}_{root}_weight"].astype(float)
            rows.append(
                {
                    "variant": variant,
                    "root": root,
                    "mean_weight": weights.mean(),
                    "median_weight": weights.median(),
                    "min_weight": weights.min(),
                    "p10_weight": weights.quantile(0.10),
                    "p90_weight": weights.quantile(0.90),
                    "max_weight": weights.max(),
                }
            )
    return pd.DataFrame(rows)


def build_turnover_summary(portfolio: pd.DataFrame) -> pd.DataFrame:
    years = years_between(portfolio.index[0], portfolio.index[-1])
    rows = []
    variants = ["EW_1N", "INV_VOL_30D_DAILY"] + [variant.name for variant in VARIANTS]
    for variant in variants:
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


def select_best_variant(metrics: pd.DataFrame) -> str:
    candidates = metrics[~metrics["variant"].isin(["EW_1N"])].copy()
    return str(candidates.sort_values(["sharpe_0rf", "cagr"], ascending=False).iloc[0]["variant"])


def plot_cumulative_returns(daily: pd.DataFrame, metrics: pd.DataFrame) -> None:
    top = (
        metrics.sort_values(["sharpe_0rf", "cagr"], ascending=False)
        .head(6)["variant"]
        .tolist()
    )
    if "EW_1N" not in top:
        top.append("EW_1N")
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for variant in top:
        ax.plot(
            daily["ts"],
            daily[f"{variant}_cum_log_return"],
            linewidth=2.2 if variant != "EW_1N" else 1.7,
            label=compact_variant_label(variant),
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Correlation-aware core metals portfolios versus 1/N")
    ax.set_ylabel("Cumulative log return")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlation_aware_cum_log_returns_2021.png", dpi=170)
    plt.close(fig)


def plot_drawdowns(portfolio: pd.DataFrame, metrics: pd.DataFrame) -> None:
    top = (
        metrics.sort_values(["sharpe_0rf", "cagr"], ascending=False)
        .head(5)["variant"]
        .tolist()
    )
    if "EW_1N" not in top:
        top.append("EW_1N")
    fig, ax = plt.subplots(figsize=(13, 5.6))
    for variant in top:
        cumulative = portfolio[f"{variant}_log_return_5m"].cumsum()
        drawdown = np.exp(cumulative - cumulative.cummax()) - 1.0
        sampled = drawdown.resample("1D").last().dropna()
        ax.plot(
            sampled.index.tz_convert(None),
            sampled,
            linewidth=1.5,
            label=compact_variant_label(variant),
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Correlation-aware portfolio drawdowns")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlation_aware_drawdowns_2021.png", dpi=170)
    plt.close(fig)


def plot_metric_bars(metrics: pd.DataFrame) -> None:
    ordered = metrics.sort_values("sharpe_0rf", ascending=True)
    labels = [compact_variant_label(item) for item in ordered["variant"]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 6.5), sharey=True)
    y = np.arange(len(ordered))
    axes[0].barh(y, ordered["sharpe_0rf"], color="#005f73")
    axes[0].set_yticks(y, labels=labels)
    axes[0].set_title("Sharpe")
    axes[0].grid(True, axis="x", alpha=0.25)
    axes[1].barh(y, ordered["cagr"] * 100.0, color="#0a9396")
    axes[1].set_title("CAGR, %")
    axes[1].grid(True, axis="x", alpha=0.25)
    axes[2].barh(y, ordered["max_drawdown"] * 100.0, color="#ae2012")
    axes[2].set_title("Max DD, %")
    axes[2].grid(True, axis="x", alpha=0.25)
    fig.suptitle("Correlation-aware benchmark metrics")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlation_aware_metric_bars.png", dpi=170)
    plt.close(fig)


def plot_best_weights(daily: pd.DataFrame, best_variant: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.2))
    x = daily["ts"]
    values = [daily[f"{best_variant}_{root}_weight"] for root in ROOTS]
    colors = ["#b68b00", "#7a8591", "#b35c2e", "#3b6ea8", "#5f8f5f"]
    ax.stackplot(x, values, labels=ROOTS, colors=colors, alpha=0.82)
    ax.set_title(f"{compact_variant_label(best_variant)} weights")
    ax.set_ylabel("Weight")
    ax.set_xlabel("Date")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", frameon=False, ncol=5)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "best_correlation_aware_weights_2021.png", dpi=170)
    plt.close(fig)


def write_report(
    metrics: pd.DataFrame,
    weight_summary: pd.DataFrame,
    turnover_summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    best_variant: str,
    portfolio: pd.DataFrame,
) -> None:
    ordered = metrics.sort_values(["sharpe_0rf", "cagr"], ascending=False)
    best_weights = weight_summary[weight_summary["variant"].eq(best_variant)].copy()
    best_turnover = turnover_summary[turnover_summary["variant"].eq(best_variant)].copy()
    diag_summary = (
        diagnostics.groupby("variant", as_index=False)
        .agg(
            success_rate=("success", "mean"),
            median_avg_pairwise_corr=("avg_pairwise_corr", "median"),
            median_condition_number=("condition_number", "median"),
            mean_window_obs=("window_obs", "mean"),
        )
        .sort_values("variant")
    )
    lines = [
        "# Core Metals Correlation-Aware Portfolio Benchmark",
        "",
        "Objective: test whether covariance-aware allocation reduces drawdowns versus the",
        "1/N and 30-day inverse-volatility baselines.",
        "",
        "Construction:",
        "",
        "- Source: HYP-0041 raw 5-minute continuous futures close-to-close log returns.",
        "- Evaluation starts at the first available bar after `2021-01-01 00:00:00+00:00`.",
        "- Weights rebalance daily and are applied to same-day 5-minute returns.",
        "- Covariance is estimated from the prior 30 calendar days of 5-minute returns.",
        "- Tested long-only equal-risk-contribution portfolios with covariance shrinkage",
        "to diagonal of 0%, 25%, and 50%.",
        "- Also tested long-only minimum-variance portfolios with 25% and 50% shrinkage.",
        "- No transaction costs are charged in this benchmark.",
        "",
        "## Metrics",
        "",
        ordered.to_markdown(index=False, floatfmt=".4f"),
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
        "## Optimization Diagnostics",
        "",
        diag_summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Input Span",
        "",
        f"- start: `{portfolio.index.min()}`",
        f"- end: `{portfolio.index.max()}`",
        f"- rows: `{len(portfolio)}`",
        "- lookback: `30D`",
        f"- minimum lookback rows: `{MIN_OBS}`",
        "",
        "## Files",
        "",
        "- `correlation_aware_cum_log_returns_2021.png`",
        "- `correlation_aware_drawdowns_2021.png`",
        "- `correlation_aware_metric_bars.png`",
        "- `best_correlation_aware_weights_2021.png`",
        "- `core_metals_5m_correlation_aware_benchmark.parquet`",
        "- `daily_rebalance_weights.csv`",
        "- `benchmark_metrics.csv`",
        "- `weights_summary.csv`",
        "- `turnover_summary.csv`",
        "- `rebalance_diagnostics.csv`",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_results_json(
    metrics: pd.DataFrame,
    weight_summary: pd.DataFrame,
    turnover_summary: pd.DataFrame,
    best_variant: str,
    portfolio: pd.DataFrame,
) -> None:
    payload = {
        "experiment_id": "HYP-0050",
        "completed_at": datetime.now(UTC).isoformat(),
        "input_path": str(INPUT_PATH),
        "start_ts": portfolio.index.min().isoformat(),
        "end_ts": portfolio.index.max().isoformat(),
        "rows_5m": len(portfolio),
        "roots": ROOTS,
        "lookback_days": 30,
        "rebalance_frequency": "1D",
        "variants": [variant.__dict__ for variant in VARIANTS],
        "best_variant": best_variant,
        "metrics": metrics.to_dict(orient="records"),
        "weights_summary": weight_summary.to_dict(orient="records"),
        "turnover_summary": turnover_summary.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def compact_variant_label(variant: str) -> str:
    return (
        variant.replace("EW_1N", "1/N")
        .replace("INV_VOL_30D_DAILY", "30D inv vol daily")
        .replace("ERC_30D_SHRINK", "ERC shrink ")
        .replace("MINVAR_30D_SHRINK", "MinVar shrink ")
    )


def years_between(start: pd.Timestamp, end: pd.Timestamp) -> float:
    return (end - start).total_seconds() / (365.25 * 24.0 * 60.0 * 60.0)


def max_drawdown_from_cum_log(cumulative: pd.Series) -> float:
    drawdown = np.exp(cumulative - cumulative.cummax()) - 1.0
    return float(drawdown.min())


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


if __name__ == "__main__":
    main()
