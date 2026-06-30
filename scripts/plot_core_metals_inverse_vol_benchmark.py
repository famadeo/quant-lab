"""Build core metals inverse-volatility portfolio from 5-minute returns."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
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
    / "HYP-0041-core-metals-5m-log-returns"
    / "core_metals_5m_log_returns_wide.parquet"
)
OUTPUT_DIR = REPO_ROOT / "experiments" / "HYP-0048-core-metals-inverse-vol-benchmark"

ROOTS = ["GC", "SI", "HG", "PL", "PA"]
START_TS = pd.Timestamp("2021-01-01", tz="UTC")
VOL_LOOKBACK = "30D"
MIN_VOL_OBS = 1_000
VOL_FLOOR = 1e-8


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    returns_full = load_returns_full()
    portfolio = build_inverse_vol_portfolio(returns_full)
    daily = daily_sample(portfolio)
    metrics = build_metrics(portfolio)
    weights_summary = build_weights_summary(portfolio)
    turnover_summary = build_turnover_summary(portfolio)

    portfolio.to_parquet(OUTPUT_DIR / "core_metals_5m_inverse_vol_benchmark.parquet")
    portfolio.to_csv(OUTPUT_DIR / "core_metals_5m_inverse_vol_benchmark.csv.gz")
    daily.to_csv(OUTPUT_DIR / "core_metals_inverse_vol_benchmark_daily.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "benchmark_metrics.csv", index=False)
    weights_summary.to_csv(OUTPUT_DIR / "weights_summary.csv", index=False)
    turnover_summary.to_csv(OUTPUT_DIR / "turnover_summary.csv", index=False)

    plot_cumulative_returns(daily)
    plot_relative_cumulative_return(daily)
    plot_weights(daily)
    plot_weight_summary(weights_summary)
    write_report(portfolio, metrics, weights_summary, turnover_summary)
    write_results_json(portfolio, metrics, weights_summary, turnover_summary)

    print(metrics.to_string(index=False))
    print(f"Wrote {OUTPUT_DIR}", flush=True)


def load_returns_full() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)
    frame = pd.read_parquet(INPUT_PATH)
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    frame = frame.sort_values("ts").set_index("ts")[ROOTS].astype("float64")
    frame = frame.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    return frame


def build_inverse_vol_portfolio(returns_full: pd.DataFrame) -> pd.DataFrame:
    rolling_vol = returns_full.rolling(VOL_LOOKBACK, min_periods=MIN_VOL_OBS).std().shift(1)
    weights_full = inverse_vol_weights(rolling_vol)
    returns = returns_full[returns_full.index >= START_TS].copy()
    weights = weights_full.reindex(returns.index).ffill().fillna(1.0 / len(ROOTS))
    returns.iloc[0] = 0.0

    simple_returns = np.expm1(returns)
    inv_vol_simple = (weights * simple_returns).sum(axis=1)
    inv_vol_log = np.log1p(inv_vol_simple)
    equal_weight_simple = simple_returns.mean(axis=1)
    equal_weight_log = np.log1p(equal_weight_simple)

    output = returns.add_suffix("_log_return_5m").copy()
    for root in ROOTS:
        output[f"{root}_weight"] = weights[root]
        output[f"{root}_rolling_vol_30d"] = rolling_vol.reindex(returns.index)[root]
        output[f"{root}_cum_log_return"] = returns[root].cumsum()
    output["INV_VOL_30D_log_return_5m"] = inv_vol_log
    output["EW_1N_log_return_5m"] = equal_weight_log
    output["INV_VOL_30D_cum_log_return"] = inv_vol_log.cumsum()
    output["EW_1N_cum_log_return"] = equal_weight_log.cumsum()
    output["INV_VOL_30D_turnover"] = weights.diff().abs().sum(axis=1).fillna(0.0)
    output["INV_VOL_30D_gross"] = weights.abs().sum(axis=1)
    return output


def inverse_vol_weights(rolling_vol: pd.DataFrame) -> pd.DataFrame:
    clean_vol = rolling_vol.where(rolling_vol > VOL_FLOOR)
    inv_vol = 1.0 / clean_vol
    raw = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    weights = raw.reindex(columns=ROOTS)
    weights = weights.where(np.isfinite(weights), np.nan)
    fallback = pd.DataFrame(1.0 / len(ROOTS), index=weights.index, columns=ROOTS)
    return weights.fillna(fallback)


def daily_sample(portfolio: pd.DataFrame) -> pd.DataFrame:
    columns = (
        [f"{root}_cum_log_return" for root in ROOTS]
        + ["INV_VOL_30D_cum_log_return", "EW_1N_cum_log_return"]
        + [f"{root}_weight" for root in ROOTS]
    )
    sampled = portfolio[columns].resample("1D").last().dropna().reset_index()
    sampled["ts"] = sampled["ts"].dt.tz_convert(None)
    return sampled


def build_metrics(portfolio: pd.DataFrame) -> pd.DataFrame:
    rows = []
    years = years_between(portfolio.index[0], portfolio.index[-1])
    obs_per_year = len(portfolio) / years
    series = [(root, f"{root}_log_return_5m") for root in ROOTS] + [
        ("EW_1N", "EW_1N_log_return_5m"),
        ("INV_VOL_30D", "INV_VOL_30D_log_return_5m"),
    ]
    for name, column in series:
        returns = portfolio[column].astype(float)
        cumulative = returns.cumsum()
        annual_log_return = returns.sum() / years
        annual_vol = returns.std(ddof=1) * math.sqrt(obs_per_year)
        rows.append(
            {
                "asset": name,
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


def build_weights_summary(portfolio: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for root in ROOTS:
        weights = portfolio[f"{root}_weight"]
        rows.append(
            {
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
    turnover = portfolio["INV_VOL_30D_turnover"].astype(float)
    return pd.DataFrame(
        [
            {
                "portfolio": "INV_VOL_30D",
                "cum_turnover": turnover.sum(),
                "annual_turnover": turnover.sum() / years,
                "mean_5m_turnover": turnover.mean(),
                "p95_5m_turnover": turnover.quantile(0.95),
                "max_5m_turnover": turnover.max(),
            }
        ]
    )


def plot_cumulative_returns(daily: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.plot(
        daily["ts"],
        daily["EW_1N_cum_log_return"],
        linewidth=2.0,
        color="#444444",
        label="1/N equal weight",
    )
    ax.plot(
        daily["ts"],
        daily["INV_VOL_30D_cum_log_return"],
        linewidth=2.4,
        color="#005f73",
        label="30D inverse vol",
    )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax.set_title("Core metals 1/N vs 30-day inverse-volatility portfolio")
    ax.set_ylabel("Cumulative log return")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "core_metals_inverse_vol_vs_1n_cum_log_returns_2021.png", dpi=170)
    plt.close(fig)


def plot_relative_cumulative_return(daily: pd.DataFrame) -> None:
    relative = daily["INV_VOL_30D_cum_log_return"] - daily["EW_1N_cum_log_return"]
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.plot(daily["ts"], relative, linewidth=1.8, color="#9b2226")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("30-day inverse-vol cumulative log return minus 1/N benchmark")
    ax.set_ylabel("Relative cumulative log return")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "core_metals_inverse_vol_relative_to_1n_2021.png", dpi=170)
    plt.close(fig)


def plot_weights(daily: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.2))
    x = daily["ts"]
    values = [daily[f"{root}_weight"] for root in ROOTS]
    colors = ["#b68b00", "#7a8591", "#b35c2e", "#3b6ea8", "#5f8f5f"]
    ax.stackplot(x, values, labels=ROOTS, colors=colors, alpha=0.8)
    ax.set_title("30-day inverse-volatility portfolio weights")
    ax.set_ylabel("Weight")
    ax.set_xlabel("Date")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", frameon=False, ncol=5)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "core_metals_inverse_vol_weights_2021.png", dpi=170)
    plt.close(fig)


def plot_weight_summary(weights_summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(weights_summary["root"], weights_summary["mean_weight"], color="#005f73")
    ax.axhline(1.0 / len(ROOTS), color="black", linewidth=1.0, linestyle="--", label="1/N")
    ax.set_title("Average inverse-vol weight since 2021")
    ax.set_ylabel("Average weight")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "core_metals_inverse_vol_average_weights_2021.png", dpi=170)
    plt.close(fig)


def write_report(
    portfolio: pd.DataFrame,
    metrics: pd.DataFrame,
    weights_summary: pd.DataFrame,
    turnover_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Core Metals 30-Day Inverse-Volatility Benchmark",
        "",
        "Baseline built from the HYP-0041 raw 5-minute close-to-close continuous futures",
        "log-return panel.",
        "",
        "Construction:",
        "",
        "- Evaluation starts at the first available bar after `2021-01-01 00:00:00+00:00`.",
        "- Trailing volatility is the rolling 30-calendar-day standard deviation of each",
        "metal's 5-minute log returns.",
        "- Volatility is shifted by one bar before weight calculation to avoid lookahead.",
        "- Weights are long-only inverse-volatility weights: `w_i ∝ 1 / sigma_i`, normalized",
        "to sum to 1.",
        "- The portfolio is rebalanced every 5-minute bar before costs.",
        "- This is equal standalone-vol weighting, not correlation-aware risk parity.",
        "",
        "## Metrics",
        "",
        metrics.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Weight Summary",
        "",
        weights_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Turnover Summary",
        "",
        turnover_summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Input Span",
        "",
        f"- start: `{portfolio.index.min()}`",
        f"- end: `{portfolio.index.max()}`",
        f"- rows: `{len(portfolio)}`",
        f"- volatility lookback: `{VOL_LOOKBACK}`",
        f"- min volatility observations: `{MIN_VOL_OBS}`",
        "",
        "## Files",
        "",
        "- `core_metals_inverse_vol_vs_1n_cum_log_returns_2021.png`",
        "- `core_metals_inverse_vol_relative_to_1n_2021.png`",
        "- `core_metals_inverse_vol_weights_2021.png`",
        "- `core_metals_inverse_vol_average_weights_2021.png`",
        "- `core_metals_5m_inverse_vol_benchmark.parquet`",
        "- `core_metals_5m_inverse_vol_benchmark.csv.gz`",
        "- `core_metals_inverse_vol_benchmark_daily.csv`",
        "- `benchmark_metrics.csv`",
        "- `weights_summary.csv`",
        "- `turnover_summary.csv`",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_results_json(
    portfolio: pd.DataFrame,
    metrics: pd.DataFrame,
    weights_summary: pd.DataFrame,
    turnover_summary: pd.DataFrame,
) -> None:
    inv = metrics[metrics["asset"].eq("INV_VOL_30D")].iloc[0]
    ew = metrics[metrics["asset"].eq("EW_1N")].iloc[0]
    payload = {
        "experiment_id": "HYP-0048",
        "completed_at": datetime.now(UTC).isoformat(),
        "input_path": str(INPUT_PATH),
        "start_ts": portfolio.index.min().isoformat(),
        "end_ts": portfolio.index.max().isoformat(),
        "rows_5m": len(portfolio),
        "roots": ROOTS,
        "vol_lookback": VOL_LOOKBACK,
        "min_vol_obs": MIN_VOL_OBS,
        "inverse_vol_metrics": inv.to_dict(),
        "equal_weight_metrics": ew.to_dict(),
        "weights_summary": weights_summary.to_dict(orient="records"),
        "turnover_summary": turnover_summary.iloc[0].to_dict(),
    }
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def years_between(start: pd.Timestamp, end: pd.Timestamp) -> float:
    return (end - start).total_seconds() / (365.25 * 24.0 * 60.0 * 60.0)


def max_drawdown_from_cum_log(cumulative: pd.Series) -> float:
    drawdown = np.exp(cumulative - cumulative.cummax()) - 1.0
    return float(drawdown.min())


def json_safe(value: Any) -> Any:  # noqa: PLR0911
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
