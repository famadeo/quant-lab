"""Plot core metals equal-weight benchmark from 5-minute raw close-to-close returns."""

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
OUTPUT_DIR = REPO_ROOT / "experiments" / "HYP-0047-core-metals-equal-weight-benchmark"

ROOTS = ["GC", "SI", "HG", "PL", "PA"]
START_TS = pd.Timestamp("2021-01-01", tz="UTC")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    returns = load_returns()
    benchmark = build_benchmark(returns)
    daily = daily_sample(benchmark)
    metrics = build_metrics(benchmark)

    benchmark.to_parquet(OUTPUT_DIR / "core_metals_5m_equal_weight_benchmark.parquet")
    benchmark.to_csv(OUTPUT_DIR / "core_metals_5m_equal_weight_benchmark.csv.gz")
    daily.to_csv(OUTPUT_DIR / "core_metals_equal_weight_benchmark_daily.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "benchmark_metrics.csv", index=False)

    plot_cumulative_returns(daily)
    plot_asset_contribution(metrics)
    write_report(benchmark, metrics)
    write_results_json(benchmark, metrics)

    print(metrics.to_string(index=False))
    print(f"Wrote {OUTPUT_DIR}", flush=True)


def load_returns() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)
    frame = pd.read_parquet(INPUT_PATH)
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    frame = frame.sort_values("ts").set_index("ts")[ROOTS].astype("float64")
    frame = frame[frame.index >= START_TS].copy()
    if frame.empty:
        raise ValueError(f"No rows at or after {START_TS}")
    frame = frame.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    frame.iloc[0] = 0.0
    return frame


def build_benchmark(returns: pd.DataFrame) -> pd.DataFrame:
    simple_returns = np.expm1(returns)
    equal_weight_simple = simple_returns.mean(axis=1)
    equal_weight_log = np.log1p(equal_weight_simple)

    output = returns.add_suffix("_log_return_5m").copy()
    output["EW_1N_log_return_5m"] = equal_weight_log
    for root in ROOTS:
        output[f"{root}_cum_log_return"] = returns[root].cumsum()
    output["EW_1N_cum_log_return"] = equal_weight_log.cumsum()
    return output


def daily_sample(benchmark: pd.DataFrame) -> pd.DataFrame:
    columns = [f"{root}_cum_log_return" for root in ROOTS] + ["EW_1N_cum_log_return"]
    sampled = benchmark[columns].resample("1D").last().dropna().reset_index()
    sampled["ts"] = sampled["ts"].dt.tz_convert(None)
    return sampled


def build_metrics(benchmark: pd.DataFrame) -> pd.DataFrame:
    rows = []
    years = years_between(benchmark.index[0], benchmark.index[-1])
    obs_per_year = len(benchmark) / years
    for name, column in [(root, f"{root}_log_return_5m") for root in ROOTS] + [
        ("EW_1N", "EW_1N_log_return_5m")
    ]:
        returns = benchmark[column].astype(float)
        cumulative = returns.cumsum()
        annual_log_return = returns.sum() / years
        annual_vol = returns.std(ddof=1) * math.sqrt(obs_per_year)
        rows.append(
            {
                "asset": name,
                "start_ts": benchmark.index[0],
                "end_ts": benchmark.index[-1],
                "nobs_5m": len(benchmark),
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


def plot_cumulative_returns(daily: pd.DataFrame) -> None:
    colors = {
        "GC": "#b68b00",
        "SI": "#7a8591",
        "HG": "#b35c2e",
        "PL": "#3b6ea8",
        "PA": "#5f8f5f",
    }
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for root in ROOTS:
        ax.plot(
            daily["ts"],
            daily[f"{root}_cum_log_return"],
            linewidth=1.0,
            alpha=0.75,
            color=colors[root],
            label=root,
        )
    ax.plot(
        daily["ts"],
        daily["EW_1N_cum_log_return"],
        linewidth=2.4,
        color="black",
        label="1/N equal weight",
    )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax.set_title("Core metals cumulative log returns from 5-minute close-to-close prices")
    ax.set_ylabel("Cumulative log return")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "core_metals_1n_cum_log_returns_2021.png", dpi=170)
    plt.close(fig)


def plot_asset_contribution(metrics: pd.DataFrame) -> None:
    ordered = metrics[metrics["asset"].isin(ROOTS)].set_index("asset").loc[ROOTS]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = ["#b68b00", "#7a8591", "#b35c2e", "#3b6ea8", "#5f8f5f"]
    ax.bar(ordered.index, ordered["cum_log_return"], color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Asset cumulative log return since 2021")
    ax.set_ylabel("Cumulative log return")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "core_metals_asset_cum_log_return_bars_2021.png", dpi=170)
    plt.close(fig)


def write_report(benchmark: pd.DataFrame, metrics: pd.DataFrame) -> None:
    lines = [
        "# Core Metals 1/N Equal-Weight Benchmark",
        "",
        "Baseline built from the HYP-0041 raw 5-minute close-to-close continuous futures",
        "log-return panel.",
        "",
        "Construction:",
        "",
        "- Start timestamp: `2021-01-01 00:00:00+00:00`.",
        "- Asset cumulative returns are simple cumulative sums of 5-minute log returns.",
        "- The 1/N portfolio is rebalanced every 5-minute bar: convert each asset log",
        "return to a simple return, average the five simple returns, then convert the",
        "portfolio bar return back to log return for cumulative log wealth.",
        "- The first in-sample bar is set to zero so the cumulative series starts at zero.",
        "",
        "## Metrics",
        "",
        metrics.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Input Span",
        "",
        f"- start: `{benchmark.index.min()}`",
        f"- end: `{benchmark.index.max()}`",
        f"- rows: `{len(benchmark)}`",
        "",
        "## Files",
        "",
        "- `core_metals_1n_cum_log_returns_2021.png`",
        "- `core_metals_asset_cum_log_return_bars_2021.png`",
        "- `core_metals_5m_equal_weight_benchmark.parquet`",
        "- `core_metals_5m_equal_weight_benchmark.csv.gz`",
        "- `core_metals_equal_weight_benchmark_daily.csv`",
        "- `benchmark_metrics.csv`",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_results_json(benchmark: pd.DataFrame, metrics: pd.DataFrame) -> None:
    ew = metrics[metrics["asset"].eq("EW_1N")].iloc[0]
    payload = {
        "experiment_id": "HYP-0047",
        "completed_at": datetime.now(UTC).isoformat(),
        "input_path": str(INPUT_PATH),
        "start_ts": benchmark.index.min().isoformat(),
        "end_ts": benchmark.index.max().isoformat(),
        "rows_5m": len(benchmark),
        "roots": ROOTS,
        "equal_weight_metrics": ew.to_dict(),
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
