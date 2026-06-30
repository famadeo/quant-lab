"""Analyze autoregression in core metals hourly log returns."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = (
    REPO_ROOT
    / "experiments"
    / "HYP-0041-core-metals-5m-log-returns"
    / "core_metals_5m_log_returns_wide.parquet"
)
OUTPUT_DIR = REPO_ROOT / "experiments" / "HYP-0049-core-metals-hourly-autoregression"

ROOTS = ["GC", "SI", "HG", "PL", "PA"]
START_TS = pd.Timestamp("2021-01-01", tz="UTC")
ACF_LAGS = list(range(1, 25))
LJUNG_BOX_LAGS = [1, 6, 12, 24]
AR_ORDERS = [1, 6]
ROLLING_WINDOW = "90D"
ROLLING_MIN_OBS = 500
MIN_CORR_OBS = 3


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hourly = load_hourly_returns()
    acf_summary = build_acf_summary(hourly)
    ar_summary, ar_coefficients = build_ar_summaries(hourly)
    ljung_box = build_ljung_box(hourly)
    rolling_ar1 = build_rolling_ar1(hourly)

    hourly.reset_index().rename(columns={"index": "ts"}).to_parquet(
        OUTPUT_DIR / "core_metals_hourly_log_returns.parquet",
        index=False,
    )
    hourly.reset_index().rename(columns={"index": "ts"}).to_csv(
        OUTPUT_DIR / "core_metals_hourly_log_returns.csv.gz",
        index=False,
    )
    acf_summary.to_csv(OUTPUT_DIR / "acf_summary.csv", index=False)
    ar_summary.to_csv(OUTPUT_DIR / "ar_model_summary.csv", index=False)
    ar_coefficients.to_csv(OUTPUT_DIR / "ar_coefficients.csv", index=False)
    ljung_box.to_csv(OUTPUT_DIR / "ljung_box_summary.csv", index=False)
    rolling_ar1.to_csv(OUTPUT_DIR / "rolling_90d_ar1.csv", index=False)

    plot_acf_heatmap(acf_summary)
    plot_ar1_coefficients(ar_summary)
    plot_rolling_ar1(rolling_ar1)
    plot_lag_curves(acf_summary)
    write_report(hourly, acf_summary, ar_summary, ar_coefficients, ljung_box)
    write_results_json(hourly, acf_summary, ar_summary, ljung_box)

    print(ar_summary.to_string(index=False))
    print(f"Wrote {OUTPUT_DIR}", flush=True)


def load_hourly_returns() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)
    frame = pd.read_parquet(INPUT_PATH)
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    returns = frame.sort_values("ts").set_index("ts")[ROOTS].astype("float64")
    returns = returns[returns.index >= START_TS].replace([np.inf, -np.inf], np.nan)
    returns = returns.ffill().fillna(0.0)
    returns.iloc[0] = 0.0
    hourly = returns.groupby(returns.index.floor("1h")).sum().sort_index()
    hourly.index.name = "ts"
    hourly["gap_hours"] = hourly.index.to_series().diff().dt.total_seconds().div(3600.0)
    return hourly


def build_acf_summary(hourly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for root in ROOTS:
        series = hourly[root].astype(float)
        for sample in ["all_observed", "contiguous"]:
            for lag in ACF_LAGS:
                result = lag_correlation(series, lag, contiguous=(sample == "contiguous"))
                rows.append({"root": root, "sample": sample, "lag": lag, **result})
    return pd.DataFrame(rows)


def lag_correlation(series: pd.Series, lag: int, *, contiguous: bool) -> dict[str, float]:
    y = series.iloc[lag:]
    x = series.shift(lag).iloc[lag:]
    mask = x.notna() & y.notna()
    if contiguous:
        lag_gap = series.index.to_series().diff(lag).iloc[lag:]
        mask &= lag_gap.eq(pd.Timedelta(hours=lag)).to_numpy()
    x_values = x[mask].to_numpy(dtype=float)
    y_values = y[mask].to_numpy(dtype=float)
    nobs = len(x_values)
    if nobs < MIN_CORR_OBS or np.std(x_values) == 0 or np.std(y_values) == 0:
        return {"nobs": nobs, "autocorr": np.nan, "tstat": np.nan, "pvalue": np.nan}
    corr = float(np.corrcoef(x_values, y_values)[0, 1])
    denom = max(1.0 - corr * corr, 1e-12)
    tstat = corr * math.sqrt((nobs - 2) / denom)
    pvalue = 2.0 * stats.t.sf(abs(tstat), df=nobs - 2)
    return {"nobs": nobs, "autocorr": corr, "tstat": tstat, "pvalue": pvalue}


def build_ar_summaries(hourly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    coefficient_rows = []
    for root in ROOTS:
        for sample in ["all_observed", "contiguous"]:
            for order in AR_ORDERS:
                model_data = ar_design(hourly[root], order, contiguous=(sample == "contiguous"))
                model_result = fit_ar_model(model_data, order)
                summary_rows.append(
                    {
                        "root": root,
                        "sample": sample,
                        "order": order,
                        **model_result["summary"],
                    }
                )
                for lag, coef_data in model_result["coefficients"].items():
                    coefficient_rows.append(
                        {
                            "root": root,
                            "sample": sample,
                            "order": order,
                            "lag": lag,
                            **coef_data,
                        }
                    )
    return pd.DataFrame(summary_rows), pd.DataFrame(coefficient_rows)


def ar_design(series: pd.Series, order: int, *, contiguous: bool) -> pd.DataFrame:
    frame = pd.DataFrame({"y": series})
    for lag in range(1, order + 1):
        frame[f"lag_{lag}"] = series.shift(lag)
    frame = frame.dropna()
    if contiguous:
        gap = series.index.to_series().diff(order).reindex(frame.index)
        frame = frame[gap.eq(pd.Timedelta(hours=order))]
    return frame


def fit_ar_model(frame: pd.DataFrame, order: int) -> dict[str, Any]:
    if len(frame) <= order + 5:
        return {
            "summary": {
                "nobs": len(frame),
                "r2": np.nan,
                "adj_r2": np.nan,
                "aic": np.nan,
                "bic": np.nan,
                "intercept": np.nan,
                "intercept_tstat": np.nan,
                "intercept_pvalue": np.nan,
                "lag1_coef": np.nan,
                "lag1_tstat": np.nan,
                "lag1_pvalue": np.nan,
            },
            "coefficients": {},
        }
    y = frame["y"]
    x = sm.add_constant(frame[[f"lag_{lag}" for lag in range(1, order + 1)]])
    fit = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": min(24, max(1, len(frame) // 50))})
    coefficients = {}
    for lag in range(1, order + 1):
        name = f"lag_{lag}"
        coefficients[lag] = {
            "coef": fit.params[name],
            "tstat": fit.tvalues[name],
            "pvalue": fit.pvalues[name],
        }
    return {
        "summary": {
            "nobs": int(fit.nobs),
            "r2": fit.rsquared,
            "adj_r2": fit.rsquared_adj,
            "aic": fit.aic,
            "bic": fit.bic,
            "intercept": fit.params["const"],
            "intercept_tstat": fit.tvalues["const"],
            "intercept_pvalue": fit.pvalues["const"],
            "lag1_coef": fit.params["lag_1"],
            "lag1_tstat": fit.tvalues["lag_1"],
            "lag1_pvalue": fit.pvalues["lag_1"],
        },
        "coefficients": coefficients,
    }


def build_ljung_box(hourly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for root in ROOTS:
        series = hourly[root].dropna().astype(float)
        result = acorr_ljungbox(series, lags=LJUNG_BOX_LAGS, return_df=True)
        for lag, row in result.iterrows():
            rows.append(
                {
                    "root": root,
                    "sample": "all_observed",
                    "lag": int(lag),
                    "lb_stat": row["lb_stat"],
                    "lb_pvalue": row["lb_pvalue"],
                }
            )
    return pd.DataFrame(rows)


def build_rolling_ar1(hourly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for root in ROOTS:
        series = hourly[root].astype(float)
        lagged = series.shift(1)
        rolling = series.rolling(ROLLING_WINDOW, min_periods=ROLLING_MIN_OBS).corr(lagged)
        sampled = rolling.resample("1D").last().dropna()
        for ts, value in sampled.items():
            rows.append({"ts": ts, "root": root, "rolling_ar1": value})
    output = pd.DataFrame(rows)
    if not output.empty:
        output["ts"] = pd.to_datetime(output["ts"], utc=True)
    return output


def plot_acf_heatmap(acf_summary: pd.DataFrame) -> None:
    data = acf_summary[acf_summary["sample"].eq("contiguous")]
    matrix = data.pivot(index="root", columns="lag", values="autocorr").reindex(ROOTS)
    values = matrix.to_numpy(dtype=float)
    vmax = max(0.05, np.nanpercentile(np.abs(values), 95))
    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    image = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title("Hourly log-return autocorrelation by lag, contiguous hours")
    ax.set_xlabel("Lag, hours")
    ax.set_ylabel("Asset")
    ax.set_xticks(np.arange(len(matrix.columns)), labels=matrix.columns)
    ax.set_yticks(np.arange(len(matrix.index)), labels=matrix.index)
    for i, root in enumerate(matrix.index):
        for j, lag in enumerate(matrix.columns):
            value = matrix.loc[root, lag]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="Autocorrelation")
    fig.savefig(OUTPUT_DIR / "hourly_autocorrelation_heatmap_contiguous.png", dpi=170)
    plt.close(fig)


def plot_ar1_coefficients(ar_summary: pd.DataFrame) -> None:
    data = ar_summary[(ar_summary["order"].eq(1)) & ar_summary["sample"].eq("contiguous")]
    data = data.set_index("root").reindex(ROOTS)
    colors = ["#b68b00", "#7a8591", "#b35c2e", "#3b6ea8", "#5f8f5f"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(data.index, data["lag1_coef"], color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Hourly AR(1) coefficient, contiguous hours")
    ax.set_ylabel("AR(1) coefficient")
    ax.grid(True, axis="y", alpha=0.25)
    for i, row in enumerate(data.itertuples()):
        if np.isfinite(row.lag1_tstat):
            ax.text(i, row.lag1_coef, f"t={row.lag1_tstat:.1f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "hourly_ar1_coefficients_contiguous.png", dpi=170)
    plt.close(fig)


def plot_rolling_ar1(rolling_ar1: pd.DataFrame) -> None:
    if rolling_ar1.empty:
        return
    fig, ax = plt.subplots(figsize=(13, 6.2))
    colors = {
        "GC": "#b68b00",
        "SI": "#7a8591",
        "HG": "#b35c2e",
        "PL": "#3b6ea8",
        "PA": "#5f8f5f",
    }
    for root in ROOTS:
        data = rolling_ar1[rolling_ar1["root"].eq(root)]
        ax.plot(data["ts"].dt.tz_convert(None), data["rolling_ar1"], label=root, color=colors[root])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Rolling 90-day hourly AR(1) correlation")
    ax.set_ylabel("AR(1) correlation")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", frameon=False, ncol=5)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rolling_90d_hourly_ar1.png", dpi=170)
    plt.close(fig)


def plot_lag_curves(acf_summary: pd.DataFrame) -> None:
    data = acf_summary[acf_summary["sample"].eq("contiguous")]
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {
        "GC": "#b68b00",
        "SI": "#7a8591",
        "HG": "#b35c2e",
        "PL": "#3b6ea8",
        "PA": "#5f8f5f",
    }
    for root in ROOTS:
        subset = data[data["root"].eq(root)]
        ax.plot(
            subset["lag"],
            subset["autocorr"],
            marker="o",
            linewidth=1.3,
            label=root,
            color=colors[root],
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Hourly log-return autocorrelation curves, contiguous hours")
    ax.set_xlabel("Lag, hours")
    ax.set_ylabel("Autocorrelation")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=5)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "hourly_autocorrelation_lag_curves.png", dpi=170)
    plt.close(fig)


def write_report(
    hourly: pd.DataFrame,
    acf_summary: pd.DataFrame,
    ar_summary: pd.DataFrame,
    ar_coefficients: pd.DataFrame,
    ljung_box: pd.DataFrame,
) -> None:
    ar1 = ar_summary[(ar_summary["sample"].eq("contiguous")) & ar_summary["order"].eq(1)].copy()
    ar6 = ar_summary[(ar_summary["sample"].eq("contiguous")) & ar_summary["order"].eq(6)].copy()
    acf_top = acf_summary[
        (acf_summary["sample"].eq("contiguous")) & (acf_summary["lag"].isin([1, 2, 3, 6, 12, 24]))
    ].copy()
    ar6_coefs = ar_coefficients[
        (ar_coefficients["sample"].eq("contiguous")) & (ar_coefficients["order"].eq(6))
    ].copy()
    lines = [
        "# Core Metals Hourly Return Autoregression",
        "",
        "Objective: measure how autoregressive hourly log returns are for each core metal.",
        "",
        "Construction:",
        "",
        "- Source: HYP-0041 raw 5-minute continuous futures close-to-close log returns.",
        "- Evaluation starts at the first available bar after `2021-01-01 00:00:00+00:00`.",
        "- Hourly returns are summed 5-minute log returns by observed hourly bucket.",
        "- `contiguous` statistics only use lag pairs where the timestamp difference is exactly",
        "the stated lag in hours, avoiding Friday-to-Sunday/session-gap lag pairs.",
        "- AR model t-stats use HAC standard errors with up to 24 hourly lags.",
        "",
        "## Contiguous-Hour AR(1)",
        "",
        ar1[
            [
                "root",
                "nobs",
                "lag1_coef",
                "lag1_tstat",
                "lag1_pvalue",
                "r2",
                "adj_r2",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Contiguous-Hour AR(6) Summary",
        "",
        ar6[
            [
                "root",
                "nobs",
                "lag1_coef",
                "lag1_tstat",
                "lag1_pvalue",
                "r2",
                "adj_r2",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Selected Autocorrelations",
        "",
        acf_top.pivot(index=["root"], columns="lag", values="autocorr")
        .reindex(ROOTS)
        .to_markdown(floatfmt=".6f"),
        "",
        "## AR(6) Coefficients",
        "",
        ar6_coefs.pivot(index="root", columns="lag", values="coef")
        .reindex(ROOTS)
        .to_markdown(floatfmt=".6f"),
        "",
        "## Ljung-Box On Observed-Hour Sequence",
        "",
        ljung_box.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Input Span",
        "",
        f"- hourly start: `{hourly.index.min()}`",
        f"- hourly end: `{hourly.index.max()}`",
        f"- observed hourly rows: `{len(hourly)}`",
        "",
        "## Files",
        "",
        "- `core_metals_hourly_log_returns.parquet`",
        "- `core_metals_hourly_log_returns.csv.gz`",
        "- `acf_summary.csv`",
        "- `ar_model_summary.csv`",
        "- `ar_coefficients.csv`",
        "- `ljung_box_summary.csv`",
        "- `rolling_90d_ar1.csv`",
        "- `hourly_autocorrelation_heatmap_contiguous.png`",
        "- `hourly_ar1_coefficients_contiguous.png`",
        "- `rolling_90d_hourly_ar1.png`",
        "- `hourly_autocorrelation_lag_curves.png`",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_results_json(
    hourly: pd.DataFrame,
    acf_summary: pd.DataFrame,
    ar_summary: pd.DataFrame,
    ljung_box: pd.DataFrame,
) -> None:
    ar1 = ar_summary[(ar_summary["sample"].eq("contiguous")) & ar_summary["order"].eq(1)]
    payload = {
        "experiment_id": "HYP-0049",
        "completed_at": datetime.now(UTC).isoformat(),
        "input_path": str(INPUT_PATH),
        "start_ts": hourly.index.min().isoformat(),
        "end_ts": hourly.index.max().isoformat(),
        "hourly_rows": len(hourly),
        "roots": ROOTS,
        "acf_lags": ACF_LAGS,
        "ar_orders": AR_ORDERS,
        "contiguous_ar1": ar1.to_dict(orient="records"),
        "ljung_box": ljung_box.to_dict(orient="records"),
        "max_abs_contiguous_acf": float(
            acf_summary[acf_summary["sample"].eq("contiguous")]["autocorr"].abs().max()
        ),
    }
    (OUTPUT_DIR / "results.json").write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


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
