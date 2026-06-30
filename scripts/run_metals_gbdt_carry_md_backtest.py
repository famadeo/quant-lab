# ruff: noqa: PLR0911
from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

ROOTS = ["GC", "SI", "HG", "PL", "PA"]
HORIZONS = {"15m": 3, "1h": 12, "4h": 48, "1d": 288}
THRESHOLD_QUANTILES = [0.80, 0.90, 0.95]
MIN_TRAIN_ROWS = 5_000
TRAIN_LOOKBACK = pd.Timedelta("365D")
TEST_WINDOW = pd.Timedelta("90D")
FREQUENCY = "5min"
PERIODS_PER_YEAR = 365.25 * 24 * 12

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "experiments" / "HYP-0025-metals-gbdt-carry-md-backtest"
CONTINUOUS_DIR = Path(
    "/home/famadeo/research/databento-asset-browser/data/metals_1m_10y/continuous"
)
CARRY_PATH = (
    REPO_ROOT
    / "notebooks"
    / "explorations"
    / "assets"
    / "2026-06-25_metals_5m_carry_timeseries"
    / "annualized_carry_5m_exact_observations.csv"
)
MD_DIR = (
    REPO_ROOT / "notebooks" / "explorations" / "assets" / "2026-06-25_metals_5m_trade_mahalanobis"
)
MD_PATH = MD_DIR / "trade_mahalanobis_5m.csv"
COST_PATH = (
    REPO_ROOT
    / "experiments"
    / "HYP-0014-metals-flow-filtered-residual-reversion-3y"
    / "cost_estimates.csv"
)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    returns = prices.diff().fillna(0.0)
    md = load_md()
    carry = load_carry()
    costs_bps = load_costs()

    feature_frames = {root: build_root_features(root, prices, returns, md, carry) for root in ROOTS}
    predictions, diagnostics = walk_forward_predictions(feature_frames, prices)
    strategy_frames, metrics = run_strategy_variants(predictions, returns, costs_bps)

    predictions.to_parquet(OUT_DIR / "walk_forward_predictions.parquet", index=False)
    diagnostics.to_csv(OUT_DIR / "model_diagnostics.csv", index=False)
    metrics.to_csv(OUT_DIR / "variant_metrics.csv", index=False)

    best_gross = metrics.sort_values(["gross_return", "net_return"], ascending=False).iloc[0]
    best_net = metrics.sort_values(["net_return", "gross_to_cost"], ascending=False).iloc[0]
    for label, row in [("best_gross", best_gross), ("best_net", best_net)]:
        frame = strategy_frames[row["variant"]]
        frame.to_csv(OUT_DIR / f"{label}_strategy_returns.csv", index=False)
        plot_equity(frame, OUT_DIR / f"{label}_equity.png", title=f"{label}: {row['variant']}")
        monthly_returns(frame).to_csv(OUT_DIR / f"{label}_monthly_returns.csv", index=False)

    summary = {
        "experiment_id": "HYP-0025",
        "title": "Metals GBDT carry/MD 5m walk-forward backtest",
        "completed_at": datetime.now(UTC).isoformat(),
        "data": {
            "roots": ROOTS,
            "horizons": HORIZONS,
            "threshold_quantiles": THRESHOLD_QUANTILES,
            "train_lookback": str(TRAIN_LOOKBACK),
            "test_window": str(TEST_WINDOW),
            "feature_rows": {root: len(frame) for root, frame in feature_frames.items()},
            "predictions": len(predictions),
            "start": str(predictions["ts"].min()),
            "end": str(predictions["ts"].max()),
            "costs_bps_per_side": costs_bps.to_dict(),
        },
        "best_gross": best_gross.to_dict(),
        "best_net": best_net.to_dict(),
        "top_metrics": metrics.sort_values(["net_return", "gross_to_cost"], ascending=False)
        .head(20)
        .to_dict(orient="records"),
    }
    (OUT_DIR / "results.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    write_report(summary, metrics, diagnostics)
    print(metrics.sort_values(["net_return", "gross_to_cost"], ascending=False).head(15))
    print(f"Wrote {OUT_DIR}")


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


def load_prices() -> pd.DataFrame:
    parts = []
    for root in ROOTS:
        path = CONTINUOUS_DIR / f"{root}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = (
            pl.scan_parquet(path)
            .filter(pl.col("ts") >= pd.Timestamp("2023-06-20", tz="UTC"))
            .select("ts", "cont_logprice")
            .collect()
            .to_pandas()
        )
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
        series = (
            frame.set_index("ts")["cont_logprice"]
            .sort_index()
            .resample(FREQUENCY)
            .last()
            .ffill()
            .rename(root)
        )
        parts.append(series)
    return pd.concat(parts, axis=1).sort_index().dropna(how="all").ffill()


def load_md() -> pd.DataFrame:
    if not MD_PATH.exists():
        raise FileNotFoundError(MD_PATH)
    md = pd.read_csv(MD_PATH, parse_dates=["ts"]).set_index("ts").sort_index()
    md.index = pd.to_datetime(md.index, utc=True)
    signed = pd.read_parquet(MD_DIR / "signed_flow_share_5m.parquet").add_prefix("signed_share_")
    notional = pd.read_parquet(MD_DIR / "notional_share_5m.parquet").add_prefix("notional_share_")
    signed.index = pd.to_datetime(signed.index, utc=True)
    notional.index = pd.to_datetime(notional.index, utc=True)
    md = md.join([signed, notional], how="left")
    md["log_complex_trades"] = np.log1p(md["complex_trades"])
    md["log_complex_notional"] = np.log1p(md["complex_notional"])
    for column in ["md_signed_flow_ewma", "md_notional_share_ewma"]:
        md[f"{column}_change_1"] = md[column].diff()
        md[f"{column}_z_1d"] = rolling_z(md[column], window=288, min_periods=100)
    return md


def load_carry() -> pd.DataFrame:
    if not CARRY_PATH.exists():
        raise FileNotFoundError(CARRY_PATH)
    carry = pd.read_csv(CARRY_PATH, parse_dates=["ts"]).sort_values(["root", "ts"])
    carry["ts"] = pd.to_datetime(carry["ts"], utc=True)
    return carry


def load_costs() -> pd.Series:
    if COST_PATH.exists():
        costs = pd.read_csv(COST_PATH)
        return costs.set_index("root")["per_side_cost_bps"].reindex(ROOTS).astype(float)
    return pd.Series({"GC": 0.6, "SI": 1.9, "HG": 0.8, "PL": 2.6, "PA": 5.6})


def build_root_features(
    root: str,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    md: pd.DataFrame,
    carry: pd.DataFrame,
) -> pd.DataFrame:
    root_carry = carry[carry["root"] == root].set_index("ts").sort_index()
    frame = pd.DataFrame(index=root_carry.index)
    frame["carry_level"] = root_carry["annualized_log_carry_pct"]
    frame["months_from_anchor"] = root_carry["months_from_anchor"]
    frame["anchor_volume_5m"] = np.log1p(root_carry["anchor_volume_5m"])
    frame["far_volume_5m"] = np.log1p(root_carry["far_volume_5m"])
    for lag in [1, 3, 12, 48, 288]:
        frame[f"carry_change_{lag}"] = frame["carry_level"].diff(lag)
    frame["carry_z_1d"] = rolling_z(frame["carry_level"], window=288, min_periods=100)
    frame["carry_z_5d"] = rolling_z(frame["carry_level"], window=288 * 5, min_periods=400)

    aligned_returns = returns.reindex(frame.index).ffill()
    aligned_prices = prices.reindex(frame.index).ffill()
    for candidate in ROOTS:
        frame[f"ret_{candidate}_1"] = aligned_returns[candidate]
        frame[f"ret_{candidate}_3"] = aligned_prices[candidate].diff(3)
        frame[f"ret_{candidate}_12"] = aligned_prices[candidate].diff(12)
        frame[f"ret_{candidate}_48"] = aligned_prices[candidate].diff(48)
    frame["own_vol_1h"] = aligned_returns[root].rolling(12, min_periods=6).std()
    frame["own_vol_1d"] = aligned_returns[root].rolling(288, min_periods=80).std()

    md_aligned = md.reindex(frame.index).ffill(limit=3)
    md_columns = [
        "md_signed_flow_ewma",
        "md_notional_share_ewma",
        "md_signed_flow_ewma_change_1",
        "md_notional_share_ewma_change_1",
        "md_signed_flow_ewma_z_1d",
        "md_notional_share_ewma_z_1d",
        "complex_signed_notional_ratio",
        "log_complex_trades",
        "log_complex_notional",
    ]
    for column in md_columns:
        frame[column] = md_aligned[column]
    for candidate in ROOTS:
        frame[f"signed_share_{candidate}"] = md_aligned[f"signed_share_{candidate}"]
        frame[f"notional_share_{candidate}"] = md_aligned[f"notional_share_{candidate}"]
    frame["own_signed_share"] = md_aligned[f"signed_share_{root}"]
    frame["own_notional_share"] = md_aligned[f"notional_share_{root}"]

    ts = frame.index
    hour = ts.hour + ts.minute / 60.0
    dow = ts.dayofweek
    frame["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    frame["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    frame["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    frame["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)

    for horizon_name, steps in HORIZONS.items():
        frame[f"target_{horizon_name}"] = aligned_prices[root].shift(-steps) - aligned_prices[root]

    frame.insert(0, "root", root)
    frame.insert(1, "ts", frame.index)
    return frame.reset_index(drop=True).replace([np.inf, -np.inf], np.nan)


def rolling_z(series: pd.Series, *, window: int, min_periods: int) -> pd.Series:
    mean = series.rolling(window, min_periods=min_periods).mean().shift(1)
    std = series.rolling(window, min_periods=min_periods).std(ddof=1).shift(1)
    return (series - mean) / std.replace(0.0, np.nan)


def walk_forward_predictions(
    feature_frames: dict[str, pd.DataFrame], prices: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_frames = []
    diagnostic_rows = []
    feature_cols = [
        column
        for column in feature_frames[ROOTS[0]].columns
        if column not in {"root", "ts"} and not column.startswith("target_")
    ]
    start = max(frame["ts"].min() for frame in feature_frames.values()) + TRAIN_LOOKBACK
    end = min(prices.index.max(), max(frame["ts"].max() for frame in feature_frames.values()))
    split_starts = pd.date_range(start.floor("D"), end, freq=TEST_WINDOW)

    for root in ROOTS:
        frame = feature_frames[root].copy()
        for horizon_name, steps in HORIZONS.items():
            target = f"target_{horizon_name}"
            horizon_delta = pd.Timedelta(minutes=5 * steps)
            for test_start in split_starts:
                test_end = min(test_start + TEST_WINDOW, end)
                train_start = test_start - TRAIN_LOOKBACK
                train_end = test_start - horizon_delta
                train_mask = (
                    (frame["ts"] >= train_start) & (frame["ts"] < train_end) & frame[target].notna()
                )
                test_mask = (frame["ts"] >= test_start) & (frame["ts"] < test_end)
                train = frame.loc[train_mask, [*feature_cols, target]].dropna()
                test = frame.loc[test_mask, ["root", "ts", *feature_cols]].dropna()
                if len(train) < MIN_TRAIN_ROWS or len(test) == 0:
                    continue

                model = HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=0.05,
                    max_iter=120,
                    max_leaf_nodes=31,
                    min_samples_leaf=50,
                    l2_regularization=0.05,
                    random_state=17,
                )
                model.fit(train[feature_cols], train[target])
                train_pred = model.predict(train[feature_cols])
                test_pred = model.predict(test[feature_cols])
                train_r2 = r2_score(train[target], train_pred)
                pred_abs = pd.Series(np.abs(train_pred))
                thresholds = {
                    f"q{int(q * 100)}": float(pred_abs.quantile(q)) for q in THRESHOLD_QUANTILES
                }
                diagnostic_rows.append(
                    {
                        "root": root,
                        "horizon": horizon_name,
                        "test_start": test_start,
                        "test_end": test_end,
                        "train_rows": len(train),
                        "test_rows": len(test),
                        "train_target_mean": train[target].mean(),
                        "train_target_std": train[target].std(),
                        "train_pred_std": np.std(train_pred),
                        "test_pred_std": np.std(test_pred),
                        "train_r2": train_r2,
                        **{f"threshold_{key}": value for key, value in thresholds.items()},
                    }
                )
                pred = test[["root", "ts"]].copy()
                pred["horizon"] = horizon_name
                pred["prediction"] = test_pred
                for key, value in thresholds.items():
                    pred[f"threshold_{key}"] = value
                prediction_frames.append(pred)

    if not prediction_frames:
        raise RuntimeError("No predictions were generated.")
    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["horizon", "root", "ts"]
    )
    diagnostics = pd.DataFrame(diagnostic_rows)
    return predictions, diagnostics


def run_strategy_variants(
    predictions: pd.DataFrame, returns: pd.DataFrame, costs_bps: pd.Series
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    strategy_frames = {}
    metric_rows = []
    for horizon in HORIZONS:
        pred_h = predictions[predictions["horizon"] == horizon]
        for threshold_label in [f"q{int(q * 100)}" for q in THRESHOLD_QUANTILES]:
            positions = prediction_positions(pred_h, threshold_label, returns.index)
            frame = returns_frame(positions, returns, costs_bps)
            variant = f"horizon_{horizon}_{threshold_label}"
            frame.insert(0, "variant", variant)
            strategy_frames[variant] = frame.reset_index(names="ts")
            metrics = metrics_for_returns(frame, variant)
            metric_rows.append(metrics)
    return strategy_frames, pd.DataFrame(metric_rows).sort_values(
        ["net_return", "gross_to_cost"], ascending=False
    )


def prediction_positions(
    pred_h: pd.DataFrame, threshold_label: str, index: pd.DatetimeIndex
) -> pd.DataFrame:
    columns = []
    for root in ROOTS:
        root_pred = pred_h[pred_h["root"] == root].set_index("ts").sort_index()
        threshold = root_pred[f"threshold_{threshold_label}"]
        signal = pd.Series(0.0, index=root_pred.index)
        signal[root_pred["prediction"] > threshold] = 1.0
        signal[root_pred["prediction"] < -threshold] = -1.0
        columns.append(signal.reindex(index).ffill().fillna(0.0).rename(root))
    positions = pd.concat(columns, axis=1)
    return positions.shift(1).fillna(0.0)


def returns_frame(
    positions: pd.DataFrame, returns: pd.DataFrame, costs_bps: pd.Series
) -> pd.DataFrame:
    common = positions.index.intersection(returns.index)
    positions = positions.reindex(common).fillna(0.0)
    returns = returns.reindex(common).fillna(0.0)
    turnover = positions.diff().abs().fillna(positions.abs())
    gross_by_root = positions * returns
    cost_by_root = turnover.mul(costs_bps, axis=1) / 10_000.0
    gross = gross_by_root.sum(axis=1) / len(ROOTS)
    cost = cost_by_root.sum(axis=1) / len(ROOTS)
    frame = pd.DataFrame(
        {
            "gross_return": gross,
            "cost_return": cost,
            "net_return": gross - cost,
            "turnover": turnover.sum(axis=1) / len(ROOTS),
            "gross_exposure": positions.abs().sum(axis=1) / len(ROOTS),
        },
        index=common,
    )
    for root in ROOTS:
        frame[f"pos_{root}"] = positions[root]
    return frame


def metrics_for_returns(frame: pd.DataFrame, variant: str) -> dict[str, float | str]:
    net = frame["net_return"]
    gross = frame["gross_return"]
    cost = frame["cost_return"]
    equity = net.cumsum()
    drawdown = equity - equity.cummax()
    vol = net.std(ddof=1)
    total_cost = cost.sum()
    return {
        "variant": variant,
        "gross_return": gross.sum(),
        "cost_return": total_cost,
        "net_return": net.sum(),
        "ann_return": net.mean() * PERIODS_PER_YEAR,
        "ann_vol": vol * np.sqrt(PERIODS_PER_YEAR),
        "sharpe": (net.mean() / vol) * np.sqrt(PERIODS_PER_YEAR) if vol > 0 else np.nan,
        "tstat": net.mean() / (vol / np.sqrt(len(net))) if vol > 0 else np.nan,
        "max_drawdown": drawdown.min(),
        "hit_rate": (net > 0).mean(),
        "mean_gross_exposure": frame["gross_exposure"].mean(),
        "mean_turnover": frame["turnover"].mean(),
        "gross_to_cost": gross.sum() / total_cost if total_cost > 0 else np.nan,
        "bars": len(frame),
    }


def monthly_returns(frame: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        frame.set_index("ts").resample("ME")[["gross_return", "cost_return", "net_return"]].sum()
    )
    monthly["month"] = monthly.index.strftime("%Y-%m")
    return monthly.reset_index(drop=True)[["month", "gross_return", "cost_return", "net_return"]]


def plot_equity(frame: pd.DataFrame, path: Path, *, title: str) -> None:
    data = frame.set_index("ts")
    equity = data["net_return"].cumsum()
    gross = data["gross_return"].cumsum()
    drawdown = equity - equity.cummax()
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    axes[0].plot(gross.index, gross, label="gross", color="#6f7f8f")
    axes[0].plot(equity.index, equity, label="net", color="#1f2933")
    axes[0].set_title(title)
    axes[0].set_ylabel("cumulative log return")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)
    axes[1].fill_between(drawdown.index, drawdown, 0.0, color="#c43d3d", alpha=0.35)
    axes[1].set_ylabel("drawdown")
    axes[1].grid(True, alpha=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report(summary: dict[str, Any], metrics: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    best_net = summary["best_net"]
    best_gross = summary["best_gross"]
    top = metrics.sort_values(["net_return", "gross_to_cost"], ascending=False).head(10)
    diag = (
        diagnostics.groupby(["root", "horizon"])
        .agg(
            models=("train_rows", "count"),
            train_rows=("train_rows", "median"),
            test_rows=("test_rows", "median"),
            train_r2=("train_r2", "median"),
            test_pred_std=("test_pred_std", "median"),
        )
        .reset_index()
    )
    text = [
        "# HYP-0025 Metals GBDT Carry/MD Backtest",
        "",
        f"Completed at `{summary['completed_at']}`.",
        "",
        "## Design",
        "",
        (
            "- Features: 5m annualized carry level/changes, 5m trade-flow MD, "
            "signed/notional flow shares, lagged returns, volatility, and session features."
        ),
        "- Model: per-root `HistGradientBoostingRegressor`.",
        (
            "- Validation: 365-day rolling train window, 90-day walk-forward test windows, "
            "horizon embargo between train and test labels."
        ),
        (
            "- Trading: sign of predicted forward return, thresholded by in-train absolute "
            "prediction quantile; position enters next 5m bar."
        ),
        "- Costs: MBP1 per-side estimates divided across equal-root capital.",
        "",
        "## Best Variants",
        "",
        (
            f"- Best net: `{best_net['variant']}`, net `{best_net['net_return']:.4f}`, "
            f"gross `{best_net['gross_return']:.4f}`, cost `{best_net['cost_return']:.4f}`, "
            f"Sharpe `{best_net['sharpe']:.2f}`."
        ),
        (
            f"- Best gross: `{best_gross['variant']}`, net `{best_gross['net_return']:.4f}`, "
            f"gross `{best_gross['gross_return']:.4f}`, "
            f"cost `{best_gross['cost_return']:.4f}`, Sharpe `{best_gross['sharpe']:.2f}`."
        ),
        "",
        "## Top Net Metrics",
        "",
        top.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Model Diagnostics",
        "",
        diag.to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(text), encoding="utf-8")


if __name__ == "__main__":
    main()
