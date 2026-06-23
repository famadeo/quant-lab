from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import polars as pl
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

month_codes = "FGHJKMNQUVXZ"
minimum_trading_dates = 2
minimum_fit_observations = 3


@dataclass(frozen=True)
class FitResult:
    beta: float
    tstat: float
    corr: float
    observations: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HYP-0005 vol-clock OFI test.")
    parser.add_argument("config", type=Path)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def utc_timestamp(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def as_datetime(value: str) -> datetime:
    return utc_timestamp(value).to_pydatetime()


def restrict_to_session(df: pl.DataFrame, ts_col: str, start: str, end: str) -> pl.DataFrame:
    start_hour, start_minute = (int(part) for part in start.split(":"))
    end_hour, end_minute = (int(part) for part in end.split(":"))
    lower_bound = start_hour * 60 + start_minute
    upper_bound = end_hour * 60 + end_minute
    minutes = pl.col(ts_col).dt.hour().cast(pl.Int32) * 60 + pl.col(ts_col).dt.minute().cast(
        pl.Int32
    )
    return df.filter((minutes > lower_bound) & (minutes <= upper_bound)).sort(ts_col)


def outright_pattern(root: str) -> str:
    return rf"^{re.escape(root)}[{month_codes}]\d{{1,2}}$"


def add_external_package_path(config: dict[str, Any]) -> None:
    src = Path(config["data"]["databento_asset_browser_src"])
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def load_return_bars(root: str, config: dict[str, Any]) -> pd.DataFrame:
    c5 = importlib.import_module("databento_asset_browser.continuous_5m")
    start = config["data"]["start"]
    end = config["data"]["end"]
    start_dt = as_datetime(start)
    end_dt = as_datetime(end)
    bars = c5.build_continuous_5m(root, start, end, use_cache=True)
    bars = c5.restrict_to_session(bars, config["data"]["rth_start"], config["data"]["rth_end"])
    bars = bars.filter((pl.col("ts") >= start_dt) & (pl.col("ts") < end_dt))
    out = (
        bars.select("ts", pl.col("cont_logret").fill_null(0.0).alias("ret")).sort("ts").to_pandas()
    )
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    return out.dropna(subset=["ret"]).reset_index(drop=True)


def split_dates(return_bars: pd.DataFrame, train_fraction: float) -> tuple[set[date], set[date]]:
    all_dates = sorted(return_bars["ts"].dt.date.unique())
    if len(all_dates) < minimum_trading_dates:
        msg = "Need at least two trading dates for train/test split."
        raise ValueError(msg)
    train_count = math.floor(len(all_dates) * train_fraction)
    train_count = min(max(train_count, 1), len(all_dates) - 1)
    return set(all_dates[:train_count]), set(all_dates[train_count:])


def load_bucketed_trade_flow(
    root: str,
    config: dict[str, Any],
    train_dates: set[date],
) -> tuple[pd.DataFrame, dict[str, float]]:
    raw_dir = Path(config["data"]["raw_trades_dir"])
    path = raw_dir / f"{root}.parquet"
    if not path.exists():
        msg = f"Missing raw trades for {root}: {path}"
        raise FileNotFoundError(msg)

    start_dt = as_datetime(config["data"]["start"])
    end_dt = as_datetime(config["data"]["end"])
    work = (
        pl.read_parquet(path)
        .filter(pl.col("symbol").str.contains(outright_pattern(root)))
        .filter(pl.col("side").is_in(["B", "A"]))
        .filter((pl.col("ts_event") >= start_dt) & (pl.col("ts_event") < end_dt))
        .rename({"ts_event": "ts"})
    )
    work = restrict_to_session(work, "ts", config["data"]["rth_start"], config["data"]["rth_end"])
    train = work.filter(pl.col("ts").dt.date().is_in(list(train_dates)))
    if train.is_empty():
        msg = f"No training trades available for {root}."
        raise ValueError(msg)

    low_q = float(config["trade_size_buckets"]["low_quantile"])
    high_q = float(config["trade_size_buckets"]["high_quantile"])
    low_threshold = float(train["size"].quantile(low_q))
    high_threshold = float(train["size"].quantile(high_q))

    signed_side = (
        pl.when(pl.col("side") == "B")
        .then(1.0)
        .when(pl.col("side") == "A")
        .then(-1.0)
        .otherwise(0.0)
    )
    work = work.with_columns(
        signed_side.alias("side_sign"),
        (pl.col("size") <= low_threshold).alias("is_low"),
        (pl.col("size") >= high_threshold).alias("is_high"),
    ).with_columns((pl.col("side_sign") * pl.col("size")).alias("signed_size"))

    grouped = (
        work.group_by_dynamic(
            "ts",
            every="5m",
            period="5m",
            closed="left",
            label="right",
        )
        .agg(
            pl.col("signed_size").filter(pl.col("is_low")).sum().alias("signed_size_low"),
            pl.col("signed_size").filter(pl.col("is_high")).sum().alias("signed_size_high"),
            pl.col("size").filter(pl.col("is_low")).sum().alias("tot_size_low"),
            pl.col("size").filter(pl.col("is_high")).sum().alias("tot_size_high"),
            pl.col("size").filter(pl.col("is_low")).count().alias("n_trades_low"),
            pl.col("size").filter(pl.col("is_high")).count().alias("n_trades_high"),
        )
        .sort("ts")
        .with_columns(
            pl.col("signed_size_low").fill_null(0.0),
            pl.col("signed_size_high").fill_null(0.0),
            pl.col("tot_size_low").fill_null(0.0),
            pl.col("tot_size_high").fill_null(0.0),
            pl.col("n_trades_low").fill_null(0),
            pl.col("n_trades_high").fill_null(0),
        )
        .with_columns(
            pl.when(pl.col("tot_size_low") > 0)
            .then(pl.col("signed_size_low") / pl.col("tot_size_low"))
            .otherwise(0.0)
            .alias("ofi_low"),
            pl.when(pl.col("tot_size_high") > 0)
            .then(pl.col("signed_size_high") / pl.col("tot_size_high"))
            .otherwise(0.0)
            .alias("ofi_high"),
        )
        .with_columns((pl.col("ofi_high") - pl.col("ofi_low")).alias("ofi_spread"))
    )

    out = grouped.to_pandas()
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    thresholds = {
        "low_size_threshold": low_threshold,
        "high_size_threshold": high_threshold,
        "train_trade_count": float(train.height),
        "all_trade_count": float(work.height),
    }
    return out, thresholds


def join_returns_and_flow(return_bars: pd.DataFrame, flow_bars: pd.DataFrame) -> pd.DataFrame:
    joined = return_bars.merge(flow_bars, on="ts", how="left")
    fill_zero_columns = [
        "signed_size_low",
        "signed_size_high",
        "tot_size_low",
        "tot_size_high",
        "n_trades_low",
        "n_trades_high",
        "ofi_low",
        "ofi_high",
        "ofi_spread",
    ]
    for column in fill_zero_columns:
        if column not in joined:
            joined[column] = 0.0
        joined[column] = joined[column].fillna(0.0)
    joined["date"] = joined["ts"].dt.date
    return joined.sort_values("ts").reset_index(drop=True)


def volatility_threshold(
    bars_5m: pd.DataFrame,
    train_dates: set[date],
    target_avg_5m_bars: float,
) -> float:
    train = bars_5m[bars_5m["date"].isin(train_dates)]
    mean_var = float(np.square(train["ret"].to_numpy(dtype=float)).mean())
    if not np.isfinite(mean_var) or mean_var <= 0:
        msg = "Training realized variance is not positive."
        raise ValueError(msg)
    return mean_var * target_avg_5m_bars


def build_volatility_bars(
    bars_5m: pd.DataFrame,
    var_threshold: float,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for _, day_frame in bars_5m.groupby("date", sort=True):
        accumulator = {
            "ret": 0.0,
            "realized_var": 0.0,
            "signed_size_low": 0.0,
            "signed_size_high": 0.0,
            "tot_size_low": 0.0,
            "tot_size_high": 0.0,
            "n_trades_low": 0.0,
            "n_trades_high": 0.0,
        }
        ts_start: pd.Timestamp | None = None
        n_5m = 0

        for row in day_frame.itertuples(index=False):
            if ts_start is None:
                ts_start = row.ts
            ret = float(row.ret)
            accumulator["ret"] += ret
            accumulator["realized_var"] += ret * ret
            accumulator["signed_size_low"] += float(row.signed_size_low)
            accumulator["signed_size_high"] += float(row.signed_size_high)
            accumulator["tot_size_low"] += float(row.tot_size_low)
            accumulator["tot_size_high"] += float(row.tot_size_high)
            accumulator["n_trades_low"] += float(row.n_trades_low)
            accumulator["n_trades_high"] += float(row.n_trades_high)
            n_5m += 1

            if accumulator["realized_var"] >= var_threshold:
                append_volatility_bar(records, accumulator, ts_start, row.ts, n_5m)
                ts_start = None
                n_5m = 0
                for key in accumulator:
                    accumulator[key] = 0.0

        if ts_start is not None and n_5m > 0:
            append_volatility_bar(records, accumulator, ts_start, day_frame.iloc[-1]["ts"], n_5m)

    if not records:
        return pd.DataFrame()

    bars = pd.DataFrame.from_records(records)
    bars.insert(0, "vol_bar", np.arange(len(bars)))
    bars["date"] = pd.to_datetime(bars["ts_end"], utc=True).dt.date
    return bars


def append_volatility_bar(
    records: list[dict[str, Any]],
    accumulator: dict[str, float],
    ts_start: pd.Timestamp,
    ts_end: pd.Timestamp,
    n_5m: int,
) -> None:
    tot_low = accumulator["tot_size_low"]
    tot_high = accumulator["tot_size_high"]
    ofi_low = accumulator["signed_size_low"] / tot_low if tot_low > 0 else 0.0
    ofi_high = accumulator["signed_size_high"] / tot_high if tot_high > 0 else 0.0
    wall_minutes = (ts_end - ts_start).total_seconds() / 60.0 + 5.0
    records.append(
        {
            "ts_start": ts_start,
            "ts_end": ts_end,
            "n_5m": n_5m,
            "wall_minutes": wall_minutes,
            "ret": accumulator["ret"],
            "realized_var": accumulator["realized_var"],
            "signed_size_low": accumulator["signed_size_low"],
            "signed_size_high": accumulator["signed_size_high"],
            "tot_size_low": tot_low,
            "tot_size_high": tot_high,
            "n_trades_low": int(accumulator["n_trades_low"]),
            "n_trades_high": int(accumulator["n_trades_high"]),
            "ofi_low": ofi_low,
            "ofi_high": ofi_high,
            "ofi_spread": ofi_high - ofi_low,
        }
    )


def fit_slope(x_values: pd.Series, y_values: pd.Series) -> FitResult:
    frame = pd.DataFrame({"x": x_values, "y": y_values}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < minimum_fit_observations or frame["x"].std(ddof=1) == 0:
        return FitResult(beta=0.0, tstat=0.0, corr=0.0, observations=len(frame))

    x = frame["x"].to_numpy(dtype=float)
    y = frame["y"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residuals = y - design @ beta
    dof = len(x) - 2
    sigma_sq = float(residuals @ residuals / dof)
    cov = sigma_sq * np.linalg.inv(design.T @ design)
    se = math.sqrt(max(float(cov[1, 1]), 0.0))
    tstat = float(beta[1] / se) if se > 0 else 0.0
    corr = float(np.corrcoef(x, y)[0, 1]) if len(x) >= minimum_fit_observations else 0.0
    return FitResult(beta=float(beta[1]), tstat=tstat, corr=corr, observations=len(frame))


def evaluate_feature(
    root: str,
    feature: str,
    vol_bars: pd.DataFrame,
    train_dates: set[date],
    test_dates: set[date],
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    work = vol_bars.copy()
    work["next_ret"] = work["ret"].shift(-1)
    work["next_date"] = work["date"].shift(-1)
    train_mask = work["date"].isin(train_dates) & work["next_date"].isin(train_dates)
    fit = fit_slope(work.loc[train_mask, feature], work.loc[train_mask, "next_ret"])

    clip_abs = float(config["strategy"]["feature_clip_abs"])
    cost_rate = float(config["strategy"]["cost_bps_per_unit_turnover"]) / 10_000.0
    direction = float(np.sign(fit.beta))
    work["feature"] = feature
    work["feature_prev"] = work[feature].shift(1).clip(lower=-clip_abs, upper=clip_abs)
    work["position"] = direction * work["feature_prev"]
    work["position"] = work["position"].fillna(0.0)
    work["turnover"] = work["position"].diff().abs().fillna(work["position"].abs())
    work["gross_return"] = work["position"] * work["ret"]
    work["cost_return"] = cost_rate * work["turnover"]
    work["net_return"] = work["gross_return"] - work["cost_return"]
    work["split"] = np.where(work["date"].isin(train_dates), "train", "test")
    test = work[work["date"].isin(test_dates)].copy()

    metric = summarize_returns(root, feature, test, config)
    coefficient = {
        "root": root,
        "feature": feature,
        "train_beta": fit.beta,
        "train_tstat": fit.tstat,
        "train_corr": fit.corr,
        "train_observations": fit.observations,
        "direction": direction,
    }
    return metric, test, coefficient


def summarize_returns(
    root: str,
    feature: str,
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    min_test = int(config["validation"]["min_test_observations"])
    returns = frame["net_return"].to_numpy(dtype=float)
    gross = frame["gross_return"].to_numpy(dtype=float)
    cost = frame["cost_return"].to_numpy(dtype=float)
    turnover = frame["turnover"].to_numpy(dtype=float)
    observations = len(frame)
    std = float(np.std(returns, ddof=1)) if observations > 1 else 0.0
    mean = float(np.mean(returns)) if observations else 0.0
    event_tstat = float(mean / (std / math.sqrt(observations))) if std > 0 else 0.0
    active_fraction = float((frame["position"].abs() > 0).mean()) if observations else 0.0
    unique_dates = max(int(frame["date"].nunique()), 1)
    bars_per_year = observations / unique_dates * 252.0
    sharpe = float(mean / std * math.sqrt(bars_per_year)) if std > 0 else 0.0
    return {
        "root": root,
        "feature": feature,
        "observations": observations,
        "sufficient_test_observations": observations >= min_test,
        "total_gross_return": float(gross.sum()) if observations else 0.0,
        "total_cost_return": float(cost.sum()) if observations else 0.0,
        "total_net_return": float(returns.sum()) if observations else 0.0,
        "mean_net_bps_per_bar": mean * 10_000.0,
        "event_tstat": event_tstat,
        "annualized_sharpe": sharpe,
        "hit_rate": float((returns > 0).mean()) if observations else 0.0,
        "active_fraction": active_fraction,
        "total_turnover": float(turnover.sum()) if observations else 0.0,
    }


def pooled_metrics(root_returns: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    records = []
    for feature, frame in root_returns.groupby("feature", sort=True):
        records.append(summarize_returns("POOLED", feature, frame, config))
    return pd.DataFrame.from_records(records).sort_values("feature").reset_index(drop=True)


def decide(
    root_metrics: pd.DataFrame,
    pooled: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    primary = config["strategy"]["primary_feature"]
    rules = config["decision_rules"]
    pooled_row = pooled[pooled["feature"] == primary].iloc[0]
    primary_roots = root_metrics[root_metrics["feature"] == primary]
    positive_fraction = float((primary_roots["total_net_return"] > 0).mean())
    passes = (
        pooled_row["total_net_return"] > float(rules["min_pooled_net_return"])
        and pooled_row["event_tstat"] >= float(rules["min_pooled_event_tstat"])
        and positive_fraction >= float(rules["min_positive_root_fraction"])
    )
    status = str(rules["pass_status"] if passes else rules["fail_status"])
    return {
        "status": status,
        "passed": bool(passes),
        "primary_feature": primary,
        "positive_root_fraction": positive_fraction,
        "pooled_total_net_return": float(pooled_row["total_net_return"]),
        "pooled_event_tstat": float(pooled_row["event_tstat"]),
        "notes": (
            "Primary low-size OFI validation passed the preregistered thresholds."
            if passes
            else "Primary low-size OFI validation failed the preregistered thresholds."
        ),
    }


def plot_outputs(root_returns: pd.DataFrame, root_metrics: pd.DataFrame, output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 5))
    for feature, frame in root_returns.sort_values("ts_end").groupby("feature", sort=True):
        axis.plot(frame["ts_end"], frame["net_return"].cumsum() * 100.0, label=feature)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_title("Out-of-sample cumulative net return by OFI feature")
    axis.set_ylabel("Cumulative net return (%)")
    axis.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "oos_cumulative_net_by_feature.png", dpi=150)
    plt.close(fig)

    pivot = root_metrics.pivot(index="root", columns="feature", values="total_net_return") * 100.0
    fig, axis = plt.subplots(figsize=(8, 4.5))
    pivot.plot(kind="bar", ax=axis)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_title("Out-of-sample net return by root and OFI feature")
    axis.set_ylabel("Net return (%)")
    axis.legend(title="feature")
    fig.tight_layout()
    fig.savefig(output_dir / "root_net_return_by_feature.png", dpi=150)
    plt.close(fig)


def save_outputs(
    config: dict[str, Any],
    output_dir: Path,
    root_metrics: pd.DataFrame,
    pooled: pd.DataFrame,
    coefficients: pd.DataFrame,
    diagnostics: pd.DataFrame,
    returns: pd.DataFrame,
    vol_bars: pd.DataFrame,
    decision: dict[str, Any],
) -> dict[str, Any]:
    root_metrics_path = output_dir / "root_metrics.csv"
    pooled_metrics_path = output_dir / "pooled_metrics.csv"
    coefficients_path = output_dir / "train_coefficients.csv"
    diagnostics_path = output_dir / "root_diagnostics.csv"
    returns_path = output_dir / "root_bar_returns.parquet"
    vol_bars_path = output_dir / "volatility_bars.parquet"

    root_metrics.to_csv(root_metrics_path, index=False)
    pooled.to_csv(pooled_metrics_path, index=False)
    coefficients.to_csv(coefficients_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)
    returns.to_parquet(returns_path, index=False)
    vol_bars.to_parquet(vol_bars_path, index=False)
    plot_outputs(returns, root_metrics, output_dir)

    return {
        "experiment_id": config["experiment"]["id"],
        "title": config["experiment"]["title"],
        "completed_at": datetime.now().astimezone().isoformat(),
        "decision": decision,
        "roots": list(config["universe"]["roots"]),
        "primary_feature": config["strategy"]["primary_feature"],
        "pooled_metrics": pooled.to_dict(orient="records"),
        "artifacts": {
            "root_metrics": str(root_metrics_path),
            "pooled_metrics": str(pooled_metrics_path),
            "train_coefficients": str(coefficients_path),
            "root_diagnostics": str(diagnostics_path),
            "root_bar_returns": str(returns_path),
            "volatility_bars": str(vol_bars_path),
            "oos_cumulative_plot": str(output_dir / "oos_cumulative_net_by_feature.png"),
            "root_net_return_plot": str(output_dir / "root_net_return_by_feature.png"),
        },
    }


def run(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    add_external_package_path(config)
    output_dir = config_path.parent

    roots = list(config["universe"]["roots"])
    features = list(config["strategy"]["features"])
    train_fraction = float(config["validation"]["train_fraction_dates"])
    target_avg_bars = float(config["volatility_clock"]["target_avg_5m_bars"])

    root_metrics: list[dict[str, Any]] = []
    coefficients: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    all_returns: list[pd.DataFrame] = []
    all_vol_bars: list[pd.DataFrame] = []

    for root in roots:
        return_bars = load_return_bars(root, config)
        train_dates, test_dates = split_dates(return_bars, train_fraction)
        flow_bars, thresholds = load_bucketed_trade_flow(root, config, train_dates)
        bars_5m = join_returns_and_flow(return_bars, flow_bars)
        var_threshold = volatility_threshold(bars_5m, train_dates, target_avg_bars)
        vol_bars = build_volatility_bars(bars_5m, var_threshold)
        if vol_bars.empty:
            msg = f"No volatility bars built for {root}."
            raise ValueError(msg)

        vol_bars.insert(0, "root", root)
        all_vol_bars.append(vol_bars.copy())
        diagnostics.append(
            {
                "root": root,
                "train_dates": len(train_dates),
                "test_dates": len(test_dates),
                "return_bars_5m": len(return_bars),
                "vol_bars": len(vol_bars),
                "train_vol_bars": int(vol_bars["date"].isin(train_dates).sum()),
                "test_vol_bars": int(vol_bars["date"].isin(test_dates).sum()),
                "vol_var_threshold": var_threshold,
                "avg_5m_per_vol_bar": float(vol_bars["n_5m"].mean()),
                **thresholds,
            }
        )

        for feature in features:
            metric, feature_returns, coefficient = evaluate_feature(
                root,
                feature,
                vol_bars,
                train_dates,
                test_dates,
                config,
            )
            root_metrics.append(metric)
            coefficients.append(coefficient)
            feature_returns["root"] = root
            all_returns.append(feature_returns)

    root_metrics_df = pd.DataFrame.from_records(root_metrics).sort_values(["feature", "root"])
    coefficients_df = pd.DataFrame.from_records(coefficients).sort_values(["feature", "root"])
    diagnostics_df = pd.DataFrame.from_records(diagnostics).sort_values("root")
    returns_df = pd.concat(all_returns, ignore_index=True)
    vol_bars_df = pd.concat(all_vol_bars, ignore_index=True)
    pooled_df = pooled_metrics(returns_df, config)
    decision = decide(root_metrics_df, pooled_df, config)
    result = save_outputs(
        config,
        output_dir,
        root_metrics_df,
        pooled_df,
        coefficients_df,
        diagnostics_df,
        returns_df,
        vol_bars_df,
        decision,
    )
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def main() -> None:
    args = parse_args()
    result = run(args.config)
    print(json.dumps(result["decision"], indent=2))


if __name__ == "__main__":
    main()
