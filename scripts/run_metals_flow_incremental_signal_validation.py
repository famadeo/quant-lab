# ruff: noqa: PLR2004
from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantlab.metals_flow.forward import future_returns
from quantlab.metals_flow.strategy import (
    backtest_positions,
    benjamini_hochberg,
    convergence_state_positions,
    demean_and_normalize_positions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate whether corrected metals flow features add predictive value beyond "
            "residual dislocation."
        )
    )
    parser.add_argument("config", type=Path, help="Experiment config YAML.")
    return parser.parse_args()


def main() -> None:
    config_path = parse_args().config
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    out_dir = Path(payload["outputs"]["directory"])
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(exist_ok=True)
    _run(payload, out_dir, plot_dir)


def _run(payload: dict[str, Any], out_dir: Path, plot_dir: Path) -> None:
    source_dir = Path(payload["data"]["source_experiment"])
    roots = tuple(payload["universe"]["roots"])
    horizons = tuple(int(value) for value in payload["research"]["horizons"])
    strategy_cfg = payload["strategy"]

    bars = pd.read_parquet(source_dir / "bars.parquet")
    bars["start_ts"] = pd.to_datetime(bars["start_ts"], utc=True)
    bars["end_ts"] = pd.to_datetime(bars["end_ts"], utc=True)
    returns = pd.read_parquet(source_dir / "returns.parquet").reindex(columns=roots)
    residual_z = pd.read_parquet(source_dir / "residual_z.parquet").reindex(columns=roots)
    anomalies = pd.read_parquet(source_dir / "anomalies.parquet")
    size_disagreement = pd.read_parquet(source_dir / "size_disagreement.parquet")
    price_validity = pd.read_parquet(source_dir / "price_validity.parquet")
    cost_estimates = pd.read_csv(source_dir / "cost_estimates.csv")

    valid_mask = price_validity["valid_price_mask"].fillna(False).astype(bool)
    threshold_window = int(strategy_cfg["threshold_window"])
    threshold_min_periods = int(strategy_cfg["threshold_min_periods"])
    md_threshold = shifted_rolling_quantile(
        anomalies["md_rolling"],
        float(strategy_cfg["md_quantile"]),
        threshold_window,
        threshold_min_periods,
    )
    disagreement_threshold = shifted_rolling_quantile(
        size_disagreement["large_small_l1_distance"],
        float(strategy_cfg["large_small_quantile"]),
        threshold_window,
        threshold_min_periods,
    )
    md_gate = (anomalies["md_rolling"] >= md_threshold) & valid_mask
    size_gate = (
        size_disagreement["large_small_l1_distance"] >= disagreement_threshold
    ) & valid_mask
    combined_flow_gate = md_gate & size_gate
    root_size_masks = root_large_small_masks_walk_forward(
        size_disagreement,
        roots,
        quantile=float(strategy_cfg["large_small_quantile"]),
        window=threshold_window,
        min_periods=threshold_min_periods,
    ).where(valid_mask, False)

    periods_per_year = periods_per_year_from_bars(bars)
    cost_bps = cost_estimates.set_index("root")["per_side_cost_bps"].reindex(roots)
    variant_frames, variant_metrics, variant_positions = run_strategy_variants(
        residual_z=residual_z,
        returns=returns,
        valid_mask=valid_mask,
        md_gate=md_gate,
        size_gate=size_gate,
        combined_flow_gate=combined_flow_gate,
        root_size_masks=root_size_masks,
        cost_bps=cost_bps,
        periods_per_year=periods_per_year,
        strategy_cfg=strategy_cfg,
    )

    future = future_returns(returns, horizons)
    regression_results = run_predictive_regressions(
        residual_z=residual_z,
        anomalies=anomalies,
        size_disagreement=size_disagreement,
        future=future,
        valid_mask=valid_mask,
        roots=roots,
        horizons=horizons,
    )
    regression_results = (
        benjamini_hochberg(regression_results)
        if not regression_results.empty
        else regression_results
    )
    event_study = run_conditioned_event_study(
        residual_z=residual_z,
        future=future,
        valid_mask=valid_mask,
        flow_gate=combined_flow_gate,
        root_flow_masks=root_size_masks,
        roots=roots,
        horizons=horizons,
        entry_z=float(strategy_cfg["entry_z"]),
    )
    portfolio_scatter = run_portfolio_forward_scatter(
        positions=variant_positions,
        future=future,
        roots=roots,
        horizons=horizons,
    )
    root_scatter = run_root_forward_scatter(
        residual_z=residual_z,
        future=future,
        roots=roots,
        horizons=horizons,
        valid_mask=valid_mask,
    )
    gate_diagnostics = gate_summary(
        bars=bars,
        valid_mask=valid_mask,
        md_gate=md_gate,
        size_gate=size_gate,
        combined_flow_gate=combined_flow_gate,
        root_size_masks=root_size_masks,
    )

    write_outputs(
        out_dir=out_dir,
        plot_dir=plot_dir,
        bars=bars,
        returns_by_variant=variant_frames,
        positions_by_variant=variant_positions,
        variant_metrics=variant_metrics,
        regression_results=regression_results,
        event_study=event_study,
        portfolio_scatter=portfolio_scatter,
        root_scatter=root_scatter,
        gate_diagnostics=gate_diagnostics,
        residual_z=residual_z,
        anomalies=anomalies,
        size_disagreement=size_disagreement,
        future=future,
        roots=roots,
        primary_horizon=int(payload["research"].get("primary_horizon", 20)),
        payload=payload,
        source_dir=source_dir,
    )


def run_strategy_variants(
    *,
    residual_z: pd.DataFrame,
    returns: pd.DataFrame,
    valid_mask: pd.Series,
    md_gate: pd.Series,
    size_gate: pd.Series,
    combined_flow_gate: pd.Series,
    root_size_masks: pd.DataFrame,
    cost_bps: pd.Series,
    periods_per_year: float,
    strategy_cfg: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, pd.DataFrame]]:
    all_roots = pd.DataFrame(True, index=residual_z.index, columns=residual_z.columns)
    all_roots = all_roots.where(valid_mask, False)
    variant_specs = {
        "rv_only": (valid_mask, all_roots),
        "rv_md_only": (md_gate, all_roots),
        "rv_large_small_l1_only": (size_gate, all_roots),
        "rv_md_and_l1": (combined_flow_gate, all_roots),
        "rv_md_l1_root_filtered": (combined_flow_gate, root_size_masks),
    }

    frames: dict[str, pd.DataFrame] = {}
    positions_by_variant: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for name, (entry_mask, root_masks) in variant_specs.items():
        raw_positions = convergence_state_positions(
            residual_z,
            entry_mask,
            root_masks,
            entry_z=float(strategy_cfg["entry_z"]),
            exit_z=float(strategy_cfg["exit_z"]),
            stop_z=float(strategy_cfg["stop_z"]),
        )
        raw_positions = raw_positions.where(valid_mask, 0.0, axis=0)
        positions = demean_and_normalize_positions(raw_positions)
        frame, metrics = backtest_positions(
            positions,
            returns,
            cost_bps,
            periods_per_year=periods_per_year,
        )
        frame["variant"] = name
        frames[name] = frame
        positions_by_variant[name] = positions
        rows.append({"variant": name, **metrics.to_dict()})

    return frames, pd.DataFrame(rows), positions_by_variant


def run_predictive_regressions(
    *,
    residual_z: pd.DataFrame,
    anomalies: pd.DataFrame,
    size_disagreement: pd.DataFrame,
    future: pd.DataFrame,
    valid_mask: pd.Series,
    roots: tuple[str, ...],
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for root in roots:
        root_flow = size_disagreement[f"{root}_large_minus_small_share"]
        base = pd.DataFrame(
            {
                "rv_reversion_signal": -residual_z[root],
                "md_rolling": anomalies["md_rolling"],
                "large_small_l1_distance": size_disagreement["large_small_l1_distance"],
                "root_large_minus_small_share": root_flow,
            }
        ).where(valid_mask, np.nan)
        standardized = standardize_frame(base)
        standardized["rv_x_md"] = (
            standardized["rv_reversion_signal"] * standardized["md_rolling"]
        )
        standardized["rv_x_l1"] = (
            standardized["rv_reversion_signal"] * standardized["large_small_l1_distance"]
        )
        standardized["rv_x_root_flow"] = (
            standardized["rv_reversion_signal"] * standardized["root_large_minus_small_share"]
        )
        for horizon in horizons:
            target = future[(root, horizon)]
            frame = pd.concat([target.rename("future_return"), standardized], axis=1).dropna()
            if len(frame) < 500:
                continue
            x = sm.add_constant(frame.drop(columns="future_return"), has_constant="add")
            y = frame["future_return"]
            fit = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": max(1, horizon)})
            for term in x.columns:
                if term == "const":
                    continue
                rows.append(
                    {
                        "root": root,
                        "horizon": horizon,
                        "term": term,
                        "coef": float(fit.params[term]),
                        "coef_bps": float(fit.params[term] * 10_000.0),
                        "tstat": float(fit.tvalues[term]),
                        "pvalue": float(fit.pvalues[term]),
                        "rsquared": float(fit.rsquared),
                        "observations": int(fit.nobs),
                    }
                )
    return pd.DataFrame(rows)


def run_conditioned_event_study(
    *,
    residual_z: pd.DataFrame,
    future: pd.DataFrame,
    valid_mask: pd.Series,
    flow_gate: pd.Series,
    root_flow_masks: pd.DataFrame,
    roots: tuple[str, ...],
    horizons: tuple[int, ...],
    entry_z: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for root in roots:
        signal = -residual_z[root]
        residual_extreme = residual_z[root].abs() >= entry_z
        conditions = {
            "rv_extreme_no_flow": residual_extreme & valid_mask & ~flow_gate,
            "rv_extreme_flow_gate": residual_extreme & valid_mask & flow_gate,
            "rv_extreme_flow_and_root": residual_extreme
            & valid_mask
            & flow_gate
            & root_flow_masks[root],
        }
        for horizon in horizons:
            signed_future = future[(root, horizon)] * np.sign(signal)
            for condition, mask in conditions.items():
                stats = return_stats(signed_future[mask])
                rows.append(
                    {
                        "root": root,
                        "horizon": horizon,
                        "condition": condition,
                        **stats,
                    }
                )
    return pd.DataFrame(rows)


def run_portfolio_forward_scatter(
    *,
    positions: dict[str, pd.DataFrame],
    future: pd.DataFrame,
    roots: tuple[str, ...],
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant, position_frame in positions.items():
        for horizon in horizons:
            future_by_root = future.xs(horizon, axis=1, level="horizon").reindex(columns=roots)
            expected_direction_return = (
                position_frame.reindex(columns=roots) * future_by_root
            ).sum(axis=1)
            signal_strength = position_frame.abs().sum(axis=1)
            frame = pd.DataFrame(
                {
                    "signal_strength": signal_strength,
                    "forward_strategy_return": expected_direction_return,
                }
            ).replace([np.inf, -np.inf], np.nan)
            frame = frame.loc[frame["signal_strength"] > 0].dropna()
            rows.append(
                {
                    "variant": variant,
                    "horizon": horizon,
                    **correlation_stats(frame["signal_strength"], frame["forward_strategy_return"]),
                    **return_stats(frame["forward_strategy_return"]),
                }
            )
    return pd.DataFrame(rows)


def run_root_forward_scatter(
    *,
    residual_z: pd.DataFrame,
    future: pd.DataFrame,
    roots: tuple[str, ...],
    horizons: tuple[int, ...],
    valid_mask: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for root in roots:
        signal = -residual_z[root]
        signal_abs = signal.abs().where(valid_mask)
        for horizon in horizons:
            signed_future = future[(root, horizon)] * np.sign(signal)
            frame = pd.DataFrame(
                {"signal_abs": signal_abs, "signed_future_return": signed_future}
            ).dropna()
            rows.append(
                {
                    "root": root,
                    "horizon": horizon,
                    **correlation_stats(frame["signal_abs"], frame["signed_future_return"]),
                    **return_stats(frame["signed_future_return"]),
                }
            )
    return pd.DataFrame(rows)


def gate_summary(
    *,
    bars: pd.DataFrame,
    valid_mask: pd.Series,
    md_gate: pd.Series,
    size_gate: pd.Series,
    combined_flow_gate: pd.Series,
    root_size_masks: pd.DataFrame,
) -> pd.DataFrame:
    root_mask_share = root_size_masks.mean(axis=0).rename("root_size_mask_fraction")
    rows = [
        {"metric": "bars", "value": float(len(bars))},
        {"metric": "valid_price_fraction", "value": float(valid_mask.mean())},
        {"metric": "md_gate_fraction", "value": float(md_gate.mean())},
        {"metric": "size_gate_fraction", "value": float(size_gate.mean())},
        {"metric": "combined_flow_gate_fraction", "value": float(combined_flow_gate.mean())},
    ]
    rows.extend(
        {"metric": f"{root}_root_size_mask_fraction", "value": float(value)}
        for root, value in root_mask_share.items()
    )
    return pd.DataFrame(rows)


def write_outputs(
    *,
    out_dir: Path,
    plot_dir: Path,
    bars: pd.DataFrame,
    returns_by_variant: dict[str, pd.DataFrame],
    positions_by_variant: dict[str, pd.DataFrame],
    variant_metrics: pd.DataFrame,
    regression_results: pd.DataFrame,
    event_study: pd.DataFrame,
    portfolio_scatter: pd.DataFrame,
    root_scatter: pd.DataFrame,
    gate_diagnostics: pd.DataFrame,
    residual_z: pd.DataFrame,
    anomalies: pd.DataFrame,
    size_disagreement: pd.DataFrame,
    future: pd.DataFrame,
    roots: tuple[str, ...],
    primary_horizon: int,
    payload: dict[str, Any],
    source_dir: Path,
) -> None:
    variant_metrics.to_csv(out_dir / "strategy_variant_metrics.csv", index=False)
    regression_results.to_csv(out_dir / "predictive_regressions.csv", index=False)
    event_study.to_csv(out_dir / "conditioned_event_study.csv", index=False)
    portfolio_scatter.to_csv(out_dir / "portfolio_forward_scatter.csv", index=False)
    root_scatter.to_csv(out_dir / "root_forward_scatter.csv", index=False)
    gate_diagnostics.to_csv(out_dir / "gate_diagnostics.csv", index=False)

    for variant, frame in returns_by_variant.items():
        frame.to_parquet(out_dir / f"{variant}_returns.parquet", index=False)
    for variant, frame in positions_by_variant.items():
        frame.to_parquet(out_dir / f"{variant}_positions.parquet", index=False)

    timestamps = pd.to_datetime(bars["end_ts"], utc=True)
    plot_variant_equity(plot_dir, timestamps, returns_by_variant)
    plot_variant_drawdowns(plot_dir, timestamps, returns_by_variant)
    plot_regression_tstats(plot_dir, regression_results, primary_horizon)
    plot_event_study(plot_dir, event_study, primary_horizon)
    plot_portfolio_scatter(
        plot_dir,
        positions_by_variant,
        future,
        roots,
        primary_horizon,
    )
    plot_root_scatter(plot_dir, residual_z, future, roots, primary_horizon)
    plot_flow_gate_diagnostics(
        plot_dir,
        timestamps,
        anomalies,
        size_disagreement,
    )

    summary = build_summary(
        payload=payload,
        source_dir=source_dir,
        variant_metrics=variant_metrics,
        regression_results=regression_results,
        event_study=event_study,
        portfolio_scatter=portfolio_scatter,
        root_scatter=root_scatter,
        gate_diagnostics=gate_diagnostics,
        primary_horizon=primary_horizon,
    )
    (out_dir / "results.json").write_text(
        json.dumps(json_safe(summary), indent=2),
        encoding="utf-8",
    )


def plot_variant_equity(
    plot_dir: Path,
    timestamps: pd.Series,
    returns_by_variant: dict[str, pd.DataFrame],
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for variant, frame in returns_by_variant.items():
        ax.plot(timestamps, frame["net_return"].cumsum() * 100.0, label=variant)
    ax.set_title("Corrected residual strategy variants: net cumulative return")
    ax.set_ylabel("Cumulative log return (%)")
    ax.legend(loc="best", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(plot_dir / "variant_net_equity.png", dpi=160)
    plt.close(fig)


def plot_variant_drawdowns(
    plot_dir: Path,
    timestamps: pd.Series,
    returns_by_variant: dict[str, pd.DataFrame],
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for variant, frame in returns_by_variant.items():
        equity = frame["net_return"].cumsum()
        drawdown = equity - equity.cummax()
        ax.plot(timestamps, drawdown * 100.0, label=variant)
    ax.set_title("Corrected residual strategy variants: net drawdown")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="best", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(plot_dir / "variant_net_drawdowns.png", dpi=160)
    plt.close(fig)


def plot_regression_tstats(
    plot_dir: Path,
    regression_results: pd.DataFrame,
    primary_horizon: int,
) -> None:
    data = regression_results.loc[regression_results["horizon"] == primary_horizon].copy()
    if data.empty:
        return
    pivot = data.pivot_table(index="term", columns="root", values="tstat", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 5))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
    ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns)
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
    ax.set_title(f"HAC t-stats by term, horizon {primary_horizon} bars")
    fig.colorbar(image, ax=ax, label="t-stat")
    fig.tight_layout()
    fig.savefig(plot_dir / f"predictive_regression_tstats_h{primary_horizon}.png", dpi=160)
    plt.close(fig)


def plot_event_study(plot_dir: Path, event_study: pd.DataFrame, primary_horizon: int) -> None:
    data = event_study.loc[event_study["horizon"] == primary_horizon].copy()
    if data.empty:
        return
    pivot = data.pivot_table(index="root", columns="condition", values="mean_bps", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(f"Signed forward reversion return by condition, horizon {primary_horizon}")
    ax.set_ylabel("Mean signed future return (bps)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / f"conditioned_event_returns_h{primary_horizon}.png", dpi=160)
    plt.close(fig)


def plot_portfolio_scatter(
    plot_dir: Path,
    positions_by_variant: dict[str, pd.DataFrame],
    future: pd.DataFrame,
    roots: tuple[str, ...],
    primary_horizon: int,
) -> None:
    variants = list(positions_by_variant)
    cols = 2
    rows = math.ceil(len(variants) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), squeeze=False)
    future_by_root = future.xs(primary_horizon, axis=1, level="horizon").reindex(columns=roots)
    for ax, variant in zip(axes.ravel(), variants, strict=False):
        positions = positions_by_variant[variant].reindex(columns=roots)
        signal_strength = positions.abs().sum(axis=1)
        forward_return = (positions * future_by_root).sum(axis=1)
        frame = pd.DataFrame({"signal": signal_strength, "future": forward_return})
        frame = frame.loc[frame["signal"] > 0].replace([np.inf, -np.inf], np.nan).dropna()
        if len(frame) > 20_000:
            frame = frame.sample(n=20_000, random_state=7)
        ax.scatter(frame["signal"], frame["future"] * 10_000.0, s=4, alpha=0.15)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(variant)
        ax.set_xlabel("Gross signal strength")
        ax.set_ylabel(f"Forward strategy return, h={primary_horizon} (bps)")
    for ax in axes.ravel()[len(variants) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(plot_dir / f"portfolio_signal_forward_scatter_h{primary_horizon}.png", dpi=160)
    plt.close(fig)


def plot_root_scatter(
    plot_dir: Path,
    residual_z: pd.DataFrame,
    future: pd.DataFrame,
    roots: tuple[str, ...],
    primary_horizon: int,
) -> None:
    fig, axes = plt.subplots(len(roots), 1, figsize=(10, 3.2 * len(roots)), squeeze=False)
    for ax, root in zip(axes.ravel(), roots, strict=True):
        signal = -residual_z[root]
        signed_future = future[(root, primary_horizon)] * np.sign(signal)
        frame = pd.DataFrame({"signal_abs": signal.abs(), "future": signed_future})
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        if len(frame) > 20_000:
            frame = frame.sample(n=20_000, random_state=7)
        ax.scatter(frame["signal_abs"], frame["future"] * 10_000.0, s=3, alpha=0.12)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(root)
        ax.set_xlabel("|residual reversion signal|")
        ax.set_ylabel(f"Signed future return, h={primary_horizon} (bps)")
    fig.tight_layout()
    fig.savefig(plot_dir / f"root_signal_forward_scatter_h{primary_horizon}.png", dpi=160)
    plt.close(fig)


def plot_flow_gate_diagnostics(
    plot_dir: Path,
    timestamps: pd.Series,
    anomalies: pd.DataFrame,
    size_disagreement: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(timestamps, anomalies["md_rolling"], linewidth=0.8)
    axes[0].set_title("Rolling Mahalanobis flow distance")
    axes[0].set_ylabel("MD")
    axes[1].plot(timestamps, size_disagreement["large_small_l1_distance"], linewidth=0.8)
    axes[1].set_title("Large-small contribution-vector distance")
    axes[1].set_ylabel("L1 distance")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(plot_dir / "flow_gate_diagnostics.png", dpi=160)
    plt.close(fig)


def build_summary(
    *,
    payload: dict[str, Any],
    source_dir: Path,
    variant_metrics: pd.DataFrame,
    regression_results: pd.DataFrame,
    event_study: pd.DataFrame,
    portfolio_scatter: pd.DataFrame,
    root_scatter: pd.DataFrame,
    gate_diagnostics: pd.DataFrame,
    primary_horizon: int,
) -> dict[str, Any]:
    best_net = variant_metrics.sort_values("net_return", ascending=False).head(1)
    flow_terms = {
        "md_rolling",
        "large_small_l1_distance",
        "root_large_minus_small_share",
        "rv_x_md",
        "rv_x_l1",
        "rv_x_root_flow",
    }
    flow_regressions = regression_results.loc[regression_results["term"].isin(flow_terms)].copy()
    primary_events = event_study.loc[event_study["horizon"] == primary_horizon].copy()
    return {
        "completed_at": datetime.now(UTC).isoformat(),
        "experiment_id": payload["experiment"]["id"],
        "source_experiment": str(source_dir),
        "status": "incremental_signal_validation",
        "best_net_variant": best_net.to_dict(orient="records"),
        "variant_metrics": variant_metrics.to_dict(orient="records"),
        "gate_diagnostics": gate_diagnostics.to_dict(orient="records"),
        "flow_regression_tests": len(flow_regressions),
        "flow_regression_significant_5pct_uncorrected": int(
            (flow_regressions["pvalue"] <= 0.05).sum()
        )
        if not flow_regressions.empty
        else 0,
        "flow_regression_significant_5pct_fdr": int(
            (flow_regressions["bh_qvalue"] <= 0.05).sum()
        )
        if "bh_qvalue" in flow_regressions
        else 0,
        "max_abs_flow_tstat": float(flow_regressions["tstat"].abs().max())
        if not flow_regressions.empty
        else np.nan,
        "primary_horizon_event_study": primary_events.to_dict(orient="records"),
        "portfolio_forward_scatter": portfolio_scatter.to_dict(orient="records"),
        "root_forward_scatter": root_scatter.to_dict(orient="records"),
    }


def shifted_rolling_quantile(
    series: pd.Series,
    quantile: float,
    window: int,
    min_periods: int,
) -> pd.Series:
    return series.shift(1).rolling(window=window, min_periods=min_periods).quantile(quantile)


def root_large_small_masks_walk_forward(
    disagreement: pd.DataFrame,
    roots: tuple[str, ...],
    *,
    quantile: float,
    window: int,
    min_periods: int,
) -> pd.DataFrame:
    masks = pd.DataFrame(False, index=disagreement.index, columns=roots)
    for root in roots:
        column = f"{root}_large_minus_small_share"
        threshold = shifted_rolling_quantile(disagreement[column], quantile, window, min_periods)
        masks[root] = disagreement[column] >= threshold
    return masks


def periods_per_year_from_bars(bars: pd.DataFrame) -> float:
    elapsed_years = (
        pd.to_datetime(bars["end_ts"], utc=True).max()
        - pd.to_datetime(bars["start_ts"], utc=True).min()
    ).total_seconds() / (365.25 * 86_400.0)
    return len(bars) / elapsed_years if elapsed_years > 0 else float(len(bars))


def standardize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.replace([np.inf, -np.inf], np.nan).astype(float)
    mean = out.mean(axis=0)
    std = out.std(axis=0, ddof=1).replace(0.0, np.nan)
    return (out - mean) / std


def return_stats(values: pd.Series) -> dict[str, float | int]:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    observations = len(clean)
    if observations == 0:
        return {
            "observations": 0,
            "mean_bps": np.nan,
            "median_bps": np.nan,
            "vol_bps": np.nan,
            "tstat": np.nan,
            "hit_rate": np.nan,
        }
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if observations > 1 else 0.0
    return {
        "observations": observations,
        "mean_bps": mean * 10_000.0,
        "median_bps": float(clean.median()) * 10_000.0,
        "vol_bps": std * 10_000.0,
        "tstat": mean / (std / math.sqrt(observations)) if std > 0 else np.nan,
        "hit_rate": float((clean > 0.0).mean()),
    }


def correlation_stats(x_values: pd.Series, y_values: pd.Series) -> dict[str, float | int]:
    frame = pd.DataFrame({"x": x_values, "y": y_values}).replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna()
    if len(frame) < 3 or frame["x"].std(ddof=1) == 0.0 or frame["y"].std(ddof=1) == 0.0:
        return {"scatter_observations": len(frame), "correlation": np.nan, "slope": np.nan}
    x = frame["x"].to_numpy(dtype=float)
    y = frame["y"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return {
        "scatter_observations": len(frame),
        "correlation": float(np.corrcoef(x, y)[0, 1]),
        "slope": float(beta[1]),
    }


def json_safe(value: Any) -> Any:  # noqa: PLR0911
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    return value


if __name__ == "__main__":
    main()
