from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path("/home/famadeo/research/databento-asset-browser/data/metals_1m_10y")
OUT_DIR = Path("experiments/HYP-0011-expanded-metals-residual-basket")
ROOTS = ["GC", "SI", "HG", "PL", "PA", "ALI"]
DEFAULT_COST_BPS = 1.5


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    roots: tuple[str, ...]
    lookback_bars: int
    entry_z: float = 2.0
    deadband: float = 0.0
    roll_pause_bars: int = 0
    cost_bps: float = DEFAULT_COST_BPS


VARIANTS = [
    Variant(
        name="daily_highest_oos_sharpe_roll_pause_3_no_PA",
        description=(
            "Literal highest OOS Sharpe from daily scan: GC/SI/HG/PL/ALI, "
            "PA excluded, roll-pause 3 bars"
        ),
        roots=("GC", "SI", "HG", "PL", "ALI"),
        lookback_bars=126,
        roll_pause_bars=3,
    ),
    Variant(
        name="train_positive_highest_sharpe_roll_pause_1_core3",
        description=(
            "Highest daily OOS Sharpe among variants with positive train CAGR: "
            "GC/SI/HG, roll-pause 1 bar"
        ),
        roots=("GC", "SI", "HG"),
        lookback_bars=126,
        roll_pause_bars=1,
    ),
    Variant(
        name="selected_deadband_0p5_core4",
        description=(
            "Selected robust drawdown-control variant: GC/SI/HG/PL with 0.5 z-score deadband"
        ),
        roots=("GC", "SI", "HG", "PL"),
        lookback_bars=126,
        deadband=0.5,
    ),
    Variant(
        name="baseline_all_six_1m",
        description="Baseline six-metal basket on 1-minute bars",
        roots=("GC", "SI", "HG", "PL", "PA", "ALI"),
        lookback_bars=126,
    ),
]


SENSITIVITY_LOOKBACKS = [63, 126, 252, 504, 630, 1260]
SENSITIVITY_COSTS = [0.0, 1.5, 3.0]
SEGMENTS = {
    "full": (None, None),
    "recent_12m": (pd.Timestamp("2025-06-22", tz="UTC"), None),
    "post_2021": (pd.Timestamp("2021-01-01", tz="UTC"), None),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retest HYP-0011 metals residual basket variants on 1-minute data."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Reuse existing 1-minute retest CSV artifacts and only rebuild plots/report.",
    )
    return parser.parse_args()


def load_root(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["ts", "cont_logprice", "is_roll"])
    frame = frame.dropna(subset=["cont_logprice"]).drop_duplicates("ts", keep="last")
    frame = frame.sort_values("ts").set_index("ts")
    return frame


def load_panel(data_dir: Path, roots: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices: list[pd.Series] = []
    rolls: list[pd.Series] = []
    for root in roots:
        frame = load_root(data_dir / "continuous" / f"{root}.parquet")
        prices.append(frame["cont_logprice"].rename(root))
        rolls.append(frame["is_roll"].fillna(False).astype(bool).rename(root))

    price_panel = pd.concat(prices, axis=1, join="inner").sort_index()
    roll_panel = pd.concat(rolls, axis=1, join="inner").reindex(price_panel.index).fillna(False)
    return price_panel, roll_panel.astype(bool)


def roll_pause_mask(rolls: pd.DataFrame, pause_bars: int) -> pd.DataFrame:
    if pause_bars <= 0:
        return pd.DataFrame(False, index=rolls.index, columns=rolls.columns)

    paused = pd.DataFrame(False, index=rolls.index, columns=rolls.columns)
    for col in rolls.columns:
        roll_values = rolls[col].to_numpy(dtype=bool)
        paused_values = np.zeros(len(roll_values), dtype=bool)
        roll_idx = np.flatnonzero(roll_values)
        for idx in roll_idx:
            end = min(len(paused_values), idx + pause_bars + 1)
            paused_values[idx:end] = True
        paused[col] = paused_values
    return paused


def compute_positions(
    prices: pd.DataFrame,
    rolls: pd.DataFrame,
    lookback_bars: int,
    entry_z: float,
    deadband: float,
    roll_pause_bars: int,
) -> pd.DataFrame:
    residuals = prices.sub(prices.mean(axis=1), axis=0)
    min_periods = max(2, lookback_bars // 2)
    mean = residuals.rolling(lookback_bars, min_periods=min_periods).mean()
    std = residuals.rolling(lookback_bars, min_periods=min_periods).std()
    zscores = (residuals - mean) / std.replace(0.0, np.nan)

    signal = -zscores.clip(-entry_z, entry_z) / entry_z
    if deadband > 0.0:
        signal = signal.mask(zscores.abs() < deadband, 0.0)

    paused = roll_pause_mask(rolls, roll_pause_bars)
    signal = signal.mask(paused, 0.0).fillna(0.0)
    gross = signal.abs().sum(axis=1)
    positions = signal.div(gross.where(gross > 0.0), axis=0).fillna(0.0)
    return positions


def metric_block(
    returns: pd.DataFrame,
    positions: pd.DataFrame,
    variant: Variant,
    segment: str,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> dict[str, object]:
    if start is not None:
        returns = returns.loc[returns.index >= start]
        positions = positions.loc[positions.index >= start]
    if end is not None:
        returns = returns.loc[returns.index < end]
        positions = positions.loc[positions.index < end]

    obs = len(returns)
    net = returns["net_return"]
    gross = returns["gross_return"]
    cost = returns["cost_return"]
    turnover = returns["turnover"]
    equity = net.cumsum()
    drawdown = equity - equity.cummax()
    calendar_days = (
        (returns.index[-1] - returns.index[0]).total_seconds() / 86_400.0 if obs > 1 else 0.0
    )
    bars_per_year = obs / calendar_days * 365.25 if calendar_days > 0.0 else np.nan
    net_std = net.std(ddof=1)
    tstat = float(net.mean() / net_std * np.sqrt(obs)) if obs > 1 and net_std > 0.0 else np.nan
    sharpe = (
        float(net.mean() / net_std * np.sqrt(bars_per_year))
        if obs > 1 and net_std > 0.0 and np.isfinite(bars_per_year)
        else np.nan
    )
    net_log = float(net.sum()) if obs else 0.0
    ann_return = np.expm1(net_log / calendar_days * 365.25) if calendar_days > 0.0 else np.nan

    return {
        "variant": variant.name,
        "description": variant.description,
        "segment": segment,
        "roots": ",".join(variant.roots),
        "lookback_bars": variant.lookback_bars,
        "entry_z": variant.entry_z,
        "deadband": variant.deadband,
        "roll_pause_bars": variant.roll_pause_bars,
        "cost_bps": variant.cost_bps,
        "common_bars": int(positions.shape[0]),
        "eval_bars": int(obs),
        "start": returns.index[0] if obs else pd.NaT,
        "end": returns.index[-1] if obs else pd.NaT,
        "calendar_days": calendar_days,
        "bars_per_year_sample": bars_per_year,
        "gross_log_return": float(gross.sum()) if obs else 0.0,
        "cost_log_return": float(cost.sum()) if obs else 0.0,
        "net_log_return": net_log,
        "compounded_net_return": float(np.expm1(net_log)),
        "annualized_return_sample": float(ann_return) if np.isfinite(ann_return) else np.nan,
        "mean_net_bps_per_bar": float(net.mean() * 10_000.0) if obs else np.nan,
        "bar_tstat": tstat,
        "annualized_sharpe_sample": sharpe,
        "hit_rate": float((net > 0.0).mean()) if obs else np.nan,
        "max_drawdown_log": float(drawdown.min()) if obs else 0.0,
        "avg_turnover": float(turnover.mean()) if obs else np.nan,
        "p95_turnover": float(turnover.quantile(0.95)) if obs else np.nan,
        "avg_gross_exposure": float(positions.abs().sum(axis=1).mean())
        if len(positions)
        else np.nan,
    }


def backtest_variant(
    data_dir: Path,
    variant: Variant,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    prices, rolls = load_panel(data_dir, variant.roots)
    positions = compute_positions(
        prices,
        rolls,
        variant.lookback_bars,
        variant.entry_z,
        variant.deadband,
        variant.roll_pause_bars,
    )
    forward_returns = prices.diff().shift(-1).fillna(0.0)
    gross_return = (positions * forward_returns).sum(axis=1)
    turnover = positions.diff().abs().sum(axis=1).fillna(positions.abs().sum(axis=1))
    cost_return = turnover * (variant.cost_bps / 10_000.0)
    returns = pd.DataFrame(
        {
            "gross_return": gross_return,
            "cost_return": cost_return,
            "net_return": gross_return - cost_return,
            "turnover": turnover,
            "gross_exposure": positions.abs().sum(axis=1),
            "variant": variant.name,
        }
    )

    valid = positions.abs().sum(axis=1) > 0.0
    first_valid = valid.idxmax() if valid.any() else returns.index[-1]
    positions = positions.loc[first_valid:]
    returns = returns.loc[first_valid:]

    metrics = []
    for segment, (start, end) in SEGMENTS.items():
        segment_returns = returns
        if start is not None:
            segment_returns = segment_returns.loc[segment_returns.index >= start]
        if end is not None:
            segment_returns = segment_returns.loc[segment_returns.index < end]
        if len(segment_returns):
            metrics.append(metric_block(returns, positions, variant, segment, start, end))
    return returns, positions, metrics


def metrics_for_sensitivity(
    data_dir: Path,
    template: Variant,
    lookback_bars: int,
    cost_bps: float,
) -> dict[str, object]:
    variant = Variant(
        name=f"{template.name}_lb{lookback_bars}_cost{str(cost_bps).replace('.', 'p')}",
        description=template.description,
        roots=template.roots,
        lookback_bars=lookback_bars,
        entry_z=template.entry_z,
        deadband=template.deadband,
        roll_pause_bars=template.roll_pause_bars,
        cost_bps=cost_bps,
    )
    returns, positions, metrics = backtest_variant(data_dir, variant)
    full = next(row for row in metrics if row["segment"] == "full")
    full["variant"] = template.name
    full["lookback_bars"] = lookback_bars
    full["cost_bps"] = cost_bps
    full["common_bars"] = len(positions)
    full["eval_bars"] = len(returns)
    return full


def plot_equity_drawdown(returns: pd.DataFrame, out_path: Path) -> None:
    variants = list(dict.fromkeys(returns["variant"]))
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for variant in variants:
        series = returns.loc[returns["variant"].eq(variant), "net_return"]
        equity = series.cumsum()
        drawdown = equity - equity.cummax()
        daily = (
            pd.DataFrame({"equity": equity, "drawdown": drawdown}).resample("1D").last().dropna()
        )
        axes[0].plot(daily.index, np.expm1(daily["equity"]), linewidth=1.1, label=variant)
        axes[1].plot(daily.index, daily["drawdown"], linewidth=1.0, label=variant)
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[0].set_title("1-Minute Metals Residual Basket: Cumulative Net Return", loc="left")
    axes[0].set_ylabel("compounded return")
    axes[1].set_title("Log Drawdown", loc="left")
    axes[1].set_ylabel("log drawdown")
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_equity_drawdown_from_csv(returns_path: Path, out_path: Path) -> None:
    state: dict[str, tuple[float, float]] = {}
    daily_parts: list[pd.DataFrame] = []

    chunks = pd.read_csv(
        returns_path,
        usecols=["ts", "net_return", "variant"],
        parse_dates=["ts"],
        chunksize=500_000,
    )
    for chunk in chunks:
        chunk["ts"] = pd.to_datetime(chunk["ts"], utc=True)
        for variant, group in chunk.groupby("variant", sort=False):
            last_equity, last_peak = state.get(variant, (0.0, 0.0))
            equity = last_equity + group["net_return"].cumsum()
            peaks = np.maximum.accumulate(np.maximum(equity.to_numpy(), last_peak))
            drawdown = equity.to_numpy() - peaks
            state[variant] = (float(equity.iloc[-1]), float(peaks[-1]))

            daily = pd.DataFrame(
                {
                    "ts": group["ts"].to_numpy(),
                    "variant": variant,
                    "equity": equity.to_numpy(),
                    "drawdown": drawdown,
                }
            )
            daily["date"] = daily["ts"].dt.floor("D")
            daily_parts.append(
                daily.groupby(["variant", "date"], sort=False)
                .tail(1)[["ts", "variant", "equity", "drawdown"]]
                .copy()
            )

    daily_all = pd.concat(daily_parts, ignore_index=True)
    daily_all["date"] = daily_all["ts"].dt.floor("D")
    daily_all = (
        daily_all.sort_values(["variant", "ts"]).groupby(["variant", "date"], sort=False).tail(1)
    )

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for variant, group in daily_all.groupby("variant", sort=False):
        variant_daily = group.sort_values("ts")
        axes[0].plot(
            variant_daily["ts"],
            np.expm1(variant_daily["equity"]),
            linewidth=1.1,
            label=variant,
        )
        axes[1].plot(variant_daily["ts"], variant_daily["drawdown"], linewidth=1.0, label=variant)
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[0].set_title("1-Minute Metals Residual Basket: Cumulative Net Return", loc="left")
    axes[0].set_ylabel("compounded return")
    axes[1].set_title("Log Drawdown", loc="left")
    axes[1].set_ylabel("log drawdown")
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metric_bars(metrics: pd.DataFrame, out_path: Path) -> None:
    full = metrics.loc[metrics["segment"].eq("full")].copy()
    labels = full["variant"].tolist()
    fields = [
        ("annualized_sharpe_sample", "Sample Annualized Sharpe"),
        ("annualized_return_sample", "Sample Annualized Return"),
        ("max_drawdown_log", "Max Log Drawdown"),
        ("cost_log_return", "Total Cost Drag"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (field, title) in zip(axes.ravel(), fields, strict=True):
        ax.bar(labels, full[field])
        ax.set_title(title, loc="left")
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_report(metrics: pd.DataFrame, sensitivity: pd.DataFrame, out_dir: Path) -> None:
    full = metrics.loc[metrics["segment"].eq("full")].copy()
    full["roots_label"] = full["roots"].str.replace(",", "/")

    def pct(value: float) -> str:
        return f"{value:.2%}"

    def log_drawdown_pct(value: float) -> str:
        return pct(float(np.expm1(value)))

    def num(value: float) -> str:
        return f"{value:.2f}"

    rows = [
        "| "
        + " | ".join(
            [
                f"`{row.variant}`",
                f"`{row.roots_label}`",
                f"{row.eval_bars:,}",
                pct(row.compounded_net_return),
                pct(row.annualized_return_sample),
                num(row.bar_tstat),
                num(row.annualized_sharpe_sample),
                log_drawdown_pct(row.max_drawdown_log),
                pct(row.cost_log_return),
            ]
        )
        + " |"
        for row in full.itertuples(index=False)
    ]

    best = full.sort_values("annualized_sharpe_sample", ascending=False).iloc[0]
    selected = full.loc[full["variant"].eq("selected_deadband_0p5_core4")].iloc[0]
    sens = sensitivity.loc[
        sensitivity["variant"].eq("daily_highest_oos_sharpe_roll_pause_3_no_PA")
        & sensitivity["cost_bps"].isin([0.0, 1.5, 3.0])
        & sensitivity["lookback_bars"].isin([63, 126, 630])
    ].sort_values(["lookback_bars", "cost_bps"])

    sens_rows = [
        "| "
        + " | ".join(
            [
                str(row.lookback_bars),
                f"{row.cost_bps:.1f}",
                f"{row.eval_bars:,}",
                pct(row.compounded_net_return),
                num(row.bar_tstat),
                log_drawdown_pct(row.max_drawdown_log),
                pct(row.cost_log_return),
            ]
        )
        + " |"
        for row in sens.itertuples(index=False)
    ]

    best_label = "Best" if best["annualized_sharpe_sample"] > 0.0 else "Least-negative"

    report = f"""---
title: "HYP-0011 1-Minute Retest"
format: html
---

## Objective

Retest the fixed HYP-0011 metals residual mean-reversion variants on the newly
downloaded 10-year `ohlcv-1m` continuous metals data.

The implementation keeps the 5-minute retest mechanics: common timestamps across
the variant roots, residual log prices, rolling z-scores, one-bar forward
returns, `1.5` bps cost per unit turnover, and optional roll-pause/deadband
controls.

## Main Result

{best_label} full-sample Sharpe in this fixed set: `{best["variant"]}` with sample
annualized Sharpe `{best["annualized_sharpe_sample"]:.2f}`.

Selected drawdown-control variant: `selected_deadband_0p5_core4` compounded
`{selected["compounded_net_return"]:.2%}` net with sample annualized Sharpe
`{selected["annualized_sharpe_sample"]:.2f}` and max compounded drawdown
`{np.expm1(selected["max_drawdown_log"]):.2%}`.

| Variant | Roots | Eval bars | Net compounded | Ann. return | T-stat | Sharpe | Max DD | Cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

![](one_min_strategy_equity_drawdown.png)

![](one_min_strategy_metric_bars.png)

## Lookback And Cost Sensitivity

Sensitivity below is for the literal highest daily-Sharpe variant
`GC/SI/HG/PL/ALI`, with `PA` excluded and a 3-bar roll pause. `630` 1-minute bars
is the rough clock-time equivalent of the earlier `126` 5-minute-bar lookback.

| Lookback bars | Cost bps | Eval bars | Net compounded | T-stat | Max DD | Cost |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(sens_rows)}

## Caveats

- Annualization uses the realized common-bar frequency of each variant, so it is
  useful for comparing these 1-minute variants but should not be read as a
  capacity-adjusted production Sharpe.
- `ALI` and `PA` make common timestamps much sparser. Variants including `ALI`
  are tested on a materially different event grid than `GC/SI/HG/PL`.
- The data pull had Databento quality warnings on a small number of dates; this
  retest does not adjust for those dates beyond using the continuous files as
  downloaded.

## Artifacts

- Metrics: `one_min_strategy_metrics.csv`
- Returns: `one_min_strategy_returns.csv`
- Positions: `one_min_strategy_positions.parquet`
- Sensitivity: `one_min_strategy_lookback_cost_sensitivity.csv`
"""
    (out_dir / "one_min_strategy_retest_report.qmd").write_text(report)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = args.out_dir / "one_min_strategy_metrics.csv"
    returns_path = args.out_dir / "one_min_strategy_returns.csv"
    positions_path = args.out_dir / "one_min_strategy_positions.parquet"
    sensitivity_path = args.out_dir / "one_min_strategy_lookback_cost_sensitivity.csv"
    report_path = args.out_dir / "one_min_strategy_retest_report.qmd"

    if args.plots_only:
        metrics_out = pd.read_csv(metrics_path)
        sensitivity_out = pd.read_csv(sensitivity_path)
        plot_equity_drawdown_from_csv(
            returns_path, args.out_dir / "one_min_strategy_equity_drawdown.png"
        )
        plot_metric_bars(metrics_out, args.out_dir / "one_min_strategy_metric_bars.png")
        write_report(metrics_out, sensitivity_out, args.out_dir)
        summary = {
            "metrics": str(metrics_path),
            "returns": str(returns_path),
            "positions": str(positions_path),
            "sensitivity": str(sensitivity_path),
            "report": str(report_path),
        }
        (args.out_dir / "one_min_strategy_retest_results.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(metrics_out.loc[metrics_out["segment"].eq("full")].to_string(index=False))
        return

    all_returns: list[pd.DataFrame] = []
    all_positions: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []

    for variant in VARIANTS:
        print(f"running {variant.name} on {','.join(variant.roots)}")
        returns, positions, metrics = backtest_variant(args.data_dir, variant)
        all_returns.append(returns.reset_index(names="ts"))
        positions_out = positions.copy()
        positions_out["variant"] = variant.name
        all_positions.append(positions_out.reset_index(names="ts"))
        metric_rows.extend(metrics)

    returns_out = pd.concat(all_returns, ignore_index=True)
    positions_out = pd.concat(all_positions, ignore_index=True)
    metrics_out = pd.DataFrame(metric_rows)

    print("running lookback/cost sensitivity")
    sensitivity_rows = [
        metrics_for_sensitivity(args.data_dir, VARIANTS[0], lookback, cost)
        for lookback in SENSITIVITY_LOOKBACKS
        for cost in SENSITIVITY_COSTS
    ]
    sensitivity_out = pd.DataFrame(sensitivity_rows)

    metrics_out.to_csv(metrics_path, index=False)
    returns_out.to_csv(returns_path, index=False)
    positions_out.to_parquet(positions_path, index=False)
    sensitivity_out.to_csv(sensitivity_path, index=False)

    plot_equity_drawdown(
        returns_out.set_index("ts"), args.out_dir / "one_min_strategy_equity_drawdown.png"
    )
    plot_metric_bars(metrics_out, args.out_dir / "one_min_strategy_metric_bars.png")
    write_report(metrics_out, sensitivity_out, args.out_dir)

    summary = {
        "metrics": str(metrics_path),
        "returns": str(returns_path),
        "positions": str(positions_path),
        "sensitivity": str(sensitivity_path),
        "report": str(report_path),
    }
    (args.out_dir / "one_min_strategy_retest_results.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(metrics_out.loc[metrics_out["segment"].eq("full")].to_string(index=False))


if __name__ == "__main__":
    main()
