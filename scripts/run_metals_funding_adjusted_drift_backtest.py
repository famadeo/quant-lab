from __future__ import annotations

import math
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = (
    REPO_ROOT
    / "experiments"
    / "HYP-0037-metals-funding-vs-realized-returns"
    / "cumulative_hourly_accounting.parquet"
)
COST_PATH = (
    REPO_ROOT
    / "experiments"
    / "HYP-0015-metals-flow-corrected-residual-reversion"
    / "cost_estimates.csv"
)
OUTPUT_DIR = REPO_ROOT / "experiments" / "HYP-0039-metals-funding-adjusted-drift-backtest"

ROOTS = ["GC", "SI", "HG", "PL", "PA"]
TARGET_MONTHS = [1, 3, 6]
LOOKBACKS = {"1d": 24, "3d": 72, "1w": 168, "2w": 336}
SCORE_METHODS = ["pressure_vol_scaled", "pressure_z"]
ENTRY_THRESHOLDS = [0.50, 1.00, 1.50]
EXIT_THRESHOLDS = [0.00, 0.25, 0.50]
FUNDING_SCALED_FILTERS = [0.00, 0.02, 0.05]
COST_MULTIPLIERS = [0.0, 1.0, 2.0, 3.0]

ROLLING_Z_BARS = 24 * 126
ROLLING_Z_MIN_BARS = 24 * 30
TRAIN_FRACTION = 0.70
EMBARGO_DAYS = 7
MIN_OBS = 100
MIN_PERIODS_PER_YEAR_OBS = 2
MIN_ACTIVE_FRACTION = 0.01
REPORT_TARGET_MONTHS = 3
REPORT_LOOKBACK = "1w"
ROBUST_MIN_TRAIN_TEST_NET = 0.05
ROBUST_MIN_GROSS_TO_COST = 1.5


@dataclass(frozen=True)
class Variant:
    target_months: int
    lookback_label: str
    lookback_bars: int
    score_method: str
    entry_threshold: float
    exit_threshold: float
    funding_scaled_filter: float
    cost_multiplier: float

    @property
    def name(self) -> str:
        entry = fmt_num(self.entry_threshold)
        exit_ = fmt_num(self.exit_threshold)
        filt = fmt_num(self.funding_scaled_filter)
        cost = fmt_num(self.cost_multiplier)
        return (
            f"target{self.target_months}m_lb{self.lookback_label}_{self.score_method}"
            f"_entry{entry}_exit{exit_}_filt{filt}_costx{cost}"
        )


def fmt_num(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def load_costs() -> pd.Series:
    if COST_PATH.exists():
        costs = pd.read_csv(COST_PATH).set_index("root")["per_side_cost_bps"]
    else:
        costs = pd.Series(
            {
                "GC": 0.55078817788255,
                "SI": 1.8695083193124462,
                "HG": 0.8003841844080726,
                "PL": 2.563182447326601,
                "PA": 5.593884020137983,
            },
            name="per_side_cost_bps",
        )
    costs = costs.reindex(ROOTS).astype(float)
    if costs.isna().any():
        missing = costs[costs.isna()].index.tolist()
        raise ValueError(f"Missing cost estimates for {missing}")
    return costs


def load_source() -> pd.DataFrame:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(SOURCE_PATH)
    cols = [
        "root",
        "target_months",
        "ts",
        "funding_rate",
        "funding_pct_ann",
        "log_return_next_bar",
        "funding_paid_next_bar",
        "excess_after_funding_next_bar",
        "funding_observed",
    ]
    frame = pd.read_parquet(SOURCE_PATH, columns=cols)
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(
        subset=[
            "log_return_next_bar",
            "funding_paid_next_bar",
            "excess_after_funding_next_bar",
        ]
    )
    return frame.sort_values(["root", "target_months", "ts"])


def add_pressure_signals(frame: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for (root, target), group in frame.groupby(["root", "target_months"], sort=True):
        data = group.sort_values("ts").copy()
        ret = data["log_return_next_bar"]
        funding = data["funding_paid_next_bar"]
        for label, bars in LOOKBACKS.items():
            trailing_return = ret.rolling(bars, min_periods=bars).sum().shift(1)
            trailing_funding = funding.rolling(bars, min_periods=bars).sum().shift(1)
            trailing_hourly_std = ret.rolling(bars, min_periods=bars).std(ddof=1).shift(1)
            trailing_vol = trailing_hourly_std * math.sqrt(bars)
            pressure = trailing_return - trailing_funding
            pressure_mean = pressure.shift(1).rolling(
                ROLLING_Z_BARS, min_periods=ROLLING_Z_MIN_BARS
            ).mean()
            pressure_std = pressure.shift(1).rolling(
                ROLLING_Z_BARS, min_periods=ROLLING_Z_MIN_BARS
            ).std(ddof=1)
            data[f"{label}_trailing_return"] = trailing_return
            data[f"{label}_trailing_funding_paid"] = trailing_funding
            data[f"{label}_pressure"] = pressure
            data[f"{label}_trailing_vol"] = trailing_vol
            data[f"{label}_funding_scaled"] = trailing_funding.abs() / trailing_vol.replace(
                0.0, np.nan
            )
            data[f"{label}_pressure_vol_scaled"] = pressure / trailing_vol.replace(
                0.0, np.nan
            )
            data[f"{label}_pressure_z"] = (pressure - pressure_mean) / pressure_std.replace(
                0.0, np.nan
            )
        frames.append(data)
        print(f"Prepared signals {root} {target}M", flush=True)
    return pd.concat(frames, ignore_index=True)


def generate_position(
    score: pd.Series, active_filter: pd.Series, entry: float, exit_: float
) -> pd.Series:
    values = score.to_numpy(dtype=float)
    filters = active_filter.fillna(False).to_numpy(dtype=bool)
    positions = np.zeros(len(values), dtype=float)
    current = 0.0
    for idx, value in enumerate(values):
        if not filters[idx] or not np.isfinite(value):
            current = 0.0
        elif value >= entry:
            current = 1.0
        elif value <= -entry:
            current = -1.0
        elif current > 0.0 and value > exit_:
            current = 1.0
        elif current < 0.0 and value < -exit_:
            current = -1.0
        else:
            current = 0.0
        positions[idx] = current
    return pd.Series(positions, index=score.index)


def infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < MIN_PERIODS_PER_YEAR_OBS:
        return np.nan
    elapsed_years = (index[-1] - index[0]).total_seconds() / (365.25 * 24 * 60 * 60)
    if elapsed_years <= 0:
        return np.nan
    return float(len(index) / elapsed_years)


def max_drawdown(returns: pd.Series) -> float:
    equity = returns.fillna(0.0).cumsum()
    drawdown = equity - equity.cummax()
    return float(drawdown.min()) if len(drawdown) else np.nan


def summarize_returns(frame: pd.DataFrame, *, label: str) -> dict[str, object]:
    if frame.empty:
        return {
            "split": label,
            "observations": 0,
            "active_fraction": np.nan,
            "gross_excess_return": 0.0,
            "cost_return": 0.0,
            "net_return": 0.0,
            "annualized_net_return": np.nan,
            "sharpe": np.nan,
            "tstat": np.nan,
            "max_drawdown": np.nan,
            "turnover": 0.0,
            "gross_to_cost": np.nan,
        }
    net = frame["net_return"].fillna(0.0)
    gross = frame["gross_excess_return"].fillna(0.0)
    cost = frame["cost_return"].fillna(0.0)
    periods_per_year = infer_periods_per_year(pd.DatetimeIndex(frame["ts"]))
    mean = float(net.mean())
    std = float(net.std(ddof=1)) if len(net) > 1 else np.nan
    cost_sum = float(cost.sum())
    return {
        "split": label,
        "observations": len(frame),
        "active_fraction": float(frame["active"].mean()),
        "gross_excess_return": float(gross.sum()),
        "price_return": float(frame["price_return"].sum()),
        "carry_return": float(frame["carry_return"].sum()),
        "cost_return": cost_sum,
        "net_return": float(net.sum()),
        "annualized_net_return": mean * periods_per_year
        if np.isfinite(periods_per_year)
        else np.nan,
        "sharpe": mean / std * math.sqrt(periods_per_year)
        if std and std > 0 and np.isfinite(periods_per_year)
        else np.nan,
        "tstat": mean / (std / math.sqrt(len(net))) if std and std > 0 else np.nan,
        "max_drawdown": max_drawdown(net),
        "turnover": float(frame["turnover"].sum()),
        "gross_to_cost": float(gross.sum() / cost_sum) if cost_sum > 0 else np.inf,
    }


def split_frames(strategy: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ordered = strategy.sort_values("ts")
    if ordered.empty:
        return {"full": ordered, "train": ordered, "test": ordered}
    cut_index = int(len(ordered) * TRAIN_FRACTION)
    cut_index = min(max(cut_index, 0), len(ordered) - 1)
    split_ts = ordered["ts"].iloc[cut_index]
    test_start = split_ts + pd.Timedelta(days=EMBARGO_DAYS)
    return {
        "full": ordered,
        "train": ordered[ordered["ts"] <= split_ts],
        "test": ordered[ordered["ts"] >= test_start],
    }


def build_variant_strategy(
    panel: pd.DataFrame,
    variant: Variant,
    costs_bps: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_panel = panel[panel["target_months"] == variant.target_months].copy()
    score_col = f"{variant.lookback_label}_{variant.score_method}"
    filter_col = f"{variant.lookback_label}_funding_scaled"
    pressure_col = f"{variant.lookback_label}_pressure"
    trailing_return_col = f"{variant.lookback_label}_trailing_return"
    trailing_funding_col = f"{variant.lookback_label}_trailing_funding_paid"
    positions = []

    for _root, group in target_panel.groupby("root", sort=True):
        data = group.sort_values("ts").copy()
        active_filter = data[filter_col] >= variant.funding_scaled_filter
        data["position"] = generate_position(
            data[score_col],
            active_filter,
            entry=variant.entry_threshold,
            exit_=variant.exit_threshold,
        )
        data["weight"] = data["position"] / len(ROOTS)
        data["score"] = data[score_col]
        data["pressure"] = data[pressure_col]
        data["trailing_return"] = data[trailing_return_col]
        data["trailing_funding_paid"] = data[trailing_funding_col]
        data["funding_scaled"] = data[filter_col]
        positions.append(data)

    positioned = pd.concat(positions, ignore_index=True).sort_values(["root", "ts"])
    positioned["prev_weight"] = positioned.groupby("root")["weight"].shift(1).fillna(0.0)
    positioned["turnover"] = (positioned["weight"] - positioned["prev_weight"]).abs()
    positioned["price_return"] = positioned["weight"] * positioned["log_return_next_bar"]
    positioned["carry_return"] = -positioned["weight"] * positioned["funding_paid_next_bar"]
    positioned["gross_excess_return"] = (
        positioned["weight"] * positioned["excess_after_funding_next_bar"]
    )
    positioned["cost_return"] = (
        positioned["turnover"]
        * positioned["root"].map(costs_bps)
        * variant.cost_multiplier
        / 10_000.0
    )
    positioned["net_return"] = positioned["gross_excess_return"] - positioned["cost_return"]
    positioned["abs_weight"] = positioned["weight"].abs()

    by_ts = (
        positioned.groupby("ts", as_index=False)
        .agg(
            gross_excess_return=("gross_excess_return", "sum"),
            price_return=("price_return", "sum"),
            carry_return=("carry_return", "sum"),
            cost_return=("cost_return", "sum"),
            net_return=("net_return", "sum"),
            turnover=("turnover", "sum"),
            gross_position=("abs_weight", "sum"),
        )
        .sort_values("ts")
    )
    by_ts["active"] = by_ts["gross_position"] > 0.0
    by_ts["variant"] = variant.name
    positioned["variant"] = variant.name
    return by_ts, positioned


def position_variant_grid() -> list[Variant]:
    variants = []
    for target in TARGET_MONTHS:
        for lookback_label, lookback_bars in LOOKBACKS.items():
            variants.extend(
                Variant(
                    target_months=target,
                    lookback_label=lookback_label,
                    lookback_bars=lookback_bars,
                    score_method=score_method,
                    entry_threshold=entry,
                    exit_threshold=exit_,
                    funding_scaled_filter=funding_filter,
                    cost_multiplier=1.0,
                )
                for score_method, entry, exit_, funding_filter in product(
                    SCORE_METHODS,
                    ENTRY_THRESHOLDS,
                    EXIT_THRESHOLDS,
                    FUNDING_SCALED_FILTERS,
                )
                if exit_ < entry
            )
    return variants


def apply_cost_multiplier(frame: pd.DataFrame, variant: Variant) -> pd.DataFrame:
    scaled = frame.copy()
    base_cost_multiplier = 1.0
    scale = variant.cost_multiplier / base_cost_multiplier
    scaled["cost_return"] = scaled["cost_return"] * scale
    scaled["net_return"] = scaled["gross_excess_return"] - scaled["cost_return"]
    scaled["variant"] = variant.name
    return scaled


def apply_root_cost_multiplier(root_frame: pd.DataFrame, variant: Variant) -> pd.DataFrame:
    scaled = root_frame.copy()
    scaled["cost_return"] = scaled["cost_return"] * variant.cost_multiplier
    scaled["net_return"] = scaled["gross_excess_return"] - scaled["cost_return"]
    scaled.insert(0, "variant", variant.name)
    scaled["target_months"] = variant.target_months
    scaled["lookback"] = variant.lookback_label
    scaled["score_method"] = variant.score_method
    scaled["cost_multiplier"] = variant.cost_multiplier
    return scaled


def run_backtests(
    panel: pd.DataFrame, costs_bps: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    metrics_rows = []
    root_rows = []
    variants = position_variant_grid()
    variant_by_name: dict[str, Variant] = {}

    for idx, variant in enumerate(variants, start=1):
        if idx == 1 or idx % 50 == 0:
            print(f"Backtesting {idx}/{len(variants)}", flush=True)
        base_strategy, positioned = build_variant_strategy(panel, variant, costs_bps)

        base_root = (
            positioned.groupby("root", as_index=False)
            .agg(
                observations=("ts", "size"),
                active_bars=("weight", lambda values: int((values.abs() > 0).sum())),
                gross_excess_return=("gross_excess_return", "sum"),
                price_return=("price_return", "sum"),
                carry_return=("carry_return", "sum"),
                cost_return=("cost_return", "sum"),
                net_return=("net_return", "sum"),
                turnover=("turnover", "sum"),
                mean_score=("score", "mean"),
                mean_pressure_bp=("pressure", lambda values: values.mean() * 10_000.0),
            )
        )
        for cost_multiplier in COST_MULTIPLIERS:
            costed = replace(variant, cost_multiplier=cost_multiplier)
            variant_by_name[costed.name] = costed
            strategy = apply_cost_multiplier(base_strategy, costed)
            for split, split_frame in split_frames(strategy).items():
                metrics_rows.append(
                    {
                        "variant": costed.name,
                        "target_months": costed.target_months,
                        "lookback": costed.lookback_label,
                        "lookback_bars": costed.lookback_bars,
                        "score_method": costed.score_method,
                        "entry_threshold": costed.entry_threshold,
                        "exit_threshold": costed.exit_threshold,
                        "funding_scaled_filter": costed.funding_scaled_filter,
                        "cost_multiplier": costed.cost_multiplier,
                        **summarize_returns(split_frame, label=split),
                    }
                )
            root_rows.append(apply_root_cost_multiplier(base_root, costed))

    metrics = pd.DataFrame(metrics_rows)
    root_metrics = pd.concat(root_rows, ignore_index=True) if root_rows else pd.DataFrame()

    one_x = metrics[(metrics["split"] == "train") & (metrics["cost_multiplier"] == 1.0)].copy()
    train_eligible = one_x[
        (one_x["active_fraction"] > MIN_ACTIVE_FRACTION)
        & (one_x["cost_return"] > 0.0)
        & (one_x["gross_to_cost"] > 1.0)
    ]
    if train_eligible.empty:
        train_eligible = one_x[one_x["active_fraction"] > MIN_ACTIVE_FRACTION]
    best_train = train_eligible.sort_values(
        ["net_return", "sharpe", "gross_to_cost"], ascending=False
    ).head(5)
    selected_curves = {}
    for variant_name in best_train["variant"].tolist():
        selected_variant = variant_by_name[variant_name]
        base_variant = replace(selected_variant, cost_multiplier=1.0)
        base_strategy, _positioned = build_variant_strategy(panel, base_variant, costs_bps)
        selected_curves[variant_name] = apply_cost_multiplier(base_strategy, selected_variant)

    return metrics, root_metrics, selected_curves


def build_transition_diagnostics(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGET_MONTHS:
        for label in LOOKBACKS:
            pressure_col = f"{label}_pressure"
            ret_col = f"{label}_trailing_return"
            funding_col = f"{label}_trailing_funding_paid"
            data = panel[panel["target_months"] == target].dropna(
                subset=[pressure_col, ret_col, funding_col]
            )
            if data.empty:
                continue
            funding = data[funding_col]
            pressure = data[pressure_col]
            state = np.select(
                [
                    (funding > 0.0) & (pressure > 0.0),
                    (funding > 0.0) & (pressure < 0.0),
                    (funding < 0.0) & (pressure < 0.0),
                    (funding < 0.0) & (pressure > 0.0),
                ],
                [
                    "longs_pay_and_win",
                    "longs_pay_and_lose",
                    "shorts_pay_and_win",
                    "shorts_pay_and_lose",
                ],
                default="balanced_or_flat",
            )
            work = data.assign(pressure_state=state)
            summary = (
                work.groupby(["root", "pressure_state"], as_index=False)
                .agg(
                    observations=("ts", "size"),
                    next_excess_bp=(
                        "excess_after_funding_next_bar",
                        lambda values: values.mean() * 10_000.0,
                    ),
                    next_return_bp=("log_return_next_bar", lambda values: values.mean() * 10_000.0),
                    next_funding_paid_bp=(
                        "funding_paid_next_bar",
                        lambda values: values.mean() * 10_000.0,
                    ),
                    positive_excess_fraction=(
                        "excess_after_funding_next_bar",
                        lambda values: float((values > 0.0).mean()),
                    ),
                )
            )
            summary.insert(0, "lookback", label)
            summary.insert(0, "target_months", target)
            rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_equity(selected_curves: dict[str, pd.DataFrame]) -> None:
    if not selected_curves:
        return
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    for variant_name, curve in selected_curves.items():
        data = curve.sort_values("ts")
        ax.plot(data["ts"], data["net_return"].cumsum(), lw=1.1, label=variant_name[:80])
    ax.axhline(0.0, color="#333333", lw=0.8)
    ax.set_title("Top train-selected funding-adjusted drift variants, net log return")
    ax.set_ylabel("cumulative net log return")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=7)
    fig.savefig(OUTPUT_DIR / "top_train_selected_equity.png", dpi=160)
    plt.close(fig)


def plot_metric_bars(metrics: pd.DataFrame) -> None:
    one_x = metrics[(metrics["split"] == "full") & (metrics["cost_multiplier"] == 1.0)].copy()
    if one_x.empty:
        return
    top = one_x.sort_values("net_return", ascending=False).head(20)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    axes[0].bar(np.arange(len(top)), top["net_return"], color="#2f7d8c")
    axes[0].axhline(0.0, color="#333333", lw=0.8)
    axes[0].set_title("Top full-sample variants at 1x costs")
    axes[0].set_ylabel("net log return")
    axes[1].bar(np.arange(len(top)), top["sharpe"], color="#b15a2a")
    axes[1].axhline(0.0, color="#333333", lw=0.8)
    axes[1].set_ylabel("annualized Sharpe")
    axes[1].set_xticks(
        np.arange(len(top)), labels=top["variant"], rotation=80, ha="right", fontsize=7
    )
    fig.savefig(OUTPUT_DIR / "top_full_sample_variant_metrics.png", dpi=160)
    plt.close(fig)


def write_report(
    *,
    metrics: pd.DataFrame,
    root_metrics: pd.DataFrame,
    transition_diagnostics: pd.DataFrame,
    costs_bps: pd.Series,
) -> None:
    one_x_full = metrics[(metrics["split"] == "full") & (metrics["cost_multiplier"] == 1.0)]
    one_x_train = metrics[(metrics["split"] == "train") & (metrics["cost_multiplier"] == 1.0)]
    one_x_test = metrics[(metrics["split"] == "test") & (metrics["cost_multiplier"] == 1.0)]
    best_train_names = (
        one_x_train[one_x_train["active_fraction"] > MIN_ACTIVE_FRACTION]
        .sort_values(["net_return", "sharpe", "gross_to_cost"], ascending=False)
        .head(10)["variant"]
        .tolist()
    )
    best_train_compare = metrics[
        (metrics["variant"].isin(best_train_names))
        & (metrics["cost_multiplier"] == 1.0)
        & (metrics["split"].isin(["train", "test", "full"]))
    ].sort_values(["variant", "split"])
    top_full = one_x_full.sort_values(["net_return", "sharpe"], ascending=False).head(20)
    top_test = one_x_test.sort_values(["net_return", "sharpe"], ascending=False).head(20)
    robust = robust_train_test_table(metrics)
    best_variant = best_train_names[0] if best_train_names else ""
    best_roots = (
        root_metrics[root_metrics["variant"] == best_variant]
        .sort_values("net_return", ascending=False)
        if best_variant
        else pd.DataFrame()
    )
    diag = transition_diagnostics[
        (transition_diagnostics["target_months"] == REPORT_TARGET_MONTHS)
        & (transition_diagnostics["lookback"] == REPORT_LOOKBACK)
    ].copy()

    metric_cols = [
        "variant",
        "split",
        "net_return",
        "gross_excess_return",
        "cost_return",
        "sharpe",
        "tstat",
        "max_drawdown",
        "active_fraction",
        "turnover",
        "gross_to_cost",
    ]
    lines = [
        "# HYP-0039 Metals Funding-Adjusted Drift Backtest",
        "",
        "## Strategy",
        "",
        "For each metal, tenor, and rolling window:",
        "",
        "`pressure = trailing_realized_log_return - trailing_funding_paid`",
        "",
        "Positive pressure means the long side has been winning after carry. Negative "
        "pressure means the short side has been winning after carry.",
        "",
        "The backtest uses a state machine:",
        "",
        "- Enter long when pressure score is above the entry threshold.",
        "- Enter short when pressure score is below the negative entry threshold.",
        "- Hold until the score crosses the exit threshold.",
        "- Apply the position to the next hourly return after funding.",
        "",
        "## Implementation",
        "",
        "- Source accounting panel: `HYP-0037` hourly realized return minus funding paid.",
        "- Score methods: raw pressure divided by trailing realized volatility, and rolling "
        "pressure z-score.",
        "- Funding materiality filter: `abs(trailing_funding_paid) / trailing_vol`.",
        "- Portfolio construction: equal 20% capital sleeve per metal, inactive sleeve in cash.",
        "- Costs: per-side MBP1 cost estimates multiplied by turnover and cost multiplier.",
        "- Train/test split: chronological 70/30 with a "
        f"{EMBARGO_DAYS}-day embargo before test.",
        "",
        "## Cost Assumptions",
        "",
        costs_bps.rename("per_side_cost_bps").reset_index().rename(
            columns={"index": "root"}
        ).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Best Train-Selected Variants at 1x Costs",
        "",
        best_train_compare[metric_cols].to_markdown(index=False, floatfmt=".4f")
        if not best_train_compare.empty
        else "No train-selected variants.",
        "",
        "## Top Full-Sample Variants at 1x Costs",
        "",
        top_full[metric_cols].to_markdown(index=False, floatfmt=".4f")
        if not top_full.empty
        else "No full-sample variants.",
        "",
        "## Top Test Variants at 1x Costs",
        "",
        top_test[metric_cols].to_markdown(index=False, floatfmt=".4f")
        if not top_test.empty
        else "No test variants.",
        "",
        "## Strict Train/Test Robustness at 1x Costs",
        "",
        robust.to_markdown(index=False, floatfmt=".4f") if not robust.empty else "No rows.",
        "",
        "## Root Breakdown For Best Train-Selected Variant",
        "",
        best_roots.to_markdown(index=False, floatfmt=".4f")
        if not best_roots.empty
        else "No root breakdown.",
        "",
        "## Pressure State Diagnostic, 3M Target, 1W Lookback",
        "",
        diag.to_markdown(index=False, floatfmt=".4f") if not diag.empty else "No diagnostics.",
        "",
        "## Files",
        "",
        "- `variant_metrics.csv`",
        "- `root_variant_metrics.csv`",
        "- `pressure_state_diagnostics.csv`",
        "- `top_train_selected_returns.parquet`",
        "- `top_train_selected_equity.png`",
        "- `top_full_sample_variant_metrics.png`",
        "",
        "## Caveats",
        "",
        "- This uses front-futures proxy funding, not true spot/cash funding.",
        "- Hourly rebalancing is intentionally conservative on cost sensitivity, but it may "
        "overstate turnover relative to a production implementation with execution bands.",
        "- The strategy is directional, not beta-neutral. Positive results can still be trend "
        "beta unless they survive hedging or cross-sectional construction.",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def robust_train_test_table(metrics: pd.DataFrame) -> pd.DataFrame:
    one_x = metrics[metrics["cost_multiplier"].eq(1.0)].copy()
    wide = one_x.pivot_table(
        index="variant",
        columns="split",
        values=["net_return", "sharpe", "gross_to_cost", "max_drawdown", "active_fraction"],
        aggfunc="first",
    )
    rows = []
    for variant in wide.index:
        try:
            record = {
                "variant": variant,
                "full_net": wide.loc[variant, ("net_return", "full")],
                "train_net": wide.loc[variant, ("net_return", "train")],
                "test_net": wide.loc[variant, ("net_return", "test")],
                "full_sharpe": wide.loc[variant, ("sharpe", "full")],
                "train_sharpe": wide.loc[variant, ("sharpe", "train")],
                "test_sharpe": wide.loc[variant, ("sharpe", "test")],
                "train_gross_to_cost": wide.loc[variant, ("gross_to_cost", "train")],
                "test_gross_to_cost": wide.loc[variant, ("gross_to_cost", "test")],
                "active_fraction": wide.loc[variant, ("active_fraction", "full")],
                "full_max_drawdown": wide.loc[variant, ("max_drawdown", "full")],
            }
        except KeyError:
            continue
        rows.append(record)
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows)
    robust = table[
        (table["train_net"] > ROBUST_MIN_TRAIN_TEST_NET)
        & (table["test_net"] > ROBUST_MIN_TRAIN_TEST_NET)
        & (table["train_gross_to_cost"] > ROBUST_MIN_GROSS_TO_COST)
        & (table["test_gross_to_cost"] > ROBUST_MIN_GROSS_TO_COST)
    ].copy()
    return robust.sort_values(["test_sharpe", "test_net"], ascending=False).head(20)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    costs_bps = load_costs()
    source = load_source()
    panel = add_pressure_signals(source)
    transition_diagnostics = build_transition_diagnostics(panel)
    metrics, root_metrics, selected_curves = run_backtests(panel, costs_bps)

    metrics.to_csv(OUTPUT_DIR / "variant_metrics.csv", index=False)
    root_metrics.to_csv(OUTPUT_DIR / "root_variant_metrics.csv", index=False)
    transition_diagnostics.to_csv(OUTPUT_DIR / "pressure_state_diagnostics.csv", index=False)
    if selected_curves:
        selected = pd.concat(selected_curves.values(), ignore_index=True)
        selected.to_parquet(OUTPUT_DIR / "top_train_selected_returns.parquet", index=False)
    plot_equity(selected_curves)
    plot_metric_bars(metrics)
    write_report(
        metrics=metrics,
        root_metrics=root_metrics,
        transition_diagnostics=transition_diagnostics,
        costs_bps=costs_bps,
    )
    print(f"Variants: {metrics['variant'].nunique():,}", flush=True)
    print(f"Wrote {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
