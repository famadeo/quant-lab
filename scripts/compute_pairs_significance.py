# ruff: noqa: PLR2004
"""Recompute deflated-Sharpe and fixed-universe diagnostics for the pairs experiments.

The pairs experiments (HYP-0001..HYP-0004) report headline Sharpe ratios with no
multiple-testing correction and aggregate the portfolio over a time-varying set of
observed pairs. This script reads each experiment's saved artifacts and adds:

- a per-observation Sharpe and its probabilistic Sharpe ratio (PSR);
- a deflated Sharpe ratio (DSR) that benchmarks against the expected maximum Sharpe
  given the number of pairs screened (multiple-testing control);
- a fixed-universe portfolio Sharpe (constant 1/N weights) for comparison with the
  observed-pairs aggregation.

It does not re-run any backtest; it only re-scores existing results, so it is safe to
run without the upstream market data. Writes ``pairs_significance.json`` and
``pairs_significance.csv`` into each experiment folder and prints a summary table.

Caveat: PSR/DSR assume iid observations. For intraday (5-minute / 1-minute) bars the
bar count overstates the effective sample size, so these probabilities are optimistic
for the high-frequency experiments; treat a low DSR as strong evidence of luck and a
high DSR as necessary-but-not-sufficient.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import yaml

from quantlab.pairs.strategy import calculate_metrics
from quantlab.validation.significance import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    fixed_universe_portfolio_returns,
    probabilistic_sharpe_ratio,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = [
    "HYP-0001-pairs-mahalanobis-vs-zscore",
    "HYP-0002-equity-top100-pairs",
    "HYP-0003-equity-futures-1m-pairs",
    "HYP-0004-xap-xab-longer-1m",
]


def _find(directory: Path, suffix: str) -> Path | None:
    matches = sorted(directory.glob(f"*{suffix}"))
    return matches[0] if matches else None


def _periods_per_year(config_path: Path) -> float:
    config = yaml.safe_load(config_path.read_text())
    backtest = config.get("backtest", {})
    return float(backtest.get("periods_per_year", 252))


def _candidate_pairs(directory: Path, fallback: int) -> int:
    results_path = directory / "results.json"
    if results_path.exists():
        payload = json.loads(results_path.read_text())
        value = payload.get("candidate_pairs")
        if isinstance(value, int) and value > 0:
            return value
    return fallback


def score_experiment(directory: Path) -> list[dict[str, float | int | str]]:
    portfolio_path = _find(directory, "portfolio_returns.parquet")
    pair_returns_path = _find(directory, "pair_returns.parquet")
    pair_metrics_path = _find(directory, "pair_metrics.csv")
    config_path = directory / "config.yaml"
    if portfolio_path is None or pair_returns_path is None or pair_metrics_path is None:
        return []

    periods_per_year = _periods_per_year(config_path)
    annualization = math.sqrt(periods_per_year)
    portfolio = pd.read_parquet(portfolio_path)
    pair_returns = pd.read_parquet(pair_returns_path)
    pair_metrics = pd.read_csv(pair_metrics_path)
    fixed = fixed_universe_portfolio_returns(pair_returns)
    n_candidates = _candidate_pairs(directory, fallback=int(pair_metrics["pair"].nunique()))

    rows: list[dict[str, float | int | str]] = []
    for method, group in portfolio.groupby("method"):
        returns = group["portfolio_return"].astype(float).dropna()
        n_obs = len(returns)
        std = float(returns.std(ddof=1)) if n_obs > 1 else 0.0
        if n_obs < 2 or std == 0.0:
            continue
        per_obs_sharpe = float(returns.mean()) / std
        skew = float(returns.skew())
        kurtosis = float(returns.kurt()) + 3.0  # pandas reports excess kurtosis

        method_pairs = pair_metrics.loc[pair_metrics["method"] == method]
        trial_sharpes = method_pairs["sharpe_ratio"].astype(float) / annualization
        trial_variance = float(trial_sharpes.var(ddof=1)) if len(trial_sharpes) > 1 else 0.0

        psr = probabilistic_sharpe_ratio(
            per_obs_sharpe, n_observations=n_obs, skew=skew, kurtosis=kurtosis
        )
        dsr = deflated_sharpe_ratio(
            per_obs_sharpe,
            n_observations=n_obs,
            n_trials=n_candidates,
            trial_sharpe_variance=trial_variance,
            skew=skew,
            kurtosis=kurtosis,
        )
        sr_star = expected_max_sharpe(n_trials=n_candidates, trial_sharpe_variance=trial_variance)

        fixed_group = fixed.loc[fixed["method"] == method]
        fixed_metrics = calculate_metrics(
            fixed_group["portfolio_return"].astype(float),
            fixed_group["active_pairs"] > 0,
            fixed_group["turnover"].astype(float),
            int(periods_per_year),
        )

        rows.append(
            {
                "method": str(method),
                "observations": n_obs,
                "annualized_sharpe": per_obs_sharpe * annualization,
                "per_obs_sharpe": per_obs_sharpe,
                "skew": skew,
                "excess_kurtosis": kurtosis - 3.0,
                "n_trials": n_candidates,
                "expected_max_sharpe_per_obs": sr_star,
                "probabilistic_sharpe_ratio": psr,
                "deflated_sharpe_ratio": dsr,
                "fixed_universe_annualized_sharpe": fixed_metrics.sharpe_ratio,
                "fixed_universe_total_return": fixed_metrics.total_return,
            }
        )
    return rows


def main() -> None:
    summary_rows: list[dict[str, float | int | str]] = []
    for name in EXPERIMENTS:
        directory = REPO_ROOT / "experiments" / name
        if not directory.exists():
            continue
        rows = score_experiment(directory)
        if not rows:
            print(f"{name}: no scorable artifacts found")
            continue
        frame = pd.DataFrame(rows)
        (directory / "pairs_significance.csv").write_text(frame.to_csv(index=False))
        (directory / "pairs_significance.json").write_text(
            json.dumps({"experiment_id": name, "methods": rows}, indent=2, sort_keys=True)
        )
        summary_rows.extend({"experiment": name, **row} for row in rows)

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        cols = [
            "experiment",
            "method",
            "observations",
            "annualized_sharpe",
            "probabilistic_sharpe_ratio",
            "deflated_sharpe_ratio",
            "fixed_universe_annualized_sharpe",
        ]
        with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
            print(summary.loc[:, cols].to_string(index=False))


if __name__ == "__main__":
    main()
