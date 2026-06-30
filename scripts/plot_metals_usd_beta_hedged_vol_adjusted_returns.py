"""Plot USD-beta-hedged, vol-adjusted log returns for core metals."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path("/home/famadeo/research/databento-asset-browser/data/futures_continuous")
OUT_DIR = Path("experiments/HYP-0027-metals-usd-beta-hedged-vol-adjusted-returns")

METALS = ["GC", "SI", "HG", "PL", "PA"]
FX = ["6E", "6B", "6J", "6A", "6C"]
ALL_ROOTS = [*METALS, *FX]

BETA_WINDOW = 252
BETA_MIN_OBS = 189
VOL_WINDOW = 63
TARGET_ANN_VOL = 0.10
TRADE_OVERLAP_START = pd.Timestamp("2023-06-22", tz="UTC")


def read_root_returns(root: str) -> pd.Series:
    frame = pd.read_csv(DATA_DIR / f"{root}.csv", usecols=["date", "cont_logret"])
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    return frame.set_index("date")["cont_logret"].rename(root)


def load_returns() -> pd.DataFrame:
    returns = pd.concat([read_root_returns(root) for root in ALL_ROOTS], axis=1)
    returns = returns.sort_index().replace([np.inf, -np.inf], np.nan)
    return returns.dropna(how="all")


def rolling_univariate_beta(y: pd.Series, x: pd.Series) -> pd.Series:
    cov = y.rolling(BETA_WINDOW, min_periods=BETA_MIN_OBS).cov(x)
    var = x.rolling(BETA_WINDOW, min_periods=BETA_MIN_OBS).var()
    beta = cov / var
    return beta.shift(1)


def build_series(returns: pd.DataFrame) -> pd.DataFrame:
    usd = -returns[FX].mean(axis=1, skipna=False).rename("usd")
    rows = []
    for root in METALS:
        metal = returns[root]
        beta = rolling_univariate_beta(metal, usd).rename("usd_beta_lagged")
        hedged = (metal - beta * usd).rename("usd_beta_hedged_logret")
        ann_vol = (
            hedged.rolling(VOL_WINDOW, min_periods=VOL_WINDOW)
            .std(ddof=1)
            .mul(np.sqrt(252))
            .shift(1)
            .rename("hedged_ann_vol_lagged")
        )
        vol_adjusted = (hedged * TARGET_ANN_VOL / ann_vol).rename("vol_adjusted_hedged_logret")
        standardized = (
            hedged / hedged.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std(ddof=1).shift(1)
        ).rename("standardized_hedged_logret")
        root_frame = pd.concat(
            [
                metal.rename("raw_logret"),
                usd,
                beta,
                hedged,
                ann_vol,
                vol_adjusted,
                standardized,
            ],
            axis=1,
        )
        root_frame["root"] = root
        rows.append(root_frame)
    return pd.concat(rows).rename_axis("date").reset_index()


def cumulative_wide(
    frame: pd.DataFrame, value_col: str, start: pd.Timestamp | None = None
) -> pd.DataFrame:
    data = frame.copy()
    if start is not None:
        data = data[data["date"] >= start]
    wide = data.pivot(index="date", columns="root", values=value_col).sort_index()
    wide = wide.dropna(how="all")
    return wide.fillna(0.0).cumsum()


def plot_cumulative(frame: pd.DataFrame, out_path: Path) -> None:
    full = cumulative_wide(frame, "vol_adjusted_hedged_logret")
    trade = cumulative_wide(
        frame,
        "vol_adjusted_hedged_logret",
        start=TRADE_OVERLAP_START,
    )

    colors = {
        "GC": "#b68b00",
        "SI": "#7a8591",
        "HG": "#b35c2e",
        "PL": "#3b6ea8",
        "PA": "#5f8f5f",
    }
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharey=False)
    for root in METALS:
        if root in full:
            axes[0].plot(full.index, full[root], label=root, color=colors[root])
        if root in trade:
            axes[1].plot(trade.index, trade[root], label=root, color=colors[root])

    axes[0].set_title("USD-beta-hedged cumulative vol-adjusted log returns")
    axes[0].set_ylabel("Cumulative log return at 10% target vol")
    axes[1].set_title("Trade-era overlap reset")
    axes[1].set_ylabel("Cumulative log return at 10% target vol")
    axes[1].set_xlabel("Date")
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(True, alpha=0.25)
    axes[0].legend(ncol=len(METALS), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sample, sample_frame in [
        ("full_available", frame),
        ("trade_overlap", frame[frame["date"] >= TRADE_OVERLAP_START]),
    ]:
        for root, grouped_frame in sample_frame.groupby("root"):
            root_frame = grouped_frame.dropna(subset=["vol_adjusted_hedged_logret"])
            if root_frame.empty:
                continue
            ann_ret = root_frame["vol_adjusted_hedged_logret"].mean() * 252
            ann_vol = root_frame["vol_adjusted_hedged_logret"].std(ddof=1) * np.sqrt(252)
            rows.append(
                {
                    "sample": sample,
                    "root": root,
                    "start": root_frame["date"].min(),
                    "end": root_frame["date"].max(),
                    "nobs": len(root_frame),
                    "cum_vol_adjusted_logret": root_frame["vol_adjusted_hedged_logret"].sum(),
                    "ann_return": ann_ret,
                    "ann_vol": ann_vol,
                    "sharpe_like": ann_ret / ann_vol if ann_vol > 0 else np.nan,
                    "mean_usd_beta": root_frame["usd_beta_lagged"].mean(),
                    "last_usd_beta": root_frame["usd_beta_lagged"].iloc[-1],
                    "mean_hedged_ann_vol": root_frame["hedged_ann_vol_lagged"].mean(),
                }
            )
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, out_path: Path) -> None:
    trade = summary[summary["sample"] == "trade_overlap"].copy()
    full = summary[summary["sample"] == "full_available"].copy()
    report = [
        "# HYP-0027 Metals USD-Beta-Hedged Vol-Adjusted Returns",
        "",
        f"Completed at `{datetime.now(UTC).isoformat()}`.",
        "",
        "## Method",
        "",
        "- USD factor: negative equal-weight daily log return of `6E`, `6B`, `6J`, `6A`, `6C`.",
        "- Hedge: `metal_logret - lagged_rolling_252d_usd_beta * usd_factor`.",
        "- Vol adjustment: hedged log return scaled to 10% annualized volatility "
        "using lagged 63d realized vol.",
        "- Coverage follows the daily continuous factor store, which ends at `2024-11-29`.",
        "",
        "## Trade-Era Overlap",
        "",
        trade[
            [
                "root",
                "nobs",
                "cum_vol_adjusted_logret",
                "sharpe_like",
                "mean_usd_beta",
                "last_usd_beta",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Full Available History",
        "",
        full[
            [
                "root",
                "nobs",
                "cum_vol_adjusted_logret",
                "sharpe_like",
                "mean_usd_beta",
                "last_usd_beta",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Files",
        "",
        "- `usd_beta_hedged_vol_adjusted_returns.csv`",
        "- `summary.csv`",
        "- `usd_beta_hedged_vol_adjusted_cumulative.png`",
    ]
    out_path.write_text("\n".join(report) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    returns = load_returns()
    frame = build_series(returns)
    frame.to_csv(OUT_DIR / "usd_beta_hedged_vol_adjusted_returns.csv", index=False)

    summary = summarize(frame)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    plot_cumulative(frame, OUT_DIR / "usd_beta_hedged_vol_adjusted_cumulative.png")
    write_report(summary, OUT_DIR / "report.md")

    result = {
        "completed_at": datetime.now(UTC).isoformat(),
        "data_dir": str(DATA_DIR),
        "out_dir": str(OUT_DIR),
        "metals": METALS,
        "fx_contracts": FX,
        "beta_window": BETA_WINDOW,
        "beta_min_obs": BETA_MIN_OBS,
        "vol_window": VOL_WINDOW,
        "target_ann_vol": TARGET_ANN_VOL,
        "start": frame["date"].min().isoformat(),
        "end": frame["date"].max().isoformat(),
    }
    (OUT_DIR / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
