"""Estimate metals beta to a Rates + USD macro factor set."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path("/home/famadeo/research/databento-asset-browser/data/futures_continuous")
OUT_DIR = Path("experiments/HYP-0026-metals-rates-usd-beta")

METALS = ["GC", "SI", "HG", "PL", "PA"]
RATES = ["ZT", "ZF", "ZN", "ZB"]
FX = ["6E", "6B", "6J", "6A", "6C"]
ALL_ROOTS = [*METALS, *RATES, *FX]

ROLLING_WINDOW = 252
ROLLING_MIN_OBS = 189
MIN_OLS_OBS = 30
TRADE_OVERLAP_START = pd.Timestamp("2023-06-22", tz="UTC")


@dataclass(frozen=True)
class FitResult:
    root: str
    sample: str
    start: str
    end: str
    nobs: int
    alpha: float
    beta_rates_price: float
    beta_usd: float
    beta_rates_price_t: float
    beta_usd_t: float
    std_beta_rates_price: float
    std_beta_usd: float
    metal_bp_per_1pct_rates_price: float
    metal_bp_per_1pct_usd: float
    r2: float
    residual_vol_ann: float
    raw_vol_ann: float
    macro_vol_share: float


def read_root_returns(root: str) -> pd.Series:
    path = DATA_DIR / f"{root}.csv"
    frame = pd.read_csv(path, usecols=["date", "cont_logret"])
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    return frame.set_index("date")["cont_logret"].rename(root)


def load_returns() -> pd.DataFrame:
    returns = pd.concat([read_root_returns(root) for root in ALL_ROOTS], axis=1)
    returns = returns.sort_index().replace([np.inf, -np.inf], np.nan)
    return returns.dropna(how="all")


def build_factors(returns: pd.DataFrame) -> pd.DataFrame:
    factors = pd.DataFrame(index=returns.index)
    factors["rates_price"] = returns[RATES].mean(axis=1, skipna=False)
    factors["usd"] = -returns[FX].mean(axis=1, skipna=False)
    factors["rates_2y"] = returns["ZT"]
    factors["rates_5y"] = returns["ZF"]
    factors["rates_10y"] = returns["ZN"]
    factors["rates_30y"] = returns["ZB"]
    factors["usd_fx_contracts_available"] = returns[FX].notna().sum(axis=1)
    factors["rates_contracts_available"] = returns[RATES].notna().sum(axis=1)
    return factors


def fit_ols(y: pd.Series, x: pd.DataFrame, root: str, sample: str) -> FitResult | None:
    frame = pd.concat([y.rename("metal"), x[["rates_price", "usd"]]], axis=1).dropna()
    if len(frame) < MIN_OLS_OBS:
        return None

    y_arr = frame["metal"].to_numpy(dtype=float)
    x_arr = frame[["rates_price", "usd"]].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(frame)), x_arr])
    coef, *_ = np.linalg.lstsq(design, y_arr, rcond=None)
    fitted = design @ coef
    resid = y_arr - fitted

    dof = len(frame) - design.shape[1]
    sse = float(np.square(resid).sum())
    centered = y_arr - y_arr.mean()
    sst = float(np.square(centered).sum())
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    sigma2 = sse / dof if dof > 0 else np.nan
    xtx_inv = np.linalg.pinv(design.T @ design)
    se = np.sqrt(np.diag(xtx_inv) * sigma2)
    tstats = np.divide(coef, se, out=np.full_like(coef, np.nan), where=se > 0)

    y_std = frame["metal"].std(ddof=1)
    factor_std = frame[["rates_price", "usd"]].std(ddof=1)
    raw_vol = y_std * np.sqrt(252)
    residual_vol = pd.Series(resid, index=frame.index).std(ddof=1) * np.sqrt(252)
    macro_vol_share = 1.0 - residual_vol / raw_vol if raw_vol > 0 else np.nan

    return FitResult(
        root=root,
        sample=sample,
        start=frame.index.min().isoformat(),
        end=frame.index.max().isoformat(),
        nobs=len(frame),
        alpha=float(coef[0]),
        beta_rates_price=float(coef[1]),
        beta_usd=float(coef[2]),
        beta_rates_price_t=float(tstats[1]),
        beta_usd_t=float(tstats[2]),
        std_beta_rates_price=float(coef[1] * factor_std["rates_price"] / y_std),
        std_beta_usd=float(coef[2] * factor_std["usd"] / y_std),
        metal_bp_per_1pct_rates_price=float(coef[1] * 100.0),
        metal_bp_per_1pct_usd=float(coef[2] * 100.0),
        r2=float(r2),
        residual_vol_ann=float(residual_vol),
        raw_vol_ann=float(raw_vol),
        macro_vol_share=float(macro_vol_share),
    )


def rolling_fit(root: str, y: pd.Series, factors: pd.DataFrame) -> pd.DataFrame:
    frame = pd.concat([y.rename("metal"), factors[["rates_price", "usd"]]], axis=1).dropna()
    rows: list[dict[str, float | int | str | pd.Timestamp]] = []
    for end_pos in range(ROLLING_MIN_OBS, len(frame) + 1):
        window = frame.iloc[max(0, end_pos - ROLLING_WINDOW) : end_pos]
        if len(window) < ROLLING_MIN_OBS:
            continue
        result = fit_ols(
            window["metal"],
            window[["rates_price", "usd"]],
            root=root,
            sample=f"rolling_{ROLLING_WINDOW}d",
        )
        if result is None:
            continue
        row = asdict(result)
        row["date"] = window.index[-1]
        rows.append(row)
    return pd.DataFrame(rows)


def fitted_and_residuals(
    returns: pd.DataFrame, factors: pd.DataFrame, betas: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    sample_betas = betas[betas["sample"] == "trade_overlap"].set_index("root")
    for root in METALS:
        frame = pd.concat(
            [returns[root].rename("metal"), factors[["rates_price", "usd"]]],
            axis=1,
        ).dropna()
        beta = sample_betas.loc[root]
        fitted = (
            beta["alpha"]
            + beta["beta_rates_price"] * frame["rates_price"]
            + beta["beta_usd"] * frame["usd"]
        )
        root_frame = pd.DataFrame(
            {
                "root": root,
                "metal_return": frame["metal"],
                "macro_fitted_return": fitted,
                "macro_residual_return": frame["metal"] - fitted,
            },
            index=frame.index,
        )
        rows.append(root_frame)
    return pd.concat(rows).rename_axis("date").reset_index()


def plot_latest_betas(latest: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    latest = latest.sort_values("root")
    axes[0].barh(latest["root"], latest["metal_bp_per_1pct_usd"], color="#2f6f9f")
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title("USD beta")
    axes[0].set_xlabel("metal bp per +1% USD basket")

    axes[1].barh(
        latest["root"],
        latest["metal_bp_per_1pct_rates_price"],
        color="#8d5a2b",
    )
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("Rates-price beta")
    axes[1].set_xlabel("metal bp per +1% Treasury basket")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_rolling(rolling: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for root, root_group in rolling.groupby("root"):
        sorted_group = root_group.sort_values("date")
        axes[0].plot(
            sorted_group["date"],
            sorted_group["metal_bp_per_1pct_usd"],
            label=root,
        )
        axes[1].plot(
            sorted_group["date"],
            sorted_group["metal_bp_per_1pct_rates_price"],
            label=root,
        )
        axes[2].plot(sorted_group["date"], sorted_group["r2"], label=root)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Rolling USD beta")
    axes[1].set_title("Rolling rates-price beta")
    axes[2].set_title("Rolling macro R2")
    axes[0].set_ylabel("bp per +1% USD")
    axes[1].set_ylabel("bp per +1% Treasury basket")
    axes[2].set_ylabel("R2")
    axes[2].legend(ncol=len(METALS), loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_report(
    betas: pd.DataFrame,
    latest: pd.DataFrame,
    factors: pd.DataFrame,
    out_path: Path,
) -> None:
    trade = betas[betas["sample"] == "trade_overlap"].copy()
    trade = trade.sort_values("root")
    latest = latest.sort_values("root")
    factor_corr = factors[["rates_price", "usd"]].dropna().corr().iloc[0, 1]
    report = [
        "# HYP-0026 Metals Rates + USD Beta",
        "",
        f"Completed at `{datetime.now(UTC).isoformat()}`.",
        "",
        "## Definition",
        "",
        "- `rates_price`: equal-weight daily log return of `ZT`, `ZF`, `ZN`, `ZB`. "
        "Positive means Treasury futures prices up / yields lower.",
        "- `usd`: negative equal-weight daily log return of `6E`, `6B`, `6J`, "
        "`6A`, `6C`. Positive means broad USD stronger versus those FX futures.",
        "- Regression: `metal_return = alpha + beta_rates_price * rates_price "
        "+ beta_usd * usd + residual`.",
        "- Beta columns in bp are metal return bp for a +1% move in the factor.",
        "",
        "## Coverage",
        "",
        f"- Factor span: `{factors.index.min().date()}` to `{factors.index.max().date()}`.",
        f"- Trade-era overlap used here: `{TRADE_OVERLAP_START.date()}` to "
        f"`{factors.index.max().date()}`.",
        f"- Rates/USD factor correlation: `{factor_corr:.3f}`.",
        "",
        "## Trade-Era Overlap Betas",
        "",
        trade[
            [
                "root",
                "nobs",
                "metal_bp_per_1pct_usd",
                "metal_bp_per_1pct_rates_price",
                "r2",
                "beta_usd_t",
                "beta_rates_price_t",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Latest Rolling 252d Betas",
        "",
        latest[
            [
                "root",
                "date",
                "metal_bp_per_1pct_usd",
                "metal_bp_per_1pct_rates_price",
                "r2",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Files",
        "",
        "- `factor_returns_daily.csv`",
        "- `full_sample_betas.csv`",
        "- `rolling_252d_betas.csv`",
        "- `latest_rolling_252d_betas.csv`",
        "- `macro_fitted_residual_returns.parquet`",
        "- `latest_beta_bars.png`",
        "- `rolling_betas_and_r2.png`",
    ]
    out_path.write_text("\n".join(report) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    returns = load_returns()
    factors = build_factors(returns).dropna(subset=["rates_price", "usd"])
    returns = returns.reindex(factors.index)

    factors.to_csv(OUT_DIR / "factor_returns_daily.csv", index_label="date")

    sample_defs = {
        "full_history": (factors.index.min(), factors.index.max()),
        "trade_overlap": (TRADE_OVERLAP_START, factors.index.max()),
    }
    fit_rows = []
    for sample_name, (start, end) in sample_defs.items():
        sample_factors = factors.loc[start:end]
        for root in METALS:
            result = fit_ols(
                returns[root].loc[start:end],
                sample_factors,
                root=root,
                sample=sample_name,
            )
            if result is not None:
                fit_rows.append(asdict(result))
    betas = pd.DataFrame(fit_rows)
    betas.to_csv(OUT_DIR / "full_sample_betas.csv", index=False)

    rolling = pd.concat(
        [rolling_fit(root, returns[root], factors) for root in METALS],
        ignore_index=True,
    )
    rolling.to_csv(OUT_DIR / "rolling_252d_betas.csv", index=False)
    latest = rolling.sort_values("date").groupby("root", as_index=False).tail(1)
    latest.to_csv(OUT_DIR / "latest_rolling_252d_betas.csv", index=False)

    residuals = fitted_and_residuals(returns, factors, betas)
    residuals.to_parquet(OUT_DIR / "macro_fitted_residual_returns.parquet", index=False)

    plot_latest_betas(latest, OUT_DIR / "latest_beta_bars.png")
    plot_rolling(rolling, OUT_DIR / "rolling_betas_and_r2.png")
    write_report(betas, latest, factors, OUT_DIR / "report.md")

    summary = {
        "completed_at": datetime.now(UTC).isoformat(),
        "data_dir": str(DATA_DIR),
        "out_dir": str(OUT_DIR),
        "metals": METALS,
        "rates_contracts": RATES,
        "fx_contracts": FX,
        "factor_start": factors.index.min().isoformat(),
        "factor_end": factors.index.max().isoformat(),
        "rolling_window": ROLLING_WINDOW,
        "rolling_min_obs": ROLLING_MIN_OBS,
        "trade_overlap_start": TRADE_OVERLAP_START.isoformat(),
    }
    (OUT_DIR / "results.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
