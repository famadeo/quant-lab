"""Plot cost of carry for PC1-PC2 residual baskets in the core metals complex."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
PCA_DIR = REPO_ROOT / "experiments" / "HYP-0042-core-metals-robust-ewma-pca"
RETURNS_DIR = REPO_ROOT / "experiments" / "HYP-0041-core-metals-5m-log-returns"
FUNDING_DIR = REPO_ROOT / "experiments" / "HYP-0036-metals-hourly-funding"

STATE_PATH = PCA_DIR / "robust_ewma_pca_state.parquet"
WIDE_RETURNS_PATH = RETURNS_DIR / "core_metals_5m_log_returns_wide.parquet"
LONG_RETURNS_PATH = RETURNS_DIR / "core_metals_5m_log_returns_long.parquet"
FUNDING_PATH = FUNDING_DIR / "hourly_funding.parquet"

ROOTS = ["GC", "SI", "HG", "PL", "PA"]
TARGET_MONTHS = 3
BARS_PER_DAY = 288
VOL_HALFLIFE_BARS = 3 * BARS_PER_DAY
VOL_MIN_OBS = BARS_PER_DAY
FUNDING_ASOF_TOLERANCE = pd.Timedelta("14D")

COLORS = {
    "GC": "#b68b00",
    "SI": "#7a8591",
    "HG": "#b35c2e",
    "PL": "#3b6ea8",
    "PA": "#5f8f5f",
}


def load_state() -> pd.DataFrame:
    state = pd.read_parquet(STATE_PATH)
    state["ts"] = pd.to_datetime(state["ts"], utc=True)
    return state.sort_values("ts").reset_index(drop=True)


def load_returns_and_mask() -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = pd.read_parquet(WIDE_RETURNS_PATH)
    wide["ts"] = pd.to_datetime(wide["ts"], utc=True)
    returns = wide.sort_values("ts").set_index("ts")[ROOTS].astype("float64")

    long = pd.read_parquet(
        LONG_RETURNS_PATH,
        columns=["root", "ts", "had_observed_5m_bar"],
    )
    long["ts"] = pd.to_datetime(long["ts"], utc=True)
    mask = (
        long.pivot(index="ts", columns="root", values="had_observed_5m_bar")
        .reindex(index=returns.index, columns=ROOTS)
        .fillna(False)
        .astype(bool)
    )
    return returns, mask


def lagged_ewma_vol(returns: pd.DataFrame, observed_mask: pd.DataFrame) -> pd.DataFrame:
    observed_returns = returns.where(observed_mask)
    return (
        observed_returns.ewm(
            halflife=VOL_HALFLIFE_BARS,
            min_periods=VOL_MIN_OBS,
            adjust=False,
            ignore_na=True,
        )
        .std()
        .shift(1)
    )


def load_funding_at_state_timestamps(state_ts: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    funding = pd.read_parquet(
        FUNDING_PATH,
        columns=["root", "ts", "target_months", "funding_pct_ann"],
    )
    funding["ts"] = pd.to_datetime(funding["ts"], utc=True)
    funding = funding[funding["target_months"].eq(TARGET_MONTHS)].sort_values("ts")

    base = pd.DataFrame({"ts": state_ts.sort_values().drop_duplicates()})
    aligned_frames = []
    age_frames = []
    for root in ROOTS:
        root_funding = (
            funding[funding["root"].eq(root)][["ts", "funding_pct_ann"]]
            .dropna()
            .sort_values("ts")
            .rename(columns={"ts": "funding_ts", "funding_pct_ann": root})
        )
        aligned = pd.merge_asof(
            base,
            root_funding,
            left_on="ts",
            right_on="funding_ts",
            direction="backward",
            tolerance=FUNDING_ASOF_TOLERANCE,
        )
        aligned_frames.append(aligned[["ts", root]].set_index("ts"))
        age = (aligned["ts"] - aligned["funding_ts"]).dt.total_seconds() / 3600.0
        age_frames.append(pd.DataFrame({"ts": aligned["ts"], root: age}).set_index("ts"))

    return pd.concat(aligned_frames, axis=1), pd.concat(age_frames, axis=1)


def projection_matrix(row: pd.Series) -> np.ndarray:
    loadings = np.column_stack(
        [
            [row[f"pc1_loading_{root}"] for root in ROOTS],
            [row[f"pc2_loading_{root}"] for root in ROOTS],
        ]
    ).astype("float64")
    return np.eye(len(ROOTS), dtype="float64") - loadings @ loadings.T


def residual_carry_series() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    state = load_state()
    returns, observed_mask = load_returns_and_mask()
    vol = lagged_ewma_vol(returns, observed_mask).reindex(state["ts"])
    funding, funding_age = load_funding_at_state_timestamps(state["ts"])
    funding = funding.reindex(state["ts"])
    funding_age = funding_age.reindex(state["ts"])

    carry_rows = []
    weight_rows = []
    root_index = {root: index for index, root in enumerate(ROOTS)}
    for row in state.itertuples(index=False):
        ts = row.ts
        vol_row = vol.loc[ts, ROOTS].to_numpy(dtype="float64")
        funding_row = funding.loc[ts, ROOTS].to_numpy(dtype="float64")
        valid = np.isfinite(vol_row).all() and np.isfinite(funding_row).all()

        carry_row: dict[str, float | pd.Timestamp] = {"ts": ts}
        weight_row: dict[str, float | pd.Timestamp] = {"ts": ts}
        if not valid or np.any(vol_row <= 0):
            for root in ROOTS:
                carry_row[f"{root}_carry_pct_ann"] = np.nan
                for hedge_root in ROOTS:
                    weight_row[f"{root}_w_{hedge_root}"] = np.nan
            carry_rows.append(carry_row)
            weight_rows.append(weight_row)
            continue

        state_row = pd.Series(row._asdict())
        projector = projection_matrix(state_row)
        for root in ROOTS:
            standardized_weights = projector[root_index[root], :]
            raw_weights = standardized_weights / vol_row
            gross = np.abs(raw_weights).sum()
            if gross <= 0 or not np.isfinite(gross):
                carry_row[f"{root}_carry_pct_ann"] = np.nan
                normalized_weights = np.full(len(ROOTS), np.nan)
            else:
                normalized_weights = raw_weights / gross
                carry_row[f"{root}_carry_pct_ann"] = float(
                    np.dot(normalized_weights, funding_row)
                )
            for hedge_root, weight in zip(ROOTS, normalized_weights, strict=True):
                weight_row[f"{root}_w_{hedge_root}"] = float(weight)

        carry_rows.append(carry_row)
        weight_rows.append(weight_row)

    carry = pd.DataFrame(carry_rows).sort_values("ts")
    weights = pd.DataFrame(weight_rows).sort_values("ts")
    funding_age = funding_age.reset_index()
    return carry, weights, funding_age


def daily_median(carry: pd.DataFrame) -> pd.DataFrame:
    data = carry.copy()
    data["ts"] = pd.to_datetime(data["ts"], utc=True)
    return data.set_index("ts").resample("1D").median().dropna(how="all")


def plot_overlay(daily: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for root in ROOTS:
        column = f"{root}_carry_pct_ann"
        latest = daily[column].dropna().iloc[-1]
        ax.plot(
            daily.index,
            daily[column],
            label=f"{root} ({latest:+.2f}%)",
            color=COLORS[root],
            linewidth=1.2,
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(
        f"PC1-PC2 residual basket cost of carry ({TARGET_MONTHS}M funding proxy)"
    )
    ax.set_ylabel("Annualized carry cost paid by long residual basket (%)")
    ax.set_xlabel("Date")
    ax.legend(ncol=len(ROOTS), loc="upper left", frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(PCA_DIR / "pc12_residual_carry_cost_overlay.png", dpi=170)
    plt.close(fig)


def plot_panels(daily: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(ROOTS), 1, figsize=(13, 10), sharex=True)
    for ax, root in zip(axes, ROOTS, strict=True):
        column = f"{root}_carry_pct_ann"
        latest = daily[column].dropna().iloc[-1]
        ax.plot(daily.index, daily[column], color=COLORS[root], linewidth=1.1)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.text(
            0.99,
            0.82,
            f"latest {latest:+.2f}%",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=9,
        )
        ax.set_ylabel(root)
        ax.grid(True, alpha=0.25)
    axes[0].set_title(
        f"PC1-PC2 residual basket carry cost by asset ({TARGET_MONTHS}M funding proxy)"
    )
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(PCA_DIR / "pc12_residual_carry_cost_panels.png", dpi=170)
    plt.close(fig)


def summarize(carry: pd.DataFrame, funding_age: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for root in ROOTS:
        values = carry[f"{root}_carry_pct_ann"].dropna()
        age = funding_age[ROOTS].max(axis=1).dropna()
        rows.append(
            {
                "root": root,
                "nobs": len(values),
                "mean_carry_pct_ann": values.mean(),
                "median_carry_pct_ann": values.median(),
                "p10_carry_pct_ann": values.quantile(0.10),
                "p90_carry_pct_ann": values.quantile(0.90),
                "latest_carry_pct_ann": values.iloc[-1],
                "max_funding_age_hours": age.max(),
                "median_max_funding_age_hours": age.median(),
            }
        )
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame) -> None:
    lines = [
        "# Core Metals PC1-PC2 Residual Cost of Carry",
        "",
        "Definition:",
        "",
        "- For each asset residual, build the PC1-PC2-neutral projection row "
        "`e_i' (I - V V')` in standardized-return space.",
        "- Convert standardized residual weights to raw-return weights by dividing by each "
        "asset's lagged EWMA volatility.",
        "- Gross-normalize raw weights to one.",
        f"- Apply `{TARGET_MONTHS}M` annualized curve-implied funding from HYP-0036.",
        "- Positive carry means the long residual basket pays carry; negative means it earns "
        "carry under the funding sign convention.",
        "",
        "Caveats:",
        "",
        "- This is a futures-curve proxy, not true cash/spot carry.",
        "- The residual basket is a PCA/risk-space proxy, not an executable calendar-spread "
        "trade definition.",
        "- Funding is aligned by backward as-of match with a 14-day maximum tolerance.",
        "- Plots use daily medians of hourly PCA diagnostic timestamps.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Files",
        "",
        "- `pc12_residual_carry_cost_pct_ann.parquet`",
        "- `pc12_residual_carry_cost_daily.csv`",
        "- `pc12_residual_carry_weights.parquet`",
        "- `pc12_residual_carry_funding_age_hours.csv`",
        "- `pc12_residual_carry_summary.csv`",
        "- `pc12_residual_carry_cost_overlay.png`",
        "- `pc12_residual_carry_cost_panels.png`",
    ]
    (PCA_DIR / "pc12_residual_carry_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    carry, weights, funding_age = residual_carry_series()
    daily = daily_median(carry)
    summary = summarize(carry, funding_age)

    carry.to_parquet(PCA_DIR / "pc12_residual_carry_cost_pct_ann.parquet", index=False)
    daily.to_csv(PCA_DIR / "pc12_residual_carry_cost_daily.csv")
    weights.to_parquet(PCA_DIR / "pc12_residual_carry_weights.parquet", index=False)
    funding_age.to_csv(PCA_DIR / "pc12_residual_carry_funding_age_hours.csv", index=False)
    summary.to_csv(PCA_DIR / "pc12_residual_carry_summary.csv", index=False)
    plot_overlay(daily)
    plot_panels(daily)
    write_report(summary)
    print(f"Wrote PC1-PC2 residual carry outputs to {PCA_DIR}", flush=True)


if __name__ == "__main__":
    main()
