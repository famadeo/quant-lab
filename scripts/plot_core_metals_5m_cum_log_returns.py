"""Plot cumulative log returns for the core metals 5-minute return panel."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = Path("experiments/HYP-0041-core-metals-5m-log-returns")
RETURNS_PATH = OUTPUT_DIR / "core_metals_5m_log_returns_wide.parquet"

ROOTS = ["GC", "SI", "HG", "PL", "PA"]
COLORS = {
    "GC": "#b68b00",
    "SI": "#7a8591",
    "HG": "#b35c2e",
    "PL": "#3b6ea8",
    "PA": "#5f8f5f",
}


def load_cumulative_returns() -> pd.DataFrame:
    frame = pd.read_parquet(RETURNS_PATH)
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    frame = frame.sort_values("ts").set_index("ts")
    returns = frame[ROOTS].ffill().fillna(0.0)
    return returns.cumsum()


def plot_overlay(cumulative_daily: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for root in ROOTS:
        final_value = cumulative_daily[root].iloc[-1]
        ax.plot(
            cumulative_daily.index,
            cumulative_daily[root],
            label=f"{root} ({final_value:+.2f})",
            color=COLORS[root],
            linewidth=1.4,
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Core metals cumulative 5-minute log returns")
    ax.set_ylabel("Cumulative log return")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=len(ROOTS), loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "core_metals_5m_cumulative_log_returns_overlay.png", dpi=170)
    plt.close(fig)


def plot_panels(cumulative_daily: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(ROOTS), 1, figsize=(13, 10), sharex=True)
    for ax, root in zip(axes, ROOTS, strict=True):
        final_value = cumulative_daily[root].iloc[-1]
        ax.plot(
            cumulative_daily.index,
            cumulative_daily[root],
            color=COLORS[root],
            linewidth=1.2,
        )
        ax.axhline(0.0, color="black", linewidth=0.7)
        ax.set_ylabel(root)
        ax.text(
            0.99,
            0.82,
            f"final {final_value:+.2f}",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=9,
        )
        ax.grid(True, alpha=0.25)
    axes[0].set_title("Core metals cumulative 5-minute log returns by asset")
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "core_metals_5m_cumulative_log_returns_panels.png", dpi=170)
    plt.close(fig)


def main() -> None:
    cumulative = load_cumulative_returns()
    cumulative_daily = cumulative.resample("1D").last().dropna(how="all")
    cumulative_daily.to_csv(OUTPUT_DIR / "core_metals_5m_cumulative_log_returns_daily.csv")
    plot_overlay(cumulative_daily)
    plot_panels(cumulative_daily)
    print(f"Wrote cumulative return plots to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
