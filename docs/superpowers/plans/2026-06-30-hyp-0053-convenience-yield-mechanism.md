# HYP-0053 Convenience-Yield Mechanism Study — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the diagnostic analysis that decides whether the HYP-0030/0031 metals curve-carry reversion edge is genuine convenience-yield basis compression — vs a z-score re-centering artifact, disguised financing rates, or a roll-cycle effect.

**Architecture:** One analysis script (`scripts/analyze_metals_convenience_yield_mechanism.py`) with small, individually-testable pure functions for the load-bearing math (z reconstruction, Δz decomposition, convenience-yield basis, OU half-life) and thin orchestration that writes six diagnostic CSVs, four plots, and a `report.md` verdict memo into the experiment directory. Pure functions are unit-tested in `tests/test_convenience_yield_mechanism.py`; orchestration is verified by running end-to-end and sanity-checking outputs.

**Tech Stack:** Python 3.12 under `/home/famadeo/quant-lab/.venv`, pandas, numpy, statsmodels (AR/OU fit), matplotlib, pytest 9.

## Global Constraints

- **Interpreter:** always `/home/famadeo/quant-lab/.venv/bin/python` and `/home/famadeo/quant-lab/.venv/bin/pytest`. Never bare `python3`.
- **Spec authority:** `experiments/HYP-0053-metals-convenience-yield-mechanism/hypothesis.md` (D1–D6, predictions, guardrails) is the source of truth. Honor its guardrails verbatim: **no new variant scan**; analyze only the pre-declared variants; event t-stats are **descriptive only** (lead with effect sizes / shares / half-lives).
- **Signal replication is exact:** reuse the construction from `scripts/run_metals_convenience_yield_basis_backtest.py` — `carry_pct_ann = log(far/anchor)/(months_from_anchor/12)*100`; `z = (carry − μ)/σ` with `μ,σ = carry.shift(1).rolling(lookback, min_periods=40).mean()/.std(ddof=0)`. Import or copy `with_lagged_zscore` and `MIN_Z_OBSERVATIONS=40`; do not re-derive.
- **Inputs (read-only, never modify):** `experiments/HYP-0031-metals-convenience-yield-basis-5m-sync/{curve_panel.parquet,event_log.csv,strategy_metrics.csv}` (primary) and the same files under `experiments/HYP-0030-metals-convenience-yield-basis/` (daily-mark contrast for D1).
- **Pre-declared variant set (no others):** best `target3m_minv10`, liquid `target1m_minv500`, liquid `target1m_minv1000`. Read each variant's `lookback`/`entry_z`/`exit_z` from `strategy_metrics.csv` (do not hard-code; the best lookback differs between 0030 lb252 and 0031 lb126).
- **Outputs go only to:** `experiments/HYP-0053-metals-convenience-yield-mechanism/`.
- **Commits:** stage **explicit file paths** (never `git add -A`/`-u`) — the repo has unrelated pre-staged files and a `ruff` pre-commit hook that fails on pre-existing notebooks. New `.py` must be ruff-clean. If the hook still trips on unrelated staged files, commit with `--no-verify` and say so.
- **Roots:** `GC, SI, HG, PL, PA`. `GC` is the financing benchmark in D3.

## File Structure

- `scripts/analyze_metals_convenience_yield_mechanism.py` — all analysis: loaders, the four pure-function cores, six aggregations, plotting, report writer, `main()`.
- `tests/test_convenience_yield_mechanism.py` — unit tests for the four pure cores (reconstruction match, decomposition identity, cy basis, OU half-life recovery).
- `experiments/HYP-0053-metals-convenience-yield-mechanism/report.md` + six CSVs + four PNGs — generated artifacts (committed).

---

### Task 1: Loader + exact z reconstruction

**Files:**
- Create: `scripts/analyze_metals_convenience_yield_mechanism.py`
- Test: `tests/test_convenience_yield_mechanism.py`

**Interfaces:**
- Produces:
  - `SOURCES = {"sync": ".../HYP-0031-...", "daily": ".../HYP-0030-..."}` (dict of dirs)
  - `VARIANTS = ["target3m_minv10", "target1m_minv500", "target1m_minv1000"]`
  - `reconstruct_z(panel: pd.DataFrame, target_months: int, min_volume: float, lookback: int) -> pd.DataFrame` — filters `panel` to the (target_months, min_volume) slice, sorts by `(root, date)`, applies the lagged rolling z per root, returns the slice with added columns `carry_mean_lagged`, `carry_std_lagged`, `carry_z`.
  - `load_variant(source_dir, variant_name) -> tuple[pd.DataFrame, pd.DataFrame, dict]` — returns `(z_panel, events, params)` where `events` is the `event_log` rows for `variant_name` and `params` holds `{target_months,min_volume,lookback,entry_z,exit_z}` read from `strategy_metrics.csv`.

- [ ] **Step 1: Write the failing test** (reconstruction must reproduce the traded `entry_z`)

```python
# tests/test_convenience_yield_mechanism.py
import numpy as np, pandas as pd, pathlib, importlib.util
SPEC = pathlib.Path("scripts/analyze_metals_convenience_yield_mechanism.py")
mod = importlib.util.module_from_spec(importlib.util.spec_from_file_location("cym", SPEC))
importlib.util.spec_from_file_location("cym", SPEC).loader.exec_module(mod)  # noqa

def test_reconstructed_z_matches_event_log_entry_z():
    # Reconstructing the z-score from curve_panel must reproduce the entry_z
    # the backtest actually traded, for the best sync variant.
    zpanel, events, params = mod.load_variant(mod.SOURCES["sync"], "target3m_minv10")
    merged = events.merge(
        zpanel[["root", "date", "carry_z"]],
        left_on=["root", "entry_date"], right_on=["root", "date"], how="left",
    )
    ok = merged.dropna(subset=["carry_z", "entry_z"])
    assert len(ok) > 50
    assert np.allclose(ok["carry_z"], ok["entry_z"], atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/famadeo/quant-lab/.venv/bin/pytest tests/test_convenience_yield_mechanism.py::test_reconstructed_z_matches_event_log_entry_z -v`
Expected: FAIL (module/functions not defined).

- [ ] **Step 3: Implement loaders + `reconstruct_z`**

```python
# scripts/analyze_metals_convenience_yield_mechanism.py
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd

EXP = pathlib.Path(__file__).resolve().parents[1] / "experiments"
SOURCES = {
    "sync": EXP / "HYP-0031-metals-convenience-yield-basis-5m-sync",
    "daily": EXP / "HYP-0030-metals-convenience-yield-basis",
}
OUT = EXP / "HYP-0053-metals-convenience-yield-mechanism"
VARIANTS = ["target3m_minv10", "target1m_minv500", "target1m_minv1000"]
ROOTS = ["GC", "SI", "HG", "PL", "PA"]
MIN_Z_OBSERVATIONS = 40  # matches run_metals_convenience_yield_basis_backtest.py

def reconstruct_z(panel, target_months, min_volume, lookback):
    data = panel[(panel["target_months"] == target_months)
                 & (panel["min_volume"] == min_volume)].copy()
    data = data.sort_values(["root", "date"])
    def _z(g):
        roll = g["carry_pct_ann"].shift(1).rolling(lookback, min_periods=MIN_Z_OBSERVATIONS)
        g["carry_mean_lagged"] = roll.mean()
        g["carry_std_lagged"] = roll.std(ddof=0)
        g["carry_z"] = (g["carry_pct_ann"] - g["carry_mean_lagged"]) / g["carry_std_lagged"]
        g.loc[g["carry_std_lagged"] <= 0, "carry_z"] = np.nan
        return g
    return data.groupby("root", group_keys=False).apply(_z)

def _variant_params(strategy_metrics, variant_name):
    row = strategy_metrics[strategy_metrics["variant"].str.startswith(variant_name + "_")]
    row = row[row["cost_multiplier"] == 1.0].iloc[0]
    return {k: row[k] for k in ["target_months", "min_volume", "lookback", "entry_z", "exit_z"]}

def load_variant(source_dir, variant_name):
    panel = pd.read_parquet(source_dir / "curve_panel.parquet")
    sm = pd.read_csv(source_dir / "strategy_metrics.csv")
    p = _variant_params(sm, variant_name)
    zpanel = reconstruct_z(panel, int(p["target_months"]), float(p["min_volume"]), int(p["lookback"]))
    events = pd.read_csv(source_dir / "event_log.csv", parse_dates=["entry_date", "exit_date"])
    events = events[events["variant"].str.startswith(variant_name + "_")
                    & (events["cost_multiplier"] == 1.0)].copy()
    zpanel["date"] = pd.to_datetime(zpanel["date"])
    return zpanel, events, p
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/famadeo/quant-lab/.venv/bin/pytest tests/test_convenience_yield_mechanism.py::test_reconstructed_z_matches_event_log_entry_z -v`
Expected: PASS. (If it fails on the date-join, confirm both `entry_date` and `date` are tz-aware UTC midnight; align dtypes.)

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_metals_convenience_yield_mechanism.py tests/test_convenience_yield_mechanism.py
git commit -m "HYP-0053: loader + exact z reconstruction (validated vs event_log)"
```

---

### Task 2: D1 — mirror-or-money decomposition

**Files:**
- Modify: `scripts/analyze_metals_convenience_yield_mechanism.py`
- Test: `tests/test_convenience_yield_mechanism.py`

**Interfaces:**
- Consumes: `load_variant`, the entry/exit `carry`,`carry_mean_lagged`,`carry_std_lagged` from `reconstruct_z`.
- Produces: `decompose_events(zpanel, events) -> pd.DataFrame` — one row per event with columns `root, side, entry_z, exit_z, carry_move_dz, mean_drift_dz, vol_drift_dz, total_dz, carry_move_share, gross_spread_return, weighted_cost_return, is_mirror`. Definitions: with entry `(C0,μ0,σ0)` and exit `(C1,μ1,σ1)`, `carry_move_dz=(C1−C0)/σ0`, `mean_drift_dz=−(μ1−μ0)/σ0`, `vol_drift_dz=(C1−μ1)*(1/σ1−1/σ0)`, summing to `total_dz=z1−z0`; `carry_move_share=carry_move_dz/total_dz`; `is_mirror = |gross_spread_return| < |weighted_cost_return|` (move can't cover its own round-trip cost).

- [ ] **Step 1: Write the failing test** (decomposition is an exact identity)

```python
def test_decomposition_sums_to_total_dz():
    z = pd.DataFrame({
        "root": ["HG", "HG", "HG"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"], utc=True),
        "carry_pct_ann": [10.0, 6.0, 5.0],
        "carry_mean_lagged": [4.0, 4.5, 4.5],
        "carry_std_lagged": [2.0, 2.0, 2.5],
        "carry_z": [3.0, 0.75, 0.2],
    })
    ev = pd.DataFrame({
        "root": ["HG"], "side": ["fade_contango"],
        "entry_date": pd.to_datetime(["2024-01-01"], utc=True),
        "exit_date": pd.to_datetime(["2024-01-03"], utc=True),
        "entry_z": [3.0], "exit_z": [0.2],
        "gross_spread_return": [0.004], "weighted_cost_return": [0.0002],
        "weighted_net_return": [0.0038], "duration_days": [2],
    })
    out = mod.decompose_events(z, ev)
    r = out.iloc[0]
    assert abs((r.carry_move_dz + r.mean_drift_dz + r.vol_drift_dz) - r.total_dz) < 1e-9
    assert abs(r.total_dz - (0.2 - 3.0)) < 1e-9
    assert r.carry_move_share > 0.8          # carry fell 10->5, band barely moved
    assert bool(r.is_mirror) is False         # 0.004 > 0.0002
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/famadeo/quant-lab/.venv/bin/pytest tests/test_convenience_yield_mechanism.py::test_decomposition_sums_to_total_dz -v`
Expected: FAIL (`decompose_events` not defined).

- [ ] **Step 3: Implement `decompose_events`**

```python
def decompose_events(zpanel, events):
    cols = ["root", "date", "carry_pct_ann", "carry_mean_lagged", "carry_std_lagged", "carry_z"]
    z = zpanel[cols]
    e = events.merge(z.add_suffix("_0"), left_on=["root", "entry_date"],
                     right_on=["root_0", "date_0"], how="left")
    e = e.merge(z.add_suffix("_1"), left_on=["root", "exit_date"],
                right_on=["root_1", "date_1"], how="left")
    C0, m0, s0 = e["carry_pct_ann_0"], e["carry_mean_lagged_0"], e["carry_std_lagged_0"]
    C1, m1, s1 = e["carry_pct_ann_1"], e["carry_mean_lagged_1"], e["carry_std_lagged_1"]
    e["carry_move_dz"] = (C1 - C0) / s0
    e["mean_drift_dz"] = -(m1 - m0) / s0
    e["vol_drift_dz"] = (C1 - m1) * (1.0 / s1 - 1.0 / s0)
    e["total_dz"] = e["carry_z_1"] - e["carry_z_0"]
    e["carry_move_share"] = e["carry_move_dz"] / e["total_dz"]
    e["is_mirror"] = e["gross_spread_return"].abs() < e["weighted_cost_return"].abs()
    keep = ["root", "side", "entry_z", "exit_z", "carry_move_dz", "mean_drift_dz",
            "vol_drift_dz", "total_dz", "carry_move_share", "gross_spread_return",
            "weighted_cost_return", "weighted_net_return", "duration_days", "is_mirror"]
    return e[keep]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/famadeo/quant-lab/.venv/bin/pytest tests/test_convenience_yield_mechanism.py::test_decomposition_sums_to_total_dz -v`
Expected: PASS.

- [ ] **Step 5: Wire D1 into `main()` and write `reversion_decomposition.csv`**

Add a `write_decomposition(source_key)` that runs `decompose_events` for all three `VARIANTS` on a given source, tags rows with `source`+`variant`, concatenates, aggregates per `root×side` (`median carry_move_share`, `mean |gross_spread_return|`, `% is_mirror`, `n`), writes `OUT/reversion_decomposition.csv`. Call it for both `"sync"` and `"daily"`. Show event-level and aggregate.

- [ ] **Step 6: Run end-to-end for D1 and eyeball**

Run: `/home/famadeo/quant-lab/.venv/bin/python scripts/analyze_metals_convenience_yield_mechanism.py --only d1`
Expected: `reversion_decomposition.csv` exists; `carry_move_share` medians printed per root×side; sanity — sync `target3m_minv10` carry-move share should be the headline number for "money vs mirror".

- [ ] **Step 7: Commit**

```bash
git add scripts/analyze_metals_convenience_yield_mechanism.py tests/test_convenience_yield_mechanism.py experiments/HYP-0053-metals-convenience-yield-mechanism/reversion_decomposition.csv
git commit -m "HYP-0053: D1 mirror-or-money Δz decomposition + CSV"
```

---

### Task 3: D2 attribution + D4 asymmetry

**Files:**
- Modify: `scripts/analyze_metals_convenience_yield_mechanism.py`

**Interfaces:**
- Consumes: `decompose_events` output joined with `events` (`weighted_net_return`, `duration_days`, `holding_days`).
- Produces: `attribution_tables(decomp_with_events) -> tuple[pd.DataFrame, pd.DataFrame]` → `(root_side, asymmetry)`. `root_side`: per `root×side` — `n_events, genuine_pnl (sum weighted_net_return where not is_mirror), total_pnl, win_rate, mean_duration_days, mirror_pct, median_carry_move_share`. `asymmetry`: per `root × {contango,backwardation}` — `mean_event_net, win_rate, median_carry_move_share, median_duration`, plus a `side_spread` = contango−backwardation per metric.

- [ ] **Step 1: Implement `attribution_tables`** (no separate unit test — it is grouped aggregation; verified by output sanity in Step 2)

```python
def attribution_tables(decomp):
    g = decomp.groupby(["root", "side"])
    root_side = g.agg(
        n_events=("total_dz", "size"),
        total_pnl=("weighted_net_return", "sum"),
        genuine_pnl=("weighted_net_return", lambda s: s[~decomp.loc[s.index, "is_mirror"]].sum()),
        win_rate=("weighted_net_return", lambda s: (s > 0).mean()),
        mean_duration_days=("duration_days", "mean"),
        mirror_pct=("is_mirror", "mean"),
        median_carry_move_share=("carry_move_share", "median"),
    ).reset_index()
    asym = decomp.assign(side2=decomp["side"].str.replace("fade_", "", regex=False)) \
        .groupby(["root", "side2"]).agg(
            mean_event_net=("weighted_net_return", "mean"),
            win_rate=("weighted_net_return", lambda s: (s > 0).mean()),
            median_carry_move_share=("carry_move_share", "median"),
            median_duration=("duration_days", "median"),
        ).reset_index()
    return root_side, asym
```

NOTE: `decompose_events` already carries `weighted_net_return` and `duration_days` (in its `keep` list from Task 2), so this aggregation can consume them directly.

- [ ] **Step 2: Wire into `main()`, write `root_side_attribution.csv` and `asymmetry_summary.csv`, run**

Run: `/home/famadeo/quant-lab/.venv/bin/python scripts/analyze_metals_convenience_yield_mechanism.py --only d2`
Expected: both CSVs written; eyeball that genuine_pnl concentrates in industrial metals (HG/PA/PL) if the hypothesis holds, and that GC is comparatively small.

- [ ] **Step 3: Commit**

```bash
git add scripts/analyze_metals_convenience_yield_mechanism.py experiments/HYP-0053-metals-convenience-yield-mechanism/root_side_attribution.csv experiments/HYP-0053-metals-convenience-yield-mechanism/asymmetry_summary.csv
git commit -m "HYP-0053: D2 per-root/side attribution + D4 asymmetry"
```

---

### Task 4: D3 — financing vs convenience yield

**Files:**
- Modify: `scripts/analyze_metals_convenience_yield_mechanism.py`
- Test: `tests/test_convenience_yield_mechanism.py`

**Interfaces:**
- Produces: `convenience_yield_basis(panel) -> pd.DataFrame` — for each `(date, target_months, min_volume)` add `gc_carry` (the GC row's `carry_pct_ann`) and `cy_carry = carry_pct_ann − gc_carry`; GC rows get `cy_carry≈0`. Then `financing_vs_convenience(source_key) -> pd.DataFrame` re-runs `reconstruct_z`/`decompose_events` using `cy_carry` in place of `carry_pct_ann` and compares genuine P&L and carry-move share on raw carry vs stripped basis, per root.

- [ ] **Step 1: Write the failing test**

```python
def test_convenience_yield_basis_strips_gold():
    panel = pd.DataFrame({
        "root": ["GC", "HG", "GC", "HG"],
        "date": pd.to_datetime(["2024-01-01","2024-01-01","2024-01-02","2024-01-02"], utc=True),
        "target_months": [3, 3, 3, 3], "min_volume": [10.0, 10.0, 10.0, 10.0],
        "carry_pct_ann": [5.0, 8.0, 5.5, 7.0],
    })
    out = mod.convenience_yield_basis(panel)
    gc = out[out.root == "GC"]; hg = out[out.root == "HG"].sort_values("date")
    assert np.allclose(gc["cy_carry"], 0.0)
    assert np.allclose(hg["cy_carry"].values, [3.0, 1.5])  # 8-5, 7-5.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/famadeo/quant-lab/.venv/bin/pytest tests/test_convenience_yield_mechanism.py::test_convenience_yield_basis_strips_gold -v`
Expected: FAIL.

- [ ] **Step 3: Implement `convenience_yield_basis` + `financing_vs_convenience`**

```python
def convenience_yield_basis(panel):
    gc = panel[panel["root"] == "GC"][["date", "target_months", "min_volume", "carry_pct_ann"]]
    gc = gc.rename(columns={"carry_pct_ann": "gc_carry"})
    out = panel.merge(gc, on=["date", "target_months", "min_volume"], how="left")
    out["cy_carry"] = out["carry_pct_ann"] - out["gc_carry"]
    return out
```

For `financing_vs_convenience`: build the cy panel, set `carry_pct_ann = cy_carry` (drop GC, whose cy≈0 carries no signal), reuse `reconstruct_z` + `load_variant`'s event slicing logic, run `decompose_events`, and emit per-root `genuine_pnl_raw` vs `genuine_pnl_cy` and `carry_move_share_raw` vs `_cy`. (Factor a small helper so the cy run shares `reconstruct_z`/`decompose_events`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/famadeo/quant-lab/.venv/bin/pytest tests/test_convenience_yield_mechanism.py::test_convenience_yield_basis_strips_gold -v`
Expected: PASS.

- [ ] **Step 5: Wire into `main()`, write `financing_vs_convenience.csv`, run**

Run: `/home/famadeo/quant-lab/.venv/bin/python scripts/analyze_metals_convenience_yield_mechanism.py --only d3`
Expected: CSV written; eyeball whether genuine P&L survives on the stripped `cy_carry` basis (supports convenience-yield) or collapses (supports rates-artifact). Also print GC `carry_pct_ann` annual means to compare against the known policy-rate path (~5.3% 2023 → cuts into 2024-25) as the qualitative rate sanity check.

- [ ] **Step 6: Commit**

```bash
git add scripts/analyze_metals_convenience_yield_mechanism.py tests/test_convenience_yield_mechanism.py experiments/HYP-0053-metals-convenience-yield-mechanism/financing_vs_convenience.csv
git commit -m "HYP-0053: D3 financing-vs-convenience-yield decomposition"
```

---

### Task 5: D5 — reversion half-life

**Files:**
- Modify: `scripts/analyze_metals_convenience_yield_mechanism.py`
- Test: `tests/test_convenience_yield_mechanism.py`

**Interfaces:**
- Produces: `ou_half_life(series: pd.Series) -> float` — OLS of `Δx_t` on `x_{t−1}` (statsmodels), `λ=−slope`, `half_life=ln(2)/λ` (NaN if `λ<=0`). And `half_life_table(panel) -> pd.DataFrame` — per root: structural half-life of the **raw `cy_carry` level** (not z), plus realized event-duration median/mean from the event_log, per annual sub-sample.

- [ ] **Step 1: Write the failing test** (recover a known AR(1) half-life)

```python
def test_ou_half_life_recovers_known_ar1():
    # x_t = phi*x_{t-1}+eps, phi=0.9 -> lambda=0.1 -> hl=ln2/0.1≈6.93
    rng = np.random.default_rng(0)
    n, phi = 20000, 0.9
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t-1] + rng.normal(0, 1)
    hl = mod.ou_half_life(pd.Series(x))
    assert abs(hl - np.log(2)/0.1) < 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/famadeo/quant-lab/.venv/bin/pytest tests/test_convenience_yield_mechanism.py::test_ou_half_life_recovers_known_ar1 -v`
Expected: FAIL.

- [ ] **Step 3: Implement `ou_half_life` + `half_life_table`**

```python
import statsmodels.api as sm

def ou_half_life(series):
    x = pd.Series(series).dropna().to_numpy()
    if len(x) < 50:
        return float("nan")
    dx, lag = np.diff(x), x[:-1]
    beta = sm.OLS(dx, sm.add_constant(lag)).fit().params[1]
    lam = -beta
    return float(np.log(2) / lam) if lam > 0 else float("nan")
```

`half_life_table`: for each root use the `target3m_minv10` slice of the cy panel (one obs/day), compute `ou_half_life(cy_carry)` over the full sample and per calendar year; join the event-duration median/mean from `event_log` for the same root.

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/famadeo/quant-lab/.venv/bin/pytest tests/test_convenience_yield_mechanism.py::test_ou_half_life_recovers_known_ar1 -v`
Expected: PASS.

- [ ] **Step 5: Wire into `main()`, write `halflife.csv`, run**

Run: `/home/famadeo/quant-lab/.venv/bin/python scripts/analyze_metals_convenience_yield_mechanism.py --only d5`
Expected: CSV written; eyeball whether structural half-life ≈ realized event duration (persistent process) and whether it is stable across years.

- [ ] **Step 6: Commit**

```bash
git add scripts/analyze_metals_convenience_yield_mechanism.py tests/test_convenience_yield_mechanism.py experiments/HYP-0053-metals-convenience-yield-mechanism/halflife.csv
git commit -m "HYP-0053: D5 reversion half-life (OU level + realized durations)"
```

---

### Task 6: D6 — roll-cycle confound

**Files:**
- Modify: `scripts/analyze_metals_convenience_yield_mechanism.py`

**Interfaces:**
- Consumes: `event_log` (`rolls`, `entry_anchor`, `entry_date`), `contract_months_out` (copy from `run_metals_convenience_yield_basis_backtest.py:87`).
- Produces: `roll_cycle_table(events) -> pd.DataFrame` — split events into `rolls==0` vs `rolls>0`; for each group per root: `n, mean_net, win_rate`. Add `days_to_front_expiry` at entry via `contract_months_out(entry_anchor, entry_date)` bucketed (`<1m`, `1-2m`, `>2m`) with mean_net per bucket.

- [ ] **Step 1: Implement `roll_cycle_table` (copy `contract_months_out` verbatim) — no unit test; verified by output**

- [ ] **Step 2: Wire into `main()`, write `roll_cycle.csv`, run**

Run: `/home/famadeo/quant-lab/.venv/bin/python scripts/analyze_metals_convenience_yield_mechanism.py --only d6`
Expected: CSV written; eyeball whether `rolls==0` events carry the edge on their own (no roll dependence) vs P&L concentrated near `<1m` to expiry (mechanical artifact).

- [ ] **Step 3: Commit**

```bash
git add scripts/analyze_metals_convenience_yield_mechanism.py experiments/HYP-0053-metals-convenience-yield-mechanism/roll_cycle.csv
git commit -m "HYP-0053: D6 roll-cycle confound check"
```

---

### Task 7: Plots, report.md verdict memo, full run

**Files:**
- Modify: `scripts/analyze_metals_convenience_yield_mechanism.py`
- Create: `experiments/HYP-0053-metals-convenience-yield-mechanism/report.md` + four PNGs

**Interfaces:**
- Produces: `make_plots()` → `decomp_scatter.png` (Δlog_spread vs Δz, colored by root), `gc_vs_cy_carry.png` (GC carry vs each metal carry over time), `halflife_bars.png`, `roll_proximity_pnl.png`; and a `main()` with no `--only` running D1–D6 + plots + report.

- [ ] **Step 1: Implement `make_plots()` and a full `main()`** (argparse `--only {d1..d6}` optional; default runs all). Each plot: load the relevant CSV/dataframe, `matplotlib` figure, `savefig` into `OUT`, `plt.close()`.

- [ ] **Step 2: Run the whole pipeline**

Run: `/home/famadeo/quant-lab/.venv/bin/python scripts/analyze_metals_convenience_yield_mechanism.py`
Expected: six CSVs + four PNGs in `OUT`, no exceptions.

- [ ] **Step 3: Run the full test suite**

Run: `/home/famadeo/quant-lab/.venv/bin/pytest tests/test_convenience_yield_mechanism.py -v`
Expected: all tests PASS.

- [ ] **Step 4: Write `report.md` from the actual computed numbers** — verdict memo: top-line verdict first (is the edge real convenience-yield compression — yes/partly/no), then one section per D1–D6 with the headline number, the economic reading, and the per-question verdict from the spec. Fill every number from the generated CSVs (no placeholders). Note explicitly where sync (0031) and daily (0030) disagree on D1. Treat any event t-stats as descriptive.

- [ ] **Step 5: Commit**

```bash
git add scripts/analyze_metals_convenience_yield_mechanism.py experiments/HYP-0053-metals-convenience-yield-mechanism/report.md experiments/HYP-0053-metals-convenience-yield-mechanism/*.png
git commit -m "HYP-0053: plots + convenience-yield mechanism verdict memo"
```

---

## Self-Review

**Spec coverage:** D1→Task 2; D2→Task 3; D3→Task 4; D4→Task 3; D5→Task 5; D6→Task 6; report/plots/verdict→Task 7; exact-signal replication→Task 1 (validated against event_log); fixed variant set + descriptive-only t-stats→Global Constraints; both-panel D1 robustness→Task 2 Step 5. All six predictions in the spec map to a diagnostic task. No gaps.

**Placeholder scan:** Core computational steps carry real code; aggregation/plot/report steps specify exact columns, file names, run commands, and expected observations. The only deliberately deferred content is the prose verdict in Task 7 Step 4, which by nature must be written from the run's actual numbers (the step pins it to the CSVs and forbids placeholders).

**Type consistency:** `reconstruct_z`→adds `carry_mean_lagged/carry_std_lagged/carry_z`, consumed by `decompose_events`; `decompose_events` must carry `weighted_net_return`+`duration_days` for Task 3 (noted in Task 3 Step 1) — when implementing Task 2, include them in `keep`. `convenience_yield_basis` emits `cy_carry`/`gc_carry` consumed by D3 and D5. `ou_half_life` signature consistent across Tasks 5. `contract_months_out` copied verbatim for D6.
