# AGENTS.md

## Mission

This repository is a quantitative research lab for finding, falsifying, documenting,
and promoting positive expected value trading strategies. Every agent should optimize
for reproducibility, auditability, and scientific skepticism.

## Non-Negotiable Research Rules

- Do not modify `data/raw/` except through an explicit ingestion command.
- Do not claim a strategy has edge unless the experiment includes a falsifiable
  hypothesis, data assumptions, transaction costs, and out-of-sample evidence.
- Do not promote a strategy from research to paper trading without reviewer notes
  covering leakage, survivorship bias, multiple testing, slippage, fees, liquidity,
  capacity, drawdown, and regime sensitivity.
- Do not use live-trading credentials or place live orders from this repository.
- Treat notebooks as exploratory scratch space. Reusable logic belongs in `src/`.
- Every experiment must have a stable ID such as `HYP-0007`, a config file, and a
  decision: `reject`, `revise`, `paper_trade`, or `archive`.

## Required Commands

After changing Python code, run:

```bash
uv run ruff format .
uv run ruff check .
uv run pyright
uv run pytest
```

For a full local smoke run, run:

```bash
./scripts/check.sh
```

## Scientific Method Workflow

1. Create or update `experiments/<HYP-ID>/hypothesis.md`.
2. Define the experiment in `experiments/<HYP-ID>/config.yaml`.
3. Implement the smallest reusable signal/backtest change in `src/quantlab/`.
4. Add tests for date alignment, no lookahead, missing data, and cost assumptions.
5. Run `quantlab run-experiment experiments/<HYP-ID>/config.yaml`.
6. Record metrics, artifacts, and the final decision in the experiment folder.
7. Render or update a Quarto report when the experiment is material.

## Agent Roles

- Hypothesis agents should write falsifiable predictions and specify what result
  would disconfirm the idea.
- Implementation agents should keep changes narrow, tested, and reproducible.
- Reviewer agents should lead with failure modes and missing evidence.
- Report agents should summarize methods, data, assumptions, results, limitations,
  and next actions without overstating edge.

## Definition Of Done

An experiment is done only when its code runs from a clean environment, tests pass,
MLflow has a run record, and the experiment folder contains the resulting metrics
or a clear explanation of why the run was blocked.
