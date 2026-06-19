# Quant Lab

Agent-assisted quantitative research workspace for reproducible, skeptical trading
strategy research.

## Quick Start

```bash
cd ~/quant-lab
uv sync --all-extras --group dev
cp .env.example .env
uv run pre-commit install
./scripts/check.sh
```

Run the sample smoke experiment:

```bash
uv run quantlab run-experiment experiments/HYP-0000-smoke/config.yaml
```

View tracked experiments:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## What This Workspace Gives You

- Reproducible Python environment managed by `uv`.
- Research package under `src/quantlab/`.
- Scientific-method experiment folders under `experiments/`.
- MLflow experiment tracking for parameters, metrics, and artifacts.
- DuckDB/Polars-ready data layout using Parquet-friendly folders.
- Repo-scoped Codex skills under `.agents/skills/`.
- VS Code settings, extension recommendations, dev container, and CI.
- A smoke strategy and tests that prove the loop works end to end.

## Research Loop

1. Start with a falsifiable hypothesis in `experiments/<HYP-ID>/hypothesis.md`.
2. Pin the universe, dates, costs, slippage, and validation windows in `config.yaml`.
3. Implement reusable logic in `src/quantlab/`.
4. Add or update tests.
5. Run the experiment through `quantlab`.
6. Review the result skeptically before promotion.

## External Tools

This repo is usable with only Python, Git, and `uv`. For the full ecosystem:

- Install Docker Desktop or Docker Engine to use `.devcontainer/`.
- Install VS Code and open this folder with the recommended extensions.
- Install Quarto to render `.qmd` reports locally.
- Use DVC when raw data or derived datasets become too large for Git.

See `docs/vscode-setup.md` and `docs/scientific-method.md` for details.

## Public Wiki

The publishable research wiki is built from Markdown under `docs/` with MkDocs.
Experiment summary pages are generated from local experiment artifacts:

```bash
uv run python scripts/build_wiki.py
uv run --extra docs mkdocs build --strict
```

To publish externally with GitHub Pages, enable Pages for this repository and set
the source to GitHub Actions. The workflow in `.github/workflows/wiki.yml` builds
and deploys the committed wiki pages without publishing raw data, parquet files,
`.env`, or MLflow state.
