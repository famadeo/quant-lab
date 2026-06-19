# Francisco Amadeo Quant Lab Wiki

This wiki is Francisco Amadeo's proprietary research record for hypothesis-driven
quantitative strategy work. It documents hypotheses, experiment design, results,
decisions, and promotion gates without publishing raw market data, credentials,
or local-only artifacts.

Public access is provided for review and documentation only. No license is
granted to copy, reuse, redistribute, commercialize, or implement the research,
strategy logic, experiment designs, or derived conclusions.

## Start Here

- [Scientific method workflow](scientific-method.md)
- [Agent workflows](agent-workflows.md)
- [Promotion gates](promotion-gates.md)
- [Publication policy](publication-policy.md)
- [Proprietary notice](proprietary-notice.md)
- [Experiment index](experiments/index.md)
- [Overall results](results/index.md)

## Publication Workflow

Regenerate the public summaries after material experiment changes:

```bash
uv run python scripts/build_wiki.py
uv run mkdocs build --strict
```

The GitHub Pages workflow builds the committed Markdown pages in `docs/`. It
does not publish ignored files such as raw data, parquet artifacts, `.env`,
MLflow state, or local result files.
