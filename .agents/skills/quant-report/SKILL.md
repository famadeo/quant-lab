---
name: quant-report
description: Update a Quant Lab Quarto report from hypothesis, config, and results artifacts while preserving scientific caution and promotion-gate language.
---

# Quant Report

Use this skill when asked to write or update an experiment report.

## Workflow

1. Read the experiment folder:
   - `hypothesis.md`,
   - `config.yaml`,
   - `results.json`,
   - `report.qmd`.
2. Summarize the hypothesis, data, method, costs, validation approach, and metrics.
3. Include limitations before conclusions.
4. State the decision using only one of:
   - `reject`,
   - `revise`,
   - `paper_trade`,
   - `archive`.
5. If Quarto is installed, render the report:

   ```bash
   uv run quantlab render-report experiments/<HYP-ID>/report.qmd
   ```

## Output

Return the changed report path, render status, key metrics, and unresolved
limitations.

Do not describe any strategy as live-trading ready.
