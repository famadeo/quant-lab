---
name: quant-hypothesis-review
description: Review a quantitative trading experiment hypothesis or result for scientific-method rigor, leakage, overfitting, data bias, costs, and promotion readiness.
---

# Quant Hypothesis Review

Use this skill when asked to review a hypothesis, experiment folder, backtest
result, or strategy claim.

## Inputs

- Experiment folder such as `experiments/HYP-0003`.
- Optional config, report, code diff, or MLflow result reference.

## Workflow

1. Read `hypothesis.md`, `config.yaml`, `results.json` if present, and relevant
   source files.
2. Check whether the hypothesis was stated before the evidence.
3. Identify missing falsification criteria.
4. Review data assumptions:
   - point-in-time availability,
   - timestamp alignment,
   - survivorship handling,
   - corporate actions,
   - universe selection.
5. Review backtest assumptions:
   - signal lag,
   - fees,
   - slippage,
   - liquidity/capacity,
   - turnover,
   - drawdown,
   - position sizing.
6. Review statistical risk:
   - multiple testing,
   - parameter sensitivity,
   - regime dependence,
   - sample size,
   - out-of-sample separation.
7. Lead with findings. Do not bury serious issues in a summary.

## Output

Return:

- `Blockers`: issues that prevent any edge claim.
- `Concerns`: weaknesses to resolve before paper trading.
- `Evidence`: what is already well-supported.
- `Required next tests`: concrete experiments or code checks.
- `Decision`: `reject`, `revise`, `paper_trade`, or `archive`.

Never state that a strategy is live-trading ready from this repository.
