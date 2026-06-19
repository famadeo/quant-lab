# Agent Workflows

Use agents to make the research process stricter, not looser.

## Hypothesis Agent

Prompt it to create or refine `hypothesis.md`. Require explicit falsification
criteria and data assumptions.

## Implementation Agent

Prompt it to implement the smallest code change in `src/quantlab/`, add tests,
and run `./scripts/check.sh`.

## Reviewer Agent

Prompt it to review a completed experiment for leakage, survivorship bias,
multiple testing, costs, slippage, liquidity, and overfitting.

## Report Agent

Prompt it to update the Quarto report from `results.json` without overstating
the strength of evidence.

## Useful Prompts

```text
Use the quant-hypothesis-review skill to review experiments/HYP-0003.
Lead with reasons this result may be false.
```

```text
Use the quant-experiment-run skill to run HYP-0003, fix deterministic failures,
and summarize the metrics and artifacts.
```

```text
Use the quant-report skill to update the report for HYP-0003 from the latest
results.json. Do not claim live-trading readiness.
```
