# Total-recomputation objective and rerun

The first comparison exposed a mismatch: averaging percentage changes per
scenario could give a positive score to a policy that recomputed more tokens
overall. This rerun changes the objective and keeps the comparison budget fixed.

## Objective

The new default is `total_extra_v2`:

```text
LRU_total = sum of LRU extra computed tokens across all scenarios
candidate_total = sum of candidate extra computed tokens across all scenarios
score = (LRU_total - candidate_total) / max(1, LRU_total)
```

Extra computation is measured against the unlimited-cache reference. On a fixed
suite, that reference is constant, so minimizing extra computation also minimizes
total computed prompt tokens. Every saved token has equal value. We do not apply
additional dataset weights or average percentages per shard. A positive score
requires a lower total token count. Changing how a fixed workload is partitioned
into scenarios cannot by itself change the score if replay outcomes stay the same.

Each cache capacity is a distinct replay episode in the sum. This objective
represents the configured workload mix, not all possible serving traffic. Changing
the mix deliberately changes what the search optimizes.

When LRU has zero extra computation in one scenario, regressions there receive
the same suite-wide normalization as all other tokens. If the *entire suite* has
zero LRU extra tokens, the denominator is 1: a tie scores zero; any regression is
negative. No improvement over zero extra tokens is possible, and that case has
no percentage interpretation.

CPU time remains a feasibility gate, initially 50,000 µs/request. This change
does not claim to optimize GPU latency or model inference dollars.

Both discovery and controller-development prompts explain the new score. Model
feedback contains per-scenario token savings and contributions to the total score.
The former `improvement` field is retained as a diagnostic in detailed reports;
it no longer determines selection in the new mode.

## Fixed rerun design

```sh
uv run --locked python -m dream_rsi.compare \
  --config configs/dream-comparison-total.json --backend mimo
```

The default backend is `mock`; use it to check the workflow without API charges.

- Three trials per strategy; fixed/adaptive run order alternates by pair.
- MiMo V2.6 Pro, thinking disabled, 2,048 completion tokens maximum per call.
- Eight calls and $0.10 estimated spend allowed per run; at most 48 calls and
  $0.60 across the comparison. Adaptive controller-development calls count too.
- Same discovery settings: up to three cycles, three rounds per cycle, batch
  width two, maximum depth four, one adaptive-controller revision per rollout.
- Same replay settings, CPU-time gate and controller attempt-cost coefficients.
- Same training suite: ShareGPT and MASH-QA shards 1–3, at 2,048 and 4,096 blocks.
  Twelve scenarios, 1,246 distinct requests. LRU extra-token total: 83,312.
- Fresh validation suite: validation shards 4–6 from each dataset at the same
  capacities. Twelve scenarios, 1,093 distinct requests. LRU extra-token total:
  37,056. These shards were not used by the prior smoke or comparison runs.
- No test data, previous winner warm starts, shared histories across arms, or
  hyperparameter changes in response to validation results.

A regression test checks that the old and new comparison configurations differ
only in objective and validation shard selection. The former comparison config
now explicitly names `mean_relative_v1` so historical scoring is not silently
changed. New configurations default to `total_extra_v2`; unknown modes fail.
Reports refuse to aggregate different objective versions. Historical replay uses
the scores already recorded in the tree, with its original objective label.

Freeze configuration and trace hashes before generation. Choose each run's winner
using its training score; validate that winner once at the end. Report every run,
accepted controller revision, failure, actual token cost, and raw recomputation.
Under the new objective, score and raw-token rankings agree on a common suite.

The old and new validation totals are not directly comparable because the holdout
shards changed. Compare fixed, adaptive and LRU within this rerun. A separate,
adequately replicated objective ablation would be needed to measure the causal
effect of the objective change on search quality. Three stochastic pairs remain
a small pilot, not proof of a general advantage from controller evolution.

See [the completed rerun results](total-recomputation-results.md) and scoring audit.
