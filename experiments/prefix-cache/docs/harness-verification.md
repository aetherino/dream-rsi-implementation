# Harness verification — 2026-09-27

This records integration checks, not a benchmark of Dream-RSI's effectiveness.
Full local artifacts are under the ignored `runs/` directory.

- **46 tests passed**: simulator and tokenization regressions, expression constraints,
  replay visibility and objective, missing history support, controller acceptance
  and adoption on a subsequent rollout, trace/split checks, timeouts, token/call/dollar
  accounting, and rejection of truncated API responses.
- **Dataset mock integration:** `runs/dream-datasets-mock-v2`. Two discovery cycles,
  14 valid cache proposals, four controller proposals, followed by held-out validation.
  Four train and four validation scenarios: one shard from each dataset at two
  capacities. Total API cost $0. The best deterministic mock expression was
  `frequency / (1 + now - last_access)`, with mean training score 0.10593 and
  validation score 0.12750. No controller revision beat the incumbent in this run.
  The independent replay CLI reproduced the final mean value -0.02531787996097317.
- **Actual MiMo integration:** `runs/dream-mimo-live-smoke-v2`. MiMo V2.6 Pro,
  thinking disabled, two calls: one valid cache program and one valid controller
  revision. The controller tied the incumbent, which was correctly retained.
  Cache-policy score was 0.06915 on one synthetic training scenario and 0.05556
  on one separate-seed validation scenario. This tiny check cannot establish
  generalization or an advantage from controller optimization.

The actual MiMo cache program was:

```text
last_access + 1000 * log1p(frequency) / (1 + log1p(now - inserted_at + 1))
```

The successful two-call check used 2,061 input and 433 completion tokens. Estimated
cost at the configured uncached rates: **$0.001273245**. An earlier two-call check
with implicit thinking enabled exhausted the 8,192-token completion cap on both
responses, which were correctly rejected. That check cost an estimated
**$0.014850465**. Combined estimated cost: **$0.01612371** (about 1.61 US cents).
These are token-based estimates, not a provider invoice; input-cache discounts
are ignored. Thinking is now explicit in all resolved configurations: disabled
for the short transport smoke, enabled with a larger completion cap for the
dataset-search configuration.

The mock dataset winner illustrates why aggregate gains need per-scenario checks:

| Training scenario | Extra recomputed tokens | Improvement vs LRU |
| --- | ---: | ---: |
| ShareGPT, 2048 blocks | 8,720 | -2.06% |
| ShareGPT, 4096 blocks | 0 | 0% |
| MASH-QA, 2048 blocks | 8,464 | +44.43% |
| MASH-QA, 4096 blocks | 0 | 0% |

Its 10.59% mean score masks a ShareGPT regression. Expanding the fixed workload
suite and comparing controller strategies at equal total API budgets remain
necessary experiments. No full-dataset live search or serving-engine benchmark
was run in these checks.
