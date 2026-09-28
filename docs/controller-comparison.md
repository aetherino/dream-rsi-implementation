# Fixed versus adaptive controller experiment

Run the repeatable pilot from the repository root:

```sh
uv run --locked python -m dream_rsi.compare \
  --config configs/dream-comparison.json --backend mimo
```

Use `--backend mock` to exercise the same experiment machinery without API calls.
The mock is deterministic plumbing, not independent stochastic search trials.
Run directories must be new; neither automatic retries nor resume are implemented.

## Question and fixed design

Does spending some proposal budget on controller development improve the cache
policy found, compared with spending the entire allowance on discovery using the
initial fixed controller?

The 2026-09-27 pilot fixes these settings before model-generated results:

| Setting | Both strategies |
| --- | --- |
| Model | MiMo V2.6 Pro |
| Thinking | Disabled; result does not establish behavior in thinking mode |
| Per-call completion limit | 2,048 tokens |
| Per-run allowance | 8 API calls and $0.10 estimated spend |
| Replicates | 3 per strategy, 6 runs total |
| Total ceiling | 48 API calls and $0.60 estimated spend |
| Discovery schedule | Up to 3 cycles, 3 rounds/cycle, 2 proposal slots/round |
| Initial policy/controller | LRU / parallel-refinement |
| Train suite | ShareGPT and MASH-QA training shards 1, 2, 3; 2 capacities each |
| Validation suite | Separate validation shards 1, 2, 3; same capacities |
| Capacities | 2,048 and 4,096 blocks of 16 tokens |
| Test split | Unused |

There are 12 training scenarios with 1,246 distinct requests and 12 validation
scenarios with 1,076 distinct requests. Each request is replayed at both capacities.
Each shard starts from an empty cache. Every scenario has equal score weight, so
each dataset receives 50% of the weight. The earlier smoke-test shard 0 is excluded.

The fixed strategy has zero controller revisions. The adaptive strategy allows
one revision between completed rollouts. Both strategies otherwise receive the
same configuration, including access to their own best prior-rollout observations.
No history is shared between runs or strategies. The model provider does not give
us guaranteed paired random samples: trial numbers pair settings, not randomness.
Run order alternates: fixed/adaptive, adaptive/fixed, fixed/adaptive. Runs and CPU
evaluations are sequential to reduce contention in policy-time measurements.

Controller-development calls count against the adaptive run's eight-call cap.
With the ordinary first rollout consuming five discoveries, the adaptive strategy
can spend one call on controller development and two on subsequent discoveries;
the fixed strategy can spend all eight on discoveries. Counts may differ when a
controller stops early, a prompt is rejected before dispatch, or the estimated
dollar guard blocks a request. All actual counts and spending are reported.

The last partial batch takes selected actions in priority order, without racing
concurrent budget reservations. Controller development is skipped after the last
allowed cycle or when fewer than two calls remain, so a development call leaves
at least one call for later discovery. A rollout that immediately elects to stop
ends that run. The harness does not spend leftover allowance just to match totals.

This is **equal budget ceilings, not exact dollars spent**. Different prompt and
completion lengths produce different actual costs. A strict monetary-efficiency
study would require a larger cost frontier, rather than asserting that equal call
counts have identical costs.

## Outcomes and interpretation

The primary descriptive outcome is each training-selected winner's held-out score:

```text
extra = candidate computed prompt tokens − unlimited computed prompt tokens
scenario_score = (LRU extra − candidate extra) / max(1, LRU extra)
run_score = mean scenario_score
```

Higher is better. The zero-denominator convention matters: if LRU recomputes no
extra tokens on a scenario, a regression loses one score unit per extra token.
This can make aggregate scores very negative. Read per-scenario raw counts before
treating a mean score as a simple percentage reduction. Neither that convention
nor other scoring settings are tuned after viewing validation results.

CPU policy time is a feasibility gate, initially 50,000 µs/request. Evictions and
timings remain in the per-run reports. There is no claim about serving latency.

The report includes every trial, mean/min/max/sample standard deviation by arm,
paired score differences, candidate failures, accepted controller revisions, call
allocation and actual estimated token cost. Three pairs are a pilot, not enough
to establish a statistically reliable advantage. The 12 workload scenarios are
not 12 independent search replicates; several reuse the same trace at another
capacity. Do not select only the best adaptive run for comparison.

Each winner is chosen by training score; its validation feedback is never sent to
the model. No settings are adapted between trials based on validation. Future
experiments must account for the fact that these validation results are now known.

## Saved evidence

The experiment creates `plan.json` with resolved settings, trace hashes, limits
and execution order before the first API call. Existing supplied trace hashes
are checked again on subsequent runs, so refreezing a suite cannot silently accept
changed input data. Individual runs preserve all usual harness artifacts.

`comparison.json` and `report.md` are updated after each completed arm. The JSON
contains per-scenario held-out results and the selected programs. A failure or
interruption preserves completed arms and records the incomplete experiment status;
it is not reported as a completed six-run comparison.

See [the completed first-pilot results](controller-comparison-results.md) for all six runs and the scoring mismatch they exposed.
