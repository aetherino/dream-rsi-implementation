# Mixed-task pilot results

This is one paired development pilot comparing block-only numeric features (`block-v1`) with declared task flags and latest turn index (`task-v1`). Both arms used the same mixed training/validation suites, LRU start, fixed controller, prompt v2, model and call/budget ceilings. Policies were selected by training score. **The final test remains sealed: no test replay or test outcome is reported.**

The first attempt encountered growing prompt-input guard failures. It is archived as `debug-superseded.json`, not the primary feature comparison. The replacement used bounded prompt feedback. Validation was reused between attempts, and MASH-QA validation source documents had already appeared in earlier studies; these observations are development evidence, not a fresh hidden test.

| Attempt | Arm | Calls | Estimated USD |
| --- | --- | ---: | ---: |
| Superseded debug | block-v1 | 11 | $0.09011808 |
| Superseded debug | task-v1 | 11 | $0.08551317 |
| Bounded primary | block-v1 | 12 | $0.09573393 |
| Bounded primary | task-v1 | 12 | $0.09591098 |

Total estimated API spending across both attempts: **$0.36727615**. These are configured token-rate estimates, not invoices. Baseline preparation/replay cost $0 in API calls. Each primary arm had a 12-call and $0.212184 ceiling. The total authorized ceiling including the debug attempt was $0.60.

## Primary aggregate validation

| Arm | Training score | Validation score | Computed prompt tokens | Extra computed tokens | Prefix hit ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| block-v1 | 4.79477% | 3.83495% | 809,522 | 132,000 | 85.84495% |
| task-v1 | 5.69076% | 4.94230% | 808,002 | 130,480 | 85.87153% |
| LRU | — | 0% | 814,786 | 137,264 | 85.75290% |

Against LRU, total computed prompt tokens fall by 0.64606% in the block arm and 0.83261% in the task arm. The task arm computes 1,520 fewer tokens than the block arm, a 0.18776% reduction relative to block-arm total. Scores measure reductions in extra computation above the unlimited-cache reference, with LRU extra tokens as denominator. Prefix hit ratio is token weighted over the suite. Timings are CPU policy instrumentation, not GPU serving latency.

## Every profile and capacity

Two episodes are summed within each row. Positive extra-compute reduction favors the candidate; negative values identify regressions.

| Profile | Blocks | LRU computed | Block computed | Task computed | Block extra reduction | Task extra reduction | LRU hit | Block hit | Task hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| balanced | 2048 | 112,443 | 113,707 | 113,051 | -3.972% | -1.911% | 75.087% | 74.807% | 74.952% |
| balanced | 4096 | 80,619 | 80,619 | 80,619 | 0.000% | 0.000% | 82.138% | 82.138% | 82.138% |
| chat-heavy | 2048 | 105,620 | 105,828 | 106,404 | -0.602% | -2.269% | 76.758% | 76.712% | 76.585% |
| chat-heavy | 4096 | 72,468 | 72,292 | 72,292 | 12.500% | 12.500% | 84.053% | 84.092% | 84.092% |
| qa-heavy | 2048 | 141,321 | 137,401 | 135,961 | 10.062% | 13.758% | 70.813% | 71.622% | 71.920% |
| qa-heavy | 4096 | 102,649 | 102,649 | 102,649 | 0.000% | 0.000% | 78.800% | 78.800% | 78.800% |
| shift | 2048 | 113,777 | 111,137 | 111,137 | 9.086% | 9.086% | 92.258% | 92.437% | 92.437% |
| shift | 4096 | 85,889 | 85,889 | 85,889 | 0.000% | 0.000% | 94.155% | 94.155% | 94.155% |

## Per-workload validation

| Workload | Arm | Computed prompt tokens | Prefix hit ratio |
| --- | --- | ---: | ---: |
| sharegpt | LRU | 428,282 | 72.76520% |
| mashqa | LRU | 386,504 | 90.67858% |
| sharegpt | block-v1 | 429,450 | 72.69092% |
| mashqa | block-v1 | 380,072 | 90.83370% |
| sharegpt | task-v1 | 431,290 | 72.57391% |
| mashqa | task-v1 | 376,712 | 90.91474% |

## Frozen expressions

**block-v1:** recency-with-firmer-insert-freshness

```text
last_access + 0.1 * (now - inserted_at)
```

**task-v1:** leaf-lru-freq-4-chat-penalize

```text
last_access + frequency * 4 + (30 if task_qa and turn_index > 0 else 0) + (10 if task_unknown else 0)
```

The selected task-arm expression does not establish an effective metadata advantage on this prepared workload: QA has turn index zero, and all prepared requests have known chat/QA labels. Its `task_qa and turn_index > 0` and `task_unknown` terms are therefore inactive; its effective expression here is `last_access + frequency * 4`. The task-arm improvement can arise from ordinary recency/frequency coefficients.

Only one paired search was run, sequentially by arm, with stochastic API proposals. Groups may recur across profiles/episodes, so scenarios are correlated. Any difference is exploratory evidence; it does not establish statistical superiority, GPU performance, a general task-metadata benefit, or equivalence to UniCache.

## Archive and reproduction

The compact [archive](../experiment-results/mixed-task-v1/) stores training baselines, prepared settings/counts/hashes, both debug and primary bundled evidence, frozen selections and a manifest. Per-arm bundles include configs, summaries, selected policies/controllers, per-scenario train baselines and validation metrics, usage totals and source hashes. Prompts, responses, raw text/tokens, identifiers and test traces are omitted. Manifest source hashes refer to original ignored files; archive hashes refer to sanitized copies.

Follow [the protocol](mixed-task-experiment.md) for setup, verified preparation and a fresh search directory. API reruns reproduce the procedure rather than identical sampled policies. Archived source hashes record the historical code; later code/config changes may alter results. The frozen selection is evidence, not an executable resume checkpoint. No test command was run for this report.
