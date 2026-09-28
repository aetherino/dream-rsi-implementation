# Prompt v2 search continuation

**Prompts changed partway through this search.** Earlier v1 histories, controllers, attempts, and spending are retained. This is a continuation, not a controlled prompt A/B test; final arm differences cannot isolate the prompt effect. Per-run first-v2 call numbers are recorded in plan.json.

Status: **completed**. Completed runs: 19/19.
New requests reserved: 410/420. New estimated spending including pending reservations: $2.3706 / $3.00.

All scores below are reductions in total extra recomputation versus LRU. Higher is better.

## Fixed search: 40 to 100 total calls

Both policies are evaluated on the same fresh validation shards after search finishes.

| Trial | 40-call policy | Extended policy | Change | New calls |
| --- | ---: | ---: | ---: | ---: |
| 1 | 29.06% | 27.84% | -1.22% | 60 |
| 2 | 23.13% | 26.45% | +3.31% | 60 |
| 3 | 25.85% | 25.88% | +0.03% | 60 |

## Accepted controllers versus their predecessors

Every accepted revision is tested; controllers stay frozen. Both arms start from LRU with empty discovery histories.

| Transition | Replicate | Revised − predecessor, validation | Predecessor calls | Revised calls |
| --- | ---: | ---: | ---: | ---: |
| transition-01 | 1 | +13.55% | 14 | 14 |
| transition-01 | 2 | +18.09% | 14 | 15 |
| transition-02 | 1 | +0.00% | 15 | 15 |
| transition-02 | 2 | -8.11% | 14 | 15 |
| transition-03 | 1 | -6.28% | 14 | 14 |
| transition-03 | 2 | +0.78% | 14 | 14 |
| transition-04 | 1 | -5.08% | 15 | 14 |
| transition-04 | 2 | -7.25% | 14 | 15 |

transition-01: mean paired validation difference +15.82% (2 completed pairs).

transition-02: mean paired validation difference -4.06% (2 completed pairs).

transition-03: mean paired validation difference -2.75% (2 completed pairs).

transition-04: mean paired validation difference -6.17% (2 completed pairs).

Equal call and dollar ceilings do not guarantee identical realized spend or sampled proposals. Two replicates per transition are a diagnostic, not a significance test.
Only prompts/context changed at the recorded boundary. Model, training suite, evaluator, and online round limits are unchanged. Validation shards 13–18 never appear in model prompts. The original experiment is preserved.
