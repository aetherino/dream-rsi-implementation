# Fixed versus adaptive controller pilot

Status: **completed**. Backend: `mimo`.

Both strategies receive the same call-count, output-token and estimated-dollar ceilings. Controller-development calls count against the adaptive allowance. Actual token spending may differ; this is not exact dollar-spend matching.

| Trial | Strategy | Train score | Validation score | Discovery calls | Controller calls | Accepted revisions | Estimated USD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | adaptive | 0.05781 | 0.07340 | 7 | 1 | 0 | $0.025333 |
| 1 | fixed | 0.11350 | 0.07556 | 8 | 0 | 0 | $0.024461 |
| 2 | adaptive | 0.15441 | 0.18394 | 7 | 1 | 0 | $0.025675 |
| 2 | fixed | 0.00000 | 0.00000 | 8 | 0 | 0 | $0.023259 |
| 3 | adaptive | 0.00787 | 0.07858 | 7 | 1 | 0 | $0.027288 |
| 3 | fixed | 0.02055 | 0.03843 | 8 | 0 | 0 | $0.024052 |

Mean validation difference (adaptive − fixed): **+0.07398**.

Total estimated API cost: **$0.150068**.

## Held-out recomputation

| Strategy | Mean extra tokens per run | LRU reference | Change vs LRU |
| --- | ---: | ---: | ---: |
| fixed | 35,648.00 | 37,056.00 | -3.80% |
| adaptive | 32,906.67 | 37,056.00 | -11.20% |

Scoring version: `total_extra_v2`. The task score is (SUM of LRU extra tokens - SUM of candidate extra tokens) / max(1, SUM of LRU extra tokens), summed across all scenarios BEFORE dividing. Every saved token has equal value. A positive score requires fewer TOTAL recomputed tokens. Per-scenario relative improvements are diagnostics, not the optimization objective. A scenario where LRU has zero extra tokens uses the same suite-wide denominator as every other scenario.

Lower raw extra-token counts are better. These sum scenarios, including each capacity as a separate replay. Higher task scores are better. Scenarios share shards across capacities, so they are not independent replicates.

This small pilot reports all trials, not just the best run. Trial pairs share workloads and settings, not identical model samples or guaranteed random seeds. The run order alternates by pair. No significance or general superiority claim is warranted from three pairs.

Validation is evaluated once for each training-selected winner and is never fed into generation. The experiment does not tune settings after observing validation; test data is unused. Model thinking is disabled in this pilot, so results do not establish behavior in thinking mode.

See comparison.json for per-scenario outcomes, failures, policies, spending, variability and accepted controller revisions.
