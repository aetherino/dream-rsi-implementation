# Fixed versus adaptive controller longer experiment

Status: **completed**. Backend: `mimo`.

Both strategies receive the same call-count, output-token and estimated-dollar ceilings. Controller-development calls count against the adaptive allowance. Actual token spending may differ; this is not exact dollar-spend matching.

| Trial | Strategy | Train score | Validation score | Discovery calls | Controller calls | Accepted revisions | Estimated USD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | adaptive | 0.02055 | 0.05405 | 33 | 7 | 2 | $0.144120 |
| 1 | fixed | 0.26234 | 0.29566 | 40 | 0 | 0 | $0.162620 |
| 2 | adaptive | 0.16824 | 0.18495 | 34 | 6 | 0 | $0.154055 |
| 2 | fixed | 0.21087 | 0.17648 | 40 | 0 | 0 | $0.165135 |
| 3 | adaptive | 0.12810 | 0.16257 | 34 | 6 | 2 | $0.152420 |
| 3 | fixed | 0.20223 | 0.21035 | 40 | 0 | 0 | $0.164034 |

Mean validation difference (adaptive − fixed): **-0.09364**.

Total estimated API cost: **$0.942385**.

## Held-out recomputation

| Strategy | Mean extra tokens per run | LRU reference | Change vs LRU |
| --- | ---: | ---: | ---: |
| fixed | 118,224.00 | 153,040.00 | -22.75% |
| adaptive | 132,554.67 | 153,040.00 | -13.39% |

Scoring version: `total_extra_v2`. The task score is (SUM of LRU extra tokens - SUM of candidate extra tokens) / max(1, SUM of LRU extra tokens), summed across all scenarios BEFORE dividing. Every saved token has equal value. A positive score requires fewer TOTAL recomputed tokens. Per-scenario relative improvements are diagnostics, not the optimization objective. A scenario where LRU has zero extra tokens uses the same suite-wide denominator as every other scenario.

Lower raw extra-token counts are better. These sum scenarios, including each capacity as a separate replay. Higher task scores are better. Scenarios share shards across capacities, so they are not independent replicates.

This small pilot reports all trials, not just the best run. Trial pairs share workloads and settings, not identical model samples or guaranteed random seeds. Trials advance together using concurrent regular API requests. No significance or general superiority claim is warranted from three pairs.

Validation is evaluated once for each training-selected winner and is never fed into generation. The experiment does not tune settings after observing validation; test data is unused. Model thinking is disabled in this pilot, so results do not establish behavior in thinking mode.

See comparison.json for per-scenario outcomes, failures, policies, spending, variability and accepted controller revisions.

Request waves submitted/planned: 31. Pending requests: 0.
Current estimated cost including pending reservations: $0.942385.
