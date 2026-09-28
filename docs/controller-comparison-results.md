# Controller comparison results — 2026-09-27

The adaptive arm scored better in this small pilot, but the experiment does **not**
show that controller improvement caused that advantage. More practically, neither
arm beat LRU on total held-out extra recomputation. The scoring objective needs
attention before a larger paid search.

See [the fixed design](controller-comparison.md) and
[portable result data](../experiment-results/controller-comparison-20260927.json).
Full local artifacts are in `runs/comparison-mimo-pilot-20260927/`.

## Execution

- Six completed runs: three fixed-controller and three adaptive-controller runs.
- MiMo V2.6 Pro with thinking disabled; eight calls allowed per run.
- Twelve training scenarios (1,246 distinct requests) and twelve validation
  scenarios (1,076 distinct requests), balanced between ShareGPT and MASH-QA.
- The same trace snapshots, implementation hashes, model settings and API limits
  were verified across all six runs. Test data was unused. No settings were tuned
  between trials based on held-out outcomes.
- All 48 calls returned accounted token usage: 242,371 input tokens and 11,398
  completion tokens. Estimated uncached API cost: **$0.115347645**, below the $0.60
  experiment ceiling. Input-cache discounts are ignored; this is not an invoice.
- Runtime was about 12 minutes. Four of 45 discovery proposals failed program
  validation; they remained charged attempts. All three controller proposals were
  valid, and one was accepted. There were no automatic retries.

## All trial outcomes

The primary score is the mean per-scenario relative reduction in extra prompt
computation versus LRU. It is not a reduction in the sum of recomputed tokens.

| Trial | Strategy | Training score | Validation score | Validation extra tokens | Discovery / controller calls | Accepted revisions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Fixed | 0.000000 | 0.000000 | 66,400 | 8 / 0 | 0 |
| 1 | Adaptive | 0.003155 | 0.010181 | 67,584 | 7 / 1 | 0 |
| 2 | Fixed | 0.000000 | 0.000000 | 66,400 | 8 / 0 | 0 |
| 2 | Adaptive | 0.000361 | 0.032329 | 66,656 | 7 / 1 | 0 |
| 3 | Fixed | 0.003247 | -0.027497 | 80,848 | 8 / 0 | 0 |
| 3 | Adaptive | 0.000000 | 0.000000 | 66,400 | 7 / 1 | 1 |

LRU's reference total is **66,400 extra tokens per validation run**, summing the
12 separate scenario replays. Requests appear at both capacities; these totals
are workload-scenario counts, not counts from one continuous serving trace.

| Measure across three trials | Fixed | Adaptive |
| --- | ---: | ---: |
| Mean validation score | -0.009166 | 0.014170 |
| Sample standard deviation of score | 0.015875 | 0.016529 |
| Mean validation extra tokens | 71,216 | 66,880 |
| Extra tokens relative to LRU | +7.253% | +0.723% |
| Total API calls | 24 | 24 |
| Estimated API cost | $0.057149 | $0.058199 |

Adaptive minus fixed mean score was **+0.023336**, with positive differences in
all three pairs. These are only three stochastic pairs sharing a small workload
suite, without guaranteed paired model seeds. This is descriptive evidence from
a pilot, not a statistically established or causal advantage.

## What the controller improvement actually did

The first two adaptive runs rejected their proposed controller revisions. Their
positive validation scores therefore came from cache-policy samples found under
the unchanged initial controller. The development calls consumed budget without
changing subsequent search rules in those runs.

The third adaptive run accepted `diverse-branching-then-refine`. On the recorded
tree, replay value rose from **-0.091667 to -0.075000** by revealing four attempts
instead of five while retaining the same best task score of zero. The revised
controller was used in the second rollout, but the final winning cache policy
remained LRU. Better historical replay value did not yield a better cache policy
in this run.

## Why a positive score did not beat LRU in raw tokens

The objective gives equal weight to each scenario's percentage improvement,
regardless of that scenario's baseline amount of extra computation. Saving a
small absolute number of tokens on a small-baseline scenario can offset losing
many more tokens on a large-baseline scenario.

For example, adaptive trial 1 scored **+0.010181** while recomputing **1,184 more
tokens than LRU** in total. Its ShareGPT regressions outweighed its absolute
savings elsewhere. Adaptive trial 2 similarly scored positively but recomputed
256 more tokens overall. The macro score is calculating its defined objective
correctly; that objective does not directly minimize total recomputation.

There is a second discontinuity: the denominator is `max(1, LRU extra)`, so a
scenario on which LRU has zero extra computation can assign a very large penalty
to a small regression. Several rejected-by-selection candidates exhibited that
behavior during training. We did not change the scoring rule during this study.

The raw-total diagnostic was added to the report after observing this mismatch.
It is a descriptive secondary measure, not a retroactively substituted primary
endpoint. Full per-scenario counts are retained in the result JSON.

## Recommendation

Keep LRU as the practical baseline for now. Before increasing search spend,
choose an objective aligned with the intended workload: total extra recomputation,
or explicitly weighted workload totals with bounded normalization. Define how much
a ShareGPT regression can be traded against a MASH-QA gain, and how CPU cost should
enter the objective. Then freeze that choice and use fresh held-out evaluation.

A larger study should also give accepted controllers enough subsequent discovery
budget to demonstrate a benefit. This pilot left only two calls after a controller
revision. Its results apply to the restricted expression language and non-thinking
MiMo mode used here, not to unrestricted program search or serving-engine latency.

Implementation verification: **51 tests pass**, including shared-budget accounting,
trace-snapshot consistency, and a regression test demonstrating that positive
macro improvement can coexist with worse total recomputation.
