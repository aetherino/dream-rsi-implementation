# Total-recomputation rerun results — 2026-09-27

The objective now rewards lower total recomputation, and the rerun found cache
policies that reduced held-out token counts. The fixed-controller arm averaged
**3.80% fewer extra tokens than LRU**; the adaptive arm averaged **11.20% fewer**.
None of the adaptive controller revisions was accepted, so this does not establish
an advantage caused by controller evolution.

See [the frozen design](total-recomputation-rerun.md) and
[portable results](../experiment-results/controller-comparison-total-v2-20260927.json).
Full local records are in `runs/comparison-total-v2-20260927/`.

## Change and verification

The new default objective is `total_extra_v2`:

```text
score = (sum(LRU extra tokens) - sum(candidate extra tokens)) / max(1, sum(LRU extra tokens))
```

Sum first, divide once. Saving one token has the same value everywhere in the
suite. A positive score cannot accompany an increase in total recomputation.
The evaluator, discovery prompt, controller prompt, saved feedback and reports
all use this definition. The old `mean_relative_v1` mode is explicitly retained
for historical comparisons, and historical replay preserves recorded rewards.

**58 tests pass.** They include the earlier misleading-score example, randomized
checks that sign and ranking follow total tokens, zero-baseline scenarios, scenario
partition invariance, prompt consistency, and objective-version separation.

A separate audit checked **62 saved evaluations** across the six live runs,
including initial LRU observations and final validation. Every score and score
contribution matched recomputation totals, every training winner was the best
observed valid candidate, and all runs shared the same implementation hashes,
trace snapshots and API limits.

## Live experiment

Both arms had three trials, each allowed eight calls and $0.10 estimated spend.
MiMo V2.6 Pro ran with thinking disabled and a 2,048-token completion cap. The
adaptive arm's controller-development calls counted against its allowance.

Training used the same 12 scenarios as before: ShareGPT/MASH-QA training shards
1–3 at 2,048 and 4,096 blocks. Validation used fresh shards 4–6, at the same
capacities: 12 scenarios and 1,093 distinct requests. LRU needed **37,056 extra
recomputed tokens per validation run**. Each capacity is a separate replay.
No validation feedback was sent to MiMo; no test data was used.

| Trial | Strategy | Training reduction vs LRU | Validation extra tokens | Validation reduction vs LRU | Discovery / controller calls |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Fixed | 11.35% | 34,256 | 7.56% | 8 / 0 |
| 1 | Adaptive | 5.78% | 34,336 | 7.34% | 7 / 1 |
| 2 | Fixed | 0.00% | 37,056 | 0.00% | 8 / 0 |
| 2 | Adaptive | 15.44% | 30,240 | 18.39% | 7 / 1 |
| 3 | Fixed | 2.05% | 35,632 | 3.84% | 8 / 0 |
| 3 | Adaptive | 0.79% | 34,144 | 7.86% | 7 / 1 |

| Mean outcome | LRU | Fixed | Adaptive |
| --- | ---: | ---: | ---: |
| Validation extra tokens | 37,056 | 35,648 | 32,906.67 |
| Extra tokens saved per run | 0 | 1,408 | 4,149.33 |
| Reduction vs LRU | 0% | 3.80% | 11.20% |

The new aggregate score agrees exactly with these raw-total rankings. The adaptive
arm's mean score exceeded the fixed arm by 0.073978, but it lost the first pair
by a small margin and won the other two. With three pairs and no accepted
controller revisions, the arm difference should not be attributed to a better
search controller. Model sampling differences remain a sufficient explanation.

The strongest training-selected policy was adaptive trial 2:

```text
last_access + 700 * frequency / (1 + depth)
```

The simulator evicts the lowest-scoring eligible block. This expression combines
recency with a frequency bonus that decreases with prefix depth. It saved 6,816
extra tokens on its held-out suite (18.39%). It was selected using training results,
not promoted into production or established as a generally superior policy.

## Cost and limitations

- 48 API calls: 45 discovery proposals and 3 controller proposals.
- 44 valid cache proposals; one failed for returning unexpected fields. No free retry.
- All 3 controller proposals were valid; none improved replay value enough to be accepted.
- 314,353 input tokens and 15,316 completion tokens, all accounted by API usage.
- Estimated cost: **$0.150068475** (about 15 cents), below the $0.60 ceiling.
  Fixed: $0.071772825; adaptive: $0.078295650. Cached-input discounts are ignored.
- Approximately 11.8 minutes of runtime. CPU replay required no GPU.

These are three-trial pilot results on a small fixed workload suite. CPU time was
a feasibility gate, not a jointly optimized latency metric. Fresh validation
shards prevent a direct numerical comparison with the first pilot's holdout;
compare against the LRU reference within this run. The experiment demonstrates
that the objective is aligned and useful candidates were found, not that the
objective change causally improves every search or that controller evolution pays
for itself. Broader untouched workloads and more trials are needed for those claims.
