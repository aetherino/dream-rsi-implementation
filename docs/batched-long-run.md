# Longer comparison with MiMo Batch API

The new runner uses MiMo's asynchronous **provider Batch API**, including its
discounted token rates. It combines ready proposals from independent trials in
one JSONL job. It does not combine prompts into a single generation or launch
realtime requests under a batch label.

## Frozen experiment

`configs/dream-comparison-batch.json` specifies:

- Three independent trials per arm: fixed exploration versus adaptive exploration.
- At most **40 model requests per run**, including controller proposals: **240 total**,
  five times the previous experiment's 48-call allowance.
- At most ten discovery cycles, three rounds per cycle, two selected nodes per
  round, depth limit four. Adaptive runs can propose one controller revision
  between cycles; a final revision without another discovery opportunity is skipped.
- MiMo-V2.6-Pro, thinking disabled, at most 2,048 completion tokens per request.
- A **$0.50 estimated spending ceiling per run; $3 total**. Actual spend may be
  lower when controllers stop or budgets leave insufficient room for a request.
- The same training shards as the previous corrected-objective experiment:
  ShareGPT and MASH-QA train shards 1–3, at 2,048 and 4,096 blocks.
- Fresh held-out validation shards **7–12** from each dataset, at both capacities
  (24 validation scenarios). Validation runs once after selecting each run's best
  training policy. Validation never enters model prompts; test data remains unused.
- Objective `total_extra_v2`: reduce the sum of extra recomputed prompt tokens
  relative to LRU. The CPU policy-time gate remains unchanged.

These settings give each arm equal ceilings, not identical token spending or
identical random model samples. Three trials per arm still provide limited
evidence. A new validation suite also means its percentage improvements should
not be compared directly to percentages from the earlier suite.

## Batches preserve dependencies

All six trials start with an independent proposal from LRU. Their six requests
can share the first provider job. Once results arrive, CPU evaluations run
serially. The next ready requests can then be batched, including controller
proposals when a trial finishes a discovery cycle. Trials do not share policies,
histories, controller revisions, or validation results.

Every proposal in a trial's current round sees only the tree that existed before
that round. Its siblings' new results become visible in the next round. A
controller revision is evaluated locally against exactly the incumbent's history
pool, and adopted only if strictly better. Full histories remain on disk. Model
context omits past rationales; controller prompts retain every recorded outcome
and causal feature while omitting per-scenario cache metrics. Discovery context
can discard older auxiliary observations to fit the input budget, preserving the
selected parent and training-suite description.

Batch scheduling optimizes token cost, not turnaround time. MiMo schedules jobs
asynchronously/off-peak; we request a 24-hour completion window for **each wave**.
Dependent waves cannot all be submitted at once, so a whole experiment may take
much longer than a single window. Keep the coordinator running to advance waves.
The CPU simulator still requires no GPU.

## Commands

Run from the repository root. `XIAOMI_API` is read from the environment or `.env`.
The Batch API Base URL is account-region-specific: copy it from the
[MiMo Batch console](https://platform.xiaomimimo.com/console/batch). Do not assume
the China example URL works for an overseas account. Batch billing uses cash
balance, not a Token Plan quota.

Prepare the real experiment without submitting any inference:

```sh
.venv/bin/python -m dream_rsi.batch_compare \
  --config configs/dream-comparison-batch.json --backend mimo-batch \
  --output runs/comparison-batch-long-20260927 --prepare-only
```

Then supply the exact console URL and start the coordinator:

```sh
.venv/bin/python -m dream_rsi.batch_compare \
  --output runs/comparison-batch-long-20260927 --resume \
  --base-url "$MIMO_BATCH_BASE_URL" --watch
```

After a restart, the endpoint is already saved:

```sh
.venv/bin/python -m dream_rsi.batch_compare \
  --output runs/comparison-batch-long-20260927 --resume --watch
```

Without `--watch`, a resume checks the outstanding job once, processes any
completed results, submits the next ready wave, and exits when a job is pending.
`--prepare-only` never submits or polls. `--backend mock` exercises the same
coordinator with deterministic local proposals and no API calls. Mock results
are plumbing checks, not evidence about model performance.

## Recovery and accounting

The coordinator uses a process lock, an atomically replaced checkpoint, frozen
trace/source hashes, and saved source copies. It refuses resume when code or
traces have changed. A pending provider job continues if the local process stops;
resume retrieves its results instead of submitting it again.

Each request has a stable unique ID. Returned results are associated by ID, never
by file position. Out-of-order results, partial failures, and missing results are
handled explicitly. Unusable model programs still consume their request budget.
Known provider failures are marked unbilled. Successful responses with usage are
charged at the configured token rates; pending requests and unknown usage retain
a conservative reservation. Reservations use serialized UTF-8 bytes plus overhead,
not the Qwen tokenizer used for the cache workload. These estimates are not invoices.

The configured overseas Pro rates are **$0.2175 per million input tokens** and
**$0.435 per million output tokens**, the documented batch rates on 2026-09-27.
Input accounting conservatively ignores provider prompt-cache discounts.
See [MiMo's official Batch API documentation](https://mimo.mi.com/docs/en-US/quick-start/usage-guide/text-generation/batch-api).

No automatic inference retries occur. If creation times out after the server may
have accepted a job, the coordinator stops instead of risking duplicate billing.
Find the job with the saved `input_file_id` in the console, then use
`--resume --attach-batch BATCH_ID --watch`. The runner verifies the input file
matches. An explicit HTTP rejection can be retried by a later manual resume after
the account problem is resolved. Interrupting the coordinator does not cancel
an already submitted provider job.

Artifacts include `checkpoint.json`, the frozen `plan.json`, per-wave input files,
provider job IDs and normalized results, per-run `usage.json`, trees, controller
replays, prompts, and final JSON/Markdown comparisons. Returned hidden reasoning
and credentials are not saved. `progress.json` records training-best score against
API call number, counting controller calls too, so late improvements and plateaus
can be examined. Provider results are downloaded when terminal; MiMo retains
remote files for only 30 days.
