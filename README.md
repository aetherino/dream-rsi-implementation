# Dream-RSI implementation

Experiments in evolutionary program search for prefix-cache retention policies.
MiMo is the initial API model for proposing policies; policy evaluation runs
locally on CPU against fixed request traces. No local model weights or GPU are
needed to download/tokenize data or run the replay simulator.

## Current experiment and results

The current experiment mixes chat and document QA in one shared cache and compares
policies with and without declared task metadata. A separate final test remains
outside search feedback. See [mixed-workload setup and protocol](docs/mixed-task-experiment.md),
[the first pilot results](docs/mixed-task-pilot-results.md), and
[the $4 continuation protocol](docs/mixed-task-long-run.md).

[Archived experiments](docs/past-results.md) include selected policies, per-scenario
metrics, configurations and provenance hashes. The latest three extended searches
averaged **4.01% fewer total computed prompt tokens than LRU** on their validation
suite (26.72% less extra recomputation). These are simulator results, not GPU
speedups or a comparison against UniCache.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --locked
uv run --locked python scripts/setup_datasets.py
```

`uv sync` creates `.venv/`. To use the environment directly:

```sh
source .venv/bin/activate
python scripts/setup_datasets.py --verify-only
```

Downloads live under `data/raw/` and are excluded from Git. Re-running setup
reuses verified downloads. Use `--dataset sharegpt` or `--dataset mashqa` to
prepare only one dataset. `--verify-only` performs no network requests.
An interrupted download uses a `.part` file and restarts on the next run.

## Dataset sources

- **ShareGPT:** `ShareGPT_V3_unfiltered_cleaned_split.json` from
  [ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered).
  This is the source named by [UniCache's ShareGPT preparation](https://github.com/xsyslab/UniCache/tree/main/TaskData/SharedGPT-multi-turn).
  The script verifies its published SHA-256. Approximately 673 MB.
- **MASH-QA:** the original authors' `mashqa_data.zip`, linked from
  [MASHQA](https://github.com/mingzhu0527/MASHQA). The script extracts only the
  train/validation/test `*_webmd_squad_v2_full.json` files used by
  [UniCache's preprocessing](https://github.com/xsyslab/UniCache/tree/main/TaskData/mashqa_data).
  The original archive is retained (28.1 MB; extracted JSON totals 78.0 MB).
  Its SHA-256 is pinned from our first validated download on 2026-09-27;
  this is an observed checksum, not a publisher-provided checksum.

`data/raw/manifest.json` records sources, SHA-256 digests, file sizes and record
counts. The source datasets retain their respective licenses and attribution;
this repository does not redistribute them.

## Tokenize the datasets ($0 API cost)

```sh
uv run --locked python scripts/tokenize_datasets.py
uv run --locked python scripts/tokenize_datasets.py --verify-only
```

This uses the pinned **Qwen2.5-14B-Instruct tokenizer locally**, with its chat
serialization template. No model weights or MiMo calls are needed; `.env` and
`XIAOMI_API` are not used. Results are in `data/tokenized/qwen2.5/`, with a manifest
of hashes, token counts, preprocessing exclusions, run time and API cost.

Each output shard is a replay-ready JSON trace with synthetic arrival times.
Conversation/document groups stay together, and train/validation/test splits
are kept separate. See [dataset preparation](docs/dataset-preparation.md) for
filtering, provenance, cost, capacity requirements, and rerun instructions.

## Run the simulator

The CPU simulator includes LRU, LFU and FIFO baselines, a seeded synthetic
chat/document-QA workload, and an unlimited-cache reference.

```sh
uv run --locked python -m cache_sim --capacities 32 64 128 --output runs/demo.json
```

Capacities are in blocks (16 tokens per block by default). The table reports
prompt-hit ratio, computed prompt tokens, extra computation versus unlimited
cache, eviction count and policy execution time. JSON includes per-workload
breakdowns, peak cache occupancy and timing details.

Save the input trace or replay your own pretokenized trace:

```sh
uv run --locked python -m cache_sim --save-trace data/synthetic.json
uv run --locked python -m cache_sim --trace data/synthetic.json --details --output runs/replay.json
uv run --locked python -m unittest discover -s tests -v
```

See [the simulator contract](docs/simulator.md) for trace format, policy API,
metrics, and assumptions. The model is sequential, pins active-request blocks,
and permits only leaf evictions so retained prefixes remain usable. Requests
that cannot fit are rejected. Timings measure instrumented CPU policy work,
not GPU inference latency. The low-level simulator accepts trusted Python policies;
the Dream-RSI harness below restricts generated programs to a bounded expression language.

## Dream-RSI search harness

The harness has two loops: MiMo proposes cache policies for real CPU evaluation,
then proposes exploration-controller revisions evaluated by replaying recorded
discovery trees. Only observed outcomes are visible during a controller replay.
The incumbent controller is replaced only on a strict improvement against the
same history pool. A final held-out check evaluates the selected cache policy.

```sh
# No API cost: deterministic proposals, complete two-loop search.
uv run --locked python -m dream_rsi --config configs/dream-smoke.json --backend mock

# Same harness on small fixed ShareGPT/MASH-QA shard suites.
uv run --locked python -m dream_rsi --config configs/dream-datasets.json --backend mock

# Two-call MiMo integration check; uses XIAOMI_API from .env/environment.
uv run --locked python -m dream_rsi --config configs/dream-live-smoke.json --backend mimo

# Real search; includes controller-generation calls in the budget.
uv run --locked python -m dream_rsi --config configs/dream-datasets.json --backend mimo \
  --max-usd 0.50 --max-calls 24
```

The default backend is `mock`. Generated policies and controllers are validated
numeric expressions, never arbitrary Python. Each cache evaluation runs in a
child process with a wall-time limit. The API has call/completion-token limits and
an estimated-dollar guard. Local run directories record prompts, programs,
tree structure, failures, baselines, controller replay trajectories and API usage.
The realtime runner requires a new output directory. The batch comparison runner below supports resume.

See [the detailed walkthrough and flowchart](docs/dream-rsi.md) for the scoring
formulas, configuration, replay semantics, budget accounting, and differences
from the paper. This implements the paper's two-loop mechanism in a restricted
program space; smoke results are not evidence of general self-improvement.

## Compare fixed and adaptive controllers

```sh
uv run --locked python -m dream_rsi.compare \
  --config configs/dream-comparison-total.json --backend mimo
```

This runs three trials per strategy on a larger fixed suite, with eight API calls
and a $0.10 estimated budget per run ($0.60 total ceiling). Controller-development
calls count against the adaptive allowance. The default backend is `mock`; use
it to check the experiment without API costs. A frozen plan, all per-run artifacts,
and aggregate JSON/Markdown reports are saved under a new `runs/` directory.

The default objective now minimizes total extra recomputation across the suite.
A positive score requires fewer recomputed tokens overall. The previous macro
objective is retained explicitly as `mean_relative_v1` for historical comparisons.

See [the corrected-objective design](docs/total-recomputation-rerun.md) for workload selection,
scoring, actual-spend caveats, and interpretation. These small pilots do not establish
statistical superiority. Further work includes calibrating recomputation/CPU-cost
tradeoffs and validating promising policies against a serving engine.

The first live pilot is complete: [results and interpretation](docs/controller-comparison-results.md).
It found a mismatch between mean relative scenario scores and total recomputation;
neither strategy beat LRU on the latter measure.

The corrected-objective rerun is complete: [results](docs/total-recomputation-results.md).
Both arms reduced raw held-out recomputation versus LRU; no controller revision
was accepted, so the experiment does not establish a benefit from controller evolution.

## Longer run with discounted provider batches

The resumable batch coordinator groups currently ready requests across independent
trials using MiMo's actual Batch API. The longer configuration allows 240 total
requests (40 per run), three trials per strategy, ten discovery cycles, fresh
validation shards 7–12, and a $3 total estimated spending ceiling.

```sh
# Freeze the plan and prepare the first wave, without sending API requests.
.venv/bin/python -m dream_rsi.batch_compare \
  --config configs/dream-comparison-batch.json --backend mimo-batch \
  --output runs/comparison-batch-long-20260927 --prepare-only

# Copy your account's Batch API Base URL from the MiMo console first.
.venv/bin/python -m dream_rsi.batch_compare \
  --output runs/comparison-batch-long-20260927 --resume \
  --base-url "$MIMO_BATCH_BASE_URL" --watch
```

Batch jobs are asynchronous. Later search rounds depend on earlier results;
multiple provider waves are required. The coordinator saves progress and can
resume without resubmitting known jobs. See [the batch-run guide](docs/batched-long-run.md)
for experiment design, pricing, recovery, and interpretation.

### Regular API fallback

The same resumable coordinator can use concurrent regular MiMo requests when
provider batching is inconvenient. This uses standard token rates, not the batch
discount. The 240-call and $3 total ceilings are unchanged:

```sh
.venv/bin/python -m dream_rsi.batch_compare \
  --config configs/dream-comparison-long.json --backend mimo \
  --output runs/comparison-long-realtime-20260927 --watch
```

Up to four requests run concurrently across independent trials. Each request's
outcome is saved immediately. Restart with the same output path and `--resume
--watch`; completed requests are reused, and requests interrupted with an unknown
outcome are recorded as failed attempts without automatic resubmission.

## Follow-up diagnostics

The completed 240-call experiment favored the fixed controller on held-out
recomputation (22.75% average reduction versus LRU, compared with 13.39% adaptive).
The next study continues all three fixed searches to 100 total calls and tests
all four accepted controller revisions against their predecessors on fresh
searches. It allows 420 new requests with a $3 new spending ceiling.

See [the diagnostic study design](docs/controller-transfer-study.md) and
[the prompt audit](docs/prompt-audit-20260927.md). The original study protocol holds prompts unchanged; the subsequent user-directed revision is documented below.

## Revised prompts and continuation

Prompt v2 clarifies simulator mechanics and output constraints, includes the
best-ever training policy and a compact attempted-formula history, and separates
live search limits from offline replay limits. The diagnostic search continues
from its saved state with the same call and spending ceilings. Its report marks
the prompt-change boundary; mixed histories are not a controlled prompt A/B test.

See [the prompt v2 continuation](docs/prompt-v2-continuation.md) for changes,
accounting, and resume commands. New configurations default to `prompt_version:
"v2"`; explicit `"v1"` retains the legacy prompt text.
