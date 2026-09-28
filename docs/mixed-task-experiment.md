# Mixed-workload task-information experiment

The first experiment asks whether **declared task metadata improves discovered eviction policies** when chat and document QA compete for one cache. It does not yet add agent datasets, arbitrary policy memory, task queues, or a UniCache implementation. The formula language, controller, model, objective, and budget ceilings are held fixed between feature arms.

## Setup and preparation

```sh
uv sync --locked
uv run --locked python scripts/setup_datasets.py
uv run --locked python scripts/tokenize_datasets.py
uv run --locked python scripts/prepare_mixed_workloads.py
uv run --locked python scripts/prepare_mixed_workloads.py --verify-only
```

All preparation runs on CPU with zero paid API calls. It uses the existing pinned Qwen2.5 tokenizer export, not MiMo tokenization or new generated answers. See [dataset preparation](dataset-preparation.md) for source provenance, filtering and licenses.

The default export is `data/mixed/task-v1/`. It contains four profiles, two seeded episodes per profile, and independent train/validation/test partitions. Each episode targets 256 requests, rounded upward to preserve whole conversation/document groups. A group is never truncated, repeated within an episode, or split between phases. Episodes are cold starts, not contiguous sections of a production trace. Groups can recur across profiles and replicas within a split; those scenarios are correlated, not independent statistical trials.

| Profile | Target chat share of admitted-cohort requests |
| --- | --- |
| Chat-heavy | 80% |
| Balanced | 50% |
| QA-heavy | 20% |
| Shift | 80% in first admission phase, 20% in second |

Fractions count requests, not tokens or cache occupancy. The manifest records actual fractions after whole-group rounding. Session starts and independent QA arrivals are uniform within 300-second admission windows (equivalent to homogeneous Poisson arrivals conditioned on the chosen count). Chat follow-ups use synthetic lognormal gaps with `mu=4.15`, `sigma=0.971`; they remain ordered and can cross phase boundaries or drain after admissions stop. The shift changes **new admission cohorts**, not every request instantly. Timestamps are not production measurements. The cache is not reset between shift phases.

Source shards are train 20–27, validation 0–7, and test 0–7 for each dataset. Earlier studies have already exposed the MASH-QA validation split; new mixtures do **not** make those source documents fresh. Validation is a development diagnostic. The official test partition is reserved for final evaluation. Existing tokenizer preparation groups related conversations and removes exact cross-split document overlap; it does not guarantee semantic deduplication. Preparation checks source SHA-256 values, tokenizer identity, and disjoint source group identities across splits.

The manifest hashes traces, suites, and configs. Verification rejects edits. Preparation refuses an existing destination; use a new output directory for a genuinely new preparation, not to disguise reuse of an opened test set.

## Policy information and fair comparison

| Arm | Runtime variables |
| --- | --- |
| `block-v1` | Existing recency, frequency, prefix depth and access/insertion order |
| `task-v1` | The same variables plus `task_chat`, `task_qa`, `task_unknown`, `turn_index` |

Flags and the zero-based current turn describe the **most recent request touching a block**. A shared prefix takes the latest toucher's metadata; it is not permanently owned by its first task. QA always has turn index zero. Labels are declared serving categories; even a chat with only one observed turn is labeled chat. There is no future-continuation or final-turn flag.

Generated expressions cannot read raw text, token IDs, block IDs, session/document IDs, dataset names, phase membership, files, or future requests. The simulator internally needs tokens and IDs for prefix matching, but its numeric policy interpreter exposes only the selected arm's allowlist. Policy prompts receive aggregate training results, including per-workload counts. Both arms use the same data and feedback; only the runtime feature contract differs. Policies still have no custom mutable state.

Cache capacities are 2,048 and 4,096 blocks, with 16 tokens per block. Both arms start from LRU, use prompt v2, a fixed exploration controller and the total-extra-recomputation objective. Defaults are 12 generation calls and a $0.30 estimated ceiling **per arm**, maximum 2,048 completion tokens, MiMo-v2.6-pro with thinking disabled. Actual calls can stop earlier at a budget guard. Rates come from the existing harness configuration and are estimates, not a provider billing guarantee. One pair is a pilot, not a significance test; a larger study should predeclare multiple repetitions and alternate arm order.

## Run and preserve the hidden test

```sh
# Training LRU/LFU/FIFO plus unlimited-cache reference; no API or test evaluation.
uv run --locked python -m dream_rsi.mixed_experiment baselines \
  --output runs/mixed-task-v1-baselines

# End-to-end plumbing check with deterministic mock proposals; not LLM evidence.
uv run --locked python -m dream_rsi.mixed_experiment search --backend mock \
  --output runs/mixed-task-v1-mock

# Real feature comparison, using XIAOMI_API in .env/environment.
uv run --locked python -m dream_rsi.mixed_experiment search --backend mimo \
  --output runs/mixed-task-v1-study
```

Search configs contain only training and validation suites. Each arm selects its candidate using training scores; validation is evaluated at the end and never sent back to the model. After both arms finish, `frozen-selection.json` records both policies, source-code hashes, data-manifest hash, and hashes of their search configs and summaries. The final test remains unopened.

Only after deciding that the search procedure is final, explicitly run:

```sh
uv run --locked python -m dream_rsi.mixed_experiment test \
  --output runs/mixed-task-v1-study
```

This evaluates both frozen policies and baselines on the test suite. An exclusive `test-opened.json` receipt is created **before** evaluation, including on a subsequent failure; rerunning against that search directory is rejected. Changed source/data/search artifacts are rejected. Do not open the test for a mock run or repeatedly create new search directories to tune against the same holdout. This is an auditable experimental protocol, not OS-level protection from the repository owner. If test findings guide a revision, that split has become development data and a new untouched holdout is needed.

The paired search wrapper is intentionally simple and not resumable: keep interrupted artifacts, and use a new run directory for a replacement pilot. Existing long-run transport machinery remains available separately.

## Interpretation

Report total computed prompt tokens and prefix hit ratio alongside the search score, which measures reduction in extra computation over an unlimited-cache replay. Include per-workload behavior and all profiles/capacities so one dominant workload cannot hide a regression. Policy CPU time is a feasibility gate, not measured GPU latency. Unlimited cache is not a capacity-matched offline optimum.

This experiment isolates access to task metadata. It cannot establish equivalence to UniCache: our replay is sequential, ancestor-closed and has no serving scheduler, task queues or adaptive allocation between queues. A matched UniCache baseline and richer stateful policy interface are separate next steps.

Historical outcomes and reproduction instructions are preserved in [past results](past-results.md).

The first local pilot, its prompt-size debugging run, and sanitized evidence are
documented in [pilot results](mixed-task-pilot-results.md).
