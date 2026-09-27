# Dream-RSI implementation

Experiments in evolutionary program search for prefix-cache retention policies.
MiMo is the initial API model for proposing policies; policy evaluation will run
locally on CPU against fixed request traces. No local model weights or GPU are
needed to download/tokenize data or run the planned replay simulator.

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

## Scope and next steps

This repository currently contains dataset setup only. The downloaded files
are **raw text datasets, not tokenized replay traces**. No paid API calls are
made by setup.

1. Validate the simulator on tiny synthetic traces.
2. Clean/deduplicate ShareGPT; preserve MASH-QA document identity and splits.
3. Choose and pin a serving tokenizer and chat template, then tokenize on CPU.
   The serving tokenizer need not be MiMo's. Keep recorded answers as outputs.
4. Build timestamped traces with explicit, seeded arrival assumptions. Keep
   complete sessions/documents separate across search and held-out evaluation.
5. Compare LRU/LFU and candidate policies on separate and mixed workloads,
   across cache capacities. Preserve exact-prefix dependencies and legal eviction.
6. Minimize extra prompt computation versus an unlimited-cache replay, subject
   to a policy-time budget. Measure all policy bookkeeping, metadata memory,
   and cache churn. Token savings do not by themselves establish GPU latency gains.
7. Integrate MiMo proposals and the evolution loop after the evaluator is sound.

The initial sequential replay will be a simplification, not a reproduction of
UniCache's concurrent vLLM-based simulator. Concurrent scheduling and decode
memory pressure require subsequent validation.
