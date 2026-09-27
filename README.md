# Dream-RSI implementation

Experiments in evolutionary program search for prefix-cache retention policies.
MiMo is the initial API model for proposing policies; policy evaluation will run
locally on CPU against fixed request traces. No local model weights or GPU are
needed to download/tokenize data or run the replay simulator.

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
not GPU inference latency. Custom-policy isolation is not implemented yet.

## Next steps

Downloaded ShareGPT and MASH-QA remain raw text datasets. The working demo uses
synthetic tokens; it does not yet claim dataset benchmark results.

1. Clean/deduplicate ShareGPT and preserve MASH-QA document identity and splits.
2. Pin a serving tokenizer/chat template; prepare CPU-tokenized traces with
   recorded answers and seeded arrival assumptions. Keep complete sessions and
   documents separate across search and held-out evaluation.
3. Add bounded, isolated candidate evaluation and integrate MiMo proposals.
4. Record coding-attempt histories, then build Dream-RSI's separate controller
   replay and improvement loop.
5. Validate concurrent scheduling, decode pressure and promising policies against
   a serving engine before making latency claims.
