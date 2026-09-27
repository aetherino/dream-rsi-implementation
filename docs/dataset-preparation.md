# CPU dataset tokenization

```sh
uv sync --locked
uv run --locked python scripts/tokenize_datasets.py
uv run --locked python scripts/tokenize_datasets.py --verify-only
```

The default run processes both downloaded datasets and writes independent replay
episodes under `data/tokenized/qwen2.5/{sharegpt,mashqa}/{train,validation,test}/`.
`manifest.json` records raw-file hashes, tokenizer revision and asset hashes,
package versions, settings, skipped-record counts, per-split token totals,
elapsed time, each shard's hash and size, and `api_cost_usd: 0`.

Tokenization is local. It never reads `.env`, uses `XIAOMI_API`, invokes MiMo,
loads model weights, or generates new answers. Only about 7 MB of tokenizer and
configuration files are fetched from Hugging Face. Subsequent runs verify and
reuse completed artifacts. Use a new `--output-dir` for different settings or
after an interrupted preparation; existing incomplete directories are not
silently overwritten. Verification-only checks all prepared-file hashes without
network access. Regular reruns also check raw-source hashes and settings.

## Serving tokenizer and serialization

The initial serving namespace is `Qwen/Qwen2.5-14B-Instruct`, fixed at revision
`cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`. Its published Jinja chat template is
rendered locally for plain-text system/user/assistant conversations and tokenized
with the Rust `tokenizers` library, with extra special-token insertion disabled.
The template supplies those tokens itself. This does not imply the serving model
is MiMo or reproduce UniCache's exact serving configuration.

For each recorded assistant turn:

1. Serialize and tokenize all preceding messages with the assistant generation
   header to obtain the prompt.
2. Serialize and tokenize through the recorded assistant answer, including the
   template's closing markers.
3. Require the prompt IDs to be an exact prefix of the complete IDs, then take
   the remainder as output IDs. Require the previous complete turn to be an exact
   prefix of the next prompt. Any mismatch omits the whole group and is counted.

No truncation is performed. The default maximum prompt-plus-output sequence is
32,768 tokens; a conversation or document group with an oversized request is
omitted in full. This is a declared workload filter, not a property of the raw
dataset. Adjust `--max-sequence-tokens` and use a new output directory to change it.
If a source chat has no system message, add `You are a helpful assistant.`;
otherwise preserve its supplied system text. MASH-QA uses that same system text.

Recorded outputs include formatting delimiters so later turns can reuse exact
prefixes. They are a replay convention, not claimed outputs of Qwen or MiMo.
These local token counts are not MiMo billing estimates.

## Cleaning and splits

- **ShareGPT:** stream the source JSON, require complete alternating user/assistant
  text turns with an optional initial system message, normalize role names, and
  remove exact duplicate normalized conversations. Unsupported roles, empty
  messages and incomplete conversations are counted and omitted. Assistant-first
  fragments are omitted because their initiating user prompt is missing; we do
  not invent it. All languages
  are kept. A seeded hash of the first user text assigns an approximately
  80/10/10 train/validation/test split. Conversations with the same first user text
  share a split even if their later responses differ. This prevents that specific
  overlap; it is not a claim of semantic deduplication across all conversations.
- **MASH-QA:** retain the supplied split labels and document identity, merge
  identical documents within a split, remove duplicate question/answer pairs,
  and use the first recorded answer. Unanswerable/empty cases are omitted.
  Documents shared across splits belong only to the highest-priority split:
  test, then validation, then train. Lower-priority copies are omitted. Prompt
  order is instruction, document, question, answer cue; the document comes first
  so questions about it can share a prefix.

The originals stay untouched. Provenance IDs are retained in each shard's
`group_sources`, and each request has a `group_id`.

## Shards and synthetic time

The default limits are 64 groups or 250,000 prompt-plus-output token occurrences
per shard. A single group can exceed that token target: groups are never split.
Each shard is a separate cold-start replay episode; **do not concatenate them
as a continuous production trace**. Changing shard size changes the workload.

Within a chat shard, session arrivals use exponential interarrival times with
mean 1 second; inter-turn intervals are lognormal with `mu=4.15`, `sigma=0.971`.
Within a MASH-QA shard, all questions are shuffled, then assigned exponential
interarrival times with mean 1 second. All sampling is seeded. These are synthetic
arrivals, not measured source timestamps or service-time estimates. Separate
workloads are currently exported; mixed-workload composition remains future work.

To make a small independently reproducible test export:

```sh
uv run --locked python scripts/tokenize_datasets.py --limit 20 --workers 2 --output-dir data/tokenized/smoke
```

`--limit` caps ShareGPT conversations submitted to tokenization and MASH-QA
documents per original split; filtering may leave fewer accepted groups.

Replay a shard using capacities that fit its longest sequence. The manifest's
`minimum_capacity_blocks_at_16_tokens` is a feasibility floor, not an optimal
cache size. For example, a 32,768-token sequence requires 2,048 blocks at size 16.

```sh
uv run --locked python -m cache_sim --trace data/tokenized/qwen2.5/mashqa/train/trace-00000.json --capacities 2048 4096 --policies lru --output runs/mashqa-first-shard.json
```

Prepared shards use the simulator's existing schema-v1 JSON. Raw downloads,
tokenizer assets, tokenized traces and local manifests are excluded from Git.

## Initial full preparation result

With the default settings and the downloaded source files, preparation completed
in 130.126 seconds with zero paid API calls:

| Dataset | Accepted groups | Requests | Prompt tokens | Recorded output tokens | JSON bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| ShareGPT | 51,780 conversations | 172,130 | 121,181,585 | 45,317,040 | 828,635,739 |
| MASH-QA | 5,541 documents | 34,772 | 41,530,064 | 2,991,905 | 221,329,049 |

There are 998 independent trace shards. Token totals include repeated histories
and template markers. Runtime and byte counts describe this run, not guarantees
for other machines or preparation settings.

Of 94,145 ShareGPT source records, 34,494 failed role ordering (34,417 of those
started with an assistant response), 276 had invalid roles/text, 550 lacked a
conversation, 47 were incomplete, 6,993 were exact duplicates, and 5 exceeded the
context limit. No template-prefix mismatches occurred. This is a filtered export,
not a claim to have retained every raw record.

MASH-QA omitted 10 cross-split document records and 4 duplicate question/answer
pairs. The original data remains available for different filtering experiments.
The detailed local manifest is the authoritative record of this preparation.
