# Cache simulator contract (v1)

This is the **task evaluator** for cache-policy search. It is separate from
Dream-RSI's replay of coding-attempt histories. No API calls, GPU,
model weights, or tokenizer are used while replaying a prepared trace.

## Model

- One serving model/tokenizer/cache namespace per trace; token IDs must be
  integers in `[0, 2**32 - 1]`. Pin the tokenizer and chat template when preparing
  real data. Equality is exact token-prefix equality, not semantic similarity.
- Requests are supplied in nondecreasing arrival-time order. Ties retain input
  order. Processing is sequential and instantaneous in virtual time: timestamps
  are recency signals, not a model of service time, queues or overlapping requests.
- Intern each complete block by `(parent_prefix_id, block_token_ids)`. Identical
  suffix blocks under different prefixes never share KV state. Compilation and
  tokenization are outside the timed replay. The policy never receives the trace.
- All blocks on the reusable full-block prompt prefix are pinned before admission.
  Materialize remaining prompt/output blocks in sequence, pinning them for the
  active request. Recorded outputs consume space but earn no prompt-hit credit.
- The retained cache is **ancestor-closed**: only unpinned leaves may be evicted.
  A parent becomes eligible when its last resident child is removed. LRU, LFU and
  FIFO rank eligible leaves, not arbitrary blocks. This is a radix-style retention
  simplification, not vLLM's complete eviction implementation.
- The last partial block occupies one transient slot and is freed at completion.
  Complete blocks survive. A partial prompt block can become a retained complete
  block when recorded output extends it.
- A request's entire prompt plus recorded output must fit the selected capacity,
  including its partial tail. Oversized requests are rejected before the policy
  runs. There is no implicit truncation, offload, sliding window, or preemption.
- Logical output replay assumes the recorded output tokens' KV states are
  materialized. A fully cached prompt can count as zero prompt computation.
  Real engines may recompute a final token to obtain logits and differ in when the
  final output token is materialized. These details are not modeled in v1.
- Empty traces are accepted; individual prompts must be nonempty. Cache begins
  cold for every run. Admission is mandatory; there is no learned admission policy.

## Metrics

`computed_prompt_tokens = prompt_tokens - reused_prompt_tokens`.

`extra_computed_tokens` in a sweep subtracts the unlimited-cache run's computed
prompt tokens. Both runs use the same full-block semantics, recorded outputs,
order, and cold start. The unlimited baseline therefore includes first-use work
and unavoidable partial-block work. Extra computation is not a claim that every
finite-capacity miss could have been prevented. Direct `replay()` results have
`extra_computed_tokens=null` until compared with an unlimited run.

Reports include workload breakdowns, evicted/inserted blocks, peak retained and
occupied slots, total policy time, maximum instrumented operation time, and total
replay time. Output tokens are tracked separately from prefill computation.
Eviction counts describe churn, not transfers to another memory tier.

Policy timing includes metadata construction/updates, insert/hit/evict hooks,
eligible-victim enumeration, victim selection, and victim validation/removal.
It excludes token-prefix compilation and ordinary simulation accounting. This
initial implementation scans resident entries for each eviction, so measured
cost includes that O(cache size) scan. All baselines use the same mechanism.
Small operations are affected by instrumentation and system noise: repeat runs
on an idle machine before drawing speed conclusions. Invariant checking adds
replay overhead and is intended for debugging. There is no GPU-latency estimate.

`peak_retained_blocks` bounds the number of built-in Entry records. It is **not**
a measurement of total process RAM or arbitrary custom-policy memory. The Dream-RSI
harness uses a bounded numeric expression language, CPU feasibility checks and
child-process wall-time limits; these are not an arbitrary-code security sandbox.

## Trace format

The CLI accepts a JSON document, not raw ShareGPT/MASH-QA or plain text:

```json
{
  "schema_version": 1,
  "tokenizer": "synthetic-example-v1",
  "requests": [
    {
      "request_id": "chat-1-turn-1",
      "arrival_time": 0.0,
      "session_id": "chat-1",
      "workload": "chat",
      "prompt_token_ids": [1, 2, 3],
      "output_token_ids": [4]
    },
    {
      "request_id": "chat-1-turn-2",
      "arrival_time": 5.0,
      "session_id": "chat-1",
      "workload": "chat",
      "prompt_token_ids": [1, 2, 3, 4, 5],
      "output_token_ids": []
    }
  ]
}
```

With block size 2 and capacity 3, turn 1 computes 3 prompt tokens and leaves
two complete blocks after output replay. Turn 2 reuses 4 prompt tokens, computes
1, and needs a transient third slot. Session/workload labels are for reporting;
matching depends on token prefixes, not labels.

This simple JSON format repeats conversation history; it is intended for initial
experiments. Packed token arrays/shared-prefix references are a later storage
optimization. ShareGPT and MASH-QA exports use this format; see
[dataset preparation](dataset-preparation.md). Optional `task_type` (`chat`,
`document-qa`, or `unknown`) and zero-based `turn_index` fields carry current
serving metadata. Missing fields default to unknown/zero. These are separate
from the reporting-only `workload` and opaque `session_id` fields.

## Extending policies

Subclass `cache_sim.policies.Policy` and pass a **fresh instance** into
`replay(compiled_trace, capacity_blocks, policy)`. `choose(eligible, now)` must
return one eligible block ID. The read-only Entry objects expose depth,
insertion/access times, access frequency and deterministic insertion/access
ordering, plus task flags and current turn from the latest request touching the
block. The generated-policy `block-v1` interface hides task metadata; `task-v1`
exposes it. IDs are opaque handles, not predictive signals. Hooks can maintain
additional policy state; the engine times them too. Invalid victims fail fast.

Custom policy code executes in the current process with its Python privileges.
The CLI intentionally exposes only the built-in baselines. Do not plug untrusted
MiMo-generated Python into this interface. The search harness accepts only
validated numeric expressions through `dream_rsi.programs.RetentionPolicy`.

## Verification

```sh
uv run --locked python -m unittest discover -s tests -v
uv run --locked python -m cache_sim --check-invariants --output runs/check.json
```

Tests cover exact-prefix identity, hits/misses, output reuse, partial tails,
capacity and pinning, baseline differences, unlimited-cache subtraction,
invalid input/actions, deterministic generation, and timing hooks. Random small
traces are cross-checked against a separate tuple-prefix LRU reference model.
