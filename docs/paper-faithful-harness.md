# Dream-RSI Section 3 harness

This executable harness follows the user-selected [Dream-RSI Section 3 algorithm](https://arxiv.org/html/2609.14858v1#S3) with a documented prefix-cache task adaptation. It replaces the earlier expression-only policy and controller language with complete Python programs. The earlier experiment remains stopped; its cache preparation can be reused, but its search histories, candidates and spending are outside this fresh experiment.

The experiment compares one fixed-controller arm with one controller-improvement arm. Both begin with the same executable LRU policy and the same initial exploration controller, model, workloads, online limits and monetary caps. The planned fresh combined allowance is **$10**, split evenly across the two arms. Unknown API outcomes retain their full conservative reservation. Preparation and local tests do not send model requests; starting or resuming the fresh search is a separate operation. These plans are not measured results.

The initial deployment opens parallel branches over successive rounds and refines observed leaves. A round can select root at most once, so root opens one new branch per round. This is the Section 3 action interface chosen for this harness. It does not reproduce simultaneous multi-workspace initialization described elsewhere in the paper.

Discovery proposals return exactly `name`, `source` and `rationale`. Source defines a self-contained `CachePolicy` with a no-argument constructor:

```python
class CachePolicy:
    def choose(self, eligible, now):
        return min(eligible, key=lambda entry: entry.access_order).block_id
```

`choose` receives the current unpinned resident leaf entries and returns one eligible `block_id`. All blocks have the same size. Optional `observe_insert(entry)`, `observe_hit(entry)` and `observe_evict(entry)` methods receive ordered observable events. Entries are immutable; policies can keep state across events within one replay. A new replay starts a fresh instance.

`block-v1` exposes depth, insertion/access times, frequency and insertion/access event order. An opaque randomized `block_id` serves only as an identity handle. `task-v1` adds the category flags `task_chat`, `task_qa`, `task_unknown` and the latest toucher's zero-based `turn_index`. Category describes the current observed serving request and supplies no return or final-turn guarantee. Neither feature set supplies request text, tokens, token IDs, session or dataset identifiers, future requests or future hits. The fresh task adaptation uses the configured feature set recorded in its run plan.

Controller source defines `ExplorationController.select(observed_nodes, round_index, workers, max_depth)`. It returns at most `workers` distinct legal integer node IDs. Runtime receives the complete currently revealed prefix, including parent relationships and measured outcomes. Root is always legal. Nonroot actions must be current leaves below `max_depth`. A prior missing-continuation marker (`exhausted=True`) is observed feedback; that node remains legal. Selecting a missing continuation spends a replay round, reveals no invented result and adds no generation-call count. Every nonempty batch contributes to the round count, including repeated misses. Returning an empty batch stops the episode. A fresh controller instance is created for each replay world.

Offline controller evaluation uses Section 3 Equation 1:

```text
value = best_revealed_task_score
        - beta_calls * revealed_nonroot_attempts
        + beta_parallel * revealed_nonroot_attempts / max(1, nonempty_rounds)
```

Failed revealed attempts count in `revealed_nonroot_attempts`; the baseline score of zero remains available. The development objective averages this value across every recorded history. The plan records both coefficients explicitly. The current adaptation uses `beta_calls=0.001` and `beta_parallel=0.00025`, with two workers, five cycles, eleven online/replay rounds and eight revision proposals per development phase. These task-scaled settings are adaptation choices, not paper-reported tuned settings. The appendix AUC objective, its coefficient sweep and grid planner are excluded by the user's Section 3 choice.

Revision development is sequential: the next proposal receives the latest proposed controller, even when it scored worse, plus every previous version and evaluation. The harness evaluates and records every proposal. At the end of the phase it deploys the best evaluated version including the incoming incumbent; a tie retains the incumbent. Replay selection uses only revealed information, and cannot inspect future outcomes in the saved tree. Controller authors receive complete completed histories for development, which are outside the runtime worker's input.

Every proposal record available to development includes its complete source, rationale and authoritative saved `score.json`: aggregate score, validity/error, aggregate computed/saved token costs and per-scenario score/timing summaries. Current observations and completed prior histories are supplied separately. No recent-only window or formula ledger substitutes for these records. References for a selected parent or best policy point to an identical full record already in the prompt when possible. Detailed simulator counters are archived in `evaluation.json` separately from the authoritative score record; raw workload traces are never policy inputs.

The cache task score is the aggregate saved prompt computation relative to LRU, normalized by the total LRU extra computation across scenarios. CPU policy feasibility is measured; this task does not measure GPU latency. The same task score is used inside the controller value above. Validation is withheld during search, both winning policies are frozen before validation, and the final test split remains unopened.

Generated code runs in a persistent isolated child. Standard Python control flow, containers, instance state and standard-library computation are supported. On macOS the worker requires `sandbox-exec` and fails closed if it is unavailable. The sandbox denies project/dataset access, network access and process creation; trusted interpreter/runtime files remain readable. It is an OS sandbox rather than a virtual machine. Resource enforcement includes irreversible child limits, host memory polling and wall/output bounds; short memory peaks between polling intervals remain a limitation.

| Bound | Default |
| --- | ---: |
| Program source | 65,536 UTF-8 bytes |
| Program name | 1–80 characters |
| Public rationale | At most 4,000 characters |
| Per worker RPC wall time | 2 seconds |
| Worker total lifetime | 120 seconds |
| Worker memory | 256 MiB |
| Worker output per RPC | 65,536 bytes |
| Conservative model input allowance | 950,000 serialized UTF-8 bytes, including a 4,096-byte envelope reserve |
| Model completion reservation | 8,192 tokens, configurable |

The model input allowance deliberately treats each serialized UTF-8 byte as a potential input token. It leaves room for the default output under the provider's documented [1M model context](https://mimo.mi.com/models/zh-CN/mimo-v2.6-pro). An oversized complete history stops preparation of the request with a clear error. Nothing is silently truncated. Monetary reservations use this conservative input allowance plus the configured completion ceiling. Actual token usage can reduce the reservation after an accounted response. The configured [regular API rates](https://mimo.mi.com/docs/en-US/price/pay-as-you-go), checked September 28, 2026, are $0.435 per million input tokens and $0.87 per million output tokens, without assuming cache discounts.

The transport reads credentials through `backend.api_key` only. It persists and synchronizes a per-request `started` journal before HTTP, uses per-ID process locking and persists the final outcome. A resumed `started` entry is an unknown outcome and is never resent. API keys, response headers and hidden model reasoning are not journaled. Public source and measured usage use the same outcome schema as the existing coordinator. HTTP/network failures receive safe error messages and no automatic retries.

The principal remaining differences from the paper are the prefix-cache CPU simulation task, MiMo model, task-scaled limits and coefficients, two workers, one comparison pair, and a standalone source file rather than a shell coding agent with a multi-file workspace. The fixed/adaptive comparison measures this documented adaptation. It does not claim the paper's original model/task outcomes or establish a statistically reliable gain from one pair.

Implementation entry points are `dream_rsi/code_prompts.py`, `dream_rsi/code_transport.py`, `dream_rsi/code_policy.py`, `dream_rsi/code_controller.py` and `dream_rsi/paper_search.py`. The coordinator's saved `plan.json`, source hashes, checkpoint, controller revisions, proposal/score records and frozen selections are the authoritative audit artifacts for a prepared or completed run. Run locations and test totals are reported by the coordinator; this document does not start a search.


## Prepare, start, and resume

From the repository root:

```sh
# Local baseline parity and preparation only; no API calls.
.venv/bin/python -m dream_rsi.paper_search \
  --output runs/paper-section3-20260928 --max-usd 10 --prepare-only

# Start or resume the exact saved experiment and ledger.
.venv/bin/python -m dream_rsi.paper_search \
  --output runs/paper-section3-20260928 --resume
```

Preparation reads the stopped predecessor at `runs/mixed-task-dream-20260928`
only for the frozen task configuration and reference cache metrics. It imports
no candidate histories. The reset cap applies to the new experiment only.
Resume refuses changed source or data hashes. A file lock prevents concurrent
coordinators. Local artifacts are in the output directory: `report.md`,
`report.json`, `plan.json`, `checkpoint.json`, per-attempt source/score/evaluation
files, request journals, controller revisions, and final frozen selections.
Raw traces and `.env` stay excluded from Git.


## Verification before restart

All 150 tests passed on the host, including 15 executable-policy sandbox tests,
14 controller replay tests, 14 complete-history/transport tests, and 7 new
coordinator tests. The 16-scenario frozen training suite gave executable LRU
exactly the same recomputation score as the original baseline (zero relative
gain). Maximum observed host policy time was about 5.3 ms/request against a
50 ms/request feasibility bound. These checks validate the plumbing and baseline
parity; they do not establish an improvement from the new search.
