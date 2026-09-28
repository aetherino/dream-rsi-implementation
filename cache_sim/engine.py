"""Sequential, ancestor-closed prefix-cache model with active-request pinning."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from time import perf_counter_ns

from .policies import BASELINES, Entry, FIFO, Policy
from .trace import CompiledTrace


@dataclass
class Result:
    policy: str
    capacity_blocks: int | None
    block_size: int
    requests: int = 0
    prompt_tokens: int = 0
    reused_prompt_tokens: int = 0
    computed_prompt_tokens: int = 0
    output_tokens: int = 0
    evicted_blocks: int = 0
    inserted_blocks: int = 0
    peak_occupied_blocks: int = 0
    peak_retained_blocks: int = 0
    policy_time_ns: int = 0
    policy_max_operation_ns: int = 0
    policy_operations: int = 0
    replay_time_ns: int = 0
    extra_computed_tokens: int | None = None
    workloads: dict = field(default_factory=dict)
    request_results: list = field(default_factory=list)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["prompt_hit_ratio"] = self.reused_prompt_tokens / max(1, self.prompt_tokens)
        result["policy_time_ms"] = self.policy_time_ns / 1e6
        result["policy_us_per_request"] = self.policy_time_ns / max(1, self.requests) / 1e3
        return result


def replay(trace: CompiledTrace, capacity_blocks: int | None, policy: Policy | None = None,
           *, details: bool = False, check_invariants: bool = False) -> Result:
    """Use a fresh policy instance per run. None capacity means unlimited.

    Each request's whole prompt+output must fit; oversize requests are rejected
    before replay. Full blocks survive completion, partial tails do not. The
    logical model allows a completely cached prompt to need zero prefill tokens;
    it does not model final-token logits, attention kernels, or generation time.
    """
    if capacity_blocks is not None:
        if type(capacity_blocks) is not int or capacity_blocks < 1:
            raise ValueError("capacity_blocks must be a positive integer or None")
        for req in trace.requests:
            if req.required_blocks > capacity_blocks:
                raise ValueError(f"Request {req.request.request_id} requires {req.required_blocks} "
                                 f"blocks, exceeding capacity {capacity_blocks}; no truncation performed")
    policy = policy if policy is not None else FIFO()
    result = Result(policy.name, capacity_blocks, trace.block_size)
    cache: dict[int, Entry] = {}
    child_counts: dict[int, int] = {}
    order = 0
    replay_start = perf_counter_ns()

    def timed_end(start: int) -> None:
        elapsed = perf_counter_ns() - start
        result.policy_time_ns += elapsed
        result.policy_max_operation_ns = max(result.policy_max_operation_ns, elapsed)
        result.policy_operations += 1

    def access(block: int, now: float, task_type: str, turn_index: int) -> None:
        nonlocal order
        start = perf_counter_ns()
        order += 1
        old = cache.get(block)
        task = dict(task_chat=task_type == "chat", task_qa=task_type == "document-qa",
                    task_unknown=task_type == "unknown", turn_index=turn_index)
        if old is None:
            entry = Entry(block, trace.depths[block], now, now, 1, order, order, **task)
            cache[block] = entry
            policy.observe_insert(entry)
        else:
            entry = replace(old, last_access=now, frequency=old.frequency + 1, access_order=order, **task)
            cache[block] = entry
            policy.observe_hit(entry)
        timed_end(start)

    def make_room(pinned: set[int], now: float) -> None:
        if capacity_blocks is None or len(cache) < capacity_blocks:
            return
        start = perf_counter_ns()
        eligible = tuple(entry for block, entry in cache.items()
                         if block not in pinned and child_counts[block] == 0)
        if not eligible:
            raise RuntimeError("No legal victim: active capacity accounting is inconsistent")
        victim = policy.choose(eligible, now)
        if type(victim) is not int or not any(e.block_id == victim for e in eligible):
            raise ValueError(f"Policy returned illegal victim {victim!r}")
        policy.observe_evict(cache[victim])
        del cache[victim]
        timed_end(start)
        parent = trace.parents[victim]
        if parent:
            child_counts[parent] -= 1
        del child_counts[victim]
        result.evicted_blocks += 1

    for compiled in trace.requests:
        req = compiled.request
        pinned: set[int] = set()
        reused = 0
        for block in compiled.blocks[:compiled.prompt_blocks]:
            if block not in cache:
                break
            pinned.add(block)
            reused += trace.block_size
        # Materialize the whole logical sequence in order. Outputs are replayed,
        # not generated; existing output blocks are accessed but earn no prefill credit.
        evictions_before = result.evicted_blocks
        for block in compiled.blocks:
            if block not in cache:
                make_room(pinned, req.arrival_time)
                parent = trace.parents[block]
                if parent:
                    if parent not in cache:
                        raise RuntimeError("Attempt to insert a block without its prefix")
                    child_counts[parent] += 1
                child_counts[block] = 0
                result.inserted_blocks += 1
            access(block, req.arrival_time, req.task_type, req.turn_index)
            pinned.add(block)
            result.peak_retained_blocks = max(result.peak_retained_blocks, len(cache))
            result.peak_occupied_blocks = max(result.peak_occupied_blocks, len(cache))
        tail = int((len(req.prompt) + len(req.output)) % trace.block_size != 0)
        if tail:
            make_room(pinned, req.arrival_time)
            result.peak_occupied_blocks = max(result.peak_occupied_blocks, len(cache) + 1)
        # The tail is freed and full blocks become evictable at request completion.
        computed = len(req.prompt) - reused
        result.requests += 1
        result.prompt_tokens += len(req.prompt)
        result.reused_prompt_tokens += reused
        result.computed_prompt_tokens += computed
        result.output_tokens += len(req.output)
        counts = result.workloads.setdefault(req.workload, {"requests": 0, "prompt_tokens": 0,
                                                           "computed_prompt_tokens": 0,
                                                           "reused_prompt_tokens": 0})
        counts["requests"] += 1
        counts["prompt_tokens"] += len(req.prompt)
        counts["computed_prompt_tokens"] += computed
        counts["reused_prompt_tokens"] += reused
        if details:
            result.request_results.append({"request_id": req.request_id, "reused_tokens": reused,
                                           "computed_tokens": computed,
                                           "evicted_blocks": result.evicted_blocks - evictions_before})
        if check_invariants:
            assert capacity_blocks is None or len(cache) + tail <= capacity_blocks
            actual_children = dict.fromkeys(cache, 0)
            for block in cache:
                parent = trace.parents[block]
                assert not parent or parent in cache
                if parent:
                    actual_children[parent] += 1
            assert actual_children == child_counts
    result.replay_time_ns = perf_counter_ns() - replay_start
    return result


def sweep(trace: CompiledTrace, capacities: list[int], policies: list[str], *,
          details: bool = False, check_invariants: bool = False) -> dict:
    if any(name not in BASELINES for name in policies):
        raise ValueError(f"Unknown policy; choose from {list(BASELINES)}")
    unlimited = replay(trace, None, FIFO(), details=details, check_invariants=check_invariants)
    unlimited.extra_computed_tokens = 0
    runs = []
    for capacity in capacities:
        for name in policies:
            result = replay(trace, capacity, BASELINES[name](), details=details,
                            check_invariants=check_invariants)
            result.extra_computed_tokens = result.computed_prompt_tokens - unlimited.computed_prompt_tokens
            assert result.extra_computed_tokens >= 0
            for workload, counts in result.workloads.items():
                counts["extra_computed_tokens"] = (counts["computed_prompt_tokens"]
                    - unlimited.workloads[workload]["computed_prompt_tokens"])
            runs.append(result.to_dict())
    return {"schema_version": 1, "model": "sequential-ancestor-closed-full-block-v1",
            "tokenizer": trace.tokenizer, "block_size": trace.block_size,
            "unlimited": unlimited.to_dict(), "runs": runs,
            "timing_note": "One instrumented CPU run; includes metadata updates, hooks and victim selection. "
                           "Not GPU latency. Repeat on an idle machine for policy comparisons.",
            "memory_note": "peak_retained_blocks counts Entry records, not bytes of policy-owned memory."}
