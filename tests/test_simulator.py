import json
from pathlib import Path
import random
import tempfile
import unittest

from cache_sim import Request, Trace, compile_trace, replay, sweep, synthetic_trace
from cache_sim.policies import FIFO, LFU, LRU, Policy
from cache_sim.trace import load_trace, save_trace


def trace_of(prompts, outputs=None, block_size=2):
    outputs = outputs or [()] * len(prompts)
    return compile_trace(Trace(tuple(Request(str(i), i, tuple(p), tuple(o))
                                    for i, (p, o) in enumerate(zip(prompts, outputs)))), block_size)


class SimulatorTests(unittest.TestCase):
    def test_repeated_prompt(self):
        result = replay(trace_of([[1, 2, 3, 4]] * 2), 2, LRU(), details=True, check_invariants=True)
        self.assertEqual(result.computed_prompt_tokens, 4)
        self.assertEqual(result.reused_prompt_tokens, 4)
        self.assertEqual(result.evicted_blocks, 0)

    def test_same_suffix_different_prefix_does_not_hit(self):
        result = replay(trace_of([[1, 2, 8, 9], [3, 4, 8, 9]]), None)
        self.assertEqual(result.reused_prompt_tokens, 0)

    def test_shared_prefix_counts_once(self):
        result = replay(trace_of([[1, 2, 3, 4], [1, 2, 5, 6]]), 3, LRU())
        self.assertEqual(result.reused_prompt_tokens, 2)
        self.assertEqual(result.inserted_blocks, 3)

    def test_partial_prompt_tail_recomputed(self):
        result = replay(trace_of([[1, 2, 3]] * 2), 2, LRU())
        self.assertEqual(result.computed_prompt_tokens, 4)
        self.assertEqual(result.peak_retained_blocks, 1)
        self.assertEqual(result.peak_occupied_blocks, 2)

    def test_recorded_output_becomes_next_turn_prefix(self):
        result = replay(trace_of([[1, 2, 3], [1, 2, 3, 4, 5]], [[4], []]), 3, LRU(), details=True)
        self.assertEqual(result.request_results[1]["reused_tokens"], 4)
        self.assertEqual(result.output_tokens, 1)

    def test_output_consumes_capacity_but_earns_no_hit_credit(self):
        result = replay(trace_of([[1, 2], [7, 8]], [[3, 4, 5, 6], []]), 3, LRU())
        self.assertEqual(result.peak_occupied_blocks, 3)
        self.assertEqual(result.evicted_blocks, 1)
        self.assertEqual(result.reused_prompt_tokens, 0)

    def test_tail_alone_can_force_eviction(self):
        result = replay(trace_of([[1, 2, 3, 4], [9]]), 2, FIFO(), check_invariants=True)
        self.assertEqual(result.evicted_blocks, 1)

    def test_oversize_rejected_before_policy_runs(self):
        class NeverRun(Policy):
            def observe_insert(self, entry):
                raise AssertionError("Policy should not run")
        with self.assertRaisesRegex(ValueError, "exceeding capacity"):
            replay(trace_of([[1, 2]], [[3, 4, 5]]), 2, NeverRun())

    def test_pinned_prefix_and_ancestors_never_offered(self):
        class Checking(LRU):
            def choose(self, eligible, now):
                # Requests share the first block. It is never a leaf victim.
                self_outer.assertTrue(all(e.depth == 2 for e in eligible))
                return super().choose(eligible, now)
        self_outer = self
        result = replay(trace_of([[1, 2, 3, 4], [1, 2, 5, 6]]), 2, Checking(), check_invariants=True)
        self.assertEqual(result.reused_prompt_tokens, 2)
        self.assertEqual(result.evicted_blocks, 1)

    def test_illegal_policy_victim_rejected(self):
        class Bad(Policy):
            def choose(self, eligible, now):
                return -1
        with self.assertRaisesRegex(ValueError, "illegal victim"):
            replay(trace_of([[1, 2], [3, 4]]), 1, Bad())

    def test_lru_fifo_distinction(self):
        trace = trace_of([[1, 2], [1, 3], [1, 2], [1, 4], [1, 3]], block_size=1)
        self.assertEqual(replay(trace, 3, LRU()).computed_prompt_tokens, 5)
        self.assertEqual(replay(trace, 3, FIFO()).computed_prompt_tokens, 4)

    def test_lfu_lru_distinction(self):
        trace = trace_of([[1], [1], [2], [3], [1]], block_size=1)
        self.assertEqual(replay(trace, 2, LRU()).computed_prompt_tokens, 4)
        self.assertEqual(replay(trace, 2, LFU()).computed_prompt_tokens, 3)

    def test_unlimited_subtraction_and_sufficient_capacity(self):
        trace = trace_of([[1, 2], [3, 4], [1, 2]])
        report = sweep(trace, [1, 2], ["lru", "lfu", "fifo"], check_invariants=True)
        self.assertEqual(report["unlimited"]["computed_prompt_tokens"], 4)
        self.assertEqual([r["extra_computed_tokens"] for r in report["runs"]], [2, 2, 2, 0, 0, 0])

    def test_empty_trace(self):
        result = replay(compile_trace(Trace(())), 1)
        self.assertEqual(result.to_dict()["prompt_hit_ratio"], 0)

    def test_input_validation(self):
        for token in [-1, True, 2**32, 0.5]:
            with self.subTest(token=token), self.assertRaises(ValueError):
                trace_of([[token]])
        for capacity in [0, -1, True, 1.5]:
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                replay(trace_of([[1]]), capacity)
        for size in [0, -1, True, 1.5]:
            with self.subTest(size=size), self.assertRaises(ValueError):
                compile_trace(Trace(()), size)
        with self.assertRaisesRegex(ValueError, "unique"):
            compile_trace(Trace((Request("a", 0, (1,)), Request("a", 1, (2,)))))
        for timestamp in [float("nan"), float("inf"), -1]:
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                compile_trace(Trace((Request("a", timestamp, (1,)),)))
        with self.assertRaises(ValueError):
            compile_trace(Trace((Request("a", 2, (1,)), Request("b", 1, (1,)))))
        with self.assertRaises(ValueError):
            trace_of([[]])

    def test_equal_timestamps_are_deterministic(self):
        trace = compile_trace(Trace(tuple(Request(str(i), 0, (i % 3,)) for i in range(12))), 1)
        values = [replay(trace, 2, LRU(), details=True).request_results for _ in range(3)]
        self.assertEqual(values[0], values[1])
        self.assertEqual(values[1], values[2])

    def test_policy_time_includes_hooks(self):
        from time import perf_counter_ns
        class Slow(FIFO):
            def observe_insert(self, entry):
                until = perf_counter_ns() + 1_000_000
                while perf_counter_ns() < until:
                    pass
        result = replay(trace_of([[1, 2], [3, 4]]), 2, Slow())
        self.assertGreaterEqual(result.policy_time_ns, 2_000_000)
        self.assertGreaterEqual(result.replay_time_ns, result.policy_time_ns)

    def test_roundtrip_and_synthetic_reproducibility(self):
        trace = synthetic_trace(7, 4)
        self.assertEqual(trace, synthetic_trace(7, 4))
        self.assertNotEqual(trace, synthetic_trace(8, 4))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            save_trace(trace, path)
            self.assertEqual(trace, load_trace(path))
            path.write_text(json.dumps({"schema_version": 99}))
            with self.assertRaises(ValueError):
                load_trace(path)

    def test_random_traces_against_independent_prefix_oracle(self):
        # Independent representation: complete token prefixes as tuple keys;
        # discovers leaf victims using tuple prefix comparisons, no compiled IDs.
        for seed in range(12):
            rng = random.Random(seed)
            prompts = [tuple(rng.randrange(3) for _ in range(rng.randint(1, 6))) for _ in range(25)]
            outputs = [tuple(rng.randrange(3) for _ in range(rng.randint(0, 3))) for _ in prompts]
            block_size, capacity = 2, 5
            resident = {}
            expected_computed = expected_evicted = counter = 0
            for prompt, output in zip(prompts, outputs):
                reusable = 0
                for end in range(block_size, len(prompt) + 1, block_size):
                    if prompt[:end] not in resident:
                        break
                    reusable = end
                expected_computed += len(prompt) - reusable
                tokens = prompt + output
                active = {prompt[:end] for end in range(block_size, reusable + 1, block_size)}
                ends = list(range(block_size, len(tokens) + 1, block_size))
                if len(tokens) % block_size:
                    ends.append(None)
                for end in ends:
                    key = tokens[:end] if end is not None else None
                    if key not in resident and len(resident) == capacity:
                        eligible = [p for p in resident if p not in active and
                                    not any(len(q) > len(p) and q[:len(p)] == p for q in resident)]
                        victim = min(eligible, key=resident.get)
                        del resident[victim]
                        expected_evicted += 1
                    if key is not None:
                        counter += 1
                        resident[key] = counter
                        active.add(key)
            result = replay(trace_of(prompts, outputs, block_size), capacity, LRU(), check_invariants=True)
            self.assertEqual(result.computed_prompt_tokens, expected_computed, f"seed={seed}")
            self.assertEqual(result.evicted_blocks, expected_evicted, f"seed={seed}")


if __name__ == "__main__":
    unittest.main()
