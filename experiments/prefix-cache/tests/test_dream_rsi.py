import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

from cache_sim.policies import Entry
from dream_rsi.backend import BudgetExceeded, MiMo, parse_program
from dream_rsi.evaluation import run_suite, isolated_evaluate, snapshot_suite
from dream_rsi.history import evaluate_controller, observation, replay_controller, root
from dream_rsi.programs import Controller, Expression, INITIAL_CONTROLLER, LRU_SPEC, POLICY_NAMES, RetentionPolicy
from dream_rsi.runner import API_DEFAULTS, prepare_config, run


class ProgramsTest(unittest.TestCase):
    def test_rejects_access_and_unbounded_computation(self):
        for source in ("__import__('os')", "now.__class__", "[now][0]", "2 ** 999", "sum([1])",
                       "block_id", "future_score", "lambda: 1", "float('nan')", "min(now)", "1e300"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                Expression(source, POLICY_NAMES)

    def test_numeric_bounds_and_short_circuit(self):
        self.assertEqual(Expression("2 if now == 0 else 1 / now", POLICY_NAMES)({"now": 0}), 2)
        self.assertFalse(Expression("False and 1 / 0", POLICY_NAMES)({}))
        for source in ("1 / 0", "sqrt(-1)", "1000000 * 1000000 * 2"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                Expression(source, POLICY_NAMES)({})

    def test_retention_policy_uses_causal_metadata_and_ties(self):
        entries = [Entry(88, 3, 0, 2, 1, 1, 4), Entry(1, 2, 0, 2, 4, 2, 3)]
        self.assertEqual(RetentionPolicy(LRU_SPEC).choose(entries, 5), 1)
        self.assertEqual(RetentionPolicy({"name": "lfu", "retention_score": "frequency"}).choose(entries, 5), 88)

    def test_size_limits(self):
        for source in ("1+" * 100 + "1", "-" * 20 + "1", "1" * 1001):
            with self.assertRaises(ValueError):
                Expression(source, POLICY_NAMES)


class ReplayTest(unittest.TestCase):
    settings = dict(workers=2, max_rounds=5, max_depth=5, beta_calls=.02, beta_parallel=.005)

    def history(self):
        tree = [root()]
        for parent_id, score, round_index in [(0, .1, 1), (1, .3, 2), (0, .2, 2), (3, .5, 3)]:
            tree.append(observation(len(tree), tree[parent_id], round_index, LRU_SPEC,
                                    {"valid": True, "score": score, "runs": []}))
        return tree

    def test_best_minus_calls_plus_parallel_formula(self):
        result = replay_controller(INITIAL_CONTROLLER, self.history(), **self.settings)
        self.assertEqual(result["attempts"], 4)
        self.assertEqual(result["best_score"], .5)
        self.assertAlmostEqual(result["value"], .5 - .02 * 4 + .005 * 4 / result["rounds"])

    def test_unrevealed_outcomes_cannot_affect_selection(self):
        tree = self.history()
        other = copy.deepcopy(tree)
        other[-1]["score"] = other[-1]["feedback"]["score"] = 999
        a = replay_controller(INITIAL_CONTROLLER, tree, **self.settings)["trajectory"]
        b = replay_controller(INITIAL_CONTROLLER, other, **self.settings)["trajectory"]
        self.assertEqual([r["selected"] for r in a], [r["selected"] for r in b])
        self.assertEqual([r["revealed"] for r in a], [r["revealed"] for r in b])

    def test_stopping_and_failed_program(self):
        stop = INITIAL_CONTROLLER | {"root_if": "False", "leaf_if": "False"}
        result = replay_controller(stop, self.history(), **self.settings)
        self.assertEqual((result["attempts"], result["rounds"], result["value"]), (0, 0, 0))
        failed = evaluate_controller(INITIAL_CONTROLLER | {"root_priority": "1 / 0"}, [self.history()], **self.settings)
        self.assertFalse(failed["valid"])

    def test_root_fifo_and_missing_continuation(self):
        spec = INITIAL_CONTROLLER | {"leaf_priority": "100", "leaf_if": "True"}
        result = replay_controller(spec, self.history(), **(self.settings | {"workers": 1}))
        self.assertEqual(result["trajectory"][0]["revealed"], [1])
        self.assertEqual(result["trajectory"][1]["revealed"], [2])
        self.assertEqual(result["trajectory"][2]["revealed"], [])
        self.assertIn(2, result["exhausted"])
        self.assertEqual(result["trajectory"][3]["revealed"], [3])

    def test_branch_invariants_and_replay_recomputes_age(self):
        tree = self.history()
        tree.append(observation(5, tree[1], 4, LRU_SPEC, {"valid": True, "score": .8}))
        with self.assertRaises(ValueError):
            replay_controller(INITIAL_CONTROLLER, tree, **self.settings)
        ages = INITIAL_CONTROLLER | {"root_if": "attempts_seen == 0", "leaf_if": "age_rounds == 1"}
        tree = self.history()
        tree[1]["round"] = 999
        result = replay_controller(ages, tree, **self.settings)
        self.assertEqual(result["attempts"], 2)

    def test_only_root_and_leaves_can_be_selected(self):
        actions = Controller(INITIAL_CONTROLLER).select(self.history(), 4, 8, 10)
        self.assertEqual(set(actions), {0, 2, 4})


class EvaluationTest(unittest.TestCase):
    suite = [{"name": "tiny", "synthetic": {"seed": 42, "sessions": 3}, "capacity_blocks": 32}]

    def test_same_lru_has_zero_recomputation_improvement(self):
        baseline = run_suite(self.suite, 16)
        result = run_suite(self.suite, 16, candidate=LRU_SPEC, baselines=baseline["baselines"])
        self.assertTrue(result["valid"])
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["runs"][0]["evicted_blocks"], baseline["baselines"][0]["policies"]["lru"]["evicted_blocks"])

    def test_bad_policy_and_capacity_are_failed_candidates(self):
        bad = isolated_evaluate({"suite": self.suite, "block_size": 16,
                                 "candidate": {"name": "bad", "retention_score": "open('a')"}}, timeout=20)
        self.assertFalse(bad["valid"])
        with self.assertRaises(ValueError):
            run_suite([self.suite[0] | {"capacity_blocks": 1}], 16)

    def test_child_timeout_is_a_recorded_failure(self):
        import subprocess
        with patch("dream_rsi.evaluation.subprocess.run", side_effect=subprocess.TimeoutExpired("worker", .01)):
            result = isolated_evaluate({}, timeout=.01)
        self.assertFalse(result["valid"])
        self.assertIn("wall-time", result["error"])

    def test_time_gate_and_trace_hash(self):
        base = run_suite(self.suite, 16)["baselines"]
        result = run_suite(self.suite, 16, candidate=LRU_SPEC, baselines=base, max_policy_us_per_request=.000001)
        self.assertFalse(result["valid"])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trace.json"
            path.write_text("{}")
            suite = snapshot_suite([{"name": "hash", "path": str(path), "capacity_blocks": 32}], Path(temp))
            path.write_text("{\"changed\":true}")
            with self.assertRaisesRegex(ValueError, "changed"):
                run_suite(suite, 16)

    def test_split_leakage_rejected(self):
        with self.assertRaisesRegex(ValueError, "disjoint"):
            prepare_config({"train": self.suite, "validation": self.suite}, Path.cwd())
        overlapping = [self.suite[0] | {"synthetic": {"seed": 42, "sessions": 4}}]
        with self.assertRaisesRegex(ValueError, "disjoint"):
            prepare_config({"train": self.suite, "validation": overlapping}, Path.cwd())


class BackendTest(unittest.TestCase):
    def test_json_and_api_usage_limits(self):
        self.assertEqual(parse_program('```json\n{"name":"ok"}\n```'), {"name": "ok"})
        with tempfile.TemporaryDirectory() as temp, patch.dict("os.environ", {"XIAOMI_API": "test-secret"}):
            backend = MiMo(API_DEFAULTS | {"max_calls": 1}, Path(temp), Path(temp) / "usage.json")
            response = Mock(status_code=200)
            response.json.return_value = {"usage": {"prompt_tokens": 20, "completion_tokens": 10},
                                           "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(LRU_SPEC)}}]}
            with patch("dream_rsi.backend.requests.post", return_value=response) as post:
                self.assertEqual(backend.generate("discovery", "test", {}), LRU_SPEC)
                with self.assertRaises(BudgetExceeded):
                    backend.generate("controller", "test", {})
                self.assertEqual(post.call_count, 1)
                self.assertFalse(post.call_args.kwargs["allow_redirects"])
                self.assertEqual(post.call_args.kwargs["json"]["thinking"], {"type": "disabled"})
            saved = (Path(temp) / "usage.json").read_text()
            self.assertNotIn("test-secret", saved)
            self.assertAlmostEqual(backend.summary()["estimated_usd"], (20 * .435 + 10 * .87) / 1e6)

    def test_failed_requests_keep_reservation_and_no_retry(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict("os.environ", {"XIAOMI_API": "test-secret"}):
            backend = MiMo(API_DEFAULTS | {"max_calls": 1}, Path(temp), Path(temp) / "usage.json")
            with patch("dream_rsi.backend.requests.post", return_value=Mock(status_code=429)) as post:
                with self.assertRaisesRegex(RuntimeError, "HTTP 429"):
                    backend.generate("discovery", "test", {})
                self.assertEqual(post.call_count, 1)
            self.assertGreater(backend.summary()["estimated_usd"], 0)
            self.assertEqual(backend.records[0]["status"], "failed_usage_unknown_reservation_retained")

    def test_dollar_cap_prevents_request(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict("os.environ", {"XIAOMI_API": "test-secret"}):
            backend = MiMo(API_DEFAULTS | {"max_usd": .000001}, Path(temp), Path(temp) / "usage.json")
            with patch("dream_rsi.backend.requests.post") as post, self.assertRaises(BudgetExceeded):
                backend.generate("discovery", "test", {})
            post.assert_not_called()

    def test_truncated_reasoning_response_is_accounted_but_rejected(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict("os.environ", {"XIAOMI_API": "test-secret"}):
            backend = MiMo(API_DEFAULTS, Path(temp), Path(temp) / "usage.json")
            response = Mock(status_code=200)
            response.json.return_value = {"usage": {"prompt_tokens": 20, "completion_tokens": 4096},
                                           "choices": [{"finish_reason": "length", "message": {"content": ""}}]}
            with patch("dream_rsi.backend.requests.post", return_value=response), self.assertRaisesRegex(ValueError, "incomplete"):
                backend.generate("discovery", "test", {})
            self.assertEqual(backend.records[0]["finish_reason"], "length")
            self.assertEqual(backend.records[0]["completion_tokens"], 4096)
            self.assertEqual(backend.records[0]["status"], "accounted")


class IntegrationTest(unittest.TestCase):
    def test_strictly_better_controller_is_adopted(self):
        from dream_rsi.backend import Mock as MockBackend
        original = MockBackend.generate

        def propose(backend, role, prompt, context):
            result = original(backend, role, prompt, context)
            if role == "controller":
                return INITIAL_CONTROLLER | {"name": "stop", "root_if": "False", "leaf_if": "False"}
            return result

        config = {"cycles": 2, "online_rounds": 4, "controller_revisions": 1,
                  "train": [{"name": "sample", "synthetic": {"seed": 42, "sessions": 12}, "capacity_blocks": 32}]}
        with tempfile.TemporaryDirectory() as temp, patch.object(MockBackend, "generate", propose):
            out = Path(temp) / "run"
            summary = run(config, "mock", Path.cwd(), out)
            revisions = json.loads((out / "controller-revisions-001.json").read_text())
            self.assertTrue(revisions[1]["accepted"])
            self.assertGreater(revisions[1]["evaluation"]["value"], revisions[0]["evaluation"]["value"])
            next_tree = json.loads((out / "tree-002.json").read_text())
            self.assertEqual(next_tree["controller"]["name"], "stop")
            self.assertEqual(len(next_tree["nodes"]), 1)
            self.assertEqual(summary["controller"]["name"], "stop")

    def test_mock_two_level_loop_and_saved_replays(self):
        config = {"cycles": 2, "online_rounds": 2, "controller_revisions": 1,
                  "train": EvaluationTest.suite,
                  "validation": [{"name": "heldout", "synthetic": {"seed": 314, "sessions": 3}, "capacity_blocks": 32}]}
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            summary = run(config, "mock", Path.cwd(), out)
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["cycles_recorded"], 2)
            self.assertTrue(summary["validation_valid"])
            self.assertEqual(summary["usage"]["estimated_usd"], 0)
            self.assertEqual(summary["usage"]["calls"], summary["task_attempts"] + 2)
            for path in out.glob("controller-revisions-*.json"):
                previous = -float("inf")
                for revision in json.loads(path.read_text()):
                    if revision["accepted"]:
                        self.assertGreaterEqual(revision["evaluation"]["value"], previous)
                        previous = revision["evaluation"]["value"]
            for path in (out / "prompts").glob("*.json"):
                self.assertNotIn("heldout", path.read_text())
            with self.assertRaises(FileExistsError):
                run(config, "mock", Path.cwd(), out)


if __name__ == "__main__":
    unittest.main()
