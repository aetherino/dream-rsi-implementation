import copy
import json
from pathlib import Path
import unittest

from dream_rsi.batch_compare import bounded_discovery, initialize
from dream_rsi.batch_transport import allowance
from dream_rsi.diagnostics import make_prompt_continuation
from dream_rsi.history import observation, root
from dream_rsi.programs import INITIAL_CONTROLLER, LRU_SPEC
from dream_rsi.prompts import (controller_prompt, controller_prompt_v1, discovery_prompt, discovery_prompt_v1,
                               tried_memory)


def node(i, score, expression):
    return observation(i, root(), 1, {"name": f"policy-{i}", "retention_score": expression},
                       {"valid": True, "score": score, "total_extra_computed_tokens": 80,
                        "total_lru_extra_computed_tokens": 100, "saved_prompt_tokens": 20,
                        "score_denominator_tokens": 100, "runs": []})


class PromptTests(unittest.TestCase):
    def test_legacy_prompts_are_available_verbatim(self):
        args = (root(), [root()], [{"name": "train"}])
        self.assertEqual(discovery_prompt(*args, version="v1"), discovery_prompt_v1(*args))
        args = (INITIAL_CONTROLLER, {}, [], {"workers": 2, "max_depth": 4, "max_rounds": 6})
        self.assertEqual(controller_prompt(*args, version="v1"), controller_prompt_v1(*args))

    def test_remembers_old_best_and_prior_expressions_beyond_recent_context(self):
        best = node(1, .2, "last_access + 83 * frequency")
        histories = [[root(), best]] + [[root(), node(1, 0, f"last_access + {i} * depth")] for i in range(8)]
        tree = [root(), node(1, .01, "last_access + 1 * frequency")]
        prompt = bounded_discovery(tree[-1], tree, [{"name": "train"}], histories, "total_extra_v2", "v2", best)
        payload = json.loads('{"prompt_version":' + prompt.split('\n{"prompt_version":', 1)[1])
        self.assertEqual(payload["best_ever_training_policy"]["candidate"]["retention_score"], best["candidate"]["retention_score"])
        self.assertEqual(payload["best_ever_training_policy"]["feedback"]["score_denominator_tokens"], 100)
        expressions = {r["retention_score"] for r in payload["tried_expressions"]["expressions"]}
        self.assertIn("last_access + 83 * frequency", expressions)
        self.assertEqual(payload["tried_expressions"]["omitted_count"], 0)
        self.assertIn("MAXIMUMS, NOT TARGETS", prompt)
        self.assertIn("leaves A and B cached", prompt)
        self.assertNotIn("validation", payload)

    def test_memory_groups_formatting_duplicates_and_reports_compaction(self):
        a, b = node(1, .1, "last_access+frequency"), node(2, .1, "last_access + frequency")
        memory = tried_memory([[root(), a]], [root(), b])
        self.assertEqual(memory["unique_expression_count"], 1)
        self.assertEqual(memory["expressions"][0]["attempts"], 2)
        crowded = [root()] + [node(i, .01*i, f"last_access + {i} * frequency") for i in range(1, 100)]
        small = tried_memory([], crowded, max_bytes=1200)
        self.assertGreater(small["omitted_count"], 0)
        self.assertLessEqual(len(json.dumps(small["expressions"]).encode()), 1200)

    def test_controller_receives_distinct_live_and_replay_limits(self):
        settings = {"workers": 2, "max_depth": 4, "max_rounds": 6}
        prompt = controller_prompt(INITIAL_CONTROLLER, {}, [], settings, online_rounds=3)
        payload = json.loads('{"prompt_version":' + prompt.split('\n{"prompt_version":', 1)[1])
        self.assertEqual(payload["deployment"]["online_rounds_per_cycle"], 3)
        self.assertEqual(payload["replay_settings"]["max_rounds"], 6)
        self.assertIn("selects leaf A and root", prompt)
        with self.assertRaisesRegex(ValueError, "actual online"):
            controller_prompt(INITIAL_CONTROLLER, {}, [], settings)

    def test_large_memory_does_not_displace_selected_parent_or_exceed_allowance(self):
        nodes = [root()]
        for i in range(1, 101):
            n = node(i, i/1000, "last_access" + f" + {i}" * 100)
            n["feedback"]["runs"] = [{"name": "x" * 450, "extra_computed_tokens": 123} for _ in range(12)]
            nodes.append(n)
        prompt = bounded_discovery(nodes[-1], nodes, [{"name": "train"}], [nodes], "total_extra_v2", "v2", nodes[-1])
        self.assertLess(allowance(prompt), 64000)
        payload = json.loads('{"prompt_version":' + prompt.split('\n{"prompt_version":', 1)[1])
        self.assertEqual(payload["selected_parent"]["id"], 100)
        self.assertEqual(payload["best_ever_training_policy"]["id"], 100)


class ContinuationTests(unittest.TestCase):
    def source(self):
        state = initialize({"trials": 1, "run": {"cycles": 2, "controller_revisions": 1,
            "api": {"max_calls": 8, "max_usd": .5},
            "train": [{"name": "train", "synthetic": {"seed": 1, "sessions": 3}, "capacity_blocks": 32}],
            "validation": [{"name": "heldout", "synthetic": {"seed": 2, "sessions": 3}, "capacity_blocks": 32}]}}, "mock", Path.cwd())
        state["status"] = "superseded_by_prompt_v2"
        state["plan"]["ceilings"] = {"new_calls": 16, "new_usd": 1}
        for run in state["runs"]:
            run["config"].pop("prompt_version", None)
            run.update(best=root(), imported_calls=0, imported_usd=0,
                       records=[{"call": 1, "charged_estimate_usd": .01, "status": "accounted"}])
        return state

    def test_migration_preserves_histories_budgets_and_accounts_for_prior_spending(self):
        source = self.source()
        unchanged = copy.deepcopy(source)
        state = make_prompt_continuation(source, Path.cwd(), "source-digest")
        self.assertEqual(source, unchanged)
        self.assertEqual(state["plan"]["ceilings"], source["plan"]["ceilings"])
        self.assertEqual(state["plan"]["prompt_revision"]["remaining_call_allowance"], 14)
        self.assertAlmostEqual(state["plan"]["prompt_revision"]["remaining_usd_allowance"], .98)
        for old, new in zip(source["runs"], state["runs"]):
            self.assertEqual(old["config"]["api"], new["config"]["api"])
            self.assertEqual(old["histories"], new["histories"])
            self.assertEqual(old["best"], new["best"])
            self.assertEqual(new["records"][0]["prompt_version"], "v1")
            self.assertEqual(new["config"]["prompt_version"], "v2")

    def test_pending_requests_observed_validation_and_semantic_changes_block_migration(self):
        for change, message in ((lambda s: s.update(active_wave=True), "pending"),
                                (lambda s: s["runs"][0].update(validation={}), "unobserved"),
                                (lambda s: s["source_hashes"].update({"dream_rsi/scoring.py": "changed"}), "Non-prompt")):
            state = self.source(); change(state)
            with self.assertRaisesRegex(ValueError, message):
                make_prompt_continuation(state, Path.cwd(), "source")


if __name__ == "__main__":
    unittest.main()
