import json
from pathlib import Path
import random
import unittest

from dream_rsi.compare import aggregate
from dream_rsi.evaluation import run_suite
from dream_rsi.history import root
from dream_rsi.programs import INITIAL_CONTROLLER, LRU_SPEC
from dream_rsi.prompts import discovery_prompt, controller_prompt
from dream_rsi.runner import prepare_config
from dream_rsi.scoring import TOTAL_EXTRA, LEGACY_MACRO, score_recomputation


class ScoringTests(unittest.TestCase):
    def test_user_example_now_penalizes_more_total_work(self):
        new = score_recomputation([50, 11000], [100, 10000])
        old = score_recomputation([50, 11000], [100, 10000], LEGACY_MACRO)
        self.assertAlmostEqual(old["score"], .2)
        self.assertAlmostEqual(new["score"], -950 / 10100)
        self.assertEqual(new["saved_prompt_tokens"], -950)

    def test_zero_baseline_scenario_has_no_special_penalty(self):
        result = score_recomputation([16, 900], [0, 1000])
        self.assertAlmostEqual(result["score"], .084)
        self.assertEqual(result["score_contributions"], [-.016, .1])
        self.assertEqual(score_recomputation([0, 0], [0, 0])["score"], 0)
        self.assertEqual(score_recomputation([16, 0], [0, 0])["score"], -16)

    def test_score_sign_and_order_always_follow_total_tokens(self):
        rng = random.Random(42)
        for _ in range(100):
            lru = [rng.randrange(10000) for _ in range(5)]
            first = [rng.randrange(10000) for _ in range(5)]
            second = [rng.randrange(10000) for _ in range(5)]
            a = score_recomputation(first, lru)
            b = score_recomputation(second, lru)
            self.assertEqual(a["score"] > 0, sum(first) < sum(lru))
            self.assertEqual(a["score"] > b["score"], sum(first) < sum(second))
            self.assertAlmostEqual(a["score"], sum(a["score_contributions"]))

    def test_splitting_a_scenario_does_not_change_score(self):
        self.assertEqual(score_recomputation([900], [1000])["score"],
                         score_recomputation([0, 900], [100, 900])["score"])

    def test_validation_and_objective_version(self):
        for args in (([], []), ([1], []), ([-1], [2]), ([True], [2])):
            with self.assertRaises(ValueError):
                score_recomputation(*args)
        with self.assertRaises(ValueError):
            score_recomputation([1], [2], "typo")
        with self.assertRaises(ValueError):
            prepare_config({"scoring": "typo"}, Path.cwd())
        with self.assertRaises(ValueError):
            aggregate([{"scoring": TOTAL_EXTRA}, {"scoring": LEGACY_MACRO}])

    def test_worker_metrics_and_prompt_agree(self):
        suite = [{"name": "tiny", "synthetic": {"seed": 42, "sessions": 3}, "capacity_blocks": 32}]
        baseline = run_suite(suite, 16)
        result = run_suite(suite, 16, candidate=LRU_SPEC, baselines=baseline["baselines"])
        self.assertEqual(result["scoring"], TOTAL_EXTRA)
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["total_extra_computed_tokens"], result["total_lru_extra_computed_tokens"])
        self.assertEqual(result["runs"][0]["score_contribution"], 0)
        prompts = [discovery_prompt(root(), [root()], []),
                   controller_prompt(INITIAL_CONTROLLER, {}, [],
                                     {"workers": 2, "max_depth": 4, "max_rounds": 6}, online_rounds=3)]
        for prompt in prompts:
            self.assertIn("SUM of LRU extra tokens", prompt)
            self.assertNotIn("equally weighted mean", prompt)
        self.assertIn("equally weighted mean", discovery_prompt(root(), [root()], [], scoring=LEGACY_MACRO))

    def test_rerun_changes_only_objective_and_validation_shards(self):
        old = json.loads(Path("configs/dream-comparison.json").read_text())
        new = json.loads(Path("configs/dream-comparison-total.json").read_text())
        self.assertEqual(old["trials"], new["trials"])
        for key in old["run"]:
            if key not in ("scoring", "validation"):
                self.assertEqual(old["run"][key], new["run"][key])
        self.assertFalse({s["path"] for s in old["run"]["validation"]} & {s["path"] for s in new["run"]["validation"]})
        self.assertEqual(new["run"]["scoring"], TOTAL_EXTRA)


if __name__ == "__main__":
    unittest.main()
