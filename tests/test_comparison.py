import copy
import json
from pathlib import Path
import tempfile
import unittest

from dream_rsi.compare import aggregate, arm_config, compare
from dream_rsi.evaluation import snapshot_suite


class ComparisonTests(unittest.TestCase):
    def test_arms_share_all_settings_except_controller_revisions(self):
        base = {"controller_revisions": 2, "api": {"max_calls": 8, "max_usd": .1}, "train": [{"name": "a"}]}
        unchanged = copy.deepcopy(base)
        fixed, adaptive = arm_config(base, "fixed"), arm_config(base, "adaptive")
        self.assertEqual(base, unchanged)
        self.assertEqual(fixed | {"controller_revisions": 2}, adaptive)
        self.assertFalse(adaptive["revise_after_final_cycle"])
        self.assertTrue(adaptive["stop_on_empty_rollout"])

    def test_refreezing_does_not_accept_changed_trace(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trace.json"
            path.write_text("a")
            frozen = snapshot_suite([{"name": "a", "path": str(path), "capacity_blocks": 32}], Path(temp))
            path.write_text("b")
            with self.assertRaisesRegex(ValueError, "changed"):
                snapshot_suite(frozen, Path(temp))

    def test_comparison_uses_caps_and_counts_meta_calls(self):
        raw = {"trials": 1, "run": {"cycles": 3, "online_rounds": 2, "controller_revisions": 1,
                "api": {"max_calls": 8},
                "train": [{"name": "tiny", "synthetic": {"seed": 42, "sessions": 3}, "capacity_blocks": 32}],
                "validation": [{"name": "heldout", "synthetic": {"seed": 99, "sessions": 3}, "capacity_blocks": 32}]}}
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "comparison"
            result = compare(raw, "mock", Path.cwd(), out)
            self.assertEqual(result["status"], "completed")
            rows = {r["arm"]: r for r in result["runs"]}
            self.assertEqual(rows["fixed"]["calls"], 8)
            self.assertEqual(rows["fixed"]["controller_calls"], 0)
            self.assertLessEqual(rows["adaptive"]["calls"], 8)
            self.assertGreater(rows["adaptive"]["controller_calls"], 0)
            self.assertEqual(rows["adaptive"]["calls"], rows["adaptive"]["discovery_calls"] + rows["adaptive"]["controller_calls"])
            self.assertEqual(result["aggregate"]["estimated_usd"], 0)
            self.assertEqual(len(result["aggregate"]["paired_differences"]), 1)
            for arm in ("fixed", "adaptive"):
                saved = json.loads((out / f"trial-01-{arm}" / "config.json").read_text())
                self.assertEqual(saved["train"], raw["run"]["train"])
                self.assertFalse(saved["revise_after_final_cycle"])
            self.assertIn("not exact dollar-spend matching", (out / "report.md").read_text())

    def test_empty_report_does_not_invent_results(self):
        self.assertIsNone(aggregate([])["mean_adaptive_minus_fixed"])

    def test_positive_macro_score_can_still_increase_total_recomputation(self):
        row = {"trial": 1, "arm": "adaptive", "validation_score": .49, "train_score": .1,
               "calls": 8, "discovery_calls": 7, "controller_calls": 1, "accepted_revisions": 0,
               "estimated_usd": .01, "scenarios": [
                   {"extra_computed_tokens": 0, "lru_extra_computed_tokens": 10},
                   {"extra_computed_tokens": 1020, "lru_extra_computed_tokens": 1000}]}
        metrics = aggregate([row])["arms"]["adaptive"]
        self.assertGreater(metrics["mean_validation_score"], 0)
        self.assertEqual(metrics["mean_validation_extra_tokens"], 1020)
        self.assertEqual(metrics["mean_lru_validation_extra_tokens"], 1010)
        self.assertGreater(metrics["total_extra_fraction_change_vs_lru"], 0)


if __name__ == "__main__":
    unittest.main()
