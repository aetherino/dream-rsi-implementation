import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dream_rsi import batch_compare as search
from dream_rsi.diagnostics import make_state, report, transitions
from dream_rsi.programs import INITIAL_CONTROLLER


class DiagnosticTests(unittest.TestCase):
    def source(self, output):
        raw = {"trials": 1, "run": {"cycles": 3, "online_rounds": 1, "controller_revisions": 1,
               "api": {"max_calls": 2, "max_usd": .5},
               "train": [{"name": "train", "synthetic": {"seed": 1, "sessions": 3}, "capacity_blocks": 32}],
               "validation": [{"name": "old-val", "synthetic": {"seed": 2, "sessions": 3}, "capacity_blocks": 32}]}}
        state = search.initialize(raw, "mock", Path.cwd())
        for _ in range(10):
            search.tick(state, output, Path.cwd())
            if state["status"] == "completed":
                break
        # Explicit fixture: two accepted revisions with one rejected proposal
        # between them, so predecessor extraction must follow adoption, not order.
        adaptive = next(r for r in state["runs"] if r["arm"] == "adaptive")
        a = INITIAL_CONTROLLER | {"name": "revision-a", "root_priority": "2"}
        b = INITIAL_CONTROLLER | {"name": "revision-b", "leaf_priority": "4"}
        adaptive["revisions"] = [[
            {"revision": 0, "controller": INITIAL_CONTROLLER, "evaluation": {"value": -.1}},
            {"revision": 1, "controller": a, "evaluation": {"value": 0}, "accepted": True, "api_call": 1},
            {"revision": 2, "controller": {"name": "rejected"}, "evaluation": {"value": -.5}, "accepted": False, "api_call": 2},
            {"revision": 3, "controller": b, "evaluation": {"value": .1}, "accepted": True, "api_call": 3}]]
        return state

    def settings(self):
        return {"extension_total_calls": 4, "extension_new_usd": .2, "diagnostic_calls": 2,
                "diagnostic_usd": .1, "replicates": 2, "max_new_usd": 2,
                "source_checkpoint_sha256": "test",
                "validation": [{"name": "fresh-val", "synthetic": {"seed": 3, "sessions": 3}, "capacity_blocks": 32}]}

    def test_all_accepted_transitions_preserve_exact_predecessor(self):
        with tempfile.TemporaryDirectory() as temp:
            source = self.source(Path(temp))
            items = transitions(source)
            self.assertEqual(len(items), 2)
            self.assertEqual(items[0]["predecessor"], INITIAL_CONTROLLER)
            self.assertEqual(items[1]["predecessor"], items[0]["revised"])
            self.assertAlmostEqual(items[0]["historical_replay_gain"], .1)

    def test_extension_and_frozen_comparisons_complete_after_restarts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old = root / "old"; old.mkdir()
            out = root / "new"; out.mkdir()
            source = self.source(old)
            unchanged = copy.deepcopy(source)
            state = make_state(source, self.settings(), Path.cwd())
            self.assertEqual(source, unchanged)
            extension = next(r for r in state["runs"] if r["study"] == "extension")
            self.assertEqual(extension["imported_calls"], 2)
            self.assertNotIn("validation", extension)
            self.assertNotIn("validation_baselines", extension)
            self.assertEqual(len(state["runs"]), 9)
            for run in state["runs"]:
                if run["study"] == "controller-transfer":
                    self.assertEqual(run["records"], [])
                    self.assertEqual(run["histories"], [])
                    self.assertEqual(run["config"]["controller_revisions"], 0)
            for _ in range(20):
                result = search.tick(state, out, Path.cwd(), reporter=report)
                state = json.loads((out / "checkpoint.json").read_text())
                if state["status"] == "completed":
                    break
            self.assertEqual(state["status"], "completed")
            self.assertEqual(len(result["paired_controller_results"]), 4)
            extended = next(r for r in result["runs"] if r["study"] == "extension")
            self.assertEqual((extended["calls"], extended["new_calls"]), (4, 2))
            self.assertIn("reference_validation_score", extended)
            self.assertEqual(result["new_estimated_usd_including_reservations"], 0)
            for run in state["runs"]:
                self.assertEqual(run["controller"], run["initial_controller"])
                if run["study"] == "controller-transfer":
                    self.assertTrue(all(r["role"] == "discovery" for r in run["records"]))
                    saved = json.loads((out / run["id"] / "controller-initial.json").read_text())
                    self.assertEqual(saved, run["controller"])
                for prompt in (out / run["id"] / "prompts").glob("*.json"):
                    self.assertNotIn("fresh-val", prompt.read_text())

    def test_excess_spending_and_changed_semantics_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            source = self.source(Path(temp))
            settings = self.settings() | {"max_new_usd": .01}
            with self.assertRaisesRegex(ValueError, "spending"):
                make_state(source, settings, Path.cwd())
            source["source_hashes"]["dream_rsi/prompts.py"] = "changed"
            with self.assertRaisesRegex(ValueError, "semantics"):
                make_state(source, self.settings(), Path.cwd())


if __name__ == "__main__":
    unittest.main()
