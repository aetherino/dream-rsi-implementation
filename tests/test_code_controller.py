import copy
import unittest
from unittest.mock import patch

from dream_rsi.code_controller import (
    CodeController, INITIAL_CONTROLLER_SPEC, evaluate_controller_source,
    replay_controller_source,
)
from dream_rsi.history import observation, replay_controller, root
from dream_rsi.programs import INITIAL_CONTROLLER, LRU_SPEC


def spec(source):
    return {"name": "test-controller", "source": source, "rationale": "test"}


LEGACY_SOURCE = '''class ExplorationController:
    def select(self, nodes, round_index, workers, max_depth):
        parents = {node["parent"] for node in nodes[1:]}
        ranked = [] if nodes[0].get("exhausted") else [(1, 0)]
        for node in nodes[1:]:
            if node["id"] in parents or node.get("exhausted") or node["depth"] >= max_depth:
                continue
            if node["failure_streak"] < 2 and node["stagnation"] < 3:
                ranked.append((2 + node["score"] - 0.1 * node["depth"], node["id"]))
        ranked.sort(key=lambda pair: (-pair[0], pair[1]))
        return [node_id for priority, node_id in ranked[:workers]]
'''


class SourceControllerTest(unittest.TestCase):
    settings = dict(workers=2, max_rounds=5, max_depth=5,
                    beta_calls=.02, beta_parallel=.005)

    def history(self):
        tree = [root()]
        for parent_id, score, round_index in [(0, .1, 1), (1, .3, 2), (0, .2, 2), (3, .5, 3)]:
            tree.append(observation(len(tree), tree[parent_id], round_index, LRU_SPEC,
                                    {"valid": True, "score": score, "runs": []}))
        return tree

    def test_source_replay_matches_legacy_transition_and_equation(self):
        old = replay_controller(INITIAL_CONTROLLER, self.history(), **self.settings)
        new = replay_controller_source(spec(LEGACY_SOURCE), self.history(), **self.settings)
        self.assertTrue(new["valid"], new)
        self.assertGreater(new["controller_cpu_time_ns"], 0)
        for key in ("value", "best_score", "attempts", "rounds", "exhausted"):
            self.assertEqual(old[key], new[key])
        for before, after in zip(old["trajectory"], new["trajectory"]):
            for key in ("round", "selected", "revealed", "best_score"):
                self.assertEqual(before[key], after[key])

    def test_unrevealed_outcomes_do_not_change_prior_decisions(self):
        tree = self.history()
        changed = copy.deepcopy(tree)
        changed[-1]["score"] = changed[-1]["feedback"]["score"] = 999
        first = replay_controller_source(spec(LEGACY_SOURCE), tree, **self.settings)
        other = replay_controller_source(spec(LEGACY_SOURCE), changed, **self.settings)
        self.assertTrue(first["valid"] and other["valid"], (first, other))
        self.assertEqual([step["selected"] for step in first["trajectory"]],
                         [step["selected"] for step in other["trajectory"]])
        self.assertEqual(first["trajectory"][0]["observed"], [0])
        self.assertNotEqual(first["best_score"], other["best_score"])
        self.assertEqual(tree, self.history())
        self.assertEqual(set(first["trajectory"][0]["observations"][0]),
                         {"id", "round", "stagnation", "failure_streak"})

    def test_controller_state_persists_within_world_and_resets_between_worlds(self):
        source = '''class ExplorationController:
    def __init__(self):
        self.calls = 0
    def select(self, nodes, round_index, workers, max_depth):
        self.calls += 1
        return [0] if self.calls == 1 else []
'''
        result = evaluate_controller_source(spec(source), [self.history(), self.history()], **self.settings)
        self.assertTrue(result["valid"], result)
        self.assertTrue(result["determinism_checked"])
        self.assertEqual([world["attempts"] for world in result["replays"]], [1, 1])
        self.assertEqual([world["rounds"] for world in result["replays"]], [1, 1])

    def test_distinct_root_fifo_creation_order_and_no_new_outcomes(self):
        tree = [root()]
        for node_id in (9, 4):
            tree.append(observation(node_id, tree[0], 1, LRU_SPEC, {"valid": True, "score": .1 * node_id}))
        source = 'class ExplorationController:\n    def select(self, nodes, round_index, workers, max_depth):\n        return [0]\n'
        result = replay_controller_source(spec(source), tree, **self.settings)
        self.assertTrue(result["valid"], result)
        self.assertEqual([step["revealed"] for step in result["trajectory"]], [[9], [4]])
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["rounds"], 2)

    def test_missing_continuations_consume_rounds_but_not_attempts(self):
        source = '''class ExplorationController:
    def select(self, nodes, round_index, workers, max_depth):
        return [0] if len(nodes) == 1 else [1]
'''
        tree = [root()]
        for score in (.1, .2):
            tree.append(observation(len(tree), tree[0], 1, LRU_SPEC, {"valid": True, "score": score}))
        result = replay_controller_source(spec(source), tree, **(self.settings | {"max_rounds": 4}))
        self.assertTrue(result["valid"], result)
        self.assertEqual((result["attempts"], result["rounds"]), (1, 4))
        self.assertEqual([step["revealed"] for step in result["trajectory"]], [[1], [], [], []])
        self.assertEqual(result["exhausted"], [1])
        self.assertAlmostEqual(result["value"], .1 - .02 + .005 / 4)

    def test_invalid_batch_and_missing_capability_become_feedback(self):
        for batch in ("[0, 0]", "[999]", "[0, 1, 2]", "[True]", "None", "[1]"):
            source = f'class ExplorationController:\n    def select(self, nodes, round_index, workers, max_depth):\n        return {batch}\n'
            with self.subTest(batch=batch):
                result = evaluate_controller_source(spec(source), [self.history()], **self.settings)
                self.assertFalse(result["valid"], result)
                self.assertIsNone(result["value"])
                self.assertIn("error", result)
        for source in ('class WrongName:\n    pass\n', 'class ExplorationController:\n    pass\n',
                       'class ExplorationController:\n    def select(self):\n        return []\n'):
            result = evaluate_controller_source(spec(source), [self.history()], **self.settings)
            self.assertFalse(result["valid"], result)

    def test_current_nonleaf_and_depth_limit_are_illegal(self):
        for batch in ("[1]", "[2]"):
            source = f'class ExplorationController:\n    def select(self, nodes, round_index, workers, max_depth):\n        return {batch}\n'
            with self.subTest(batch=batch), CodeController(spec(source)) as controller:
                with self.assertRaisesRegex(ValueError, "illegal"):
                    controller.select(self.history(), 5, 2, 2)

    def test_child_failure_is_invalid_feedback(self):
        source = '''class ExplorationController:
    def select(self, nodes, round_index, workers, max_depth):
        raise ValueError("controller failed")
'''
        result = evaluate_controller_source(spec(source), [self.history()], **self.settings)
        self.assertFalse(result["valid"], result)
        self.assertIsNone(result["value"])
        self.assertIn("controller failed", result["error"])

    def test_copies_prefix_and_recomputes_replay_local_round(self):
        source = '''class ExplorationController:
    def select(self, nodes, round_index, workers, max_depth):
        if len(nodes) == 1:
            nodes[0]["score"] = 999
            return [0]
        return [1] if nodes[1]["round"] == 1 and round_index == 2 else []
'''
        tree = self.history()
        tree[1]["round"] = 999
        preserved = copy.deepcopy(tree)
        result = replay_controller_source(spec(source), tree, **self.settings)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["best_score"], .3)
        self.assertEqual(tree, preserved)

    def test_optional_grid_planning_never_changes_replay_caps(self):
        source = '''class ExplorationController:
    def plan_grid(self, context):
        raise ValueError("§3 replay must not call plan_grid")
    def select(self, nodes, round_index, workers, max_depth):
        return [0]
'''
        result = replay_controller_source(spec(source), self.history(), **(self.settings | {"max_rounds": 1}))
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["attempts"], 1)
        self.assertEqual(result["rounds"], 1)

    def test_initial_parallel_refinement_preserves_formal_single_root_action(self):
        result = replay_controller_source(INITIAL_CONTROLLER_SPEC, self.history(), **self.settings)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["trajectory"][0]["selected"], [0])
        self.assertEqual(set(result["trajectory"][1]["selected"]), {0, 1})
        self.assertEqual(result["attempts"], 4)
        self.assertEqual(result["rounds"], 3)

    def test_mean_across_worlds_and_empty_world_pool(self):
        result = evaluate_controller_source(INITIAL_CONTROLLER_SPEC, [self.history(), [root()]], **self.settings)
        self.assertTrue(result["valid"], result)
        self.assertAlmostEqual(result["value"], sum(item["value"] for item in result["replays"]) / 2)
        failed = evaluate_controller_source(INITIAL_CONTROLLER_SPEC, [], **self.settings)
        self.assertFalse(failed["valid"])

    def test_nondeterministic_replay_is_rejected_before_adoption(self):
        first = {"valid": True, "value": 0, "trajectory": [], "exhausted": []}
        changed = first | {"value": 1}
        with patch('dream_rsi.code_controller.replay_controller_source', side_effect=[first, changed]):
            result = evaluate_controller_source(INITIAL_CONTROLLER_SPEC, [self.history()], **self.settings)
        self.assertFalse(result["valid"])
        self.assertIn("nondeterministic", result["error"])

    def test_malformed_tree_and_coefficients_are_invalid_feedback(self):
        malformed = self.history()
        malformed.append(observation(5, malformed[1], 4, LRU_SPEC, {"valid": True, "score": .8}))
        for tree, settings in ((malformed, self.settings), (self.history(), self.settings | {"beta_calls": -1}),
                               (self.history(), self.settings | {"workers": 0})):
            result = replay_controller_source(INITIAL_CONTROLLER_SPEC, tree, **settings)
            self.assertFalse(result["valid"], result)
            self.assertIsNone(result["value"])


if __name__ == '__main__':
    unittest.main()
