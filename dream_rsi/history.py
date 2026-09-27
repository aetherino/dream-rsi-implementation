"""Append-only discovery trees and counterfactual, hidden-outcome replay."""
from .programs import Controller, LRU_SPEC


def root():
    return {"id": 0, "parent": None, "depth": 0, "round": 0, "score": 0.0,
            "valid": True, "stagnation": 0, "failure_streak": 0, "candidate": LRU_SPEC,
            "feedback": {"description": "Built-in LRU reference; normalized score zero"}}


def observation(node_id, parent, round_index, candidate, feedback):
    valid = feedback["valid"]
    score = feedback["score"] if valid else -1e6
    return {"id": node_id, "parent": parent["id"], "depth": parent["depth"] + 1,
            "round": round_index, "score": score, "valid": valid,
            "stagnation": 0 if valid and score > parent["score"] else parent["stagnation"] + 1,
            "failure_streak": 0 if valid else parent["failure_streak"] + 1,
            "candidate": candidate, "feedback": feedback}


def replay_controller(spec, tree, *, workers, max_rounds, max_depth, beta_calls, beta_parallel):
    """Replay recorded outcomes only. No proposal backend or evaluator is called.

    Root reveals the next recorded root child; a leaf reveals its unique child.
    Missing continuations are learned on selection and then marked exhausted.
    They use a round slot but incur no generation-call charge.
    """
    controller = Controller(spec)
    children = {}
    seen = {0}
    for node in tree[1:]:
        if node["id"] in seen or node["parent"] not in seen:
            raise ValueError("History must be in parent-before-child order with unique IDs")
        seen.add(node["id"])
        children.setdefault(node["parent"], []).append(node)
        if node["parent"] != 0 and len(children[node["parent"]]) > 1:
            raise ValueError("Non-root nodes can have only one child")
    visible, exhausted, trajectory = [tree[0]], set(), []
    revealed_ids = {0}
    for round_index in range(1, max_rounds + 1):
        if len(visible) == len(tree):
            break
        actions = controller.select(visible, round_index, workers, max_depth, exhausted)
        if not actions:
            break
        new = []
        for action in actions:
            child = next((n for n in children.get(action, []) if n["id"] not in revealed_ids), None)
            if child is None:
                exhausted.add(action)
            else:
                # Recompute online-visible round/age features for this replay schedule.
                parent = next(n for n in visible if n["id"] == action)
                copied = observation(child["id"], parent, round_index, child["candidate"], child["feedback"])
                new.append(copied)
                revealed_ids.add(child["id"])
        visible.extend(new)
        trajectory.append({"round": round_index, "selected": actions, "revealed": [n["id"] for n in new],
                           "best_score": max(n["score"] for n in visible if n["valid"])})
    attempts, rounds = len(visible) - 1, len(trajectory)
    best = max(n["score"] for n in visible if n["valid"])
    value = best - beta_calls * attempts + beta_parallel * attempts / max(1, rounds)
    return {"value": value, "best_score": best, "attempts": attempts, "rounds": rounds,
            "trajectory": trajectory, "exhausted": sorted(exhausted)}


def evaluate_controller(spec, histories, **kwargs):
    try:
        replays = [replay_controller(spec, history, **kwargs) for history in histories]
        return {"valid": True, "value": sum(r["value"] for r in replays) / len(replays), "replays": replays}
    except (ValueError, TypeError, KeyError) as exc:
        return {"valid": False, "value": None, "error": str(exc)}
