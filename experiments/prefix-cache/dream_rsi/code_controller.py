"""Executable, prefix-only exploration and the formal Dream-RSI §3 replay.

The formal action is a set of parent IDs, so root 0 appears at most once in a
batch. This cannot reproduce §4's simultaneous initial workspace fan-out. The
initial source below opens those independent branches over successive rounds.
Appendix B's AUC objective and grid planning do not alter this §3 evaluator.
"""
import ast
import copy
import math

from .code_policy import CodePolicyError, SandboxProgram
from .programs import metadata


INITIAL_CONTROLLER_SOURCE = '''class ExplorationController:
    """Open at most one independent branch per round, then refine in parallel."""
    def select(self, observed_nodes, round_index, workers, max_depth):
        parents = {node["parent"] for node in observed_nodes[1:]}
        leaves = [node for node in observed_nodes[1:]
                  if node["id"] not in parents and node["depth"] < max_depth
                  and not node.get("exhausted", False)]
        leaves.sort(key=lambda node: node["id"])
        batch = [node["id"] for node in leaves[:workers]]
        branches = sum(node["parent"] == 0 for node in observed_nodes[1:])
        if len(batch) < workers and branches < workers and not observed_nodes[0].get("exhausted", False):
            batch.append(0)
        return batch
'''
INITIAL_CONTROLLER_SPEC = {
    "name": "parallel-refinement-code",
    "source": INITIAL_CONTROLLER_SOURCE,
    "rationale": "Fixed parallel workspace refinement adapted to §3's distinct-parent action set; one root opening per round.",
}
BASELINE_CONTROLLER_SPEC = INITIAL_CONTROLLER_SPEC


def _source(spec):
    metadata(spec)
    if set(spec) - {"name", "source", "rationale"}:
        raise ValueError("Unexpected executable controller fields")
    source = spec.get("source")
    if not isinstance(source, str) or not source.strip() or len(source) > 100_000:
        raise ValueError("Controller requires Python source of at most 100000 characters")
    try:
        module = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"Controller source syntax error: {exc.msg}") from exc
    classes = [node for node in module.body if isinstance(node, ast.ClassDef)
               and node.name == "ExplorationController"]
    if len(classes) != 1:
        raise ValueError("Source must define exactly one ExplorationController class")
    methods = [node for node in classes[0].body if isinstance(node, ast.FunctionDef)
               and node.name == "select"]
    if len(methods) != 1:
        raise ValueError("ExplorationController must implement select")
    args = methods[0].args
    if len(args.posonlyargs) + len(args.args) < 5 and args.vararg is None:
        raise ValueError("select must accept observed_nodes, round_index, workers, max_depth")
    return source


def _legal(nodes, max_depth):
    if not isinstance(nodes, list) or not nodes or nodes[0].get("id") != 0:
        raise ValueError("Missing initial root")
    parents = {node["parent"] for node in nodes[1:]}
    return {0} | {node["id"] for node in nodes[1:]
                  if node["id"] not in parents and node["depth"] < max_depth}


def _validate_batch(batch, nodes, workers, max_depth):
    if not isinstance(batch, list) or any(type(node_id) is not int for node_id in batch):
        raise ValueError("Controller select must return a list of integer node IDs")
    if len(batch) > workers:
        raise ValueError("Controller batch exceeds worker limit")
    if len(set(batch)) != len(batch):
        raise ValueError("Controller batch contains duplicate node IDs")
    if not set(batch) <= _legal(nodes, max_depth):
        raise ValueError("Controller selected an illegal node; only root and current leaves below max_depth are legal")
    return batch


class CodeController:
    """One persistent sandbox per episode; decisions receive JSON prefixes only.

    ``exhausted`` records only prior selections that returned no continuation.
    It is observable replay feedback, never a lookahead into the saved tree.
    Formal root/leaves remain legal after a miss; controllers may retry them.
    """
    def __init__(self, spec, **sandbox_limits):
        self.name = spec.get("name") if isinstance(spec, dict) else None
        self._program = SandboxProgram(_source(spec), class_name="ExplorationController", **sandbox_limits)

    def select(self, observed_nodes, round_index, workers, max_depth, exhausted=()):
        _legal(observed_nodes, max_depth)
        prefix = copy.deepcopy(observed_nodes)
        exhausted_ids = set(exhausted)
        for node in prefix:
            if node["id"] in exhausted_ids:
                node["exhausted"] = True
        batch = self._program.call("select", args=[prefix, round_index, workers, max_depth])
        return _validate_batch(batch, observed_nodes, workers, max_depth)

    def plan_grid(self, context):
        """Explicit optional RPC; callers must validate and authorize any new caps.

        This is never called by §3 replay. Missing methods raise CodePolicyError.
        """
        return self._program.call("plan_grid", args=[copy.deepcopy(context)])

    @property
    def cpu_time_ns(self):
        return self._program.cpu_time_ns

    def close(self):
        self._program.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# Named alias for callers that distinguish executable from legacy controllers.
SourceController = CodeController


def _settings(workers, max_rounds, max_depth, beta_calls, beta_parallel):
    for name, value, minimum in (("workers", workers, 1), ("max_rounds", max_rounds, 0),
                                 ("max_depth", max_depth, 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for name, value in (("beta_calls", beta_calls), ("beta_parallel", beta_parallel)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")


def _children(tree):
    if not isinstance(tree, list) or not tree or tree[0].get("id") != 0 or tree[0].get("parent") is not None:
        raise ValueError("History must begin with root 0")
    if tree[0].get("depth") != 0:
        raise ValueError("History root depth must be zero")
    by_id, children = {}, {}
    for node in tree:
        node_id = node["id"]
        if type(node_id) is not int or node_id < 0 or node_id in by_id:
            raise ValueError("History must have unique nonnegative integer IDs")
        score = node["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("History scores must be finite")
        if type(node.get("valid")) is not bool:
            raise ValueError("History nodes require boolean valid")
        if node_id != 0:
            parent_id = node["parent"]
            if type(parent_id) is not int or parent_id not in by_id:
                raise ValueError("History must be in parent-before-child creation order")
            if node.get("depth") != by_id[parent_id]["depth"] + 1:
                raise ValueError("History depth must extend its parent by one")
            siblings = children.setdefault(parent_id, [])
            siblings.append(node)
            if parent_id != 0 and len(siblings) > 1:
                raise ValueError("Non-root nodes can have only one recorded child")
        by_id[node_id] = node
    return children


def _replay(spec, tree, *, workers, max_rounds, max_depth, beta_calls, beta_parallel):
    _settings(workers, max_rounds, max_depth, beta_calls, beta_parallel)
    children = _children(tree)
    visible = [copy.deepcopy(tree[0])]
    visible[0]["round"] = 0
    revealed_ids, exhausted, trajectory = {0}, set(), []
    with CodeController(spec) as controller:
        for round_index in range(1, max_rounds + 1):
            if len(visible) == len(tree):
                break
            actions = controller.select(visible, round_index, workers, max_depth, exhausted)
            if not actions:
                break
            observed_ids = [node["id"] for node in visible]
            by_id = {node["id"]: node for node in visible}
            new = []
            for action in actions:
                child = next((node for node in children.get(action, [])
                              if node["id"] not in revealed_ids), None)
                if child is None:
                    exhausted.add(action)
                    continue
                parent = by_id[action]
                copied = copy.deepcopy(child)
                # Preserve recorded outcome/diagnostics, recompute prefix-local age
                # and trajectory features for the counterfactual decision order.
                copied["round"] = round_index
                copied["stagnation"] = (0 if copied["valid"] and copied["score"] > parent["score"]
                                         else parent.get("stagnation", 0) + 1)
                copied["failure_streak"] = 0 if copied["valid"] else parent.get("failure_streak", 0) + 1
                new.append(copied)
                revealed_ids.add(copied["id"])
            visible.extend(new)
            trajectory.append({"round": round_index, "observed": observed_ids,
                               "selected": actions, "revealed": [node["id"] for node in new],
                               # Full immutable outcome/source records live in the
                               # corresponding historical world, referenced by ID.
                               # Store only features changed by this replay schedule.
                               "observations": [{key: node[key] for key in
                                                 ("id", "round", "stagnation", "failure_streak")}
                                                for node in new],
                               "best_score": max(node["score"] for node in visible)})
    attempts, rounds = len(visible) - 1, len(trajectory)
    best = max(node["score"] for node in visible)
    value = best - beta_calls * attempts + beta_parallel * attempts / max(1, rounds)
    if not math.isfinite(value):
        raise ValueError("Replay objective overflowed")
    return {"valid": True, "value": value, "best_score": best, "attempts": attempts,
            "rounds": rounds, "trajectory": trajectory, "exhausted": sorted(exhausted),
            "controller_cpu_time_ns": controller.cpu_time_ns}


def replay_controller_source(spec, tree, *, workers, max_rounds, max_depth, beta_calls, beta_parallel):
    """Replay Eq.1 without generating outcomes; failures become policy feedback."""
    try:
        return _replay(spec, tree, workers=workers, max_rounds=max_rounds, max_depth=max_depth,
                       beta_calls=beta_calls, beta_parallel=beta_parallel)
    except Exception as exc:
        return {"valid": False, "value": None, "error": f"{type(exc).__name__}: {exc}"}


def evaluate_controller_source(spec, histories, *, check_determinism=True, **kwargs):
    """A fresh stateful worker per world, with unweighted mean Eq.1 selection."""
    try:
        if not histories:
            raise ValueError("At least one historical world is required")
        replays = []
        for history in histories:
            result = replay_controller_source(spec, history, **kwargs)
            replays.append(result)
            if not result["valid"]:
                return {"valid": False, "value": None, "error": result["error"], "replays": replays}
            if check_determinism:
                repeated = replay_controller_source(spec, history, **kwargs)
                comparable = ("value", "trajectory", "exhausted")
                if not repeated["valid"] or any(result[key] != repeated[key] for key in comparable):
                    return {"valid": False, "value": None,
                            "error": "Controller is nondeterministic across fresh replay workers",
                            "replays": replays}
        mean = math.fsum(result["value"] / len(replays) for result in replays)
        return {"valid": True, "value": mean,
                "determinism_checked": check_determinism, "replays": replays}
    except Exception as exc:
        return {"valid": False, "value": None, "error": f"{type(exc).__name__}: {exc}"}
