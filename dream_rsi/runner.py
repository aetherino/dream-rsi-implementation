import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

from .backend import BudgetExceeded, MiMo, Mock, write_json
from .evaluation import isolated_evaluate, snapshot_suite
from .history import evaluate_controller, observation, root
from .programs import Controller, INITIAL_CONTROLLER
from .prompts import compact_node, controller_prompt, discovery_prompt


API_DEFAULTS = {"model": "mimo-v2.6-pro", "max_calls": 24, "max_usd": 0.5,
                "max_completion_tokens": 4096, "thinking": "disabled", "timeout_seconds": 120,
                "input_usd_per_million": 0.435, "output_usd_per_million": 0.87}
DEFAULTS = {"cycles": 2, "online_rounds": 4, "replay_rounds": 6, "workers": 2,
            "max_depth": 4, "controller_revisions": 2, "block_size": 16,
            "beta_calls": 0.02, "beta_parallel": 0.005, "evaluation_timeout_seconds": 120,
            "max_policy_us_per_request": 50000}


def prepare_config(raw, project_root):
    config = DEFAULTS | raw
    config["api"] = API_DEFAULTS | config.get("api", {})
    for name, upper in {"cycles": 20, "online_rounds": 100, "replay_rounds": 100,
                        "workers": 8, "max_depth": 100, "controller_revisions": 20, "block_size": 4096}.items():
        lower = 0 if name == "controller_revisions" else 1
        if type(config[name]) is not int or not lower <= config[name] <= upper:
            raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    for name in ("beta_calls", "beta_parallel", "evaluation_timeout_seconds", "max_policy_us_per_request"):
        value = config[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")
    if config["evaluation_timeout_seconds"] == 0 or config["max_policy_us_per_request"] == 0:
        raise ValueError("Evaluator limits must be positive")
    api = config["api"]
    if api["thinking"] not in ("enabled", "disabled"):
        raise ValueError("api.thinking must be enabled or disabled")
    for name in ("max_calls", "max_completion_tokens", "timeout_seconds"):
        if type(api[name]) is not int or api[name] < 1:
            raise ValueError(f"api.{name} must be a positive integer")
    for name in ("max_usd", "input_usd_per_million", "output_usd_per_million"):
        if type(api[name]) not in (int, float) or not math.isfinite(api[name]) or api[name] <= 0:
            raise ValueError(f"api.{name} must be positive and finite")
    if not isinstance(api["model"], str) or not api["model"]:
        raise ValueError("api.model must be a nonempty string")
    config["train"] = snapshot_suite(config["train"], project_root)
    config["validation"] = snapshot_suite(config["validation"], project_root) if config.get("validation") else []
    for split in ("train", "validation"):
        for item in config[split]:
            if "path" in item:
                parts = Path(item["path"]).parts
                if "test" in parts or (split == "train" and "validation" in parts) or (split == "validation" and "train" in parts):
                    raise ValueError(f"Incorrect dataset split in {split} suite")
    def identity(item):
        # Different session counts with the same RNG seed can still overlap.
        return item.get("sha256") or f"synthetic-seed:{item['synthetic'].get('seed', 42)}"
    if {identity(s) for s in config["train"]} & {identity(s) for s in config["validation"]}:
        raise ValueError("Training and validation traces must be disjoint")
    return config


def run(config, backend_name, project_root, output):
    config = prepare_config(config, project_root)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_json(output / "config.json", config | {"backend": backend_name})
    sources = [*Path(__file__).parent.glob("*.py"), *(project_root / "cache_sim").glob("*.py")]
    write_json(output / "source-hashes.json", {str(p.relative_to(project_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                              for p in sources})
    histories, controller = [], INITIAL_CONTROLLER.copy()
    best, backend, stop_reason = root(), None, "completed"

    def event(kind, **data):
        record = {"elapsed_seconds": round(time.monotonic() - started, 3), "event": kind, **data}
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record), flush=True)

    def evaluate(candidate=None, suite=None, baselines=None):
        return isolated_evaluate({"suite": config["train"] if suite is None else suite,
                                  "block_size": config["block_size"], "candidate": candidate,
                                  "baselines": baselines,
                                  "max_policy_us_per_request": config["max_policy_us_per_request"]},
                                 config["evaluation_timeout_seconds"])

    def generate(role, prompt, context):
        stem = f"{role}-c{context['cycle']:03d}-n{context.get('node', context.get('revision')):04d}"
        write_json(output / "prompts" / (stem + ".json"), {"role": role, "context": context, "prompt": prompt})
        return backend.generate(role, prompt, context)

    try:
        event("baselines_started", scenarios=len(config["train"]))
        baseline_report = evaluate()
        write_json(output / "baselines.json", baseline_report)
        if not baseline_report["valid"]:
            raise ValueError(f"Invalid evaluation suite: {baseline_report.get('error')}")
        baselines = baseline_report["baselines"]
        initial_feedback = evaluate(best["candidate"], baselines=baselines)
        write_json(output / "initial-policy-evaluation.json", initial_feedback)
        if not initial_feedback["valid"]:
            raise ValueError("Initial LRU expression fails evaluation; inspect initial-policy-evaluation.json")
        initial = root() | {"feedback": initial_feedback}
        best = initial
        backend = Mock(output / "usage.json") if backend_name == "mock" else MiMo(config["api"], project_root, output / "usage.json")
        suite_summary = [{"name": s["name"], "capacity_blocks": s["capacity_blocks"],
                          "lru_extra_tokens": r["policies"]["lru"]["extra_computed_tokens"],
                          "lru_policy_us_per_request": r["policies"]["lru"]["policy_us_per_request"]}
                         for s, r in zip(config["train"], baselines)]
        suite_summary.append({"max_policy_us_per_request": config["max_policy_us_per_request"]})
        replay_settings = {"workers": config["workers"], "max_rounds": config["replay_rounds"],
                           "max_depth": config["max_depth"], "beta_calls": config["beta_calls"],
                           "beta_parallel": config["beta_parallel"]}
        write_json(output / "controller-initial.json", controller)
        for cycle in range(1, config["cycles"] + 1):
            # Controller fixed for this rollout. Every rollout begins at the same initial LRU.
            tree = [initial.copy()]
            tree_path = output / f"tree-{cycle:03d}.json"
            write_json(tree_path, {"controller": controller, "nodes": tree})
            event("discovery_started", cycle=cycle, controller=controller["name"])
            for round_index in range(1, config["online_rounds"] + 1):
                actions = Controller(controller).select(tree, round_index, config["workers"], config["max_depth"])
                if not actions:
                    event("controller_stopped", cycle=cycle, round=round_index)
                    break
                by_id = {n["id"]: n for n in tree}
                jobs = []
                with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
                    for offset, parent_id in enumerate(actions):
                        node_id = len(tree) + offset
                        parent = by_id[parent_id]
                        prior_best = [max(history, key=lambda n: n["score"]) for history in histories[-4:]]
                        prompt = discovery_prompt(parent, tree, suite_summary, prior_best)
                        jobs.append((node_id, parent, pool.submit(generate, "discovery", prompt,
                                                                  {"cycle": cycle, "node": node_id})))
                    # Generation overlaps. Evaluations are serialized for less noisy CPU timing.
                    proposals = []
                    for node_id, parent, future in jobs:
                        try:
                            proposals.append((node_id, parent, future.result(), None))
                        except BudgetExceeded:
                            stop_reason = "api_budget_reached"
                        except Exception as exc:
                            proposals.append((node_id, parent, None, {"valid": False, "score": -1e6,
                                                                      "error": f"{type(exc).__name__}: {exc}"}))
                for node_id, parent, candidate, feedback in proposals:
                    if feedback is None:
                        feedback = evaluate(candidate, baselines=baselines)
                    node = observation(node_id, parent, round_index, candidate, feedback)
                    tree.append(node)
                    write_json(tree_path, {"controller": controller, "nodes": tree})
                    write_json(output / "candidates" / f"c{cycle:03d}-n{node_id:04d}.json", node)
                    if node["valid"] and node["score"] > best["score"]:
                        best = node | {"cycle": cycle}
                    event("candidate_evaluated", cycle=cycle, round=round_index, node=node_id,
                          parent=parent["id"], valid=node["valid"], score=node["score"],
                          error=feedback.get("error"))
                if stop_reason != "completed":
                    break
            histories.append(tree)
            write_json(output / "best-policy.json", best)
            if stop_reason != "completed":
                break
            incumbent_feedback = evaluate_controller(controller, histories, **replay_settings)
            if not incumbent_feedback["valid"]:
                raise ValueError("Incumbent controller failed historical replay")
            revision_log = [{"revision": 0, "controller": controller, "evaluation": incumbent_feedback, "accepted": True}]
            revision_path = output / f"controller-revisions-{cycle:03d}.json"
            write_json(revision_path, revision_log)
            for revision in range(1, config["controller_revisions"] + 1):
                prompt = controller_prompt(controller, incumbent_feedback,
                                           [[compact_node(n) for n in history] for history in histories], replay_settings, revision_log)
                proposal = None
                try:
                    proposal = generate("controller", prompt, {"cycle": cycle, "revision": revision})
                    feedback = evaluate_controller(proposal, histories, **replay_settings)
                except BudgetExceeded:
                    stop_reason = "api_budget_reached"
                    break
                except Exception as exc:
                    feedback = {"valid": False, "value": None, "error": f"{type(exc).__name__}: {exc}"}
                accepted = feedback["valid"] and feedback["value"] > incumbent_feedback["value"] + 1e-12
                revision_log.append({"revision": revision, "controller": proposal, "evaluation": feedback,
                                     "accepted": accepted})
                if accepted:
                    controller, incumbent_feedback = proposal, feedback
                write_json(revision_path, revision_log)
                event("controller_evaluated", cycle=cycle, revision=revision, accepted=accepted,
                      replay_value=feedback["value"])
            write_json(output / "controller-final.json", controller)
            if stop_reason != "completed":
                break
        validation = None
        # One final held-out check. No validation feedback is ever sent to the model.
        if config["validation"]:
            event("validation_started")
            validation_baselines = evaluate(suite=config["validation"])
            write_json(output / "validation-baselines.json", validation_baselines)
            validation = (evaluate(best["candidate"], suite=config["validation"],
                                   baselines=validation_baselines["baselines"])
                          if validation_baselines["valid"] else validation_baselines)
            write_json(output / "validation.json", validation)
        summary = {"status": stop_reason, "backend": backend_name, "cycles_recorded": len(histories),
                   "task_attempts": sum(len(h) - 1 for h in histories), "best_train_score": best["score"],
                   "valid_task_attempts": sum(n["valid"] for h in histories for n in h[1:]),
                   "best_policy": best["candidate"], "controller": controller,
                   "validation_score": validation.get("score") if validation else None,
                   "validation_valid": validation["valid"] if validation else None,
                   "usage": backend.summary(), "elapsed_seconds": time.monotonic() - started}
        write_json(output / "controller-final.json", controller)
        write_json(output / "summary.json", summary)
        event("run_finished", status=stop_reason, best_train_score=best["score"],
              estimated_usd=summary["usage"]["estimated_usd"])
        return summary
    except BaseException as exc:
        write_json(output / "summary.json", {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                             "error": f"{type(exc).__name__}: {exc}", "best_policy": best,
                                             "usage": backend.summary() if backend else None})
        raise
