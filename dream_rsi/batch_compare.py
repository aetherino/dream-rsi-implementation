"""Resumable fixed/adaptive comparison with provider batches or concurrent realtime requests."""
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import time

from .backend import Mock, write_json
from .batch_transport import BatchTransport, allowance, cost, request_body
from .compare import aggregate, arm_config, markdown_report, summarize_run
from .evaluation import isolated_evaluate
from .history import evaluate_controller, observation, root
from .programs import Controller, INITIAL_CONTROLLER
from .prompts import controller_prompt, discovery_prompt, tried_memory
from .runner import prepare_config
from .realtime_transport import RealtimeTransport


def source_hashes(project_root):
    return {str(p.relative_to(project_root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for package in ("dream_rsi", "cache_sim") for p in sorted((project_root / package).glob("*.py"))}


def evaluate(config, candidate=None, validation=False, baselines=None):
    return isolated_evaluate({"suite": config["validation" if validation else "train"],
                              "block_size": config["block_size"], "candidate": candidate,
                              "baselines": baselines, "scoring": config["scoring"],
                              "max_policy_us_per_request": config["max_policy_us_per_request"]},
                             config["evaluation_timeout_seconds"])


def lean_node(node):
    result = copy.deepcopy(node)
    if isinstance(result.get("candidate"), dict):
        result["candidate"].pop("rationale", None)
    return result


def bounded_discovery(parent, tree, suite, histories, scoring, prompt_version="v1", best=None):
    # Keep the parent and suite; discard only old auxiliary observations as needed.
    visible = [lean_node(n) for n in tree[-12:]]
    previous = [lean_node(max(h, key=lambda n: n["score"])) for h in histories[-4:]]
    if best is None:
        best = max([n for h in histories for n in h] + tree, key=lambda n: n["score"])
    memory = tried_memory(histories, tree) if prompt_version == "v2" else None
    while True:
        prompt = discovery_prompt(lean_node(parent), visible, suite, previous, scoring, version=prompt_version,
                                  best=lean_node(best), memory=memory)
        if len(prompt.encode()) < 54000:
            return prompt
        if len(visible) > 1:
            visible.pop(0)
        elif previous:
            previous.pop(0)
        else:
            raise ValueError("Selected parent and suite exceed prompt budget")


def controller_history(histories):
    # Meta search needs every outcome and causal feature, not 12 cache metric rows per node.
    return [[{k: lean_node(n)[k] for k in ("id", "parent", "depth", "round", "score", "valid",
                                         "stagnation", "failure_streak", "candidate")} for n in h] for h in histories]


def usage(run, backend):
    return {"backend": backend, "model": run["config"]["api"]["model"],
            "thinking": run["config"]["api"]["thinking"], "calls": len(run["records"]),
            "estimated_usd": sum(r["charged_estimate_usd"] for r in run["records"]),
            "cost_basis": "Configured uncached rates for the selected transport; pending/unknown results retain reservation",
            "records": run["records"]}


def reserve(run, role, prompt, context, parent=None):
    api = run["config"]["api"]
    input_tokens = allowance(prompt)
    amount = cost(api, input_tokens, api["max_completion_tokens"])
    spent = sum(r["charged_estimate_usd"] for r in run["records"])
    if len(run["records"]) >= api["max_calls"] or spent + amount > api["max_usd"]:
        run["stop_reason"] = "api_budget_reached"
        return None
    custom_id = f"{run['id']}-call-{len(run['records']) + 1:04d}"
    record = {"call": len(run["records"]) + 1, "id": custom_id, "role": role, "context": context,
              "prompt_version": run["config"].get("prompt_version", "v1"),
              "status": "reserved", "charged_estimate_usd": amount, "input_token_allowance": input_tokens}
    run["records"].append(record)
    return {"id": custom_id, "role": role, "context": context, "parent": parent,
            "prompt": prompt, "body": request_body(api, prompt)}


def apply_results(run, results):
    config = run["config"]
    for job in run["pending"]:
        outcome = results[job["id"]]
        record = next(r for r in run["records"] if r["id"] == job["id"])
        tokens = outcome.get("usage", {})
        if all(type(tokens.get(k)) is int and tokens[k] >= 0 for k in ("prompt_tokens", "completion_tokens")):
            record.update(status="accounted", prompt_tokens=tokens["prompt_tokens"], completion_tokens=tokens["completion_tokens"],
                          charged_estimate_usd=cost(config["api"], tokens["prompt_tokens"], tokens["completion_tokens"]))
        elif outcome.get("explicit_failure"):
            record.update(status="provider_failed_not_billed", charged_estimate_usd=0)
        else:
            record["status"] = "usage_unknown_reservation_retained"
        record["finish_reason"] = outcome.get("finish_reason")
        if outcome.get("error"):
            record["error"] = outcome["error"]
        candidate = outcome["program"]
        if job["role"] == "discovery":
            parent = next(n for n in run["tree"] if n["id"] == job["parent"])
            feedback = ({"valid": False, "score": -1e6, "error": outcome["error"]} if outcome["error"] else
                        evaluate(config, candidate=candidate, baselines=run["baselines"]["baselines"]))
            node = observation(job["context"]["node"], parent, run["round"], candidate, feedback)
            node["api_call"] = record["call"]
            run["tree"].append(node)
            if node["valid"] and node["score"] > run["best"]["score"]:
                run["best"] = node | {"cycle": run["cycle"]}
            run["progress"].append({"call": record["call"], "cycle": run["cycle"], "round": run["round"],
                                    "node": node["id"], "score": node["score"], "valid": node["valid"],
                                    "best_train_score": run["best"]["score"]})
        else:
            feedback = ({"valid": False, "value": None, "error": outcome["error"]} if outcome["error"] else
                        evaluate_controller(candidate, run["histories"], **run["replay_settings"]))
            accepted = feedback["valid"] and feedback["value"] > run["incumbent_feedback"]["value"] + 1e-12
            run["revision_log"].append({"revision": run["revision"], "controller": candidate,
                                        "evaluation": feedback, "accepted": accepted, "api_call": record["call"]})
            if accepted:
                run["controller"], run["incumbent_feedback"] = candidate, feedback
    role = run["pending"][0]["role"]
    run["pending"] = []
    run["round" if role == "discovery" else "revision"] += 1


def advance(run):
    """Perform local work until this trial needs model results or finishes."""
    config = run["config"]
    while not run["pending"] and run["phase"] != "done":
        phase = run["phase"]
        if phase == "initialize":
            run["baselines"] = evaluate(config)
            if not run["baselines"]["valid"]:
                raise ValueError("Invalid training suite")
            initial_feedback = evaluate(config, root()["candidate"], baselines=run["baselines"]["baselines"])
            if not initial_feedback["valid"]:
                raise ValueError("Initial LRU expression failed evaluation")
            run["initial"] = root() | {"feedback": initial_feedback}
            run["best"] = run["initial"]
            run["suite_summary"] = [{"name": s["name"], "capacity_blocks": s["capacity_blocks"],
                                      "lru_extra_tokens": b["policies"]["lru"]["extra_computed_tokens"]}
                                     for s, b in zip(config["train"], run["baselines"]["baselines"])]
            run["suite_summary"].append({"max_policy_us_per_request": config["max_policy_us_per_request"]})
            run["phase"] = "start_cycle"
        elif phase == "start_cycle":
            if run["cycle"] > config["cycles"] or run["stop_reason"] != "completed":
                run["phase"] = "validation"
                continue
            run.update(tree=[copy.deepcopy(run["initial"])], round=1, phase="discovery",
                       rollout_controller=copy.deepcopy(run["controller"]))
        elif phase == "discovery":
            remaining = config["api"]["max_calls"] - len(run["records"])
            if remaining <= 0:
                run["stop_reason"] = "api_budget_reached"
            actions = (Controller(run["controller"]).select(run["tree"], run["round"], config["workers"], config["max_depth"])
                       if run["round"] <= config["online_rounds"] and run["stop_reason"] == "completed" else [])
            if not actions:
                run["histories"].append(copy.deepcopy(run["tree"]))
                run["rollouts"].append({"controller": run["rollout_controller"], "nodes": copy.deepcopy(run["tree"])})
                if len(run["tree"]) == 1 and run["stop_reason"] == "completed":
                    run["stop_reason"] = "controller_stopped_run"
                if run["stop_reason"] != "completed":
                    run["phase"] = "validation"
                    continue
                feedback = evaluate_controller(run["controller"], run["histories"], **run["replay_settings"])
                if not feedback["valid"]:
                    raise ValueError("Incumbent failed historical replay")
                run.update(incumbent_feedback=feedback, revision=1, phase="controller",
                           revision_log=[{"revision": 0, "controller": run["controller"], "evaluation": feedback, "accepted": True}])
                continue
            for offset, parent_id in enumerate(actions[:remaining]):
                parent = next(n for n in run["tree"] if n["id"] == parent_id)
                prompt = bounded_discovery(parent, run["tree"], run["suite_summary"], run["histories"], config["scoring"],
                                           config.get("prompt_version", "v1"), run["best"])
                job = reserve(run, "discovery", prompt, {"cycle": run["cycle"], "node": len(run["tree"]) + offset}, parent_id)
                if job is None:
                    break
                run["pending"].append(job)
        elif phase == "controller":
            remaining = config["api"]["max_calls"] - len(run["records"])
            if (run["revision"] > config["controller_revisions"] or run["cycle"] == config["cycles"]
                    or remaining < 2 or run["stop_reason"] != "completed"):
                run["revisions"].append(copy.deepcopy(run["revision_log"]))
                run["cycle"] += 1
                run["phase"] = "start_cycle"
                continue
            prompt = controller_prompt(run["controller"], run["incumbent_feedback"], controller_history(run["histories"]),
                                       run["replay_settings"], run["revision_log"], config["scoring"],
                                       version=config.get("prompt_version", "v1"), online_rounds=config["online_rounds"])
            job = reserve(run, "controller", prompt, {"cycle": run["cycle"], "revision": run["revision"]})
            if job:
                run["pending"].append(job)
        elif phase == "validation":
            run["validation_baselines"] = evaluate(config, validation=True)
            if not run["validation_baselines"]["valid"]:
                raise ValueError("Invalid held-out suite")
            run["validation"] = evaluate(config, run["best"]["candidate"], validation=True,
                                         baselines=run["validation_baselines"]["baselines"])
            if not run["validation"]["valid"]:
                raise ValueError("Training winner failed held-out evaluation")
            run.update(phase="done", finished_at=time.time())
        else:
            raise ValueError("Unknown coordinator phase")


def save_artifacts(state, output, make_report=True):
    rows = []
    for run in state["runs"]:
        folder = output / run["id"]
        folder.mkdir(exist_ok=True)
        write_json(folder / "config.json", run["config"] | {"backend": state["backend"]})
        write_json(folder / "source-hashes.json", state["source_hashes"])
        write_json(folder / "usage.json", usage(run, state["backend"]))
        write_json(folder / "progress.json", run["progress"])
        write_json(folder / "controller-initial.json", run.get("initial_controller", INITIAL_CONTROLLER))
        write_json(folder / "controller-final.json", run["controller"])
        if "best" in run:
            write_json(folder / "best-policy.json", run["best"])
        if "baselines" in run:
            write_json(folder / "baselines.json", run["baselines"])
            write_json(folder / "initial-policy-evaluation.json", run["initial"]["feedback"])
        rollouts = run["rollouts"][:]
        if run["phase"] == "discovery":
            rollouts.append({"controller": run["rollout_controller"], "nodes": run["tree"]})
        for cycle, tree in enumerate(rollouts, 1):
            write_json(folder / f"tree-{cycle:03d}.json", tree)
        revisions = run["revisions"][:]
        if run["phase"] == "controller":
            revisions.append(run["revision_log"])
        for cycle, log in enumerate(revisions, 1):
            write_json(folder / f"controller-revisions-{cycle:03d}.json", log)
        for job in run["pending"]:
            write_json(folder / "prompts" / (job["id"] + ".json"), {k: job[k] for k in ("role", "context", "prompt")})
        if run["phase"] == "done":
            write_json(folder / "validation-baselines.json", run["validation_baselines"])
            write_json(folder / "validation.json", run["validation"])
            write_json(folder / "summary.json", {"status": run["stop_reason"], "backend": state["backend"],
                        "scoring": run["config"]["scoring"], "cycles_recorded": len(run["histories"]),
                        "task_attempts": sum(len(h) - 1 for h in run["histories"]),
                        "valid_task_attempts": sum(n["valid"] for h in run["histories"] for n in h[1:]),
                        "best_train_score": run["best"]["score"], "best_policy": run["best"]["candidate"],
                        "controller": run["controller"], "validation_valid": run["validation"]["valid"],
                        "validation_score": run["validation"]["score"], "usage": usage(run, state["backend"]),
                        "elapsed_seconds": run["finished_at"] - run["started_at"]})
            rows.append(summarize_run(folder, run["trial"], run["arm"]))
    if not make_report:
        return rows
    report = {"status": state["status"], "backend": state["backend"], "plan": state["plan"],
              "runs": rows, "aggregate": aggregate(rows), "error": state.get("error")}
    write_json(output / "comparison.json", report)
    text = markdown_report(report).replace("controller pilot", "controller longer experiment")
    scheduling = ("Trials advance together using concurrent regular API requests." if state["backend"] == "mimo"
                  else "Trials advance together; ready requests share provider batches.")
    text = text.replace("The run order alternates by pair.", scheduling)
    text += f"\nRequest waves submitted/planned: {state['wave']}. Pending requests: {sum(len(r['pending']) for r in state['runs'])}.\n"
    text += f"Current estimated cost including pending reservations: ${sum(usage(r, state['backend'])['estimated_usd'] for r in state['runs']):.6f}.\n"
    (output / "report.md").write_text(text)
    return report


def initialize(raw, backend, project_root):
    trials = raw.get("trials", 3)
    if type(trials) is not int or not 1 <= trials <= 10:
        raise ValueError("trials must be an integer in [1,10]")
    base = prepare_config(raw["run"], project_root)
    if not base["validation"] or base["controller_revisions"] < 1 or base["cycles"] < 2:
        raise ValueError("Comparison needs held-out validation and multiple controller cycles")
    plan = {"trials": trials, "backend": backend, "base_config": base,
            "estimated_usd_ceiling": 2 * trials * base["api"]["max_usd"],
            "total_call_ceiling": 2 * trials * base["api"]["max_calls"],
            "scheduling": ("Independent trials; up to four concurrent regular API requests" if backend == "mimo"
                           else "All trials advance independently; ready requests share a provider batch")}
    runs = []
    for trial in range(1, trials + 1):
        for arm in (("fixed", "adaptive") if trial % 2 else ("adaptive", "fixed")):
            config = arm_config(base, arm)
            runs.append({"id": f"trial-{trial:02d}-{arm}", "trial": trial, "arm": arm, "config": config,
                         "phase": "initialize", "cycle": 1, "records": [], "pending": [], "progress": [],
                         "histories": [], "rollouts": [], "revisions": [], "controller": INITIAL_CONTROLLER.copy(),
                         "stop_reason": "completed", "started_at": time.time(),
                         "replay_settings": {"workers": config["workers"], "max_rounds": config["replay_rounds"],
                                             "max_depth": config["max_depth"], "beta_calls": config["beta_calls"],
                                             "beta_parallel": config["beta_parallel"]}})
    return {"version": 1, "backend": backend, "plan": plan, "source_hashes": source_hashes(project_root),
            "base_url": raw.get("batch_base_url"), "wave": 0, "active_wave": False,
            "status": "prepared", "runs": runs}


def tick(state, output, project_root, transport=None, attach_batch=None, reporter=None):
    reporter = reporter or save_artifacts
    checkpoint = output / "checkpoint.json"
    if state["active_wave"]:
        jobs = [j for r in state["runs"] for j in r["pending"]]
        if state["backend"] == "mock":
            mock = Mock(output / "mock-transport.json")
            results = {j["id"]: {"program": mock.generate(j["role"], j["prompt"], j["context"]), "error": None,
                                   "usage": {"prompt_tokens": 0, "completion_tokens": 0}, "explicit_failure": False} for j in jobs}
        else:
            transport = transport or (RealtimeTransport(project_root, output / "requests") if state["backend"] == "mimo"
                                      else BatchTransport(state["base_url"], project_root, output / "batches"))
            results = transport.poll(state["wave"], jobs, attach_batch=attach_batch)
        if results is None:
            state["status"] = "waiting_for_batch"
            write_json(checkpoint, state)
            return reporter(state, output)
        updated = copy.deepcopy(state)
        for run in updated["runs"]:
            if run["pending"]:
                apply_results(run, results)
        updated["active_wave"] = False
        write_json(checkpoint, updated)
        state.clear()
        state.update(updated)
    for run in state["runs"]:
        advance(run)
        write_json(checkpoint, state)
    jobs = [j for r in state["runs"] for j in r["pending"]]
    if jobs:
        state.update(wave=state["wave"] + 1, active_wave=True, status="ready_to_submit")
    else:
        state["status"] = "completed"
    write_json(checkpoint, state)
    return reporter(state, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dream-comparison-batch.json"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", choices=["mock", "mimo", "mimo-batch"], default="mock")
    parser.add_argument("--base-url", help="Account-specific Batch API Base URL from the MiMo console")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Freeze suites and prepare first wave without API calls")
    parser.add_argument("--watch", action="store_true", help="Continue polling and advancing until complete")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--attach-batch", help="Recover an ambiguous submission with its verified provider batch ID")
    args = parser.parse_args()
    if not 10 <= args.poll_seconds <= 60:
        parser.error("--poll-seconds must be in [10,60]")
    project_root = Path(__file__).resolve().parent.parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    with (output / "coordinator.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(2, "Another coordinator owns this experiment\n")
        checkpoint = output / "checkpoint.json"
        state = None
        try:
            if args.resume:
                state = json.loads(checkpoint.read_text())
                if state["source_hashes"] != source_hashes(project_root):
                    raise ValueError("Source changed since experiment was frozen; restore the recorded source before resuming")
                prepare_config(state["plan"]["base_config"], project_root)  # Recheck frozen trace hashes.
            else:
                state = initialize(json.loads(args.config.read_text()), args.backend, project_root)
                write_json(output / "plan.json", state["plan"])
                for relative in state["source_hashes"]:
                    saved = output / "source" / relative
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    saved.write_bytes((project_root / relative).read_bytes())
            if args.base_url:
                if state.get("base_url") and state["base_url"] != args.base_url:
                    raise ValueError("Cannot change the endpoint of an existing experiment")
                state["base_url"] = args.base_url
            state.pop("error", None)
            write_json(checkpoint, state)
            while state["status"] != "completed":
                if args.prepare_only and state["active_wave"]:
                    break
                tick(state, output, project_root, attach_batch=args.attach_batch)
                args.attach_batch = None
                print(json.dumps({"status": state["status"], "wave": state["wave"],
                                  "calls": sum(len(r["records"]) for r in state["runs"]),
                                  "finished_runs": sum(r["phase"] == "done" for r in state["runs"])}), flush=True)
                if state["status"] == "waiting_for_batch":
                    if not args.watch:
                        break
                    time.sleep(min(args.poll_seconds, 60))
                elif not args.watch and state["status"] != "ready_to_submit":
                    break
            save_artifacts(state, output)
        except Exception as exc:
            if state is not None:
                state.update(status="needs_attention", error=f"{type(exc).__name__}: {exc}")
                write_json(checkpoint, state)
                save_artifacts(state, output)
            parser.exit(2, f"error: {exc}\n")
    print(f"Report: {output / 'report.md'}")


if __name__ == "__main__":
    main()
