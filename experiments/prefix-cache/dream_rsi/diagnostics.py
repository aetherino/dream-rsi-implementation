"""Continue fixed searches and prospectively compare every accepted controller revision."""
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import statistics
import time

from . import batch_compare as search
from .backend import write_json
from .history import evaluate_controller
from .programs import Controller
from .runner import prepare_config


def transitions(source):
    found = []
    for run in source["runs"]:
        for cycle, log in enumerate(run["revisions"], 1):
            predecessor = log[0]["controller"]
            old_value = log[0]["evaluation"]["value"]
            for revision in log[1:]:
                if revision["accepted"]:
                    Controller(predecessor)
                    Controller(revision["controller"])
                    found.append({"id": f"transition-{len(found) + 1:02d}", "source_run": run["id"], "cycle": cycle,
                                  "call": revision["api_call"], "predecessor": copy.deepcopy(predecessor),
                                  "revised": copy.deepcopy(revision["controller"]),
                                  "historical_replay_gain": revision["evaluation"]["value"] - old_value})
                    predecessor, old_value = revision["controller"], revision["evaluation"]["value"]
    return found


def make_state(source, raw, project_root):
    if source["status"] != "completed" or source["active_wave"]:
        raise ValueError("Diagnostics require a completed source experiment")
    if source["backend"] not in ("mimo", "mock"):
        raise ValueError("This diagnostic coordinator supports realtime or mock transport")
    changed = [name for name, digest in source["source_hashes"].items()
               if name != "dream_rsi/batch_compare.py" and
               hashlib.sha256((project_root / name).read_bytes()).hexdigest() != digest]
    if changed:
        raise ValueError(f"Source experiment semantics changed: {changed}")
    revisions = transitions(source)
    if not revisions:
        raise ValueError("No accepted controller revisions to test")
    base = copy.deepcopy(source["plan"]["base_config"])
    base["validation"] = raw["validation"]
    base = prepare_config(base, project_root)
    def identity(item):
        return item.get("sha256") or f"synthetic-seed:{item['synthetic'].get('seed', 42)}"
    old_validation = {identity(x) for x in source["plan"]["base_config"]["validation"]}
    if old_validation & {identity(x) for x in base["validation"]}:
        raise ValueError("Use fresh validation shards for the diagnostic study")
    runs = []
    for prior in source["runs"]:
        if prior["arm"] != "fixed":
            continue
        run = copy.deepcopy(prior)
        prior_cost = sum(r["charged_estimate_usd"] for r in run["records"])
        if len(run["records"]) >= raw["extension_total_calls"]:
            raise ValueError("Extension must increase the call limit")
        # The prior budget was reached after a complete discovery cycle. Refuse
        # silently dropping a partial cycle if a different source is supplied.
        if run["round"] <= run["config"]["online_rounds"]:
            raise ValueError("Extension source ended mid-cycle; explicit continuation handling required")
        config = copy.deepcopy(run["config"])
        config.update(cycles=20, validation=base["validation"])
        config["api"].update(max_calls=raw["extension_total_calls"], max_usd=prior_cost + raw["extension_new_usd"])
        run["config"] = prepare_config(config, project_root)
        run.update(id=f"extend-{prior['id']}", phase="start_cycle", cycle=len(run["histories"]) + 1,
                   stop_reason="completed", pending=[], started_at=time.time(),
                   study="extension", imported_calls=len(run["records"]), imported_usd=prior_cost,
                   source_run=prior["id"], reference_policy=copy.deepcopy(run["best"]["candidate"]),
                   initial_controller=copy.deepcopy(run["controller"]))
        for key in ("validation", "validation_baselines", "finished_at"):
            run.pop(key, None)
        # Complete the last cycle's zero-call controller log for artifact numbering.
        while len(run["revisions"]) < len(run["histories"]):
            end = len(run["revisions"]) + 1
            run["revisions"].append([{"revision": 0, "controller": run["controller"], "accepted": True,
                                      "evaluation": evaluate_controller(run["controller"], run["histories"][:end], **run["replay_settings"])}])
        runs.append(run)
    diagnostic_base = copy.deepcopy(base)
    diagnostic_base.update(cycles=10)
    diagnostic_base["api"].update(max_calls=raw["diagnostic_calls"], max_usd=raw["diagnostic_usd"])
    template = search.initialize({"trials": 1, "run": diagnostic_base}, source["backend"], project_root)["runs"][0]
    for transition in revisions:
        for trial in range(1, raw["replicates"] + 1):
            for arm in (("predecessor", "revised") if trial % 2 else ("revised", "predecessor")):
                run = copy.deepcopy(template)
                run.update(id=f"{transition['id']}-trial-{trial:02d}-{arm}", trial=trial, arm=arm,
                           controller=copy.deepcopy(transition[arm]), initial_controller=copy.deepcopy(transition[arm]),
                           study="controller-transfer", transition=transition["id"], imported_calls=0, imported_usd=0,
                           started_at=time.time())
                runs.append(run)
    ceilings = {"new_calls": sum(r["config"]["api"]["max_calls"] - r["imported_calls"] for r in runs),
                "new_usd": sum(r["config"]["api"]["max_usd"] - r["imported_usd"] for r in runs)}
    if ceilings["new_usd"] > raw["max_new_usd"] + 1e-9:
        raise ValueError("Per-run budgets exceed the study spending ceiling")
    return {"version": 1, "backend": source["backend"], "status": "prepared", "base_url": None,
            "wave": 0, "active_wave": False, "source_hashes": search.source_hashes(project_root), "runs": runs,
            "plan": {"kind": "fixed-extension-and-controller-transfer", "base_config": base,
                     "source_checkpoint_sha256": raw["source_checkpoint_sha256"], "transitions": revisions,
                     "settings": raw, "ceilings": ceilings, "prompt_version": "unchanged from source experiment"}}


def make_prompt_continuation(source, project_root, source_digest):
    """Explicit version transition; retain all paid attempts and all existing ceilings."""
    if source["active_wave"] or any(r["pending"] for r in source["runs"]):
        raise ValueError("Settle pending requests before changing prompt versions")
    if any("validation" in r or r["phase"] in ("validation", "done") for r in source["runs"]):
        raise ValueError("This continuation requires unobserved validation; completed studies need a fresh suite")
    permitted = {"dream_rsi/prompts.py", "dream_rsi/runner.py", "dream_rsi/batch_compare.py", "dream_rsi/diagnostics.py"}
    for name, digest in source["source_hashes"].items():
        if name not in permitted and hashlib.sha256((project_root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Non-prompt semantics changed: {name}")
    state = copy.deepcopy(source)
    boundaries = []
    for run in state["runs"]:
        for record in run["records"]:
            record.setdefault("prompt_version", run["config"].get("prompt_version", "v1"))
        boundaries.append({"run": run["id"], "first_v2_call": len(run["records"]) + 1,
                           "prior_calls": len(run["records"]), "prior_train_score": run["best"]["score"]})
        run["config"]["prompt_version"] = "v2"
        prepare_config(run["config"], project_root)  # Revalidate hashes and configuration without resetting state.
    spent = sum(search.usage(r, state["backend"])["estimated_usd"] - r["imported_usd"] for r in state["runs"])
    used_calls = sum(len(r["records"]) - r["imported_calls"] for r in state["runs"])
    state["plan"].update(kind="prompt-v2-continuation", prompt_version="v2 following inherited v1 history",
                         prompt_revision={"source_checkpoint_sha256": source_digest, "boundaries": boundaries,
                                          "previous_new_calls": used_calls, "previous_new_usd": spent,
                                          "remaining_call_allowance": state["plan"]["ceilings"]["new_calls"] - used_calls,
                                          "remaining_usd_allowance": state["plan"]["ceilings"]["new_usd"] - spent})
    state["plan"]["base_config"]["prompt_version"] = "v2"
    state.update(status="prepared", source_hashes=search.source_hashes(project_root))
    state.pop("error", None)
    return state


def report(state, output):
    rows = search.save_artifacts(state, output, make_report=False)
    by_id = {r["id"]: r for r in state["runs"]}
    for row in rows:
        run = by_id[Path(row["path"]).name]
        row.update(study=run["study"], transition=run.get("transition"),
                   new_calls=row["calls"] - run["imported_calls"], new_usd=row["estimated_usd"] - run["imported_usd"])
        if run["study"] == "extension":
            reference_file = output / run["id"] / "reference-policy-validation.json"
            if reference_file.exists():
                reference = json.loads(reference_file.read_text())
            else:
                reference = search.evaluate(run["config"], run["reference_policy"], validation=True,
                                            baselines=run["validation_baselines"]["baselines"])
                write_json(reference_file, reference)
            if not reference["valid"]:
                raise ValueError("The 40-call reference policy failed fresh validation")
            row.update(reference_validation_score=reference["score"],
                       extension_validation_gain=row["validation_score"] - reference["score"])
    paired = []
    for transition in state["plan"]["transitions"]:
        for trial in range(1, state["plan"]["settings"]["replicates"] + 1):
            arms = {r["arm"]: r for r in rows if r.get("transition") == transition["id"] and r["trial"] == trial}
            if len(arms) == 2:
                paired.append({"transition": transition["id"], "trial": trial,
                               "validation_gain": arms["revised"]["validation_score"] - arms["predecessor"]["validation_score"],
                               "train_gain": arms["revised"]["train_score"] - arms["predecessor"]["train_score"],
                               "predecessor_calls": arms["predecessor"]["calls"], "revised_calls": arms["revised"]["calls"]})
    new_cost = sum(search.usage(r, state["backend"])["estimated_usd"] - r["imported_usd"] for r in state["runs"])
    result = {"status": state["status"], "error": state.get("error"), "plan": state["plan"], "runs": rows,
              "paired_controller_results": paired, "new_estimated_usd_including_reservations": new_cost,
              "new_calls_reserved": sum(len(r["records"]) - r["imported_calls"] for r in state["runs"])}
    write_json(output / "comparison.json", result)
    revised_prompts = "prompt_revision" in state["plan"]
    title = "Prompt v2 search continuation" if revised_prompts else "Fixed search extension and controller transfer"
    lines = ["# " + title, "",
             f"Status: **{state['status']}**. Completed runs: {len(rows)}/{len(state['runs'])}.",
             f"New requests reserved: {result['new_calls_reserved']}/{state['plan']['ceilings']['new_calls']}. "
             f"New estimated spending including pending reservations: ${new_cost:.4f} / ${state['plan']['ceilings']['new_usd']:.2f}.", "",
             "All scores below are reductions in total extra recomputation versus LRU. Higher is better.", "",
             "## Fixed search: 40 to 100 total calls", "",
             "Both policies are evaluated on the same fresh validation shards after search finishes.", "",
             "| Trial | 40-call policy | Extended policy | Change | New calls |",
             "| --- | ---: | ---: | ---: | ---: |"]
    if revised_prompts:
        lines[2:2] = ["**Prompts changed partway through this search.** Earlier v1 histories, controllers, attempts, and spending are retained. "
                      "This is a continuation, not a controlled prompt A/B test; final arm differences cannot isolate the prompt effect. "
                      "Per-run first-v2 call numbers are recorded in plan.json.", ""]
    for r in rows:
        if r["study"] == "extension":
            lines.append(f"| {r['trial']} | {r['reference_validation_score']:.2%} | {r['validation_score']:.2%} | {r['extension_validation_gain']:+.2%} | {r['new_calls']} |")
    lines += ["", "## Accepted controllers versus their predecessors", "",
              "Every accepted revision is tested; controllers stay frozen. Both arms start from LRU with empty discovery histories.", "",
              "| Transition | Replicate | Revised − predecessor, validation | Predecessor calls | Revised calls |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for r in paired:
        lines.append(f"| {r['transition']} | {r['trial']} | {r['validation_gain']:+.2%} | {r['predecessor_calls']} | {r['revised_calls']} |")
    for transition in state["plan"]["transitions"]:
        values = [r["validation_gain"] for r in paired if r["transition"] == transition["id"]]
        if values:
            lines.append(f"\n{transition['id']}: mean paired validation difference {statistics.mean(values):+.2%} ({len(values)} completed pairs).")
    lines += ["", "Equal call and dollar ceilings do not guarantee identical realized spend or sampled proposals. Two replicates per transition are a diagnostic, not a significance test.",
              ("Only prompts/context changed at the recorded boundary. " if revised_prompts else "Prompts are unchanged. ") +
              "Model, training suite, evaluator, and online round limits are unchanged. Validation shards 13–18 never appear in model prompts. The original experiment is preserved.", ""]
    (output / "report.md").write_text("\n".join(lines))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/dream-diagnostics.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--revise-prompts", action="store_true", help="Continue a settled, unvalidated diagnostic checkpoint with prompt v2")
    args = parser.parse_args()
    if args.resume and args.revise_prompts:
        parser.error("Use --revise-prompts for a new continuation directory; use --resume thereafter")
    project_root = Path(__file__).resolve().parent.parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    with (output / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checkpoint = output / "checkpoint.json"
        state = None
        try:
            if args.resume:
                state = json.loads(checkpoint.read_text())
                if state["source_hashes"] != search.source_hashes(project_root):
                    raise ValueError("Frozen study source changed")
                for run in state["runs"]:
                    prepare_config(run["config"], project_root)
            else:
                if not args.source:
                    raise ValueError("--source checkpoint is required")
                contents = args.source.read_bytes()
                digest = hashlib.sha256(contents).hexdigest()
                if args.revise_prompts:
                    state = make_prompt_continuation(json.loads(contents), project_root, digest)
                else:
                    raw = json.loads(args.config.read_text()) | {"source_checkpoint_sha256": digest}
                    state = make_state(json.loads(contents), raw, project_root)
                write_json(output / "plan.json", state["plan"])
                for relative in state["source_hashes"]:
                    dest = output / "source" / relative
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes((project_root / relative).read_bytes())
            state.pop("error", None)
            write_json(checkpoint, state)
            while state["status"] != "completed":
                if args.prepare_only and state["active_wave"]:
                    break
                search.tick(state, output, project_root, reporter=report)
                print(json.dumps({"status": state["status"], "wave": state["wave"],
                                  "new_calls": sum(len(r["records"]) - r["imported_calls"] for r in state["runs"]),
                                  "finished_runs": sum(r["phase"] == "done" for r in state["runs"])}), flush=True)
            report(state, output)
        except Exception as exc:
            if state is not None:
                state.update(status="needs_attention", error=f"{type(exc).__name__}: {exc}")
                write_json(checkpoint, state)
            parser.exit(2, f"error: {exc}\n")
    print(f"Report: {output / 'report.md'}")


if __name__ == "__main__":
    main()
