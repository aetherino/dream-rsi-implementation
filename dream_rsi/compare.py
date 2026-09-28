"""Budget-matched fixed/adaptive controller experiment; sequential runs, no agents."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import uuid

from .backend import write_json
from .runner import prepare_config, run


def arm_config(base, arm):
    result = copy.deepcopy(base)
    if arm not in ("fixed", "adaptive"):
        raise ValueError("Unknown strategy")
    result["controller_revisions"] = 0 if arm == "fixed" else base["controller_revisions"]
    result.update(revise_after_final_cycle=False, stop_on_empty_rollout=True)
    return result


def summarize_run(directory, trial, arm):
    summary = json.loads((directory / "summary.json").read_text())
    usage = summary["usage"]
    revisions = [r for p in directory.glob("controller-revisions-*.json")
                 for r in json.loads(p.read_text()) if r["revision"] > 0]
    validation = json.loads((directory / "validation.json").read_text())
    baseline = json.loads((directory / "validation-baselines.json").read_text())
    if not validation["valid"] or not baseline["valid"]:
        raise ValueError("A comparison run failed held-out validation")
    scenarios = []
    for result, reference in zip(validation["runs"], baseline["baselines"]):
        if result["name"] != reference["name"]:
            raise ValueError("Validation scenario mismatch")
        scenarios.append({"name": result["name"], "improvement": result["improvement"],
                          "extra_computed_tokens": result["extra_computed_tokens"],
                          "lru_extra_computed_tokens": reference["policies"]["lru"]["extra_computed_tokens"],
                          "policy_us_per_request": result["policy_us_per_request"]})
    nodes = [n for p in sorted(directory.glob("tree-*.json")) for n in json.loads(p.read_text())["nodes"][1:]]
    failures = [{"cycle_file": p.name, "node": n["id"], "error": n["feedback"].get("error")}
                for p in sorted(directory.glob("tree-*.json"))
                for n in json.loads(p.read_text())["nodes"][1:] if not n["valid"]]
    return {"trial": trial, "arm": arm, "path": str(directory), "status": summary["status"],
            "train_score": summary["best_train_score"], "validation_score": summary["validation_score"],
            "calls": usage["calls"], "discovery_calls": sum(r["role"] == "discovery" for r in usage["records"]),
            "controller_calls": sum(r["role"] == "controller" for r in usage["records"]),
            "valid_candidates": sum(n["valid"] for n in nodes), "candidate_failures": failures,
            "accepted_revisions": sum(r["accepted"] for r in revisions),
            "invalid_revisions": sum(not r["evaluation"]["valid"] for r in revisions),
            "estimated_usd": usage["estimated_usd"], "elapsed_seconds": summary["elapsed_seconds"],
            "best_policy": summary["best_policy"], "scenarios": scenarios}


def aggregate(rows):
    arms = {}
    for arm in ("fixed", "adaptive"):
        selected = [r for r in rows if r["arm"] == arm]
        if not selected:
            continue
        scores = [r["validation_score"] for r in selected]
        extra = [sum(s["extra_computed_tokens"] for s in r["scenarios"]) for r in selected]
        lru_extra = [sum(s["lru_extra_computed_tokens"] for s in r["scenarios"]) for r in selected]
        arms[arm] = {"runs": len(selected), "mean_validation_score": statistics.mean(scores),
                     "mean_validation_extra_tokens": statistics.mean(extra),
                     "mean_lru_validation_extra_tokens": statistics.mean(lru_extra),
                     "total_extra_fraction_change_vs_lru": (sum(extra) / sum(lru_extra) - 1) if sum(lru_extra) else None,
                     "min_validation_score": min(scores), "max_validation_score": max(scores),
                     "sample_sd": statistics.stdev(scores) if len(scores) > 1 else None,
                     "mean_train_score": statistics.mean(r["train_score"] for r in selected),
                     "total_calls": sum(r["calls"] for r in selected),
                     "discovery_calls": sum(r["discovery_calls"] for r in selected),
                     "controller_calls": sum(r["controller_calls"] for r in selected),
                     "accepted_revisions": sum(r["accepted_revisions"] for r in selected),
                     "estimated_usd": sum(r["estimated_usd"] for r in selected)}
    differences = []
    for trial in sorted({r["trial"] for r in rows}):
        pair = {r["arm"]: r for r in rows if r["trial"] == trial}
        if len(pair) == 2:
            differences.append({"trial": trial, "adaptive_minus_fixed": pair["adaptive"]["validation_score"] - pair["fixed"]["validation_score"]})
    return {"arms": arms, "paired_differences": differences,
            "mean_adaptive_minus_fixed": statistics.mean(d["adaptive_minus_fixed"] for d in differences) if differences else None,
            "estimated_usd": sum(r["estimated_usd"] for r in rows)}


def markdown_report(report):
    lines = ["# Fixed versus adaptive controller pilot", "", f"Status: **{report['status']}**. Backend: `{report['backend']}`.", "",
             "Both strategies receive the same call-count, output-token and estimated-dollar ceilings. Controller-development calls count against the adaptive allowance. Actual token spending may differ; this is not exact dollar-spend matching.", "",
             "| Trial | Strategy | Train score | Validation score | Discovery calls | Controller calls | Accepted revisions | Estimated USD |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in sorted(report["runs"], key=lambda r: (r["trial"], r["arm"])):
        lines.append(f"| {row['trial']} | {row['arm']} | {row['train_score']:.5f} | {row['validation_score']:.5f} | {row['discovery_calls']} | {row['controller_calls']} | {row['accepted_revisions']} | ${row['estimated_usd']:.6f} |")
    stats = report["aggregate"]
    if stats["mean_adaptive_minus_fixed"] is not None:
        lines += ["", f"Mean validation difference (adaptive − fixed): **{stats['mean_adaptive_minus_fixed']:+.5f}**."]
    lines += ["", f"Total estimated API cost: **${stats['estimated_usd']:.6f}**.", "",
              "## Raw held-out recomputation (descriptive secondary measure)", "",
              "| Strategy | Mean extra tokens per run | LRU reference | Change vs LRU |",
              "| --- | ---: | ---: | ---: |"]
    for arm, metrics in stats["arms"].items():
        change = metrics["total_extra_fraction_change_vs_lru"]
        formatted = f"{change:+.2%}" if change is not None else "undefined (zero LRU extra)"
        lines.append(f"| {arm} | {metrics['mean_validation_extra_tokens']:,.2f} | {metrics['mean_lru_validation_extra_tokens']:,.2f} | {formatted} |")
    lines += ["", "Lower raw extra-token counts are better. These sum scenarios, including each capacity as a separate replay. This diagnostic is not the optimization score and was added after the pilot exposed a difference between the two measures. A positive mean relative score need not mean fewer total recomputed tokens.", "",
              "Scores are mean per-scenario reductions in extra recomputation versus LRU, using the harness's zero-denominator convention. Higher is better. Scenarios share shards across capacities, so they are not independent replicates.", "",
              "This small pilot reports all trials, not just the best run. Trial pairs share workloads and settings, not identical model samples or guaranteed random seeds. The run order alternates by pair. No significance or general superiority claim is warranted from three pairs.", "",
              "Validation is evaluated once for each training-selected winner and is never fed into generation. The experiment does not tune settings after observing validation; test data is unused. Model thinking is disabled in this pilot, so results do not establish behavior in thinking mode.", "",
              "See comparison.json for per-scenario outcomes, failures, policies, spending, variability and accepted controller revisions.", ""]
    return "\n".join(lines)


def compare(raw, backend, project_root, output):
    trials = raw.get("trials", 3)
    if type(trials) is not int or not 1 <= trials <= 10:
        raise ValueError("trials must be an integer in [1,10]")
    base = prepare_config(raw["run"], project_root)
    if not base["validation"] or base["controller_revisions"] < 1 or base["cycles"] < 2:
        raise ValueError("Comparison needs validation, controller revisions, and multiple cycles")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = {"trials": trials, "backend": backend, "base_config": base,
            "estimated_usd_ceiling": 2 * trials * base["api"]["max_usd"],
            "total_call_ceiling": 2 * trials * base["api"]["max_calls"],
            "order": [{"trial": trial, "arm": arm} for trial in range(1, trials + 1)
                      for arm in (("fixed", "adaptive") if trial % 2 else ("adaptive", "fixed"))]}
    write_json(output / "plan.json", plan)
    rows = []

    def save(status, error=None):
        report = {"status": status, "backend": backend, "plan": plan, "runs": rows,
                  "aggregate": aggregate(rows), "error": error}
        write_json(output / "comparison.json", report)
        (output / "report.md").write_text(markdown_report(report))
        return report

    save("running")
    try:
        for item in plan["order"]:
            directory = output / f"trial-{item['trial']:02d}-{item['arm']}"
            print(json.dumps({"event": "comparison_arm_started", **item}), flush=True)
            run(arm_config(base, item["arm"]), backend, project_root, directory)
            rows.append(summarize_run(directory, **item))
            save("running")
        return save("completed")
    except BaseException as exc:
        save("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", f"{type(exc).__name__}: {exc}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dream-comparison.json"))
    parser.add_argument("--backend", choices=["mock", "mimo"], default="mock")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent.parent
    output = args.output or project_root / "runs" / ("comparison-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:6])
    try:
        report = compare(json.loads(args.config.read_text()), args.backend, project_root, output)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(report["aggregate"], indent=2))
    print(f"Report: {output.resolve() / 'report.md'}")


if __name__ == "__main__":
    main()
