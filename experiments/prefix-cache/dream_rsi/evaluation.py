"""Fixed replay suites. Child processes bound wall time, not arbitrary-code security."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from cache_sim.engine import replay
from cache_sim.policies import BASELINES
from cache_sim.trace import compile_trace, load_trace, synthetic_trace
from .programs import RetentionPolicy
from .scoring import TOTAL_EXTRA, score_recomputation


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_suite(suite, project_root):
    if not isinstance(suite, list) or not suite:
        raise ValueError("Suite must contain at least one scenario")
    frozen = []
    for item in suite:
        item = dict(item)
        if type(item.get("capacity_blocks")) is not int or item["capacity_blocks"] < 1:
            raise ValueError("Each scenario requires positive capacity_blocks")
        if not isinstance(item.get("name"), str) or not item["name"]:
            raise ValueError("Each scenario requires a name")
        if "path" in item:
            path = (project_root / item["path"]).resolve()
            current_hash = digest(path)
            if "sha256" in item and item["sha256"] != current_hash:
                raise ValueError("Trace changed after suite snapshot")
            declared_split = json.loads(path.read_text()).get("split")
            if declared_split is not None:
                item["declared_split"] = declared_split
            item.update(path=str(path), sha256=current_hash)
        elif "synthetic" not in item:
            raise ValueError("Scenario requires a path or synthetic configuration")
        frozen.append(item)
    if len({s["name"] for s in frozen}) != len(frozen):
        raise ValueError("Scenario names must be unique")
    return frozen


def run_suite(suite, block_size, *, candidate=None, baselines=None, max_policy_us_per_request=50000,
              scoring=TOTAL_EXTRA, policy_features="block-v1"):
    """Minimize total recomputation; timing remains a feasibility gate."""
    if candidate is not None:
        RetentionPolicy(candidate, policy_features)  # Reject malformed expressions even if no eviction happens.
    runs = []
    for index, item in enumerate(suite):
        if "path" in item:
            if digest(item["path"]) != item["sha256"]:
                raise ValueError("Trace changed after suite snapshot")
            trace = load_trace(Path(item["path"]))
        else:
            trace = synthetic_trace(**item["synthetic"])
        compiled = compile_trace(trace, block_size)
        if not compiled.requests:
            raise ValueError("Empty traces cannot be scored")
        if candidate is None:
            unlimited = replay(compiled, None).to_dict()
            policies = {name: replay(compiled, item["capacity_blocks"], cls()).to_dict()
                        for name, cls in BASELINES.items()}
            for result in policies.values():
                result["extra_computed_tokens"] = result["computed_prompt_tokens"] - unlimited["computed_prompt_tokens"]
            runs.append({"name": item["name"], "unlimited": unlimited, "policies": policies})
        else:
            reference = baselines[index]
            if reference["name"] != item["name"]:
                raise ValueError("Baseline/scenario mismatch")
            result = replay(compiled, item["capacity_blocks"], RetentionPolicy(candidate, policy_features)).to_dict()
            extra = result["computed_prompt_tokens"] - reference["unlimited"]["computed_prompt_tokens"]
            lru_extra = reference["policies"]["lru"]["extra_computed_tokens"]
            result.update(name=item["name"], extra_computed_tokens=extra,
                          lru_extra_computed_tokens=lru_extra, saved_prompt_tokens=lru_extra - extra,
                          improvement=(lru_extra - extra) / max(1, lru_extra))
            runs.append(result)
    if candidate is None:
        return {"valid": True, "scoring": scoring, "baselines": runs}
    metrics = score_recomputation([r["extra_computed_tokens"] for r in runs],
                                  [r["lru_extra_computed_tokens"] for r in runs], scoring)
    for result, contribution in zip(runs, metrics.pop("score_contributions")):
        result["score_contribution"] = contribution
    within_budget = all(r["policy_us_per_request"] <= max_policy_us_per_request for r in runs)
    return {"valid": within_budget, **metrics,
            "error": None if within_budget else "Policy CPU time exceeds configured per-request limit",
            "runs": runs}


def isolated_evaluate(payload, timeout):
    # No key is needed in the worker. Generated programs never get access to os/subprocess.
    env = {key: value for key, value in os.environ.items()
           if key in {"PATH", "SYSTEMROOT", "LANG", "LC_ALL", "TMPDIR", "HOME"}}
    try:
        proc = subprocess.run([sys.executable, "-m", "dream_rsi.worker"], input=json.dumps(payload),
                              text=True, capture_output=True, timeout=timeout, env=env)
        if proc.returncode:
            return {"valid": False, "score": -1e6, "error": f"Evaluator process exited {proc.returncode}"}
        return json.loads(proc.stdout)
    except subprocess.TimeoutExpired:
        return {"valid": False, "score": -1e6, "error": "Evaluator wall-time limit exceeded"}


def worker_main():
    try:
        result = run_suite(**json.load(sys.stdin))
    except Exception as exc:
        result = {"valid": False, "score": -1e6, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, allow_nan=False))
