"""Reproducible mixed episodes from whole, already-split tokenized groups.

Preparation can read held-out data; search never receives its trace or metrics.
This is a protocol boundary, not an OS security boundary against the operator.
"""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random

from .evaluation import digest

PROFILES = {"chat-heavy": [0.8], "balanced": [0.5], "qa-heavy": [0.2],
            "shift": [0.8, 0.2]}
VERSION = "mixed-task-v1"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def seed_for(*parts):
    return int(hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:16], 16)


def select_groups(pool, target, rng):
    """Never truncate a conversation/document to hit an exact request fraction."""
    order = list(pool)
    rng.shuffle(order)
    selected, count = [], 0
    for key in order:
        if count >= target:
            break
        selected.append(pool.pop(key))
        count += len(selected[-1])
    if count < target:
        raise ValueError(f"Insufficient whole groups: need {target} requests, found {count}")
    return selected


def compose(pools, tokenizer, profile, seed, requests=256, phase_seconds=300):
    if profile not in PROFILES or type(requests) is not int or requests < 10:
        raise ValueError("Unknown profile or invalid request target")
    if not math.isfinite(phase_seconds) or phase_seconds <= 0:
        raise ValueError("phase_seconds must be finite and positive")
    # Copy dictionaries only; rows are copied on emission. No source timing survives.
    available = {task: dict(groups) for task, groups in pools.items()}
    rng = random.Random(seed)
    rows, phases = [], []
    fractions = PROFILES[profile]
    for phase, chat_fraction in enumerate(fractions):
        target = requests // len(fractions)
        phase_meta = {"start": phase * phase_seconds, "end": (phase + 1) * phase_seconds,
                      "target_chat_request_fraction": chat_fraction, "requests_by_task": {}}
        for workload, fraction in (("sharegpt", chat_fraction), ("mashqa", 1 - chat_fraction)):
            selected = select_groups(available[workload], max(1, round(target * fraction)), rng)
            count = 0
            for group in selected:
                # Conditional on a fixed count, homogeneous Poisson arrival times are uniform.
                arrival = phase * phase_seconds + rng.uniform(0, phase_seconds)
                for turn, original in enumerate(sorted(group, key=lambda r: r["turn_index"])):
                    row = deepcopy(original)
                    row["task_type"] = "chat" if workload == "sharegpt" else "document-qa"
                    row["turn_index"] = turn if workload == "sharegpt" else 0
                    if workload == "sharegpt":
                        if turn:
                            arrival += rng.lognormvariate(4.15, .971)
                    else:
                        arrival = phase * phase_seconds + rng.uniform(0, phase_seconds)
                    row["arrival_time"] = arrival
                    # Audit-only cohort; not exposed to the generated policy.
                    row["admission_phase"] = phase
                    rows.append(row)
                    count += 1
            phase_meta["requests_by_task"][workload] = count
        phases.append(phase_meta)
    rows.sort(key=lambda r: (r["arrival_time"], r["request_id"]))
    chat = sum(r["task_type"] == "chat" for r in rows)
    return {"schema_version": 1, "tokenizer": tokenizer, "preparation_version": VERSION,
            "profile": profile, "seed": seed, "requests": rows,
            "composition": {"requested_requests": requests, "actual_requests": len(rows),
                "actual_chat_request_fraction": chat / len(rows), "phases": phases,
                "timing": "Synthetic session admissions within phase windows; chat follow-ups use "
                          "lognormal(4.15, .971) gaps and may cross phase boundaries or drain after admission ends."}}


def load_pools(source_root, manifest, split, indices):
    listed = {f["path"]: f for f in manifest["files"]}
    pools = {"sharegpt": {}, "mashqa": {}}
    sources, tokenizer = [], None
    for workload in pools:
        for index in indices:
            rel = f"{workload}/{split}/trace-{index:05d}.json"
            path = source_root / rel
            expected = listed[rel]
            if digest(path) != expected["sha256"]:
                raise ValueError(f"Source hash mismatch: {rel}")
            data = json.loads(path.read_text())
            if data["split"] != split:
                raise ValueError("Source split mismatch")
            if tokenizer is not None and tokenizer != data["tokenizer"]:
                raise ValueError("Cannot mix tokenizer namespaces")
            tokenizer = data["tokenizer"]
            grouped = defaultdict(list)
            for row in data["requests"]:
                if row["workload"] != workload:
                    raise ValueError("Source workload mismatch")
                grouped[row["group_id"]].append(row)
            if pools[workload].keys() & grouped.keys():
                raise ValueError("Duplicate source group across shards")
            pools[workload].update(grouped)
            sources.append({"path": rel, "sha256": expected["sha256"]})
    return pools, tokenizer, sources


def verify_prepared(output):
    manifest = json.loads((output / "manifest.json").read_text())
    for file in manifest["files"]:
        if digest(output / file["path"]) != file["sha256"]:
            raise ValueError(f"Prepared file changed: {file['path']}")
    return manifest


def prepare(source_root, output, *, seed=20260928, episodes=2, requests=256,
            source_start=20, source_count=8, validation_start=0, test_start=0, capacities=(2048, 4096)):
    if output.exists():
        raise ValueError("Use a new output directory; preparation never overwrites a sealed split")
    if type(episodes) is not int or episodes < 1 or source_count < 1:
        raise ValueError("episodes and source_count must be positive")
    if any(type(c) is not int or c < 1 for c in capacities):
        raise ValueError("Capacities must be positive integer blocks")
    source_manifest = json.loads((source_root / "manifest.json").read_text())
    manifest = {"version": VERSION, "seed": seed, "episodes_per_profile": episodes,
                "target_requests_per_episode": requests, "block_size": 16,
                "capacities": list(capacities), "source_manifest_sha256": digest(source_root / "manifest.json"),
                "sources": {}, "files": [], "splits": {}, "api_cost_usd": 0,
                "test_status": "sealed; never evaluated during preparation or search"}
    starts = {"train": source_start, "validation": validation_start, "test": test_start}
    listed = {f["path"] for f in source_manifest["files"]}
    for split, start in starts.items():
        for workload in ("sharegpt", "mashqa"):
            for index in range(start, start + source_count):
                rel = f"{workload}/{split}/trace-{index:05d}.json"
                if rel not in listed:
                    raise ValueError(f"Missing source shard: {rel}")
    all_groups = set()
    for split in ("train", "validation", "test"):
        start = starts[split]
        pools, tokenizer, sources = load_pools(source_root, source_manifest, split, range(start, start + source_count))
        identities = {(task, key) for task, groups in pools.items() for key in groups}
        if all_groups & identities:
            raise ValueError("Source groups overlap across splits")
        all_groups.update(identities)
        manifest["sources"][split] = sources
        suite = []
        for profile in PROFILES:
            for replica in range(episodes):
                trace = compose(pools, tokenizer, profile, seed_for(seed, split, profile, replica), requests)
                longest = max(len(r["prompt_token_ids"]) + len(r["output_token_ids"]) for r in trace["requests"])
                if math.ceil(longest / 16) > min(capacities):
                    raise ValueError("A whole request exceeds the smallest capacity; choose larger capacities")
                trace["split"] = split
                rel = f"{split}/{profile}-{replica:02d}.json"
                path = output / rel
                write_json(path, trace)
                manifest["files"].append({"path": rel, "sha256": digest(path),
                    "requests": len(trace["requests"]), "composition": trace["composition"],
                    "groups": len({(r["workload"], r["group_id"]) for r in trace["requests"]})})
                for capacity in capacities:
                    suite.append({"name": f"{profile}-{replica:02d}-cap{capacity}",
                                  "path": str(path.resolve()), "sha256": digest(path), "capacity_blocks": capacity})
        rel = f"{split}/suite.json"
        write_json(output / rel, suite)
        manifest["files"].append({"path": rel, "sha256": digest(output / rel)})
        manifest["splits"][split] = rel
    # Search configs deliberately contain no test suite or test path.
    for features in ("block-v1", "task-v1"):
        config = {"policy_features": features, "prompt_version": "v2", "cycles": 4,
                  "online_rounds": 4, "workers": 2, "max_depth": 4, "controller_revisions": 0,
                  "evaluation_timeout_seconds": 600,
                  "api": {"max_calls": 12, "max_usd": .30, "max_completion_tokens": 2048, "thinking": "disabled"},
                  "train": json.loads((output / "train/suite.json").read_text()),
                  "validation": json.loads((output / "validation/suite.json").read_text())}
        rel = f"configs/{features}.json"
        write_json(output / rel, config)
        manifest["files"].append({"path": rel, "sha256": digest(output / rel)})
    write_json(output / "manifest.json", manifest)
    return manifest
