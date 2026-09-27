#!/usr/bin/env python3
"""Prepare local cache traces. No inference, model weights, .env, or paid APIs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import time

import ijson
from jinja2.sandbox import ImmutableSandboxedEnvironment
import requests
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-14B-Instruct"
REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
TOKENIZER_ID = f"{MODEL}@{REVISION}"
PREPARATION_VERSION = 1
SYSTEM = "You are a helpful assistant."
SPLITS = ("train", "validation", "test")


def sha_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def split_for(key: str, seed: int) -> str:
    value = int(stable_hash([seed, key])[:16], 16) / 2**64
    return "train" if value < .8 else "validation" if value < .9 else "test"


def prepare_tokenizer(directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    files = {}
    for name in ("tokenizer.json", "tokenizer_config.json"):
        path = directory / name
        if not path.exists():
            url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}"
            print(f"Downloading tokenizer asset: {name}", flush=True)
            partial = path.with_suffix(path.suffix + ".part")
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with partial.open("wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
            partial.replace(path)
        digest = sha_file(path)
        if previous.get("files", {}).get(name, {}).get("sha256", digest) != digest:
            raise ValueError(f"Tokenizer asset checksum changed: {path}")
        files[name] = {"sha256": digest, "bytes": path.stat().st_size}
    config = json.loads((directory / "tokenizer_config.json").read_text())
    result = {"model": MODEL, "revision": REVISION, "files": files,
              "chat_template_sha256": hashlib.sha256(config["chat_template"].encode()).hexdigest()}
    manifest_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


class LocalTokenizer:
    def __init__(self, directory: Path):
        self.backend = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.backend.no_truncation()
        self.backend.no_padding()
        config = json.loads((directory / "tokenizer_config.json").read_text())
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        self.template = env.from_string(config["chat_template"])

    def encode(self, messages: list[dict], generation: bool = False) -> list[int]:
        text = self.template.render(messages=messages, tools=None, add_generation_prompt=generation)
        return self.backend.encode(text, add_special_tokens=False).ids


def normalize_chat(record: dict) -> list[dict]:
    """Keep well-formed, complete text turns; reject unsupported roles/structures."""
    source = record.get("conversations")
    if not isinstance(source, list) or not source:
        raise ValueError("missing_conversation")
    aliases = {"human": "user", "user": "user", "gpt": "assistant",
               "assistant": "assistant", "system": "system"}
    messages = []
    for item in source:
        if not isinstance(item, dict):
            raise ValueError("invalid_message")
        role, content = aliases.get(item.get("from")), item.get("value")
        if role is None or not isinstance(content, str) or not content.strip():
            raise ValueError("invalid_role_or_text")
        messages.append({"role": role, "content": content})
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM})
    expected = "user"
    for message in messages[1:]:
        if message["role"] != expected:
            raise ValueError("nonalternating_roles")
        expected = "assistant" if expected == "user" else "user"
    if expected != "user" or len(messages) < 3:
        raise ValueError("incomplete_conversation")
    return messages


def encode_pairs(tokenizer, messages: list[dict], max_tokens: int) -> list[tuple[list[int], list[int]]]:
    pairs = []
    previous = []
    for index in range(2, len(messages), 2):
        prompt = tokenizer.encode(messages[:index], generation=True)
        complete = tokenizer.encode(messages[:index + 1])
        if complete[:len(prompt)] != prompt or prompt[:len(previous)] != previous:
            raise ValueError("template_prefix_mismatch")
        if len(complete) > max_tokens:
            raise ValueError("over_context_limit")
        pairs.append((prompt, complete[len(prompt):]))
        previous = complete
    return pairs


_worker_tokenizer = None


def initialize_worker(directory: str):
    global _worker_tokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _worker_tokenizer = LocalTokenizer(Path(directory))


def encode_group(job: dict) -> dict:
    try:
        if job["workload"] == "sharegpt":
            pairs = encode_pairs(_worker_tokenizer, job["messages"], job["max_tokens"])
        else:
            pairs = []
            for question, answer in job["qas"]:
                prompt = ("Please answer the question based on the texts below.\n"
                          + job["document"] + "\nQuestion: " + question + "\nAnswer: ")
                messages = [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": answer}]
                pairs.extend(encode_pairs(_worker_tokenizer, messages, job["max_tokens"]))
        return {"group_id": job["group_id"], "split": job["split"], "pairs": pairs,
                "workload": job["workload"], "source_ids": job["source_ids"]}
    except ValueError as error:
        return {"error": str(error), "workload": job["workload"], "split": job["split"]}


def chat_jobs(raw: Path, seed: int, limit: int, max_tokens: int, stats: Counter):
    seen = set()
    count = 0
    with raw.open("rb") as stream:
        for index, row in enumerate(ijson.items(stream, "item")):
            stats["source_conversations"] += 1
            try:
                messages = normalize_chat(row)
            except ValueError as error:
                stats[f"skip_{error}"] += 1
                continue
            fingerprint = stable_hash(messages)
            if fingerprint in seen:
                stats["skip_exact_duplicate"] += 1
                continue
            seen.add(fingerprint)
            # All variants with identical first user text stay in the same split,
            # including conversations whose later continuation differs.
            split = split_for(messages[1]["content"], seed)
            yield {"group_id": fingerprint, "split": split, "workload": "sharegpt",
                   "messages": messages, "max_tokens": max_tokens,
                   "source_ids": [str(row.get("id", index))]}
            count += 1
            if limit and count >= limit:
                break


def mash_jobs(raw: Path, limit: int, max_tokens: int, stats: Counter):
    documents = {}
    # Held-out splits win collisions; never move a held-out document into train.
    for split, filename in (("test", "test"), ("validation", "val"), ("train", "train")):
        data = json.loads((raw / f"{filename}_webmd_squad_v2_full.json").read_text())
        for article in data["data"]:
            for paragraph in article["paragraphs"]:
                doc = paragraph["context"]
                key = hashlib.sha256(doc.encode()).hexdigest()
                if key in documents and documents[key]["split"] != split:
                    stats["skip_cross_split_document_records"] += 1
                    continue
                group = documents.setdefault(key, {"group_id": key, "split": split,
                    "document": doc, "qas": [], "source_ids": [], "seen": set(),
                    "workload": "mashqa", "max_tokens": max_tokens})
                for qa in paragraph["qas"]:
                    stats["source_questions"] += 1
                    answers = qa.get("answers", [])
                    question = qa.get("question")
                    if qa.get("is_impossible") or not answers or not isinstance(question, str):
                        stats["skip_unanswerable"] += 1
                        continue
                    answer = answers[0].get("text")
                    if not question.strip() or not isinstance(answer, str) or not answer.strip():
                        stats["skip_empty_text"] += 1
                        continue
                    pair = (question, answer)
                    if pair in group["seen"]:
                        stats["skip_duplicate_qa"] += 1
                        continue
                    group["seen"].add(pair)
                    group["qas"].append(pair)
                    group["source_ids"].append(str(qa["id"]))
    counts = Counter()
    for group in documents.values():
        if not group["qas"]:
            continue
        if limit and counts[group["split"]] >= limit:
            continue
        counts[group["split"]] += 1
        del group["seen"]
        yield group


class Shards:
    def __init__(self, root: Path, seed: int, max_groups: int, token_budget: int):
        self.root, self.seed, self.max_groups, self.token_budget = root, seed, max_groups, token_budget
        self.buffers = defaultdict(list)
        self.token_counts = Counter()
        self.files = []
        self.totals = defaultdict(Counter)
        self.indices = Counter()

    def add(self, group: dict):
        key = (group["workload"], group["split"])
        tokens = sum(len(p) + len(o) for p, o in group["pairs"])
        if self.buffers[key] and (len(self.buffers[key]) >= self.max_groups
                                 or self.token_counts[key] + tokens > self.token_budget):
            self.flush(key)
        self.buffers[key].append(group)
        self.token_counts[key] += tokens

    def flush(self, key):
        groups = self.buffers[key]
        if not groups:
            return
        index = self.indices[key]
        self.indices[key] += 1
        rng = random.Random(int(stable_hash([self.seed, key, index])[:16], 16))
        rng.shuffle(groups)
        rows, time_cursor = [], 0.0
        for group in groups:
            time_cursor += rng.expovariate(1.0)
            arrival = time_cursor
            for turn, (prompt, output) in enumerate(group["pairs"]):
                if turn:
                    arrival += rng.lognormvariate(4.15, .971) if key[0] == "sharegpt" else 0
                row = {"request_id": f"{key[0]}-{group['group_id']}-{turn}",
                       "session_id": group["group_id"] if key[0] == "sharegpt" else f"{group['group_id']}-{turn}",
                       "group_id": group["group_id"], "turn_index": turn,
                       "workload": key[0], "arrival_time": arrival,
                       "prompt_token_ids": prompt, "output_token_ids": output}
                rows.append(row)
        if key[0] == "mashqa":
            # Independent single-turn requests, shuffled within this document shard.
            rng.shuffle(rows)
            time_cursor = 0.0
            for row in rows:
                time_cursor += rng.expovariate(1.0)
                row["arrival_time"] = time_cursor
        rows.sort(key=lambda r: (r["arrival_time"], r["request_id"]))
        path = self.root / key[0] / key[1] / f"trace-{index:05d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"schema_version": 1, "tokenizer": TOKENIZER_ID,
                    "preparation_version": PREPARATION_VERSION, "split": key[1],
                    "group_sources": {g["group_id"]: g["source_ids"] for g in groups},
                    "timing": "synthetic; see manifest.json", "requests": rows}
        partial = path.with_suffix(".json.part")
        with partial.open("w") as stream:
            json.dump(metadata, stream, ensure_ascii=False, separators=(",", ":"))
            stream.write("\n")
        partial.replace(path)
        prompt_tokens = sum(len(r["prompt_token_ids"]) for r in rows)
        output_tokens = sum(len(r["output_token_ids"]) for r in rows)
        max_sequence = max(len(r["prompt_token_ids"]) + len(r["output_token_ids"]) for r in rows)
        entry = {"path": str(path.relative_to(self.root)), "sha256": sha_file(path),
                 "bytes": path.stat().st_size, "requests": len(rows), "groups": len(groups),
                 "prompt_tokens": prompt_tokens, "output_tokens": output_tokens,
                 "max_sequence_tokens": max_sequence,
                 "minimum_capacity_blocks_at_16_tokens": math.ceil(max_sequence / 16)}
        self.files.append(entry)
        self.totals["/".join(key)].update({k: entry[k] for k in
            ("bytes", "requests", "groups", "prompt_tokens", "output_tokens")})
        self.buffers[key] = []
        self.token_counts[key] = 0

    def finish(self):
        for key in list(self.buffers):
            self.flush(key)


def verify_output(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    for entry in manifest["files"]:
        path = root / entry["path"]
        if path.stat().st_size != entry["bytes"] or sha_file(path) != entry["sha256"]:
            raise ValueError(f"Prepared trace checksum mismatch: {path}")
    return manifest


def print_summary(manifest):
    print(json.dumps({"totals": manifest["totals"], "api_cost_usd": 0,
                      "elapsed_seconds": manifest["elapsed_seconds"],
                      "files": len(manifest["files"])}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", "sharegpt", "mashqa"], default="all")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/tokenized/qwen2.5")
    parser.add_argument("--tokenizer-dir", type=Path, default=ROOT / "data/tokenizers/qwen2.5")
    parser.add_argument("--max-sequence-tokens", type=int, default=32768)
    parser.add_argument("--groups-per-shard", type=int, default=64)
    parser.add_argument("--tokens-per-shard", type=int, default=250000)
    parser.add_argument("--limit", type=int, default=0,
                        help="Smoke-test limit: ShareGPT conversations, MASH-QA documents per split; 0=all")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if min(args.workers, args.max_sequence_tokens, args.groups_per_shard, args.tokens_per_shard) < 1 or args.limit < 0:
        parser.error("Workers, token limits and shard sizes must be positive; limit must be nonnegative")
    if args.verify_only:
        print_summary(verify_output(args.output_dir))
        return
    settings = {k: v for k, v in vars(args).items()
                if k not in ("output_dir", "tokenizer_dir", "workers", "verify_only")}
    settings["raw_dir"] = str(args.raw_dir.resolve())
    settings["preparation_version"] = PREPARATION_VERSION
    settings["tokenizer"] = TOKENIZER_ID
    if (args.output_dir / "manifest.json").exists():
        prior = verify_output(args.output_dir)
        if prior["settings"] != settings:
            parser.error("Output already exists with different settings; choose a new --output-dir")
        if any(sha_file(args.raw_dir / e["path"]) != e["sha256"] for e in prior["sources"]):
            parser.error("Raw data changed; choose a new --output-dir")
        print("Reusing verified prepared traces; no tokenization needed.")
        print_summary(prior)
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory contains incomplete/unrelated files; choose a new --output-dir")
    started = time.perf_counter()
    tokenizer_meta = prepare_tokenizer(args.tokenizer_dir)
    sources = []
    if args.dataset in ("all", "sharegpt"):
        sources.append(args.raw_dir / "sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json")
    if args.dataset in ("all", "mashqa"):
        sources.extend(args.raw_dir / f"mashqa/{s}_webmd_squad_v2_full.json" for s in ("train", "val", "test"))
    source_meta = [{"path": str(p.relative_to(args.raw_dir)), "sha256": sha_file(p),
                    "bytes": p.stat().st_size} for p in sources]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    marker = args.output_dir / "INCOMPLETE"
    marker.write_text("Preparation running. A final manifest indicates completion.\n")
    stats = {"sharegpt": Counter(), "mashqa": Counter()}
    writer = Shards(args.output_dir, args.seed, args.groups_per_shard, args.tokens_per_shard)
    initialize_worker(str(args.tokenizer_dir))
    pool = ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                               initargs=(str(args.tokenizer_dir),)) if args.workers > 1 else None
    try:
        for dataset in ("mashqa", "sharegpt"):
            if args.dataset not in ("all", dataset):
                continue
            jobs = (chat_jobs(args.raw_dir / "sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json",
                             args.seed, args.limit, args.max_sequence_tokens, stats[dataset])
                    if dataset == "sharegpt" else
                    mash_jobs(args.raw_dir / "mashqa", args.limit, args.max_sequence_tokens, stats[dataset]))
            # Bounded batches: ProcessPoolExecutor.map on Python 3.11 otherwise
            # eagerly consumes the entire source generator and retains its text.
            from itertools import islice
            processed = 0
            while batch := list(islice(jobs, 64)):
                results = pool.map(encode_group, batch, chunksize=4) if pool else map(encode_group, batch)
                for group in results:
                    if "error" in group:
                        stats[dataset]["skip_" + group["error"]] += 1
                    else:
                        writer.add(group)
                        stats[dataset]["accepted_groups"] += 1
                    processed += 1
                if processed % 512 == 0:
                    print(f"{dataset}: {processed:,} groups processed; {time.perf_counter()-started:.1f}s", flush=True)
            writer.finish()
    finally:
        if pool:
            pool.shutdown()
    manifest = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "settings": settings, "sources": source_meta, "tokenizer_assets": tokenizer_meta,
        "packages": {p: importlib.metadata.version(p) for p in ("tokenizers", "jinja2", "ijson")},
        "files": writer.files, "totals": dict(writer.totals), "preprocessing": stats,
        "elapsed_seconds": round(time.perf_counter() - started, 3), "api_cost_usd": 0,
        "timing": {"source": "synthetic, not production timestamps", "session_rate_per_second": 1.0,
                   "chat_interturn_lognormal_mu": 4.15, "chat_interturn_lognormal_sigma": .971,
                   "mashqa": "shuffle requests within shard, then exponential interarrival mean=1 second",
                   "shards": "independent cold-start episodes; groups never split across shards"},
        "notes": ["MiMo API is not called. No .env is loaded.",
                  "Whole conversations/documents over the context limit are omitted, never truncated.",
                  "Outputs are recorded answers plus template closing markers, not fresh model generations.",
                  "ShareGPT exact duplicates removed; first user text groups determine seeded 80/10/10 splits.",
                  "MASH-QA official splits retained; overlapping documents excluded from lower-priority splits (test > validation > train).",
                  "Token counts include repeated history and template tokens. They are not MiMo billing estimates."]}
    partial = args.output_dir / "manifest.json.part"
    partial.write_text(json.dumps(manifest, indent=2) + "\n")
    partial.replace(args.output_dir / "manifest.json")
    marker.unlink()
    print_summary(manifest)


if __name__ == "__main__":
    main()
