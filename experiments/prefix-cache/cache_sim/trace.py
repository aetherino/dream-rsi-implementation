"""Token traces and prefix interning. No tokenizer or model is needed at replay time."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random


@dataclass(frozen=True)
class Request:
    request_id: str
    arrival_time: float
    prompt: tuple[int, ...]
    output: tuple[int, ...] = ()
    session_id: str = ""
    workload: str = "unknown"
    task_type: str = "unknown"  # Declared serving metadata, never inferred from future turns.
    turn_index: int = 0  # Zero-based current turn; independent QA requests use zero.


@dataclass(frozen=True)
class Trace:
    requests: tuple[Request, ...]
    tokenizer: str = "synthetic-integer-tokens"


@dataclass(frozen=True)
class CompiledRequest:
    request: Request
    blocks: tuple[int, ...]  # Complete blocks in prompt + recorded output.
    prompt_blocks: int
    required_blocks: int  # Includes a transient partial tail, if any.


@dataclass(frozen=True)
class CompiledTrace:
    requests: tuple[CompiledRequest, ...]
    parents: tuple[int, ...]  # Index 0 is the virtual root; each block ID >= 1.
    depths: tuple[int, ...]
    block_size: int
    tokenizer: str


def compile_trace(trace: Trace, block_size: int = 16) -> CompiledTrace:
    import math

    if type(block_size) is not int or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    seen_ids: set[str] = set()
    identities: dict[tuple[int, tuple[int, ...]], int] = {}
    parents, depths, compiled = [0], [0], []
    previous_time = -math.inf
    for req in trace.requests:
        if not isinstance(req.request_id, str) or not req.request_id or req.request_id in seen_ids:
            raise ValueError("request_id must be a unique nonempty string")
        seen_ids.add(req.request_id)
        if req.task_type not in {"unknown", "chat", "document-qa"}:
            raise ValueError("Unknown task_type")
        if type(req.turn_index) is not int or req.turn_index < 0:
            raise ValueError("turn_index must be a nonnegative integer")
        if (type(req.arrival_time) not in (int, float) or not math.isfinite(req.arrival_time)
                or req.arrival_time < 0 or req.arrival_time < previous_time):
            raise ValueError("arrival_time must be finite, nonnegative and nondecreasing")
        previous_time = req.arrival_time
        if not req.prompt:
            raise ValueError("Each request needs at least one prompt token")
        tokens = tuple(req.prompt) + tuple(req.output)
        if any(type(token) is not int or token < 0 or token > 2**32 - 1 for token in tokens):
            raise ValueError("Tokens must be unsigned 32-bit integers")
        parent, blocks = 0, []
        for start in range(0, len(tokens) - block_size + 1, block_size):
            key = (parent, tokens[start:start + block_size])
            block = identities.get(key)
            if block is None:
                block = len(parents)
                identities[key] = block
                parents.append(parent)
                depths.append(depths[parent] + 1)
            blocks.append(block)
            parent = block
        compiled.append(CompiledRequest(req, tuple(blocks), len(req.prompt) // block_size,
                                        (len(tokens) + block_size - 1) // block_size))
    return CompiledTrace(tuple(compiled), tuple(parents), tuple(depths), block_size, trace.tokenizer)


def load_trace(path: Path) -> Trace:
    data = json.loads(path.read_text())
    if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
            or data["schema_version"] != 1 or not isinstance(data.get("tokenizer"), str)
            or not isinstance(data.get("requests"), list)):
        raise ValueError("Expected schema_version=1 and a tokenizer identifier")
    requests = tuple(Request(
        request_id=row["request_id"], arrival_time=row["arrival_time"],
        prompt=tuple(row["prompt_token_ids"]), output=tuple(row.get("output_token_ids", [])),
        session_id=row.get("session_id", ""), workload=row.get("workload", "unknown"),
        task_type=row.get("task_type", "unknown"), turn_index=row.get("turn_index", 0),
    ) for row in data["requests"])
    return Trace(requests, data["tokenizer"])


def save_trace(trace: Trace, path: Path) -> None:
    data = {"schema_version": 1, "tokenizer": trace.tokenizer, "requests": [
        {"request_id": r.request_id, "arrival_time": r.arrival_time,
         "prompt_token_ids": r.prompt, "output_token_ids": r.output,
         "session_id": r.session_id, "workload": r.workload,
         "task_type": r.task_type, "turn_index": r.turn_index} for r in trace.requests
    ]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def synthetic_trace(seed: int = 42, sessions: int = 40) -> Trace:
    """Seeded chat histories plus document QA. This is not ShareGPT or MASH-QA."""
    if type(sessions) is not int or sessions < 1:
        raise ValueError("sessions must be a positive integer")
    rng = random.Random(seed)
    requests: list[Request] = []
    system = tuple(range(1, 33))
    documents = [tuple(range(1000 + i * 128, 1128 + i * 128)) for i in range(8)]
    start = 0.0
    for session in range(sessions):
        start += rng.expovariate(0.3)
        history = system
        time = start
        for turn in range(rng.randint(2, 5)):
            base = 100_000 + session * 1000 + turn * 100
            prompt = history + tuple(range(base, base + rng.randint(8, 24)))
            output = tuple(range(base + 30, base + 30 + rng.randint(8, 24)))
            requests.append(Request(f"chat-{session}-{turn}", time, prompt, output,
                                    f"chat-{session}", "chat"))
            history = prompt + output
            time += rng.lognormvariate(2.5, 0.7)
        for query in range(3):
            # Shift the popular document set halfway through the trace generation.
            doc = rng.choice(range(4) if session < sessions // 2 else range(4, 8))
            base = 1_000_000 + session * 100 + query * 20
            prompt = system + documents[doc] + tuple(range(base, base + 8))
            output = tuple(range(base + 8, base + 16))
            requests.append(Request(f"qa-{session}-{query}", start + rng.uniform(0, 30),
                                    prompt, output, f"qa-{session}-{query}", "document-qa"))
    requests.sort(key=lambda r: (r.arrival_time, r.request_id))
    return Trace(tuple(requests))
