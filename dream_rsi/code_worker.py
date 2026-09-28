"""Standalone JSONL worker. The host must apply an OS sandbox before launch.

This file imports no project modules and receives no suite or trace paths.
Normal Python is executed; the security boundary is the OS sandbox, not AST
validation or restricted builtins. Stdout/stderr from candidate code is bounded
by the host and treated as a protocol error.
"""
import json
import resource
import sys
import time
from dataclasses import make_dataclass

BLOCK_FIELDS = ("block_id", "depth", "inserted_at", "last_access", "frequency",
                "insertion_order", "access_order")
TASK_FIELDS = ("task_chat", "task_qa", "task_unknown", "turn_index")


def main():
    policy = None
    entry_type = None
    for line in sys.stdin:
        try:
            message = json.loads(line)
            started = time.process_time_ns()
            operation = message["op"]
            if operation == "init":
                limit = message["memory_bytes"]
                try:
                    resource.setrlimit(resource.RLIMIT_DATA, (limit, limit))
                except (ValueError, OSError):
                    pass  # Host RSS monitor remains mandatory on Darwin.
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
                resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
                # RLIMIT_AS is ineffective on some macOS releases; the host also
                # checks resident memory while waiting for every response.
                try:
                    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
                except (ValueError, OSError):
                    pass
                fields = BLOCK_FIELDS + (TASK_FIELDS if message["policy_features"] == "task-v1" else ())
                entry_type = make_dataclass("Entry", [(f, object) for f in fields], frozen=True, slots=True)
                namespace = {"__name__": "generated_policy", "Entry": entry_type}
                exec(compile(message["source"], "<generated-policy>", "exec"), namespace)
                cls = namespace[message["class_name"]]
                policy = cls(**message.get("init_kwargs", {}))
                if not callable(getattr(policy, "choose" if message["class_name"] == "CachePolicy" else "select", None)):
                    raise TypeError("Generated class lacks its required method")
                value = None
            elif operation == "cache":
                for event in message["events"]:
                    hook = getattr(policy, event["method"], None)
                    if hook is not None:
                        hook(entry_type(**event["entry"]))
                value = (policy.choose(tuple(entry_type(**e) for e in message["eligible"]), message["now"])
                         if "eligible" in message else None)
            elif operation == "call":
                method = message["method"]
                if method.startswith("_"):
                    raise ValueError("Private methods cannot be called")
                value = getattr(policy, method)(*message.get("args", []), **message.get("kwargs", {}))
            else:
                raise ValueError("Unknown RPC operation")
            response = {"ok": True, "result": value, "cpu_ns": time.process_time_ns() - started}
            encoded = json.dumps(response, allow_nan=False, separators=(",", ":"))
        except BaseException as exc:
            encoded = json.dumps({"ok": False, "error": type(exc).__name__ + ": " + str(exc)[:1000]})
        sys.stdout.write(encoded + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
