"""Executable Python policies behind a fail-closed macOS Seatbelt sandbox.

Only current metadata enters the JSONL worker. Token IDs, prompt text, trace
paths, request/session/workload IDs, suite descriptions and future inputs never
enter it. Per-run random string block handles hide trace registry indices.

Security depends on Apple's deprecated sandbox-exec/Seatbelt facility and the
installed Python/system runtime being trusted. Runtime files and their ancestor
directory names are readable; project files and network/process creation are
denied. This is not a VM and does not claim protection from OS sandbox exploits.
Memory uses host RSS polling (10ms, so short allocation peaks can occur) plus
resource limits where supported. Darwin data/address-space limits can be
unavailable; host memory/wall/output enforcement is mandatory.
Kernel child CPU includes interpreter startup and RPC conversion. Engine timing
includes host metadata and IPC overhead, and remains the scoring feasibility gate.
"""
from __future__ import annotations

import ctypes
import json
import math
import os
from pathlib import Path
import secrets
import selectors
import signal
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time

from cache_sim.policies import Policy
from .scoring import TOTAL_EXTRA

BLOCK_FIELDS = ("depth", "inserted_at", "last_access", "frequency", "insertion_order", "access_order")
TASK_FIELDS = ("task_chat", "task_qa", "task_unknown", "turn_index")


class CodePolicyError(RuntimeError):
    """An executable candidate or its required sandbox failed."""


def _python_runtime():
    executable = Path(sys.executable).resolve()
    base = Path(sys.base_prefix).resolve()
    app = base / "Resources/Python.app/Contents/MacOS/Python"
    # macOS framework launchers spawn the real interpreter. Launch it directly
    # so process-fork can remain denied, including during initialization.
    if app.is_file():
        executable = app.resolve()
    stdlib = Path(sysconfig.get_path("stdlib")).resolve()
    paths = {str(stdlib), "/System/Library", "/usr/lib", str(base / "lib")}
    if base.name.startswith("3.") and "Python.framework" in str(base):
        paths.add(str(base))
    return executable, paths


def _sandbox_profile(executable, runtime_paths, worker):
    # Runtime ancestors need metadata for realpath(); Darwin startup also
    # requires read permission on the filesystem root itself.
    readable = {str(executable), str(worker), "/", "/dev/null", "/dev/urandom", "/dev/random"}
    ancestors = set()
    for path in [executable, worker, *map(Path, runtime_paths)]:
        ancestors.update(str(parent) for parent in path.parents)
    q = json.dumps
    filters = ["(subpath " + q(p) + ")" for p in sorted(runtime_paths)]
    filters += ["(literal " + q(p) + ")" for p in sorted(readable)]
    return ("(version 1)(deny default)"
            "(allow process-exec (literal " + q(str(executable)) + "))"
            "(allow file-read* " + " ".join(filters) + ")"
            "(allow file-read-metadata " + " ".join("(literal " + q(p) + ")" for p in sorted(ancestors)) + ")"
            "(allow sysctl-read)"
            '(allow mach-lookup (global-name "com.apple.system.notification_center"))')


class _TaskInfo(ctypes.Structure):
    _fields_ = [("virtual_size", ctypes.c_uint64), ("resident_size", ctypes.c_uint64),
                ("total_user", ctypes.c_uint64), ("total_system", ctypes.c_uint64),
                ("threads_user", ctypes.c_uint64), ("threads_system", ctypes.c_uint64),
                *[(name, ctypes.c_int32) for name in ("policy", "faults", "pageins", "cow_faults",
                    "messages_sent", "messages_received", "syscalls_mach", "syscalls_unix",
                    "csw", "threadnum", "numrunning", "priority")]]


def _resident_reader():
    library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    function = library.proc_pidinfo
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
    function.restype = ctypes.c_int
    def read(pid):
        info = _TaskInfo()
        count = function(pid, 4, 0, ctypes.byref(info), ctypes.sizeof(info))
        if count != ctypes.sizeof(info):
            return None
        return info.resident_size
    return read


class SandboxProgram:
    """Persistent sandboxed class instance with plain JSON method RPC.

    Construct a fresh instance per replay/episode. cpu_time_ns is authoritative
    kernel CPU after close; reported_cpu_time_ns during RPC is only diagnostic
    because generated code shares the worker's interpreter.
    """
    def __init__(self, source, class_name="CachePolicy", init_kwargs=None, *,
                 policy_features="block-v1", wall_timeout=2.0,
                 memory_bytes=256 * 1024 * 1024, max_output_bytes=65536,
                 total_timeout=120.0):
        self._process = None
        self._temp = None
        self._selector = None
        self.cpu_time_ns = 0
        self.reported_cpu_time_ns = 0
        self._closed = False
        self._failure = None
        self._watchdog = None
        self._stop = threading.Event()
        if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").is_file():
            raise CodePolicyError("Required macOS sandbox-exec is unavailable; refusing unsandboxed execution")
        if not isinstance(source, str) or not 0 < len(source.encode()) <= 65536:
            raise ValueError("Python policy source must contain 1–65536 bytes")
        if class_name not in {"CachePolicy", "ExplorationController"}:
            raise ValueError("Unsupported generated class")
        if policy_features not in {"block-v1", "task-v1"}:
            raise ValueError("Unknown policy feature set")
        if (not math.isfinite(wall_timeout) or wall_timeout <= 0 or
                not math.isfinite(total_timeout) or total_timeout <= 0 or
                type(memory_bytes) is not int or memory_bytes < 16 * 1024 * 1024 or
                type(max_output_bytes) is not int or max_output_bytes < 128):
            raise ValueError("Invalid sandbox resource limit")
        self.wall_timeout = wall_timeout
        self.memory_bytes = memory_bytes
        self.max_output_bytes = max_output_bytes
        self._deadline = time.monotonic() + total_timeout
        self._buffer = bytearray()
        try:
            self._rss = _resident_reader()
            self._temp = tempfile.TemporaryDirectory(prefix="dream-code-")
            worker = Path(self._temp.name).resolve() / "worker.py"
            worker.write_bytes(Path(__file__).with_name("code_worker.py").read_bytes())
            executable, runtime = _python_runtime()
            profile = _sandbox_profile(executable, runtime, worker)
            self._process = subprocess.Popen(
                ["/usr/bin/sandbox-exec", "-p", profile, str(executable), "-I", "-S", "-B", "-u", str(worker)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                cwd=worker.parent, env={"LANG": "C", "LC_ALL": "C"}, close_fds=True, start_new_session=True)
            os.set_blocking(self._process.stdin.fileno(), False)
            os.set_blocking(self._process.stdout.fileno(), False)
            self._selector = selectors.DefaultSelector()
            self._selector.register(self._process.stdout, selectors.EVENT_READ)
            self._watchdog = threading.Thread(target=self._monitor, daemon=True)
            self._watchdog.start()
            self._rpc({"op": "init", "source": source, "class_name": class_name,
                       "init_kwargs": init_kwargs or {}, "policy_features": policy_features,
                       "memory_bytes": memory_bytes})
            if self._rss(self._process.pid) is None:
                raise CodePolicyError("Required worker memory accounting is unavailable")
        except BaseException:
            self.close()
            raise

    def _monitor(self):
        # Active between RPCs too: background candidate threads cannot evade
        # memory/lifetime caps while the trusted host replays cache accesses.
        while not self._stop.wait(.01):
            rss = self._rss(self._process.pid)
            reason = None
            if time.monotonic() >= self._deadline:
                reason = "Policy total wall-time limit exceeded"
            elif rss is None:
                reason = "Policy memory accounting unavailable or worker exited"
            elif rss > self.memory_bytes:
                reason = "Policy resident-memory limit exceeded"
            if reason:
                self._failure = reason
                try:
                    os.killpg(self._process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    try:
                        os.kill(self._process.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass  # Darwin returns EPERM for an already dead child.
                return

    def _check(self, deadline):
        if self._failure:
            raise CodePolicyError(self._failure)
        if time.monotonic() >= deadline:
            raise CodePolicyError("Policy wall-time limit exceeded")
        rss = self._rss(self._process.pid)
        if rss is None:
            raise CodePolicyError("Policy memory accounting unavailable or worker exited")
        if rss > self.memory_bytes:
            raise CodePolicyError("Policy resident-memory limit exceeded")

    def _rpc(self, message):
        if self._closed:
            raise CodePolicyError("Policy worker is closed")
        deadline = min(time.monotonic() + self.wall_timeout, self._deadline)
        try:
            payload = json.dumps(message, allow_nan=False, separators=(",", ":")).encode() + b"\n"
            # Nonblocking write also bounds a worker that refuses to read input.
            offset = 0
            while offset < len(payload):
                self._check(deadline)
                try:
                    offset += os.write(self._process.stdin.fileno(), payload[offset:])
                except BlockingIOError:
                    self._selector.select(.01)
            while b"\n" not in self._buffer:
                self._check(deadline)
                for key, _ in self._selector.select(min(.01, max(0, deadline - time.monotonic()))):
                    chunk = os.read(key.fd, min(65536, self.max_output_bytes + 1))
                    if not chunk:
                        raise CodePolicyError("Sandbox worker exited or sandbox initialization failed")
                    self._buffer.extend(chunk)
                    if len(self._buffer) > self.max_output_bytes:
                        raise CodePolicyError("Policy output limit exceeded")
            line, _, remainder = self._buffer.partition(b"\n")
            self._buffer = bytearray(remainder)
            response = json.loads(line)
            if not isinstance(response, dict) or response.get("ok") is not True:
                raise CodePolicyError(str(response.get("error", "Invalid policy response")) if isinstance(response, dict)
                                      else "Invalid policy response")
            cpu = response.get("cpu_ns", 0)
            if type(cpu) is int and cpu >= 0:
                self.reported_cpu_time_ns += cpu
            return response.get("result")
        except BaseException as exc:
            self.close()
            if isinstance(exc, (CodePolicyError, KeyboardInterrupt, SystemExit)):
                raise
            raise CodePolicyError("Policy RPC failed: " + type(exc).__name__) from exc

    def call(self, method, args=None, kwargs=None):
        if not isinstance(method, str) or method.startswith("_"):
            raise ValueError("RPC requires a public method name")
        return self._rpc({"op": "call", "method": method, "args": args or [], "kwargs": kwargs or {}})

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._watchdog is not None:
            self._watchdog.join(timeout=.1)
        if self._process is not None:
            process = self._process
            if process.stdin:
                process.stdin.close()
            try:
                # Child process creation is denied; group kill also covers a
                # launcher on runtimes where initialization never succeeded.
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                try:
                    os.kill(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass  # Already terminated by the watchdog.
            try:
                _, status, usage = os.wait4(process.pid, 0)
                process.returncode = os.waitstatus_to_exitcode(status)
                self.cpu_time_ns = round((usage.ru_utime + usage.ru_stime) * 1e9)
            except ChildProcessError:
                pass
            if process.stdout:
                process.stdout.close()
        if self._selector is not None:
            self._selector.close()
        if self._temp is not None:
            self._temp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()


class CodePolicy(Policy):
    """Policy proxy; hooks are buffered in order until choose()/flush()."""
    name = "executable-python"

    def __init__(self, source, policy_features="block-v1", **limits):
        self._events = []
        self._handles = {}
        self._features = policy_features
        self._worker = SandboxProgram(source, policy_features=policy_features, **limits)

    @property
    def cpu_time_ns(self):
        return self._worker.cpu_time_ns

    def _entry(self, entry):
        handle = self._handles.get(entry.block_id)
        if handle is None:
            handle = self._handles[entry.block_id] = secrets.token_hex(16)
        fields = BLOCK_FIELDS + (TASK_FIELDS if self._features == "task-v1" else ())
        return {"block_id": handle, **{field: getattr(entry, field) for field in fields}}

    def _observe(self, method, entry):
        self._events.append({"method": method, "entry": self._entry(entry)})
        if len(self._events) >= 1024:
            self.flush()

    def observe_insert(self, entry):
        self._observe("observe_insert", entry)

    def observe_hit(self, entry):
        self._observe("observe_hit", entry)

    def observe_evict(self, entry):
        self._observe("observe_evict", entry)

    def choose(self, eligible, now):
        entries = [self._entry(entry) for entry in eligible]
        events, self._events = self._events, []
        token = self._worker._rpc({"op": "cache", "events": events, "eligible": entries, "now": now})
        by_handle = {e["block_id"]: original.block_id for e, original in zip(entries, eligible)}
        if not isinstance(token, str) or token not in by_handle:
            self._worker.close()
            raise CodePolicyError("Policy returned an illegal victim; choose must return an eligible opaque block_id")
        return by_handle[token]

    def flush(self):
        if self._events:
            events, self._events = self._events, []
            self._worker._rpc({"op": "cache", "events": events})

    def close(self):
        self._worker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def evaluate_code_suite(suite, block_size, *, candidate=None, baselines=None,
                        max_policy_us_per_request=50000, scoring=TOTAL_EXTRA,
                        policy_features="block-v1", **limits):
    """Trusted replay driver; the generated child receives only cache RPCs.

    Like evaluation.run_suite, replay errors propagate to the caller. Callers
    that record failed candidates should catch CodePolicyError/ValueError.
    """
    from cache_sim.engine import replay
    from cache_sim.trace import compile_trace, load_trace, synthetic_trace
    from .evaluation import digest, run_suite
    from .programs import metadata
    from .scoring import score_recomputation
    if candidate is None:
        return run_suite(suite, block_size, scoring=scoring, policy_features=policy_features)
    metadata(candidate)
    if set(candidate) - {"name", "source", "rationale"}:
        raise ValueError("Unexpected executable policy fields")
    if baselines is None or len(baselines) != len(suite):
        raise ValueError("Executable evaluation requires matching baselines")
    runs = []
    for index, item in enumerate(suite):
        reference = baselines[index]
        if reference["name"] != item["name"]:
            raise ValueError("Baseline/scenario mismatch")
        if "path" in item:
            if digest(item["path"]) != item["sha256"]:
                raise ValueError("Trace changed after suite snapshot")
            trace = load_trace(Path(item["path"]))
        else:
            trace = synthetic_trace(**item["synthetic"])
        compiled = compile_trace(trace, block_size)
        if not compiled.requests:
            raise ValueError("Empty traces cannot be scored")
        with CodePolicy(candidate.get("source"), policy_features, **limits) as policy:
            policy.name = candidate["name"]
            result = replay(compiled, item["capacity_blocks"], policy).to_dict()
            started = time.perf_counter_ns()
            policy.flush()  # Validate final hooks even when no eviction occurred.
            flush_ns = time.perf_counter_ns() - started
        result["policy_time_ns"] += flush_ns
        result["policy_max_operation_ns"] = max(result["policy_max_operation_ns"], flush_ns)
        result["policy_time_ms"] = result["policy_time_ns"] / 1e6
        result["policy_us_per_request"] = result["policy_time_ns"] / result["requests"] / 1e3
        result["worker_cpu_time_ns"] = policy.cpu_time_ns
        result["worker_cpu_us_per_request"] = policy.cpu_time_ns / result["requests"] / 1e3
        result["policy_timing_note"] = "Host engine timing includes metadata and JSONL IPC; child CPU includes Python startup."
        extra = result["computed_prompt_tokens"] - reference["unlimited"]["computed_prompt_tokens"]
        lru_extra = reference["policies"]["lru"]["extra_computed_tokens"]
        result.update(name=item["name"], extra_computed_tokens=extra,
                      lru_extra_computed_tokens=lru_extra, saved_prompt_tokens=lru_extra - extra,
                      improvement=(lru_extra - extra) / max(1, lru_extra))
        runs.append(result)
    metrics = score_recomputation([r["extra_computed_tokens"] for r in runs],
                                  [r["lru_extra_computed_tokens"] for r in runs], scoring)
    for result, contribution in zip(runs, metrics.pop("score_contributions")):
        result["score_contribution"] = contribution
    within_budget = all(max(r["policy_us_per_request"], r["worker_cpu_us_per_request"])
                        <= max_policy_us_per_request for r in runs)
    return {"valid": within_budget, **metrics,
            "error": None if within_budget else "Policy CPU/host time exceeds configured per-request limit",
            "runs": runs}
