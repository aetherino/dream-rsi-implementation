"""Concurrent regular MiMo requests with persistent per-request outcomes."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re

import requests

from .backend import api_key, write_json
from .batch_transport import normalize_result


class RealtimeTransport:
    def __init__(self, project_root, directory, workers=4):
        self.key = api_key(project_root)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.workers = workers

    def _generate(self, job):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", job["id"]):
            raise ValueError("Invalid request ID")
        folder = self.directory / job["id"]
        folder.mkdir(exist_ok=True)
        path = folder / "state.json"
        unknown = {"program": None, "usage": {}, "explicit_failure": False,
                   "error": "Interrupted request; outcome unknown, reservation retained, no retry"}
        if path.exists():
            state = json.loads(path.read_text())
            if state["body"] != job["body"]:
                raise ValueError("Saved request body changed")
            if state["phase"] == "finished":
                return state["result"]
            # A crash after marking started can precede or follow server acceptance.
            # Charge the reserved attempt and never blindly send it again.
            state.update(phase="finished", result=unknown)
            write_json(path, state)
            return unknown
        state = {"phase": "started", "body": job["body"]}
        write_json(path, state)
        try:
            response = requests.post("https://api.xiaomimimo.com/v1/chat/completions",
                                     headers={"api-key": self.key, "Content-Type": "application/json"},
                                     json=job["body"], timeout=(10, 120), allow_redirects=False)
            if response.status_code != 200:
                result = unknown | {"error": f"MiMo HTTP {response.status_code}; no retry"}
            else:
                result = normalize_result({"response": {"status_code": 200, "body": response.json()}})
        except (requests.RequestException, ValueError) as exc:
            result = unknown | {"error": f"MiMo request failed ({type(exc).__name__}); no retry"}
        state.update(phase="finished", result=result)
        write_json(path, state)
        return result

    def poll(self, wave_number, jobs, attach_batch=None):
        if attach_batch:
            raise ValueError("Provider batch IDs do not apply to realtime requests")
        if len({j["id"] for j in jobs}) != len(jobs):
            raise ValueError("Duplicate request IDs")
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            results = list(pool.map(self._generate, jobs))
        return {job["id"]: result for job, result in zip(jobs, results)}
