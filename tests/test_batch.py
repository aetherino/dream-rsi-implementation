import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests

from dream_rsi.backend import write_json
from dream_rsi.batch_compare import initialize, tick, reserve, apply_results, bounded_discovery, controller_history
from dream_rsi.batch_transport import BatchTransport, BatchHTTPError, normalize_result, allowance
from dream_rsi.compare import compare
from dream_rsi.history import root
from dream_rsi.programs import LRU_SPEC


BASE = "https://batch-api-cn.xiaomimimo.com/v1"


def response(data, status=200, text=None):
    value = Mock(status_code=status)
    value.json.return_value = data
    value.text = text if text is not None else json.dumps(data)
    return value


def result(custom_id, program=LRU_SPEC, finish="stop", prompt_tokens=20):
    return {"custom_id": custom_id, "response": {"status_code": 200, "body": {
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 10},
        "choices": [{"finish_reason": finish, "message": {"content": json.dumps(program),
                                                          "reasoning_content": "do not persist this reasoning"}}]}}, "error": None}


class TransportTests(unittest.TestCase):
    def test_real_batch_payload_out_of_order_results_and_restart(self):
        jobs = [{"id": x, "body": {"model": "mimo-v2.6-pro", "messages": []}} for x in ("a", "b", "c")]
        with tempfile.TemporaryDirectory() as temp, patch.dict("os.environ", {"XIAOMI_API": "test-secret"}):
            path = Path(temp)
            api = BatchTransport(BASE, path, path / "batches")
            replies = [response({"id": "file-in"}), response({"id": "batch-1"}), response({"status": "in_progress"})]
            with patch("dream_rsi.batch_transport.requests.request", side_effect=replies) as call:
                self.assertIsNone(api.poll(1, jobs))
                self.assertEqual([c.args[0] for c in call.call_args_list], ["POST", "POST", "GET"])
                self.assertEqual(call.call_args_list[1].kwargs["json"]["input_file_id"], "file-in")
                self.assertTrue(all(c.kwargs["allow_redirects"] is False for c in call.call_args_list))
            api = BatchTransport(BASE, path, path / "batches")
            data = "\n".join(json.dumps(r) for r in (result("b", {"name": "b"}), result("a")))
            failure = {"custom_id": "c", "response": None, "error": {"code": "inference_failed"}}
            replies = [response({"status": "completed", "output_file_id": "file-out", "error_file_id": "file-err"}),
                       response(None, text=data), response(None, text=json.dumps(failure))]
            with patch("dream_rsi.batch_transport.requests.request", side_effect=replies) as call:
                values = api.poll(1, jobs)
                self.assertTrue(all(c.args[0] == "GET" for c in call.call_args_list))
            self.assertEqual(values["a"]["program"], LRU_SPEC)
            self.assertEqual(values["b"]["program"]["name"], "b")
            self.assertTrue(values["c"]["explicit_failure"])
            with patch("dream_rsi.batch_transport.requests.request") as call:
                self.assertEqual(api.poll(1, jobs), values)
                call.assert_not_called()
            saved = "".join(p.read_text() for p in path.rglob("*") if p.is_file())
            self.assertNotIn("test-secret", saved)
            self.assertNotIn("do not persist this reasoning", saved)

    def test_unknown_submission_is_not_retried_and_can_be_attached(self):
        jobs = [{"id": "a", "body": {}}]
        with tempfile.TemporaryDirectory() as temp, patch.dict("os.environ", {"XIAOMI_API": "test"}):
            path = Path(temp)
            api = BatchTransport(BASE, path, path / "batches")
            with patch("dream_rsi.batch_transport.requests.request", side_effect=[response({"id": "file-in"}), requests.Timeout()]):
                with self.assertRaises(RuntimeError):
                    api.poll(1, jobs)
            with patch("dream_rsi.batch_transport.requests.request") as call:
                with self.assertRaisesRegex(RuntimeError, "unknown"):
                    api.poll(1, jobs)
                call.assert_not_called()
            with patch("dream_rsi.batch_transport.requests.request", return_value=response({"input_file_id": "different"})):
                with self.assertRaisesRegex(ValueError, "different"):
                    api.poll(1, jobs, attach_batch="batch-1")
            with patch("dream_rsi.batch_transport.requests.request", side_effect=[response({"input_file_id": "file-in"}), response({"status": "in_progress"})]) as call:
                self.assertIsNone(api.poll(1, jobs, attach_batch="batch-1"))
                self.assertTrue(all(c.args[0] == "GET" for c in call.call_args_list))

    def test_rejection_can_be_retried_only_by_later_resume(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict("os.environ", {"XIAOMI_API": "test"}):
            path = Path(temp)
            api = BatchTransport(BASE, path, path / "batches")
            with patch("dream_rsi.batch_transport.requests.request", side_effect=[response({"id": "file-in"}), response({"error": {"code": "insufficient_balance"}}, 403)]) as call:
                with self.assertRaises(BatchHTTPError):
                    api.poll(1, [{"id": "a", "body": {}}])
                self.assertEqual(call.call_count, 2)
            self.assertEqual(json.loads((path / "batches/wave-0001/state.json").read_text())["phase"], "uploaded")

    def test_duplicate_or_foreign_result_ids_fail(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict("os.environ", {"XIAOMI_API": "test"}):
            path = Path(temp)
            api = BatchTransport(BASE, path, path / "batches")
            replies = [response({"id": "file-in"}), response({"id": "batch-1"}),
                       response({"status": "completed", "output_file_id": "out"}),
                       response(None, text=json.dumps(result("foreign")))]
            with patch("dream_rsi.batch_transport.requests.request", side_effect=replies), self.assertRaisesRegex(ValueError, "custom_id"):
                api.poll(1, [{"id": "a", "body": {}}])

    def test_partial_expiration_keeps_success_and_marks_missing(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict("os.environ", {"XIAOMI_API": "test"}):
            path = Path(temp)
            api = BatchTransport(BASE, path, path / "batches")
            replies = [response({"id": "file-in"}), response({"id": "batch-1"}),
                       response({"status": "expired", "output_file_id": "out"}), response(None, text=json.dumps(result("a")))]
            with patch("dream_rsi.batch_transport.requests.request", side_effect=replies):
                values = api.poll(1, [{"id": x, "body": {}} for x in ("a", "b")])
            self.assertEqual(values["a"]["program"], LRU_SPEC)
            self.assertIn("expired", values["b"]["error"])
            self.assertFalse(values["b"]["explicit_failure"])

    def test_truncation_and_bad_json_still_return_usage(self):
        for row in (result("a", finish="length"), result("a", program=[])):
            value = normalize_result(row)
            self.assertIsNone(value["program"])
            self.assertTrue(value["error"])
            self.assertEqual(value["usage"]["prompt_tokens"], 20)

    def test_untrusted_endpoints_and_changed_requests_rejected(self):
        for url in ("http://batch-api-cn.xiaomimimo.com/v1", "https://evil.com/v1", "https://batch-api-cn.xiaomimimo.com.evil.com/v1"):
            with self.assertRaises(ValueError):
                BatchTransport(url, Path.cwd(), Path("unused"))


class CoordinatorTests(unittest.TestCase):
    def config(self):
        return {"trials": 1, "run": {"cycles": 3, "online_rounds": 2, "controller_revisions": 1,
            "api": {"max_calls": 8, "max_usd": .5, "input_usd_per_million": .2175, "output_usd_per_million": .435},
            "train": [{"name": "tiny", "synthetic": {"seed": 42, "sessions": 3}, "capacity_blocks": 32}],
            "validation": [{"name": "never-show-heldout", "synthetic": {"seed": 99, "sessions": 3}, "capacity_blocks": 32}]}}

    def test_restart_every_wave_matches_existing_runner(self):
        raw = self.config()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            output = path / "batch"
            output.mkdir()
            state = initialize(raw, "mock", Path.cwd())
            for _ in range(30):
                report = tick(state, output, Path.cwd())
                state = json.loads((output / "checkpoint.json").read_text())
                for run in state["runs"]:
                    observed = {n["id"] for n in run.get("tree", [])}
                    for job in run["pending"]:
                        self.assertNotIn("never-show-heldout", job["prompt"])
                        if job["role"] == "discovery":
                            self.assertIn(job["parent"], observed)
                if state["status"] == "completed":
                    break
            self.assertEqual(state["status"], "completed")
            direct = compare(raw, "mock", Path.cwd(), path / "direct")
            for actual, expected in zip(report["runs"], direct["runs"]):
                for key in ("arm", "calls", "discovery_calls", "controller_calls", "accepted_revisions", "best_policy", "train_score", "validation_score"):
                    self.assertEqual(actual[key], expected[key], key)
            self.assertEqual(report["aggregate"]["estimated_usd"], 0)

    def test_dollar_cap_prevents_submission(self):
        raw = self.config()
        raw["run"]["api"]["max_usd"] = .000001
        with tempfile.TemporaryDirectory() as temp:
            state = initialize(raw, "mimo-batch", Path.cwd())
            transport = Mock()
            tick(state, Path(temp), Path.cwd(), transport)
            self.assertEqual(state["status"], "completed")
            self.assertEqual(sum(len(r["records"]) for r in state["runs"]), 0)
            transport.poll.assert_not_called()

    def test_result_application_is_atomic_on_local_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            state = initialize(self.config(), "mimo-batch", Path.cwd())
            tick(state, output, Path.cwd())
            saved = copy.deepcopy(state)
            transport = Mock()
            transport.poll.return_value = {j["id"]: normalize_result(result(j["id"])) for r in state["runs"] for j in r["pending"]}
            with patch("dream_rsi.batch_compare.apply_results", side_effect=OSError("disk problem")), self.assertRaises(OSError):
                tick(state, output, Path.cwd(), transport)
            self.assertEqual(state, saved)
            self.assertEqual(json.loads((output / "checkpoint.json").read_text()), saved)

    def test_unknown_usage_retains_reservation_and_failure_costs_zero(self):
        run = initialize(self.config(), "mimo-batch", Path.cwd())["runs"][0]
        run.update(tree=[root()], best=root(), round=1)
        job = reserve(run, "discovery", "test", {"cycle": 1, "node": 1}, 0)
        amount = run["records"][0]["charged_estimate_usd"]
        run["pending"] = [job]
        apply_results(run, {job["id"]: {"program": None, "error": "missing", "usage": {}, "explicit_failure": False}})
        self.assertEqual(run["records"][0]["charged_estimate_usd"], amount)
        job = reserve(run, "discovery", "test", {"cycle": 1, "node": 2}, 0)
        run["pending"] = [job]
        apply_results(run, {job["id"]: {"program": None, "error": "failed", "usage": {}, "explicit_failure": True}})
        self.assertEqual(run["records"][1]["charged_estimate_usd"], 0)

    def test_prompt_growth_retains_parent_and_all_controller_outcomes(self):
        nodes = []
        for i in range(40):
            node = root() | {"id": i, "candidate": LRU_SPEC | {"rationale": "x" * 4000},
                            "feedback": {"runs": [{"name": "s" * 500, "extra_computed_tokens": 123} for _ in range(12)]}}
            nodes.append(node)
        prompt = bounded_discovery(nodes[-1], nodes, [{"name": "train"}], [nodes] * 10, "total_extra_v2")
        self.assertLess(allowance(prompt), 64000)
        self.assertIn('"selected_parent": {"id": 39', prompt)
        history = controller_history([nodes] * 10)
        self.assertEqual(sum(map(len, history)), 400)
        self.assertNotIn("rationale", json.dumps(history))

    def test_long_horizon_exercises_repeated_resume_and_call_limits(self):
        config = self.config()
        config["run"].update(cycles=10, online_rounds=3)
        config["run"]["api"]["max_calls"] = 40
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            state = initialize(config, "mock", Path.cwd())
            for _ in range(70):
                report = tick(state, output, Path.cwd())
                state = json.loads((output / "checkpoint.json").read_text())
                if state["status"] == "completed":
                    break
            self.assertEqual(state["status"], "completed")
            rows = {r["arm"]: r for r in report["runs"]}
            self.assertEqual(rows["fixed"]["calls"], 40)
            self.assertLessEqual(rows["adaptive"]["calls"], 40)
            for run in state["runs"]:
                ids = [r["id"] for r in run["records"]]
                self.assertEqual(len(ids), len(set(ids)))
                best = [r["best_train_score"] for r in run["progress"]]
                self.assertEqual(best, sorted(best))


if __name__ == "__main__":
    unittest.main()
