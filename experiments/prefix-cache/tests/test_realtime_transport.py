import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests

from dream_rsi.backend import write_json
from dream_rsi.batch_compare import initialize, tick
from dream_rsi.realtime_transport import RealtimeTransport


class RealtimeTests(unittest.TestCase):
    def test_saved_results_are_reused_without_calls_or_secrets(self):
        jobs = [{"id": f"trial-{i}", "body": {"model": "mimo-v2.6-pro", "messages": []}} for i in range(6)]
        reply = Mock(status_code=200)
        reply.json.return_value = {"usage": {"prompt_tokens": 20, "completion_tokens": 10},
                                  "choices": [{"finish_reason": "stop", "message": {
                                      "content": '{"name":"ok"}', "reasoning_content": "hidden-reasoning"}}]}
        with tempfile.TemporaryDirectory() as temp, patch.dict("os.environ", {"XIAOMI_API": "test-secret"}):
            root = Path(temp)
            with patch("dream_rsi.realtime_transport.requests.post", return_value=reply) as call:
                results = RealtimeTransport(root, root / "requests").poll(1, jobs)
                self.assertEqual(call.call_count, 6)
                self.assertTrue(all(c.kwargs["allow_redirects"] is False for c in call.call_args_list))
            with patch("dream_rsi.realtime_transport.requests.post") as call:
                self.assertEqual(RealtimeTransport(root, root / "requests").poll(1, jobs), results)
                call.assert_not_called()
            saved = "".join(p.read_text() for p in root.rglob("*.json"))
            self.assertNotIn("test-secret", saved)
            self.assertNotIn("hidden-reasoning", saved)

    def test_interrupted_request_is_not_resent(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict("os.environ", {"XIAOMI_API": "test"}):
            root = Path(temp)
            transport = RealtimeTransport(root, root / "requests")
            write_json(root / "requests/a/state.json", {"phase": "started", "body": {}})
            with patch("dream_rsi.realtime_transport.requests.post") as call:
                result = transport.poll(1, [{"id": "a", "body": {}}])["a"]
                call.assert_not_called()
            self.assertIn("Interrupted", result["error"])
            self.assertFalse(result["explicit_failure"])

    def test_network_failures_are_persisted_without_retry(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict("os.environ", {"XIAOMI_API": "test"}):
            root = Path(temp)
            transport = RealtimeTransport(root, root / "requests")
            with patch("dream_rsi.realtime_transport.requests.post", side_effect=requests.Timeout()) as call:
                result = transport.poll(1, [{"id": "a", "body": {}}])
                self.assertEqual(transport.poll(1, [{"id": "a", "body": {}}]), result)
                self.assertEqual(call.call_count, 1)
            self.assertFalse(result["a"]["explicit_failure"])

    def test_duplicate_ids_and_body_changes_rejected(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict("os.environ", {"XIAOMI_API": "test"}):
            root = Path(temp)
            transport = RealtimeTransport(root, root / "requests")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                transport.poll(1, [{"id": "a", "body": {}}] * 2)
            write_json(root / "requests/a/state.json", {"phase": "started", "body": {"different": True}})
            with patch("dream_rsi.realtime_transport.requests.post") as call, self.assertRaisesRegex(ValueError, "changed"):
                transport.poll(1, [{"id": "a", "body": {}}])
            call.assert_not_called()

    def test_coordinator_uses_regular_rates_and_transport(self):
        config = {"trials": 1, "run": {"cycles": 2, "online_rounds": 1, "controller_revisions": 1,
                  "api": {"max_calls": 2},
                  "train": [{"name": "t", "synthetic": {"seed": 1, "sessions": 3}, "capacity_blocks": 32}],
                  "validation": [{"name": "v", "synthetic": {"seed": 2, "sessions": 3}, "capacity_blocks": 32}]}}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            state = initialize(config, "mimo", Path.cwd())
            tick(state, output, Path.cwd())
            expected = {j["id"]: {"program": None, "error": "test failure", "usage": {"prompt_tokens": 1000000, "completion_tokens": 0}}
                        for r in state["runs"] for j in r["pending"]}
            with patch("dream_rsi.batch_compare.RealtimeTransport") as transport:
                transport.return_value.poll.return_value = expected
                tick(state, output, Path.cwd())
                transport.assert_called_once()
            self.assertEqual(state["runs"][0]["records"][0]["charged_estimate_usd"], .435)


if __name__ == "__main__":
    unittest.main()
