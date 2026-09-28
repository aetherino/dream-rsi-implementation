"""No model calls: complete history, executable schemas and durable transport tests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests

from dream_rsi.backend import write_json
from dream_rsi.code_prompts import controller_messages, discovery_messages
from dream_rsi.code_transport import (CodeTransport, MAX_INPUT_ALLOWANCE, normalize_result,
                                      parse_program, request_body, source_allowance)


def node(index):
    return {"id": index, "parent": index - 1 if index else None, "depth": index,
            "score": index / 100, "valid": index % 3 != 0,
            "candidate": {"name": f"candidate-{index}", "source": f"class CachePolicy:\n    marker = {index}\n",
                          "rationale": f"unique rationale {index}"},
            "feedback": {"error": f"unique error {index}", "runs": [{"scenario": f"scenario-{index}",
                         "measured_time": index + .1, "score_contribution": index / 1000}],
                         "full_saved_score_field": f"diagnostics-{index}"}}


def payload(messages):
    return json.loads(messages[1]["content"].split("\nFULL_HISTORY_JSON\n", 1)[1])


def program(source="class CachePolicy:\n    pass\n"):
    return {"name": "generated", "source": source, "rationale": "hypothesis"}


def reply(spec=None):
    response = Mock(status_code=200)
    response.json.return_value = {"usage": {"prompt_tokens": 123, "completion_tokens": 45},
                                 "choices": [{"finish_reason": "stop", "message": {
                                     "content": json.dumps(spec or program()), "reasoning_content": "private-reasoning"}}]}
    return response


class CodePromptTests(unittest.TestCase):
    def test_discovery_preserves_every_proposal_and_full_saved_diagnostics(self):
        prior = [[node(i) for i in range(25)], [node(i) for i in range(25, 50)]]
        current = [node(i) for i in range(50, 80)]
        messages = discovery_messages(current[-1], current, {"scenario": "bounded"}, history=prior,
                                      prior_rollouts=[{"marker": "must not duplicate"}], best=prior[0][2])
        data = payload(messages)
        self.assertEqual(data["completed_histories"], prior)
        self.assertEqual(data["current_observations"], current)
        self.assertNotIn("must not duplicate", messages[1]["content"])
        for record in [*prior[0], *prior[1], *current]:
            self.assertIn(record["candidate"]["source"], json.dumps(data, ensure_ascii=False).replace("\\n", "\n"))
            self.assertIn(record["candidate"]["rationale"], messages[1]["content"])
            self.assertIn(record["feedback"]["error"], messages[1]["content"])
            self.assertIn(record["feedback"]["full_saved_score_field"], messages[1]["content"])
        self.assertEqual(data["selected_parent"], {"id": 79, "full_record_path": ["current_observations", 29]})

    def test_baseline_context_uses_existing_full_record_once(self):
        baseline = node(0)
        data = payload(discovery_messages(baseline, [baseline], {}, best=baseline))
        self.assertEqual(data["selected_parent"]["full_record_path"], ["current_observations", 0])
        self.assertEqual(data["best_ever_training_policy"]["full_record_path"], ["current_observations", 0])
        self.assertEqual(json.dumps(data).count(baseline["candidate"]["rationale"]), 1)

    def test_policy_feature_contract_and_hook_signatures(self):
        block = discovery_messages(node(0), [node(0)], {})[1]["content"]
        self.assertIn("block-v1 does not expose task flags", block)
        self.assertIn("observe_insert(self, entry)", block)
        self.assertIn("choose(self, eligible, now)", block)
        task = discovery_messages(node(0), [node(0)], {}, policy_features="task-v1")[1]["content"]
        self.assertIn("task_chat, task_qa, task_unknown, turn_index", task)
        self.assertIn("no final-turn", task)
        with self.assertRaisesRegex(ValueError, "feature"):
            discovery_messages(node(0), [], {}, policy_features="arbitrary")

    def test_controller_preserves_sequential_current_base_and_all_revisions(self):
        latest = program("class ExplorationController:\n    version = 9\n")
        revisions = [{"controller": program(f"class ExplorationController:\n    version = {i}\n"),
                      "accepted": i % 2 == 0, "evaluation": {"value": i / 10, "error": f"revision-error-{i}"},
                      "parent_version": i - 1} for i in range(9)]
        histories = [[node(i) for i in range(20)]]
        settings = {"workers": 4, "max_depth": 5, "beta_calls": .01, "beta_parallel": .03}
        data = payload(controller_messages(latest, {"value": .9}, histories, settings, revisions, online_rounds=7))
        self.assertEqual(data["current_version"], latest)
        text = controller_messages(latest, {}, histories, settings)[1]["content"]
        self.assertIn("Root is always legal regardless of max_depth", text)
        self.assertIn("including a worse-scoring", text)
        self.assertNotIn("Acceptance requires", text)
        self.assertEqual(data["previous_revisions"], revisions)
        self.assertEqual(data["completed_histories"], histories)
        self.assertEqual(data["deployment"]["online_rounds_per_cycle"], 7)
        with self.assertRaisesRegex(ValueError, "beta_calls"):
            controller_messages(latest, {}, histories, {})

    def test_bounds_reject_instead_of_truncating_history(self):
        records = [node(i) for i in range(5)]
        records[0]["candidate"]["source"] += "x" * MAX_INPUT_ALLOWANCE
        with self.assertRaisesRegex(ValueError, "no history was truncated"):
            discovery_messages(records[-1], records, {})
        self.assertEqual(len(records[0]["candidate"]["source"]), MAX_INPUT_ALLOWANCE + len(node(0)["candidate"]["source"]))
        with self.assertRaisesRegex(ValueError, str(MAX_INPUT_ALLOWANCE)):
            source_allowance("small", MAX_INPUT_ALLOWANCE + 1)
        messages = [{"role": "user", "content": "é" * 100}]
        self.assertEqual(source_allowance(messages), len(json.dumps(messages, ensure_ascii=False).encode()) + 4096)

    def test_request_body_accepts_messages_and_configurable_output(self):
        messages = discovery_messages(node(0), [node(0)], {})
        body = request_body({"model": "model", "thinking": "enabled", "max_completion_tokens": 16000}, messages)
        self.assertEqual(body["messages"], messages)
        self.assertEqual(body["max_completion_tokens"], 16000)
        self.assertEqual(request_body({"model": "model"}, "prompt")["max_completion_tokens"], 8192)
        with self.assertRaisesRegex(ValueError, "positive"):
            request_body({"model": "model", "max_completion_tokens": 0}, "prompt")

    def test_parser_accepts_large_source_and_rejects_nonpublic_fields(self):
        spec = program("class CachePolicy:\n    pass\n" + "# source\n" * 4000)
        self.assertGreater(len(json.dumps(spec)), 20000)
        self.assertEqual(parse_program(json.dumps(spec)), spec)
        with self.assertRaisesRegex(ValueError, "metadata"):
            parse_program(json.dumps(program("é" * 32769)))
        with self.assertRaisesRegex(ValueError, "exactly"):
            parse_program(json.dumps(spec | {"reasoning": "secret"}))
        with self.assertRaisesRegex(ValueError, "oversized"):
            parse_program(json.dumps(spec), max_response_bytes=100)

    def test_normalization_preserves_usage_on_invalid_completion(self):
        response = reply()
        response.json.return_value["choices"][0]["finish_reason"] = "length"
        result = normalize_result(response.json())
        self.assertIsNone(result["program"])
        self.assertEqual(result["usage"], {"prompt_tokens": 123, "completion_tokens": 45})
        self.assertIn("Invalid", result["error"])


class CodeTransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.key_patch = patch("dream_rsi.code_transport.backend.api_key", return_value="test-secret")
        self.key_patch.start()
        self.transport = CodeTransport(self.root, self.root / "requests")
        self.body = request_body({"model": "test-model"}, "entire saved history")

    def tearDown(self):
        self.key_patch.stop()
        self.temp.cleanup()

    def test_saved_results_are_reused_and_never_store_reasoning_or_key(self):
        jobs = [{"id": f"trial-{i}", "body": self.body} for i in range(4)]
        with patch("dream_rsi.code_transport.requests.post", return_value=reply()) as call:
            results = self.transport.poll(1, jobs)
            self.assertEqual(call.call_count, 4)
            self.assertTrue(all(c.kwargs["allow_redirects"] is False for c in call.call_args_list))
            self.assertEqual(self.transport.poll(1, jobs), results)
            self.assertEqual(call.call_count, 4)
        saved = "".join(p.read_text() for p in self.root.rglob("*.json"))
        self.assertNotIn("test-secret", saved)
        self.assertNotIn("private-reasoning", saved)
        self.assertEqual(results["trial-0"]["program"], program())

    def test_before_http_journal_already_marks_started(self):
        def accept(*args, **kwargs):
            state = json.loads((self.root / "requests/a/state.json").read_text())
            self.assertEqual(state, {"phase": "started", "body": self.body})
            return reply()
        with patch("dream_rsi.code_transport.requests.post", side_effect=accept) as call:
            self.transport.poll(1, [{"id": "a", "body": self.body}])
            call.assert_called_once()

    def test_unknown_journal_is_never_resent(self):
        write_json(self.root / "requests/a/state.json", {"phase": "started", "body": self.body})
        with patch("dream_rsi.code_transport.requests.post") as call:
            result = self.transport.poll(1, [{"id": "a", "body": self.body}])["a"]
            call.assert_not_called()
            self.assertEqual(self.transport.poll(2, [{"id": "a", "body": self.body}])["a"], result)
        self.assertIn("outcome unknown", result["error"])
        self.assertFalse(result["explicit_failure"])

    def test_network_failure_has_one_attempt_and_safe_error(self):
        with patch("dream_rsi.code_transport.requests.post", side_effect=requests.Timeout("test-secret")) as call:
            result = self.transport.poll(1, [{"id": "a", "body": self.body}])
            self.assertEqual(self.transport.poll(2, [{"id": "a", "body": self.body}]), result)
            call.assert_called_once()
        self.assertNotIn("test-secret", result["a"]["error"])
        self.assertFalse(result["a"]["explicit_failure"])

    def test_all_wave_bounds_validated_before_any_http(self):
        huge = request_body({"model": "model"}, "small")
        huge["messages"][1]["content"] = "x" * MAX_INPUT_ALLOWANCE
        with patch("dream_rsi.code_transport.requests.post") as call, self.assertRaisesRegex(ValueError, "truncated"):
            self.transport.poll(1, [{"id": "valid", "body": self.body}, {"id": "too-large", "body": huge}])
        call.assert_not_called()

    def test_duplicate_ids_unsafe_ids_and_body_changes_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.transport.poll(1, [{"id": "a", "body": self.body}] * 2)
        with self.assertRaisesRegex(ValueError, "Invalid request ID"):
            self.transport.poll(1, [{"id": "../unsafe", "body": self.body}])
        write_json(self.root / "requests/a/state.json", {"phase": "started", "body": self.body})
        changed = request_body({"model": "model"}, "different")
        with patch("dream_rsi.code_transport.requests.post") as call, self.assertRaisesRegex(ValueError, "changed"):
            self.transport.poll(1, [{"id": "a", "body": changed}])
        call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
