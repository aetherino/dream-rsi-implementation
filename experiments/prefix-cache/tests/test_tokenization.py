from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from cache_sim.trace import compile_trace, load_trace
from scripts.tokenize_datasets import Shards, chat_jobs, encode_pairs, mash_jobs, normalize_chat, split_for


class CharacterTokenizer:
    def encode(self, messages, generation=False):
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        if generation:
            text += "<assistant>"
        return list(map(ord, text))


class TokenizationTests(unittest.TestCase):
    def test_normalization_and_incomplete_rejection(self):
        messages = normalize_chat({"conversations": [
            {"from": "human", "value": "question"}, {"from": "gpt", "value": "answer"}]})
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant"])
        for raw in [[], [{"from": "human", "value": "q"}],
                    [{"from": "tool", "value": "q"}],
                    [{"from": "human", "value": "q"}, {"from": "human", "value": "q2"}]]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                normalize_chat({"conversations": raw})

    def test_full_prompt_and_answer_share_prefix(self):
        messages = [{"role": "system", "content": "system"},
                    {"role": "user", "content": "one"},
                    {"role": "assistant", "content": "two"},
                    {"role": "user", "content": "three"},
                    {"role": "assistant", "content": "four"}]
        pairs = encode_pairs(CharacterTokenizer(), messages, 1000)
        prior = pairs[0][0] + pairs[0][1]
        self.assertEqual(pairs[1][0][:len(prior)], prior)
        self.assertEqual(pairs[-1][0] + pairs[-1][1], CharacterTokenizer().encode(messages))
        with self.assertRaisesRegex(ValueError, "over_context"):
            encode_pairs(CharacterTokenizer(), messages, 5)

    def test_template_mismatch_rejected(self):
        class Bad:
            def encode(self, messages, generation=False):
                return [1] if generation else [2]
        with self.assertRaisesRegex(ValueError, "prefix_mismatch"):
            encode_pairs(Bad(), [{}, {}, {}], 100)

    def test_chat_dedup_and_first_user_split(self):
        rows = [{"id": str(i), "conversations": [
            {"from": "human", "value": "shared question"},
            {"from": "gpt", "value": answer}]} for i, answer in enumerate(["a", "a", "b"])]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.json"
            path.write_text(json.dumps(rows))
            stats = Counter()
            jobs = list(chat_jobs(path, 42, 0, 1000, stats))
            self.assertEqual(len(jobs), 2)
            self.assertEqual(stats["skip_exact_duplicate"], 1)
            self.assertEqual(jobs[0]["split"], jobs[1]["split"])
            self.assertEqual(jobs[0]["split"], split_for("shared question", 42))

    def test_mashqa_document_overlap_favors_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "val", "test"):
                docs = ["shared", split]
                data = {"data": [{"paragraphs": [{"context": d, "qas": [
                    {"id": f"{split}-{d}", "question": "q", "answers": [{"text": "a"}]}]}]} for d in docs]}
                (root / f"{split}_webmd_squad_v2_full.json").write_text(json.dumps(data))
            stats = Counter()
            jobs = list(mash_jobs(root, 0, 1000, stats))
            self.assertEqual(len(jobs), 4)
            self.assertEqual([j["split"] for j in jobs if j["document"] == "shared"], ["test"])
            self.assertEqual(stats["skip_cross_split_document_records"], 2)

    def test_shards_roundtrip_keep_groups_and_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = Shards(root, 42, 2, 30)
            for i in range(4):
                writer.add({"group_id": str(i), "split": "train", "workload": "sharegpt",
                            "source_ids": [str(i)], "pairs": [([1, 2], [3, 4]), ([1, 2, 3, 4, 5], [6])]})
            writer.finish()
            self.assertEqual(len(writer.files), 2)
            seen = set()
            for entry in writer.files:
                path = root / entry["path"]
                trace = load_trace(path)
                compile_trace(trace, 2)
                groups = set(r.session_id for r in trace.requests)
                self.assertFalse(seen & groups)
                seen.update(groups)
                for group in groups:
                    requests = [r for r in trace.requests if r.session_id == group]
                    self.assertEqual(len(requests), 2)
                    self.assertLess(requests[0].arrival_time, requests[1].arrival_time)
            self.assertEqual(seen, {"0", "1", "2", "3"})


if __name__ == "__main__":
    unittest.main()
