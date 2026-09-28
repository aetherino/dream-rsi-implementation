from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cache_sim.engine import replay
from cache_sim.policies import Entry, LRU
from cache_sim.trace import Request, Trace, compile_trace, load_trace, save_trace
from dream_rsi.evaluation import digest, run_suite
from dream_rsi.history import root
from dream_rsi.mixed import compose, prepare, verify_prepared, write_json
from dream_rsi.mixed_experiment import final_test, freeze_selection
from dream_rsi.programs import RetentionPolicy, LRU_SPEC
from dream_rsi.prompts import discovery_prompt
from dream_rsi.runner import prepare_config


def pools():
    result = {'sharegpt': {}, 'mashqa': {}}
    for kind in result:
        for group in range(30):
            rows = []
            for turn in range(3):
                rows.append({'request_id': f'{kind}-{group}-{turn}', 'group_id': str(group),
                    'session_id': str(group) if kind == 'sharegpt' else f'{group}-{turn}',
                    'turn_index': turn, 'workload': kind, 'arrival_time': turn,
                    'prompt_token_ids': [1, 2] + [3] * turn, 'output_token_ids': [3]})
            result[kind][str(group)] = rows
    return result


class MixedTests(unittest.TestCase):
    def test_composition_whole_groups_order_and_shift(self):
        source = pools()
        unchanged = deepcopy(source)
        first = compose(source, 'test', 'shift', 42, requests=60)
        self.assertEqual(first, compose(source, 'test', 'shift', 42, requests=60))
        self.assertNotEqual(first, compose(source, 'test', 'shift', 43, requests=60))
        self.assertEqual(source, unchanged)
        rows = first['requests']
        self.assertEqual(len({r['request_id'] for r in rows}), len(rows))
        self.assertEqual([r['arrival_time'] for r in rows], sorted(r['arrival_time'] for r in rows))
        groups = {}
        for r in rows:
            groups.setdefault((r['workload'], r['group_id']), []).append(r)
        for (kind, _), group in groups.items():
            self.assertEqual(len(group), 3)
            self.assertEqual(len({r['admission_phase'] for r in group}), 1)
            if kind == 'sharegpt':
                self.assertEqual([r['turn_index'] for r in group], [0, 1, 2])
            else:
                self.assertEqual({r['turn_index'] for r in group}, {0})
        phases = first['composition']['phases']
        self.assertGreater(phases[0]['requests_by_task']['sharegpt'], phases[0]['requests_by_task']['mashqa'])
        self.assertLess(phases[1]['requests_by_task']['sharegpt'], phases[1]['requests_by_task']['mashqa'])

    def test_task_flags_latest_touch_and_no_future_knowledge(self):
        class Observe(LRU):
            seen = []
            def observe_insert(self, entry): self.seen.append(entry)
            def observe_hit(self, entry): self.seen.append(entry)
        past = [Request('a', 0, (1,), task_type='chat', turn_index=0),
                Request('b', 1, (1,), task_type='document-qa', turn_index=0)]
        p = Observe(); p.seen = []
        replay(compile_trace(Trace(tuple(past)), 1), 2, p)
        self.assertTrue(p.seen[0].task_chat)
        self.assertTrue(p.seen[1].task_qa)
        self.assertFalse(p.seen[1].task_chat)
        q = Observe(); q.seen = []
        replay(compile_trace(Trace(tuple(past + [Request('future', 100, (1, 2), task_type='chat', turn_index=8)])), 1), 2, q)
        self.assertEqual(p.seen, q.seen[:len(p.seen)])

    def test_feature_boundary_and_real_eviction_difference(self):
        spec = {'name': 'protect-chat', 'retention_score': 'last_access + 100 * task_chat'}
        with self.assertRaisesRegex(ValueError, 'Unknown variable'):
            RetentionPolicy(spec)
        policy = RetentionPolicy(spec, 'task-v1')
        a = Entry(1, 1, 0, 0, 1, 1, 1, task_chat=True, task_unknown=False)
        b = Entry(2, 1, 1, 1, 1, 2, 2, task_qa=True, task_unknown=False)
        self.assertEqual(policy.choose([a, b], 2), 2)
        self.assertEqual(LRU().choose([a, b], 2), 1)
        for leaked in ('block_id', 'session_id', 'group_id', 'request_id', 'future_hits', 'prompt_token_ids'):
            with self.assertRaises(ValueError):
                RetentionPolicy({'name': 'bad', 'retention_score': leaked}, 'task-v1')

    def test_metadata_roundtrip_and_validation(self):
        trace = Trace((Request('a', 0, (1,), task_type='chat', turn_index=2),))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 't.json'
            save_trace(trace, path)
            self.assertEqual(load_trace(path), trace)
        for kwargs in ({'turn_index': -1}, {'task_type': 'dataset-row-12'}):
            with self.assertRaises(ValueError):
                compile_trace(Trace((Request('a', 0, (1,), **kwargs),)))

    def test_prompt_contract_and_hidden_split_rejection(self):
        prompt = discovery_prompt(root(), [root()], [], policy_features='task-v1')
        self.assertIn('task_chat', prompt)
        self.assertIn('MOST RECENT', prompt)
        self.assertNotIn('task_chat', discovery_prompt(root(), [root()], []))
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            trace = directory / 'test' / 'trace.json'
            save_trace(Trace((Request('a', 0, (1,)),)), trace)
            with self.assertRaisesRegex(ValueError, 'Incorrect dataset split'):
                prepare_config({'train': [{'name': 'leak', 'path': str(trace), 'capacity_blocks': 2}]}, directory)

    def test_declared_test_split_rejected_even_if_renamed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'renamed.json'
            write_json(path, {'schema_version': 1, 'tokenizer': 'x', 'split': 'test', 'requests': []})
            with self.assertRaisesRegex(ValueError, 'Incorrect dataset split'):
                prepare_config({'train': [{'name': 'leak', 'path': str(path), 'capacity_blocks': 2}]}, Path(tmp))

    def test_prepare_disjoint_splits_hashes_and_unchanged_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); source = directory / 'source'; output = directory / 'prepared'
            files = []
            for split in ('train', 'validation', 'test'):
                for kind, groups in pools().items():
                    rows = []
                    for group in groups.values():
                        for original in group:
                            row = deepcopy(original)
                            for key in ('group_id', 'session_id', 'request_id'):
                                row[key] = split + '-' + row[key]
                            rows.append(row)
                    rel = f'{kind}/{split}/trace-00000.json'
                    write_json(source / rel, {'split': split, 'tokenizer': 'x', 'requests': rows})
                    files.append({'path': rel, 'sha256': digest(source / rel)})
            write_json(source / 'manifest.json', {'files': files})
            prepare(source, output, source_start=0, source_count=1, episodes=1, requests=20)
            self.assertEqual(verify_prepared(output)['test_status'], 'sealed; never evaluated during preparation or search')
            for features in ('block-v1', 'task-v1'):
                config = json.loads((output / f'configs/{features}.json').read_text())
                self.assertNotIn('/test/', json.dumps(config))
                prepare_config(config, directory)
            trace = output / 'train/balanced-00.json'
            trace.write_text(trace.read_text() + ' ')
            with self.assertRaisesRegex(ValueError, 'changed'):
                verify_prepared(output)

    def test_realistic_large_mixed_feedback_fits_backend_envelope(self):
        from dream_rsi.prompts import tried_memory
        node = root()
        node['feedback']['runs'] = [
            {'name': f'balanced-{i}-cap2048', 'extra_computed_tokens': 12000,
             'lru_extra_computed_tokens': 15000, 'saved_prompt_tokens': 3000,
             'score_contribution': .01, 'evicted_blocks': 1200, 'policy_us_per_request': 500,
             'workloads': {kind: {'requests': 130, 'prompt_tokens': 100000,
                    'computed_prompt_tokens': 20000, 'reused_prompt_tokens': 80000}
                    for kind in ('sharegpt', 'mashqa')}} for i in range(16)]
        nodes = [deepcopy(node) for _ in range(12)]
        for i, n in enumerate(nodes): n['id'] = i
        for features in ('block-v1', 'task-v1'):
            prompt = discovery_prompt(node, nodes, [{'name': 'suite'}], nodes[:4],
                     best=node, memory=tried_memory([], nodes), policy_features=features)
            self.assertLessEqual(len(json.dumps([{'role': 'user', 'content': prompt}]).encode()), 54000)
            payload = json.loads('{"prompt_version":' + prompt.split('\n{"prompt_version":', 1)[1])
            self.assertEqual(payload['selected_parent']['id'], 0)
            self.assertIsNotNone(payload['best_ever_training_policy'])

    def test_feature_lru_preserves_results(self):
        suite = [{'name': 's', 'synthetic': {'sessions': 3}, 'capacity_blocks': 32}]
        baseline = run_suite(suite, 16)
        for features in ('block-v1', 'task-v1'):
            result = run_suite(suite, 16, candidate=LRU_SPEC, baselines=baseline['baselines'], policy_features=features)
            self.assertEqual(result['score'], 0)

    def test_final_gate_one_use_and_source_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            rootdir = Path(tmp)
            prepared = rootdir / 'data'
            search = rootdir / 'search'
            write_json(prepared / 'manifest.json', {'files': []})
            write_json(prepared / 'test/suite.json', [{'name': 'tiny', 'synthetic': {'sessions': 1}, 'capacity_blocks': 32}])
            for features in ('block-v1', 'task-v1'):
                write_json(search / features / 'summary.json', {'status': 'api_budget_reached',
                           'best_policy': LRU_SPEC, 'validation_valid': True})
                write_json(search / features / 'config.json', {'policy_features': features})
            with patch('dream_rsi.mixed_experiment.source_hashes', return_value={'a': 'hash'}):
                freeze_selection(prepared, search, rootdir)
                with patch('dream_rsi.mixed_experiment.source_hashes', return_value={'a': 'changed'}):
                    with self.assertRaisesRegex(ValueError, 'source changed'):
                        final_test(prepared, search, rootdir)
                with patch('dream_rsi.mixed_experiment.isolated_evaluate', return_value={'valid': True, 'baselines': []}):
                    final_test(prepared, search, rootdir)
                    with self.assertRaises(FileExistsError):
                        final_test(prepared, search, rootdir)
            self.assertEqual(json.loads((search / 'test-opened.json').read_text())['status'], 'completed')


if __name__ == '__main__':
    unittest.main()
