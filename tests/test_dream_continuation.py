import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest

import test_mixed_long
from dream_rsi import mixed_long as study
from dream_rsi.backend import write_json
from dream_rsi.evaluation import digest


class DreamContinuationTests(unittest.TestCase):
    fixture = test_mixed_long.MixedLongTests.fixture
    def test_migration_preserves_costs_reconciles_without_http_and_runs_dreams(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pilot, prepared = self.fixture(tmp)
            old = tmp / 'old'; old.mkdir()
            state = study.initialize(pilot, prepared, Path.cwd(), trials=1, total_calls=32, max_new_usd=1)
            study.tick(state, old, Path.cwd())
            for rel in state['source_hashes']:
                p = old / 'source' / rel; p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes((Path.cwd() / rel).read_bytes())
            jobs = [j for r in state['runs'] for j in r['pending']]
            self.assertEqual(len(jobs), 4)
            # Finished, possibly billed with unknown outcome, and two never-submitted requests.
            write_json(old / 'requests' / jobs[0]['id'] / 'state.json', {
                'phase': 'finished', 'body': jobs[0]['body'], 'result': {
                    'program': {'name': 'test', 'retention_score': 'last_access'}, 'error': None,
                    'usage': {'prompt_tokens': 100, 'completion_tokens': 10}, 'explicit_failure': False}})
            write_json(old / 'requests' / jobs[1]['id'] / 'state.json', {'phase': 'started', 'body': jobs[1]['body']})
            before_cap = [r['config']['api']['max_usd'] for r in state['runs']]
            retained = state['runs'][0]['records'][-1]['charged_estimate_usd']
            out = tmp / 'new'; out.mkdir()
            with patch('requests.post', side_effect=AssertionError('Migration must not make HTTP calls')):
                new = study.continue_with_dreams(old, out, Path.cwd())
            self.assertFalse(new['active_wave'])
            self.assertEqual(before_cap, [r['config']['api']['max_usd'] for r in new['runs']])
            self.assertEqual(new['plan']['max_new_usd'], 1)
            self.assertAlmostEqual(study.new_spend(new), retained + .0000522)
            self.assertTrue((old / 'superseded.json').exists())
            self.assertTrue(all(r['config']['controller_revisions'] == 2 for r in new['runs']))
            self.assertTrue(all(not r['pending'] for r in new['runs']))
            with self.assertRaisesRegex(ValueError, 'already continued'):
                study.continue_with_dreams(old, out, Path.cwd())
            saw_dreams = False
            with patch.object(study.search, 'evaluate', wraps=study.search.evaluate) as evaluate:
                for _ in range(50):
                    study.tick(new, out, Path.cwd())
                    if any(j['role'] == 'controller' for r in new['runs'] for j in r['pending']):
                        saw_dreams = True
                    if any(r['phase'] in ('discovery', 'controller') for r in new['runs']):
                        self.assertFalse(any(c.kwargs.get('validation') for c in evaluate.call_args_list))
                    new = json.loads((out / 'checkpoint.json').read_text())
                    if new['status'] == 'completed': break
            self.assertTrue(saw_dreams)
            self.assertEqual(new['status'], 'completed')
            for r in new['runs']:
                revisions = [rev for log in r['revisions'] for rev in log[1:]]
                self.assertTrue(revisions)
                self.assertTrue(any(x['role'] == 'controller' for x in r['records']))
                for log in r['revisions']:
                    if not log: continue
                    incumbent = log[0]['evaluation']['value']
                    for revision in log[1:]:
                        if revision['accepted']:
                            self.assertGreater(revision['evaluation']['value'], incumbent)
                            incumbent = revision['evaluation']['value']
                self.assertLessEqual(len(r['records']), 32)
            self.assertLessEqual(study.new_spend(new), 1)
            self.assertFalse(json.loads((out / 'report.json').read_text())['test_opened'])
            self.assertIn('adaptive Dream-RSI', (out / 'report.md').read_text())
