import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dream_rsi import mixed_long as study
from dream_rsi.backend import write_json
from dream_rsi.evaluation import digest
from dream_rsi.programs import INITIAL_CONTROLLER
from dream_rsi.runner import run


class MixedLongTests(unittest.TestCase):
    def fixture(self, tmp):
        pilot = tmp / 'pilot'; prepared = tmp / 'prepared'
        prepared.mkdir()
        write_json(prepared / 'manifest.json', {'files': []})
        config = {'cycles': 4, 'online_rounds': 4, 'workers': 2, 'max_depth': 4,
                  'controller_revisions': 0, 'api': {'max_calls': 12, 'max_usd': .3},
                  'train': [{'name': 'train', 'synthetic': {'sessions': 3, 'seed': 1}, 'capacity_blocks': 32}],
                  'validation': [{'name': 'validation-secret', 'synthetic': {'sessions': 3, 'seed': 99}, 'capacity_blocks': 32}]}
        for features in study.FEATURES:
            run(config | {'policy_features': features}, 'mock', Path.cwd(), pilot / features)
            usage_path = pilot / features / 'usage.json'
            usage = json.loads(usage_path.read_text())
            for index, r in enumerate(usage['records'], 1):
                r.update(call=index, status='accounted', charged_estimate_usd=0)
            write_json(usage_path, usage)
        write_json(pilot / 'frozen-selection.json', {'source_hashes': {},
                   'prepared_manifest_sha256': digest(prepared / 'manifest.json')})
        return pilot, prepared

    def test_partial_pilot_continues_and_independent_trials_stay_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); pilot, prepared = self.fixture(tmp)
            state = study.initialize(pilot, prepared, Path.cwd(), trials=2, total_calls=16, max_new_usd=4)
            self.assertEqual(len(state['runs']), 4)
            self.assertEqual(state['plan']['maximum_new_calls'], 40)
            for r in state['runs']:
                self.assertEqual(r['controller'], INITIAL_CONTROLLER)
                self.assertNotIn('validation', r)
                if r['trial'] == 1:
                    self.assertEqual(len(r['records']), 12)
                    self.assertEqual(r['cycle'], 2)
                    self.assertEqual(r['round'], 4)
                    self.assertEqual(len(r['histories']), 1)
                    self.assertEqual(len(r['tree']), 6)
                    self.assertIn('12', r['milestones'])
                else:
                    self.assertEqual(r['records'], [])
                    self.assertEqual(r['histories'], [])
                    self.assertEqual(r['best']['score'], 0)
            self.assertLessEqual(sum(r['new_usd_ceiling'] for r in state['runs']), 4)
            out = tmp / 'out'; out.mkdir()
            with patch.object(study.search, 'evaluate', wraps=study.search.evaluate) as evaluate:
                for _ in range(30):
                    study.tick(state, out, Path.cwd())
                    state = json.loads((out / 'checkpoint.json').read_text())
                    if any(r['phase'] == 'discovery' for r in state['runs']):
                        self.assertFalse(any(c.kwargs.get('validation') for c in evaluate.call_args_list))
                    if state['status'] == 'completed': break
            self.assertEqual(state['status'], 'completed')
            self.assertEqual(study.new_spend(state), 0)
            for r in state['runs']:
                self.assertEqual(len(r['records']), 16)
                self.assertEqual(r['controller'], INITIAL_CONTROLLER)
                self.assertEqual(sum(len(h)-1 for h in r['histories']), 16)
                self.assertIn('12', r['milestones'])
            self.assertTrue((out / 'frozen-selection.json').exists())
            for p in out.glob('*/prompts/*.json'):
                self.assertNotIn('validation-secret', p.read_text())

    def test_budget_guard_retains_unknown_requests_and_does_not_resend_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); pilot, prepared = self.fixture(tmp)
            state = study.initialize(pilot, prepared, Path.cwd(), trials=1, total_calls=16, max_new_usd=.1)
            out = tmp / 'out'; out.mkdir()
            study.tick(state, out, Path.cwd())
            jobs = [j for r in state['runs'] for j in r['pending']]
            self.assertTrue(jobs)
            self.assertTrue(all(j['id'].endswith(('-0013', '-0014')) for j in jobs))
            reserved = study.new_spend(state)
            self.assertGreater(reserved, 0)
            class Unknown:
                def poll(self, wave, submitted):
                    return {j['id']: {'program': None, 'error': 'unknown', 'usage': {}, 'explicit_failure': False} for j in submitted}
            state['backend'] = 'mimo'
            study.tick(state, out, Path.cwd(), transport=Unknown())
            self.assertGreaterEqual(study.new_spend(state), reserved)
            self.assertLessEqual(study.new_spend(state), .1)
            self.assertTrue(any(r['status']=='usage_unknown_reservation_retained' for run in state['runs'] for r in run['records']))
            state['plan']['max_new_usd'] = .000001
            with self.assertRaisesRegex(ValueError, 'ceilings'):
                study.check_budget(state)

    def test_tiny_cap_stops_before_new_calls_and_changed_semantics_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); pilot, prepared = self.fixture(tmp)
            state = study.initialize(pilot, prepared, Path.cwd(), trials=1, total_calls=16, max_new_usd=.000002)
            out = tmp / 'out'; out.mkdir()
            study.tick(state, out, Path.cwd())
            self.assertEqual(state['status'], 'completed')
            self.assertEqual(study.new_spend(state), 0)
            self.assertTrue(all(len(r['records']) == 12 for r in state['runs']))
            frozen = json.loads((pilot / 'frozen-selection.json').read_text())
            frozen['source_hashes']['dream_rsi/programs.py'] = 'wrong'
            write_json(pilot / 'frozen-selection.json', frozen)
            with self.assertRaisesRegex(ValueError, 'semantics changed'):
                study.initialize(pilot, prepared, Path.cwd())


if __name__ == '__main__':
    unittest.main()
