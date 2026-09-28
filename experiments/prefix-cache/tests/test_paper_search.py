"""Coordinator behavior with local program responses and no API access."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dream_rsi import paper_search as study
from dream_rsi.backend import write_json
from dream_rsi.code_controller import INITIAL_CONTROLLER_SPEC
from dream_rsi.code_transport import source_allowance
from dream_rsi.evaluation import digest
from dream_rsi.history import observation
from dream_rsi.scoring import TOTAL_EXTRA


def author_payload(job):
    return json.loads(job['prompt'][1]['content'].split('FULL_HISTORY_JSON\n', 1)[1])


class LocalTransport:
    def __init__(self, unknown=False):
        self.unknown = unknown
        self.sent = []
        self.results = {}

    def poll(self, wave, jobs):
        for job in jobs:
            if job['id'] not in self.results:
                self.sent.append(job['id'])
                self.results[job['id']] = (
                    {'program': None, 'error': 'unknown outcome', 'usage': {}}
                    if self.unknown else
                    {'program': study.LRU_CODE | {'name': 'proposal-' + job['id']}, 'error': None,
                     'usage': {'prompt_tokens': 10, 'completion_tokens': 10}, 'finish_reason': 'stop'})
        return {job['id']: copy.deepcopy(self.results[job['id']]) for job in jobs}


class PaperSearchTest(unittest.TestCase):
    def fixture(self, directory, *, cycles=2, rounds=2, revisions=3, workers=2, max_usd=10):
        predecessor = directory / 'predecessor'
        prepared = directory / 'prepared'
        prepared.mkdir()
        write_json(prepared / 'manifest.json', {'files': []})
        config = {
            'train': [{'name': 'train-local', 'capacity_blocks': 32}],
            'validation': [{'name': 'validation-secret', 'capacity_blocks': 32}],
            'block_size': 16, 'scoring': TOTAL_EXTRA, 'policy_features': 'task-v1',
            'max_policy_us_per_request': 10000, 'evaluation_timeout_seconds': 20,
            'api': {'model': 'local-fixture', 'thinking': 'disabled',
                    'input_usd_per_million': .435, 'output_usd_per_million': .87},
        }
        baselines = {'valid': True, 'baselines': [
            {'policies': {'lru': {'extra_computed_tokens': 100}}}]}
        write_json(predecessor / 'checkpoint.json', {
            'status': 'paused_by_user',
            'plan': {'prepared': str(prepared), 'prepared_manifest_sha256': digest(prepared / 'manifest.json')},
            'runs': [{'arm': 'task-v1', 'config': config, 'baselines': baselines,
                      'records': [{'charged_estimate_usd': 100}],
                      'histories': [{'old': 'must never enter fresh search'}]}],
        })
        with patch.object(study, 'verify_prepared'), patch.object(study, 'prepare_config', side_effect=lambda config, root: config), \
             patch.object(study, 'evaluate_policy', return_value={'valid': True, 'score': 0}):
            state = study.initialize(predecessor, Path.cwd(), max_usd=max_usd, cycles=cycles,
                                     online_rounds=rounds, controller_revisions=revisions, workers=workers)
        output = directory / 'output'
        output.mkdir()
        return state, output

    def test_fresh_paired_initialization_excludes_old_spending_and_history(self):
        with tempfile.TemporaryDirectory() as temp:
            state, output = self.fixture(Path(temp))
            self.assertEqual(study.spent(state), 0)
            self.assertEqual([run['usd_cap'] for run in state['runs']], [5, 5])
            for run in state['runs']:
                self.assertEqual(run['histories'], [])
                self.assertEqual(run['records'], [])
                self.assertEqual(len(run['tree']), 1)
                self.assertEqual(run['controller'], INITIAL_CONTROLLER_SPEC)
                self.assertEqual(run['config']['max_depth'], run['config']['online_rounds'])
            self.assertEqual(state['plan']['controller_versions_per_phase'], 4)
            self.assertEqual(state['runs'][0]['config'], state['runs'][1]['config'])

    def start_dream(self, state):
        run = state['runs'][1]
        run['tree'].append(observation(1, run['tree'][0], 1, study.LRU_CODE,
                                       {'valid': True, 'score': .1}))
        run['round'] = run['config']['online_rounds'] + 1
        return run

    def test_sequential_worse_then_invalid_base_and_best_final_deployment(self):
        with tempfile.TemporaryDirectory() as temp:
            state, output = self.fixture(Path(temp))
            run = self.start_dream(state)
            incumbent = copy.deepcopy(run['controller'])
            proposals = [incumbent | {'name': 'worse'},
                         incumbent | {'name': 'invalid', 'source': 'class ExplorationController:\n    pass\n'},
                         incumbent | {'name': 'winner'}]

            def replay(program, histories, **settings):
                if program['name'] == 'invalid':
                    return {'valid': False, 'value': None, 'error': 'missing select'}
                return {'valid': True, 'value': {'worse': .5, 'winner': 2}.get(program['name'], 1), 'replays': []}

            with patch.object(study, 'evaluate_controller_source', side_effect=replay), \
                 patch.object(study, 'live_actions', return_value=[0]):
                study.advance(run)
                self.assertEqual(author_payload(run['pending'][0])['current_version'], incumbent)
                for index, program in enumerate(proposals, 1):
                    job = run['pending'][0]
                    study.apply_results(run, {job['id']: {'program': program, 'error': None,
                        'usage': {'prompt_tokens': 10, 'completion_tokens': 10}}}, output)
                    self.assertEqual(run['development_current'], program)
                    self.assertEqual(run['controller'], incumbent)
                    study.advance(run)
                    if index < len(proposals):
                        payload = author_payload(run['pending'][0])
                        self.assertEqual(payload['current_version'], program)
                        self.assertEqual(len(payload['previous_revisions']), index + 1)
                        self.assertEqual(payload['completed_histories'], run['histories'])
                        if index == 2:
                            self.assertFalse(payload['current_version_evaluation']['valid'])
                            self.assertEqual(payload['current_version_evaluation']['error'], 'missing select')
                self.assertEqual(run['controller']['name'], 'winner')
                self.assertEqual(run['rollout_controller']['name'], 'winner')
                self.assertEqual(run['cycle'], 2)
                self.assertEqual(len(run['revisions'][0]), 4)
                self.assertEqual(run['rollouts'][0]['controller'], incumbent)
                self.assertEqual([row['revision'] for row in run['revisions'][0]], [0, 1, 2, 3])

    def test_worse_revisions_keep_incumbent_in_candidate_set(self):
        with tempfile.TemporaryDirectory() as temp:
            state, output = self.fixture(Path(temp), revisions=1)
            run = self.start_dream(state)
            incumbent = copy.deepcopy(run['controller'])
            with patch.object(study, 'evaluate_controller_source', side_effect=[
                    {'valid': True, 'value': 1, 'replays': []},
                    {'valid': True, 'value': .5, 'replays': []}]), \
                 patch.object(study, 'live_actions', return_value=[0]):
                study.advance(run)
                job = run['pending'][0]
                study.apply_results(run, {job['id']: {'program': incumbent | {'name': 'worse'},
                    'error': None, 'usage': {'prompt_tokens': 1, 'completion_tokens': 1}}}, output)
                study.advance(run)
            self.assertEqual(run['controller'], incumbent)
            self.assertEqual(run['development_current']['name'], 'worse')
            self.assertEqual(run['revisions'][0][1]['controller']['name'], 'worse')

    def test_live_controller_private_state_reconstructed_after_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            state, output = self.fixture(Path(temp), cycles=1, rounds=3)
            run = state['runs'][0]
            controller = {'name': 'stateful', 'rationale': 'local test', 'source': '''class ExplorationController:
    def __init__(self):
        self.calls = 0
    def select(self, nodes, round_index, workers, max_depth):
        self.calls += 1
        if self.calls == 1:
            return [0]
        return [1] if self.calls == 2 else []
'''}
            run['controller'] = copy.deepcopy(controller)
            run['rollout_controller'] = copy.deepcopy(controller)
            with patch.object(study, 'evaluate_policy', return_value={'valid': True, 'score': .1}):
                study.advance(run)
                job = run['pending'][0]
                study.apply_results(run, {job['id']: {'program': study.LRU_CODE, 'error': None,
                    'usage': {'prompt_tokens': 1, 'completion_tokens': 1}}}, output)
                write_json(output / 'live.json', run)
                resumed = study.read_json(output / 'live.json')
                self.assertEqual(study.live_actions(resumed), [1])
                study.advance(resumed)
                job = resumed['pending'][0]
                self.assertEqual(job['parent'], 1)
                study.apply_results(resumed, {job['id']: {'program': study.LRU_CODE, 'error': None,
                    'usage': {'prompt_tokens': 1, 'completion_tokens': 1}}}, output)
                study.advance(resumed)
            self.assertEqual(resumed['phase'], 'search_done')
            self.assertEqual(resumed['controller'], controller)
            self.assertEqual(resumed['rollouts'][0]['controller'], controller)
            self.assertEqual([decision['actions'] for decision in resumed['rollouts'][0]['decisions']], [[0], [1]])
            changed = copy.deepcopy(run)
            changed['decisions'][0]['actions'] = [1]
            with self.assertRaisesRegex(ValueError, 'deterministic'):
                study.live_actions(changed)

    def test_both_training_winners_are_frozen_before_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            state, output = self.fixture(Path(temp), cycles=1, revisions=0)
            state['runs'][0]['config']['api']['max_calls'] = 1
            validation_calls = []

            def evaluate(config, program, baselines, *, validation=False):
                if validation:
                    frozen = study.read_json(output / 'frozen-selection.json')
                    self.assertEqual(len(frozen['selections']), 2)
                    self.assertTrue(all(run['phase'] in ('search_done', 'done') for run in state['runs']))
                    self.assertEqual([row['candidate'] for row in frozen['selections']],
                                     [run['best']['candidate'] for run in state['runs']])
                    validation_calls.append(program['name'])
                else:
                    self.assertFalse((output / 'frozen-selection.json').exists())
                return {'valid': True, 'score': .05 if validation else .1}

            transport = LocalTransport()
            with patch.object(study, 'evaluate_policy', side_effect=evaluate), \
                 patch.object(study, 'isolated_evaluate', return_value={'valid': True, 'baselines': []}):
                study.tick(state, output, Path.cwd(), transport)
                self.assertEqual(validation_calls, [])
                study.tick(state, output, Path.cwd(), transport)
                self.assertEqual(state['runs'][0]['phase'], 'search_done')
                self.assertEqual(validation_calls, [])
                self.assertFalse((output / 'frozen-selection.json').exists())
                study.tick(state, output, Path.cwd(), transport)
            self.assertEqual(state['status'], 'completed')
            self.assertEqual(len(validation_calls), 2)
            self.assertFalse(study.read_json(output / 'report.json')['test_opened'])
            for path in output.glob('*/prompts/*.json'):
                self.assertNotIn('validation-secret', path.read_text())

    def test_resume_retains_unknown_reservations_and_enforces_caps(self):
        with tempfile.TemporaryDirectory() as temp:
            state, output = self.fixture(Path(temp), cycles=1)
            for run in state['runs']:
                run['config']['api']['max_calls'] = 1
            transport = LocalTransport(unknown=True)
            with patch.object(study, 'evaluate_policy', return_value={'valid': True, 'score': 0}), \
                 patch.object(study, 'isolated_evaluate', return_value={'valid': True, 'baselines': []}):
                study.tick(state, output, Path.cwd(), transport)
                reservation = study.spent(state)
                self.assertGreater(reservation, 0)
                self.assertEqual(transport.sent, [])
                submitted_ids = [job['id'] for run in state['runs'] for job in run['pending']]
                resumed = study.read_json(output / 'checkpoint.json')
                study.tick(resumed, output, Path.cwd(), transport)
            self.assertEqual(transport.sent, submitted_ids)
            self.assertEqual(resumed['status'], 'completed')
            self.assertEqual(study.spent(resumed), reservation)
            self.assertLessEqual(study.spent(resumed), resumed['plan']['max_usd'])
            for run in resumed['runs']:
                self.assertEqual(len(run['records']), 1)
                self.assertEqual(run['records'][0]['status'], 'usage_unknown_reservation_retained')
                self.assertEqual(run['stop_reason'], 'budget_reached')
                self.assertFalse(run['histories'][0][1]['valid'])
            broken = copy.deepcopy(resumed)
            broken['plan']['max_usd'] = .000001
            with self.assertRaisesRegex(ValueError, 'budgets exceed'):
                study.check_budget(broken)
            broken = copy.deepcopy(resumed)
            broken['runs'][0]['records'][0]['charged_estimate_usd'] = 20
            with self.assertRaisesRegex(ValueError, 'spending cap exceeded'):
                study.check_budget(broken)

    def test_budget_partial_batch_records_requested_and_submitted_actions(self):
        with tempfile.TemporaryDirectory() as temp:
            state, output = self.fixture(Path(temp), cycles=1)
            run = state['runs'][0]
            with patch.object(study, 'evaluate_policy', return_value={'valid': True, 'score': .1}):
                study.advance(run)
                job = run['pending'][0]
                study.apply_results(run, {job['id']: {'program': study.LRU_CODE, 'error': None,
                    'usage': {'prompt_tokens': 1, 'completion_tokens': 1}}}, output)
                parent = run['tree'][1]
                messages = study.discovery_messages(parent, run['tree'], run['suite_summary'], run['histories'],
                    run['config']['scoring'], policy_features=run['config']['policy_features'], best=run['best'])
                amount = study.cost(run['config']['api'], source_allowance(messages),
                                    run['config']['api']['max_completion_tokens'])
                run['usd_cap'] = study.run_cost(run) + amount + 1e-9
                study.advance(run)
                self.assertEqual(run['decisions'][-1]['actions'], [1, 0])
                self.assertEqual(run['decisions'][-1]['submitted_actions'], [1])
                self.assertEqual(len(run['pending']), 1)
                self.assertEqual(run['stop_reason'], 'budget_reached')
                job = run['pending'][0]
                study.apply_results(run, {job['id']: {'program': study.LRU_CODE, 'error': None,
                    'usage': {'prompt_tokens': 1, 'completion_tokens': 1}}}, output)
                study.advance(run)
            self.assertEqual(run['phase'], 'search_done')
            self.assertEqual(len(run['histories']), 1)
            self.assertEqual(len(run['histories'][0]), 3)
            self.assertLessEqual(study.run_cost(run), run['usd_cap'])


if __name__ == '__main__':
    unittest.main()
