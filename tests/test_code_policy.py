import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from cache_sim.policies import Entry
from dream_rsi.code_policy import CodePolicy, CodePolicyError, SandboxProgram, evaluate_code_suite
from dream_rsi.evaluation import run_suite, snapshot_suite

LRU_SOURCE = '''class CachePolicy:
    def choose(self, eligible, now):
        return min(eligible, key=lambda e: (e.last_access, e.access_order)).block_id
'''


def entry(block=1, **changes):
    return Entry(**dict(dict(block_id=block, depth=1, inserted_at=0, last_access=0,
                             frequency=1, insertion_order=block, access_order=block), **changes))


class SandboxAvailabilityTest(unittest.TestCase):
    def test_fail_closed_without_supported_platform(self):
        with patch('dream_rsi.code_policy.sys.platform', 'linux'):
            with self.assertRaisesRegex(CodePolicyError, 'refusing unsandboxed'):
                CodePolicy(LRU_SOURCE)


@unittest.skipUnless(sys.platform == 'darwin' and Path('/usr/bin/sandbox-exec').is_file(),
                     'Requires actual macOS Seatbelt sandbox')
class ExecutablePolicyTest(unittest.TestCase):
    def test_stateful_loops_containers_hooks_and_opaque_handles(self):
        source = '''class CachePolicy:
    def __init__(self):
        self.resident = {}
        self.events = []
    def observe_insert(self, entry):
        assert isinstance(entry.block_id, str) and len(entry.block_id) == 32
        self.resident[entry.block_id] = entry.frequency
        self.events.append('insert')
    def observe_hit(self, entry):
        self.resident[entry.block_id] = entry.frequency
        self.events.append('hit')
    def observe_evict(self, entry):
        del self.resident[entry.block_id]
        self.events.append('evict')
    def choose(self, eligible, now):
        assert all(e.block_id in self.resident for e in eligible)
        if len(self.events) == 3:
            assert self.events == ['insert','insert','hit']
        else:
            assert self.events[-2:] == ['evict','insert']
        best = None
        for e in eligible:
            if best is None or self.resident[e.block_id] < self.resident[best.block_id]:
                best = e
        return best.block_id
'''
        with CodePolicy(source) as policy:
            policy.observe_insert(entry(1))
            policy.observe_insert(entry(2))
            policy.observe_hit(entry(1, frequency=2))
            self.assertEqual(policy.choose([entry(1), entry(2)], 1), 2)
            policy.observe_evict(entry(2))
            policy.observe_insert(entry(3))
            self.assertEqual(policy.choose([entry(1), entry(3)], 2), 3)
            policy.flush()
        self.assertGreater(policy.cpu_time_ns, 0)

    def test_block_features_omit_task_and_raw_request_fields(self):
        source = '''from dataclasses import fields
class CachePolicy:
    def choose(self, eligible, now):
        assert {f.name for f in fields(eligible[0])} == {
            'block_id','depth','inserted_at','last_access','frequency','insertion_order','access_order'}
        for e in eligible:
            for field in ('task_chat','task_qa','task_unknown','turn_index','prompt','output',
                          'token_ids','request_id','session_id','workload','future','trace','path'):
                assert not hasattr(e, field), field
        return eligible[0].block_id
'''
        with CodePolicy(source) as policy:
            self.assertEqual(policy.choose([entry(task_chat=True, task_unknown=False, turn_index=5)], 2), 1)

    def test_task_features_are_explicit_opt_in(self):
        source = '''class CachePolicy:
    def choose(self, eligible, now):
        assert eligible[0].task_chat is True and eligible[0].turn_index == 5
        assert not hasattr(eligible[0], 'request_id')
        return eligible[0].block_id
'''
        with CodePolicy(source, 'task-v1') as policy:
            self.assertEqual(policy.choose([entry(task_chat=True, turn_index=5)], 2), 1)

    def test_files_directory_listing_and_future_trace_access_denied(self):
        with tempfile.TemporaryDirectory() as temporary:
            sentinel = Path(temporary) / 'secret-future-trace.json'
            sentinel.write_text('future input and credentials must stay outside worker')
            forbidden = [str(sentinel), str(Path(__file__).resolve()),
                         str(Path(__file__).resolve().parents[1] / 'cache_sim/trace.py')]
            directories = [str(Path.home()), str(Path(__file__).resolve().parents[1]), temporary]
            source = f'''import os
class CachePolicy:
    def __init__(self):
        for path in {forbidden!r}:
            try:
                open(path).read()
            except PermissionError:
                pass
            else:
                raise AssertionError('Unauthorized file read')
        for directory in {directories!r}:
            try:
                os.listdir(directory)
            except PermissionError:
                pass
            else:
                raise AssertionError('Unauthorized directory listing')
        try:
            open({str(sentinel)!r}, 'w').write('changed')
        except PermissionError:
            pass
        else:
            raise AssertionError('Unauthorized write')
    def choose(self, eligible, now): return eligible[0].block_id
'''
            with CodePolicy(source) as policy:
                self.assertEqual(policy.choose([entry()], 1), 1)
            self.assertIn('future input', sentinel.read_text())

    def test_network_subprocess_and_environment_secrets_denied(self):
        source = '''import os, socket, subprocess
class CachePolicy:
    def __init__(self):
        assert 'POLICY_TEST_SECRET' not in os.environ
        assert 'HOME' not in os.environ and 'PATH' not in os.environ
        try:
            socket.create_connection(('127.0.0.1', 9), timeout=.1)
        except OSError:
            pass
        else:
            raise AssertionError('Network access succeeded')
        try:
            subprocess.run(['/bin/sh','-c','true'], check=True)
        except OSError:
            pass
        else:
            raise AssertionError('Process creation succeeded')
    def choose(self, eligible, now): return eligible[0].block_id
'''
        with patch.dict(os.environ, {'POLICY_TEST_SECRET': 'never-send-me'}), CodePolicy(source) as policy:
            self.assertEqual(policy.choose([entry()], 0), 1)

    def test_jsonl_contains_current_metadata_only(self):
        with CodePolicy(LRU_SOURCE) as policy:
            with patch.object(policy._worker, '_rpc', wraps=policy._worker._rpc) as rpc:
                policy.observe_insert(entry())
                policy.choose([entry()], 0)
                messages = [args.args[0] for args in rpc.call_args_list]
                self.assertEqual(set(messages[0]), {'op', 'events', 'eligible', 'now'})
                encoded = json.dumps(messages)
                for prohibited in ('prompt', 'request_id', 'session_id', 'trace', 'synthetic', 'suite', 'secret'):
                    self.assertNotIn(prohibited, encoded)
                self.assertEqual(len(messages[0]['events']), 1)

    def test_invalid_ids_and_candidate_exceptions_fail_and_close(self):
        for expression in ('None', 'True', '1', '"made-up"', 'eligible[0]', '1/0'):
            with self.subTest(expression=expression):
                with CodePolicy('class CachePolicy:\n def choose(self,eligible,now): return ' + expression) as policy:
                    with self.assertRaises(CodePolicyError):
                        policy.choose([entry()], 0)
                    self.assertTrue(policy._worker._closed)

    def test_entry_is_frozen(self):
        with CodePolicy('class CachePolicy:\n def choose(self,es,now):\n  es[0].frequency=9\n  return es[0].block_id') as policy:
            with self.assertRaisesRegex(CodePolicyError, 'FrozenInstanceError'):
                policy.choose([entry()], 0)

    def test_missing_class_and_malformed_source_fail_on_construction(self):
        for source in ('garbage syntax !!!', 'class Other: pass', 'class CachePolicy: pass'):
            with self.subTest(source=source), self.assertRaises(CodePolicyError):
                CodePolicy(source)

    def test_hangs_and_output_flood_are_bounded(self):
        started = time.monotonic()
        with self.assertRaisesRegex(CodePolicyError, 'wall-time'):
            CodePolicy('while True: pass', wall_timeout=.1)
        self.assertLess(time.monotonic() - started, 2)
        with CodePolicy('class CachePolicy:\n def choose(self, es, now):\n  while True: pass', wall_timeout=.1) as policy:
            with self.assertRaisesRegex(CodePolicyError, 'wall-time'):
                policy.choose([entry()], 0)
        with self.assertRaisesRegex(CodePolicyError, 'output limit'):
            CodePolicy('print("x"*100000)\n' + LRU_SOURCE, max_output_bytes=1024)

    def test_host_memory_monitor_and_total_timeout_between_calls(self):
        with CodePolicy(LRU_SOURCE, total_timeout=.12) as policy:
            time.sleep(.2)
            with self.assertRaisesRegex(CodePolicyError, 'wall-time'):
                policy.choose([entry()], 0)
        source = '''import threading, time
class CachePolicy:
    def __init__(self):
        def allocate():
            time.sleep(.05)
            self.data = bytearray(100*1024*1024)
            while True: time.sleep(.01)
        threading.Thread(target=allocate, daemon=True).start()
    def choose(self, es, now): return es[0].block_id
'''
        with CodePolicy(source, memory_bytes=64*1024*1024) as policy:
            time.sleep(.25)
            with self.assertRaisesRegex(CodePolicyError, 'memory'):
                policy.choose([entry()], 0)

    def test_generic_program_is_stateful_and_resets(self):
        source = '''class ExplorationController:
    def __init__(self, initial=0): self.count=initial
    def select(self, nodes, round_index, workers, max_depth):
        self.count += 1
        return [self.count]
    def plan_grid(self, context): return {'workers': context['workers']}
'''
        for _ in range(2):
            with SandboxProgram(source, 'ExplorationController', {'initial': 3}) as program:
                self.assertEqual(program.call('select', [[], 1, 2, 3]), [4])
                self.assertEqual(program.call('select', [[], 2, 2, 3]), [5])
                self.assertEqual(program.call('plan_grid', [{'workers': 2}]), {'workers': 2})
            self.assertGreater(program.cpu_time_ns, 0)

    def test_executable_lru_suite_parity_and_final_hook_validation(self):
        suite = [{'name': 'tiny', 'synthetic': {'seed': 42, 'sessions': 3}, 'capacity_blocks': 32}]
        baseline = run_suite(suite, 16)['baselines']
        candidate = {'name': 'python-lru', 'source': LRU_SOURCE, 'rationale': 'LRU'}
        result = evaluate_code_suite(suite, 16, candidate=candidate, baselines=baseline)
        self.assertTrue(result['valid'])
        self.assertEqual(result['score'], 0)
        self.assertEqual(result['runs'][0]['evicted_blocks'], baseline[0]['policies']['lru']['evicted_blocks'])
        self.assertGreater(result['runs'][0]['worker_cpu_time_ns'], 0)
        rejected = evaluate_code_suite(suite, 16, candidate=candidate, baselines=baseline,
                                       max_policy_us_per_request=.001)
        self.assertFalse(rejected['valid'])
        broken_hook = {'name':'broken','source':LRU_SOURCE + '\n    def observe_insert(self, entry): raise ValueError("hook failed")\n'}
        roomy = [suite[0] | {'capacity_blocks': 4096}]
        with self.assertRaisesRegex(CodePolicyError, 'hook failed'):
            evaluate_code_suite(roomy, 16, candidate=broken_hook, baselines=run_suite(roomy, 16)['baselines'])

    def test_candidate_shape_baseline_identity_and_trace_snapshot(self):
        suite = [{'name': 'tiny', 'synthetic': {'seed': 42, 'sessions': 1}, 'capacity_blocks': 32}]
        baseline = run_suite(suite, 16)['baselines']
        with self.assertRaisesRegex(ValueError, 'Unexpected'):
            evaluate_code_suite(suite, 16, candidate={'name':'x','source':LRU_SOURCE,'bad':True}, baselines=baseline)
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            evaluate_code_suite(suite, 16, candidate={'name':'x','source':LRU_SOURCE}, baselines=[baseline[0]|{'name':'other'}])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'trace.json'
            path.write_text('{}')
            frozen = snapshot_suite([{'name':'trace','path':str(path),'capacity_blocks':32}], Path(temporary))
            path.write_text('{"modified":true}')
            with self.assertRaisesRegex(ValueError, 'changed'):
                evaluate_code_suite(frozen, 16, candidate={'name':'x','source':LRU_SOURCE},
                                    baselines=[{'name':'trace'}])


if __name__ == '__main__':
    unittest.main()
