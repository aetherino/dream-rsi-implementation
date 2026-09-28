"""Continue the task-feature pilot and run independent replications under one new-spend cap."""
import argparse
import copy
import fcntl
import json
import math
import shutil
from pathlib import Path
import time

from . import batch_compare as search
from .backend import Mock, write_json
from .evaluation import digest
from .history import evaluate_controller, root
from .mixed import verify_prepared
from .programs import Controller, INITIAL_CONTROLLER
from .prompts import controller_prompt, discovery_prompt, tried_memory
from .realtime_transport import RealtimeTransport
from .runner import prepare_config

FEATURES = ('block-v1', 'task-v1')
MILESTONES = (12, 25, 50, 100)


def new_spend(state):
    return sum(search.usage(r, state['backend'])['estimated_usd'] - r['imported_usd'] for r in state['runs'])


def check_budget(state):
    ceilings = sum(r['config']['api']['max_usd'] - r['imported_usd'] for r in state['runs'])
    if ceilings > state['plan']['max_new_usd'] + 1e-9:
        raise ValueError('Per-run ceilings exceed global new-spend cap')
    if new_spend(state) > state['plan']['max_new_usd'] + 1e-9:
        raise ValueError('Global new-spend cap exceeded; no more requests permitted')


def all_nodes(run):
    return [n for history in run['histories'] for n in history[1:]] + (
        run['tree'][1:] if run['phase'] == 'discovery' else [])


def record_milestones(run):
    nodes = all_nodes(run)
    revealed = max((r['call'] for r in run['records'] if r['status'] != 'reserved'), default=0)
    for limit in MILESTONES:
        if limit > revealed or str(limit) in run['milestones']:
            continue
        candidates = [run['initial']] + [n for n in nodes if n['api_call'] <= limit and n['valid']]
        best = max(candidates, key=lambda n: n['score'])
        run['milestones'][str(limit)] = {'calls': limit, 'train_score': best['score'],
                                         'candidate': copy.deepcopy(best['candidate'])}


def finish_tree(run):
    run['histories'].append(copy.deepcopy(run['tree']))
    run['rollouts'].append({'controller': copy.deepcopy(run['rollout_controller']), 'nodes': copy.deepcopy(run['tree'])})


def next_cycle(run):
    run['cycle'] += 1
    run.update(tree=[copy.deepcopy(run['initial'])], round=1, phase='discovery',
               rollout_controller=copy.deepcopy(run['controller']))


def dream_prompt(run):
    config = run['config']
    def render(feedback, revisions):
        return controller_prompt(run['controller'], feedback, search.controller_history(run['histories']),
            run['replay_settings'], revisions, config['scoring'], version=config['prompt_version'],
            online_rounds=config['online_rounds'])
    prompt = render(run['incumbent_feedback'], run['revision_log'])
    try:
        search.allowance(prompt)
    except ValueError:
        # Historical trees still contain every outcome. Remove redundant replay trajectories
        # from the prompt only; acceptance always evaluates the complete history pool.
        def compact(feedback):
            result = copy.deepcopy(feedback)
            for replay in result.get('replays', []):
                replay.pop('trajectory', None)
            return result
        revisions = [r | {'evaluation': compact(r['evaluation'])} for r in run['revision_log']]
        prompt = render(compact(run['incumbent_feedback']), revisions)
        search.allowance(prompt)
    return prompt


def advance(run):
    """Preserve discovery prompts; optionally revise the controller between complete cycles."""
    config = run['config']
    while run['phase'] in ('discovery', 'controller') and not run['pending']:
        remaining = config['api']['max_calls'] - len(run['records'])
        if remaining <= 0:
            run['stop_reason'] = 'api_budget_reached'
        if run['phase'] == 'controller':
            if (run['revision'] > config['controller_revisions'] or remaining < 2
                    or run['stop_reason'] != 'completed'):
                run['revisions'].append(copy.deepcopy(run['revision_log']))
                next_cycle(run)
                continue
            job = search.reserve(run, 'controller', dream_prompt(run),
                                 {'cycle': run['cycle'], 'revision': run['revision']})
            if job is not None:
                run['pending'].append(job)
            continue
        if run['stop_reason'] != 'completed' or run['cycle'] > config['cycles']:
            if len(run['tree']) > 1:
                finish_tree(run)
            run['phase'] = 'await_validation'
            break
        actions = (Controller(run['controller']).select(run['tree'], run['round'], config['workers'], config['max_depth'])
                   if run['round'] <= config['online_rounds'] else [])
        if not actions:
            finish_tree(run)
            if config['controller_revisions'] and run['cycle'] < config['cycles'] and remaining >= 2:
                feedback = evaluate_controller(run['controller'], run['histories'], **run['replay_settings'])
                if not feedback['valid']:
                    raise ValueError('Incumbent failed historical replay')
                run.update(incumbent_feedback=feedback, revision=1, phase='controller',
                           revision_log=[{'revision': 0, 'cycle': run['cycle'], 'controller': copy.deepcopy(run['controller']),
                                          'evaluation': feedback, 'accepted': True}])
            else:
                next_cycle(run)
            continue
        for offset, parent_id in enumerate(actions[:remaining]):
            parent = next(n for n in run['tree'] if n['id'] == parent_id)
            previous = [max(history, key=lambda n: n['score']) for history in run['histories'][-4:]]
            prompt = discovery_prompt(parent, run['tree'], run['suite_summary'], previous, config['scoring'],
                version=config['prompt_version'], best=run['best'], memory=tried_memory(run['histories'], run['tree']),
                policy_features=config['policy_features'])
            job = search.reserve(run, 'discovery', prompt,
                {'cycle': run['cycle'], 'node': len(run['tree']) + offset}, parent_id)
            if job is None:
                break
            run['pending'].append(job)


def continue_with_dreams(previous, output, project_root, revisions=2):
    """Copy a stopped search, reconcile saved responses offline, and preserve its budget."""
    if type(revisions) is not int or not 1 <= revisions <= 20:
        raise ValueError('Dream revisions must be in [1,20]')
    previous = previous.resolve()
    with (previous / 'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (previous / 'superseded.json').exists():
            raise ValueError('Search already continued elsewhere')
        original = json.loads((previous / 'checkpoint.json').read_text())
        if original['status'] == 'completed' or any(r['phase'] != 'discovery' for r in original['runs']):
            raise ValueError('Migration requires a stopped discovery stage before validation')
        current_hashes = search.source_hashes(project_root)
        for rel, sha in original['source_hashes'].items():
            if digest(previous / 'source' / rel) != sha:
                raise ValueError(f'Original source snapshot changed: {rel}')
            if rel != 'dream_rsi/mixed_long.py' and current_hashes.get(rel) != sha:
                raise ValueError(f'Search semantics changed: {rel}')
        prepared = Path(original['plan']['prepared'])
        if digest(prepared / 'manifest.json') != original['plan']['prepared_manifest_sha256']:
            raise ValueError('Prepared manifest changed')
        verify_prepared(prepared)
        state = copy.deepcopy(original)
        # Finished responses are reused locally; a missing journal proves no HTTP submission.
        # Started responses with no saved outcome remain charged and are never resent.
        outcomes = {}
        for run in state['runs']:
            if run['config']['controller_revisions'] != 0:
                raise ValueError('Expected the fixed-controller stage')
            for job in run['pending']:
                path = previous / 'requests' / job['id'] / 'state.json'
                if path.exists():
                    saved = json.loads(path.read_text())
                    if saved['body'] != job['body']:
                        raise ValueError('Saved request body changed')
                    outcome = (saved['result'] if saved['phase'] == 'finished' else
                               {'program': None, 'error': 'Interrupted before dream transition; usage unknown',
                                'usage': {}, 'explicit_failure': False})
                else:
                    outcome = {'program': None, 'error': 'Cancelled before submission at dream transition',
                               'usage': {}, 'explicit_failure': True}
                outcomes[job['id']] = outcome
            if run['pending']:
                search.apply_results(run, outcomes)
                record_milestones(run)
            # Keep revision artifact indices aligned with the original rollout cycles.
            run['revisions'] = [[] for _ in run['histories']]
            run['config']['controller_revisions'] = revisions
            run['config']['revise_after_final_cycle'] = False
            prepare_config(run['config'], project_root)
            run['dream_transition'] = {'after_call': len(run['records']), 'cycle': run['cycle'],
                                       'next_round': run['round'], 'new_usd': search.usage(run, state['backend'])['estimated_usd'] - run['imported_usd']}
        state.update(active_wave=False, status='prepared', source_hashes=current_hashes)
        state['plan'].update(controller_revisions=revisions, continuation_from=str(previous),
            predecessor_checkpoint_sha256=digest(previous / 'checkpoint.json'),
            protocol='Fixed-controller warmup followed by adaptive Dream-RSI; not a controlled fixed/adaptive comparison',
            call_limit_basis='Total API attempts, including discovery and controller proposals',
            dream_transition={r['id']: r['dream_transition'] for r in state['runs']})
        check_budget(state)
        for folder in [r['id'] for r in state['runs']] + ['requests']:
            if (previous / folder).exists():
                shutil.copytree(previous / folder, output / folder)
        write_json(output / 'predecessor-checkpoint.json', original)
        write_json(output / 'plan.json', state['plan'])
        for rel in current_hashes:
            path = output / 'source' / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((project_root / rel).read_bytes())
        write_json(output / 'checkpoint.json', state)
        report(state, output)
        write_json(previous / 'superseded.json', {'continued_in': str(output), 'new_spending_cap_unchanged': state['plan']['max_new_usd']})
        original['status'] = 'superseded'
        write_json(previous / 'checkpoint.json', original)
        report(original, previous)
        return state


def initialize(pilot, prepared, project_root, *, trials=3, total_calls=100, max_new_usd=4.0):
    if type(trials) is not int or not 1 <= trials <= 10:
        raise ValueError('trials must be in [1,10]')
    if type(total_calls) is not int or not 13 <= total_calls <= 100:
        raise ValueError('total_calls must be in [13,100]')
    if not math.isfinite(max_new_usd) or max_new_usd <= 0:
        raise ValueError('max_new_usd must be finite and positive')
    verify_prepared(prepared)
    frozen = json.loads((pilot / 'frozen-selection.json').read_text())
    if frozen['prepared_manifest_sha256'] != digest(prepared / 'manifest.json'):
        raise ValueError('Prepared data changed since pilot')
    changed = [p for p, sha in frozen['source_hashes'].items() if digest(project_root / p) != sha]
    if changed:
        raise ValueError(f'Pilot semantics changed: {changed}')
    templates, evidence = {}, {}
    backend = None
    for features in FEATURES:
        folder = pilot / features
        def read(name):
            path = folder / name
            evidence[f'{features}/{name}'] = digest(path)
            return json.loads(path.read_text())
        config = prepare_config(read('config.json'), project_root)
        if config['policy_features'] != features or config['controller_revisions'] != 0:
            raise ValueError('Expected a fixed-controller feature pilot')
        if backend is not None and backend != config['backend']:
            raise ValueError('Pilot backends disagree')
        backend = config['backend']
        if backend not in ('mimo', 'mock'):
            raise ValueError('Use regular realtime or mock pilot')
        usage = read('usage.json')
        records = usage['records']
        if len(records) != 12 or any(r['role'] != 'discovery' or r['status'] != 'accounted' for r in records):
            raise ValueError('Import requires the 12 fully accounted pilot discovery calls')
        trees = [read(p.name) for p in sorted(folder.glob('tree-*.json'))]
        controller = read('controller-final.json')
        if controller != INITIAL_CONTROLLER or any(t['controller'] != controller for t in trees):
            raise ValueError('Pilot must use the unchanged initial controller')
        calls = {(r['context']['cycle'], r['context']['node']): r['call'] for r in records}
        progress = []
        for cycle, tree in enumerate(trees, 1):
            for n in tree['nodes'][1:]:
                n['api_call'] = calls[(cycle, n['id'])]
                progress.append({'call': n['api_call'], 'cycle': cycle, 'round': n['round'],
                                 'node': n['id'], 'score': n['score'], 'valid': n['valid']})
        if sorted(p['call'] for p in progress) != list(range(1, 13)):
            raise ValueError('Pilot histories and paid calls do not match')
        progress.sort(key=lambda p: p['call'])
        best_score = 0
        for p in progress:
            if p['valid']: best_score = max(best_score, p['score'])
            p['best_train_score'] = best_score
        baseline = read('baselines.json')
        initial = root() | {'feedback': read('initial-policy-evaluation.json')}
        best = read('best-policy.json')
        if best['score'] != best_score:
            raise ValueError('Pilot best policy differs from training history')
        summary = [{'name': s['name'], 'capacity_blocks': s['capacity_blocks'],
                    'lru_extra_tokens': b['policies']['lru']['extra_computed_tokens'],
                    'lru_policy_us_per_request': b['policies']['lru']['policy_us_per_request']}
                   for s, b in zip(config['train'], baseline['baselines'])]
        summary.append({'max_policy_us_per_request': config['max_policy_us_per_request']})
        templates[features] = dict(config=config, records=records, trees=trees, initial=initial,
                                    best=best, baselines=baseline, suite_summary=summary, progress=progress)
    configs = [copy.deepcopy(t['config']) for t in templates.values()]
    for c in configs:
        c.pop('policy_features')
    if configs[0] != configs[1]:
        raise ValueError('Feature arms differ in non-feature settings')
    new_calls = 2 * (total_calls - 12) + 2 * (trials - 1) * total_calls
    runs = []
    for trial in range(1, trials + 1):
        for features in (FEATURES if trial % 2 else FEATURES[::-1]):
            template = copy.deepcopy(templates[features])
            imported = trial == 1
            prior_records = template['records'] if imported else []
            old_usd = sum(r['charged_estimate_usd'] for r in prior_records)
            remaining = total_calls - len(prior_records)
            # Allocate integer microdollars, rounding DOWN so total ceilings cannot exceed the cap.
            allocated = math.floor(max_new_usd * 1_000_000 * remaining / new_calls) / 1_000_000
            config = template['config']
            config.update(cycles=20)
            config['api'].update(max_calls=total_calls, max_usd=old_usd + allocated)
            config = prepare_config(config, project_root)
            run_id = f'trial-{trial:02d}-{features}'
            for r in prior_records:
                r['id'] = f'{run_id}-imported-{r["call"]:04d}'
                r['prompt_version'] = config['prompt_version']
            completed = template['trees'][:-1] if imported else []
            tree = template['trees'][-1]['nodes'] if imported else [copy.deepcopy(template['initial'])]
            run = {'id': run_id, 'trial': trial, 'arm': features, 'config': config,
                   'phase': 'discovery', 'cycle': len(completed) + 1,
                   'round': max(n['round'] for n in tree) + 1, 'tree': tree,
                   'records': prior_records, 'pending': [], 'progress': template['progress'] if imported else [],
                   'histories': [t['nodes'] for t in completed], 'rollouts': completed, 'revisions': [],
                   'controller': copy.deepcopy(INITIAL_CONTROLLER), 'initial_controller': copy.deepcopy(INITIAL_CONTROLLER),
                   'rollout_controller': copy.deepcopy(INITIAL_CONTROLLER), 'initial': template['initial'],
                   'best': template['best'] if imported else copy.deepcopy(template['initial']),
                   'baselines': template['baselines'], 'suite_summary': template['suite_summary'],
                   'stop_reason': 'completed', 'started_at': time.time(),
                   'imported_calls': len(prior_records), 'imported_usd': old_usd,
                   'new_usd_ceiling': allocated, 'milestones': {},
                   'replay_settings': {'workers': config['workers'], 'max_rounds': config['replay_rounds'],
                        'max_depth': config['max_depth'], 'beta_calls': config['beta_calls'], 'beta_parallel': config['beta_parallel']}}
            record_milestones(run)
            runs.append(run)
    state = {'version': 1, 'backend': backend, 'runs': runs, 'wave': 0, 'active_wave': False,
             'status': 'prepared', 'source_hashes': search.source_hashes(project_root),
             'plan': {'kind': 'mixed-feature-long-search', 'trials': trials, 'total_calls_per_arm': total_calls,
                      'max_new_usd': max_new_usd, 'maximum_new_calls': new_calls,
                      'pilot': str(pilot.resolve()), 'pilot_evidence_hashes': evidence,
                      'prepared': str(prepared.resolve()), 'prepared_manifest_sha256': digest(prepared / 'manifest.json'),
                      'milestones': list(MILESTONES), 'validation': 'Only after every search finishes; never fed to discovery',
                      'test': 'Not opened', 'budget_basis': 'Uncached overseas MiMo rates, including pending/unknown reservations',
                      'pricing_source': 'https://mimo.mi.com/docs/en-US/price/pay-as-you-go',
                      'pricing_checked': '2026-09-28'}}
    check_budget(state)
    return state


def report(state, output):
    search.save_artifacts(state, output, make_report=False)
    rows = []
    for run in state['runs']:
        row = {'id': run['id'], 'phase': run['phase'], 'calls': len(run['records']),
               'new_calls': len(run['records']) - run['imported_calls'],
               'new_usd': search.usage(run, state['backend'])['estimated_usd'] - run['imported_usd'],
               'new_usd_ceiling': run['new_usd_ceiling'], 'best_train_score': run['best']['score'],
               'validation_score': run.get('validation', {}).get('score'), 'milestones': run['milestones'],
               'controller_attempts': sum(r['role'] == 'controller' for r in run['records']),
               'controller_updates_accepted': sum(bool(r['accepted']) for log in run['revisions'] + ([run['revision_log']] if run['phase'] == 'controller' else []) for r in log[1:])}
        rows.append(row)
        write_json(output / run['id'] / 'milestones.json', run['milestones'])
    value = {'status': state['status'], 'wave': state['wave'], 'new_usd': new_spend(state),
             'new_usd_cap': state['plan']['max_new_usd'], 'test_opened': False, 'runs': rows}
    write_json(output / 'report.json', value)
    adaptive = any(r['config']['controller_revisions'] for r in state['runs'])
    revisions = max(r['config']['controller_revisions'] for r in state['runs'])
    protocol = (f'Fixed-controller warmup followed by adaptive Dream-RSI, with up to {revisions} controller proposals between cycles. Discovery prompts and runtime features are unchanged. This is a staged continuation, not a fixed/adaptive comparison.' if adaptive else 'All controllers are fixed. No prompt guidance or runtime features changed.')
    lines = ['# Mixed-feature longer search', '', f"Status: **{state['status']}**. New spending, including reservations: **${value['new_usd']:.6f} / ${value['new_usd_cap']:.2f}**.", '',
             f'Trial 1 continues the 12-call pilot; trials 2 and 3 started independently from LRU. {protocol} Validation remains development data; the final test is unopened.', '',
             '| Trial/arm | Phase | Total calls | New calls | New USD | Best training score | Final validation score | Controller attempts / accepted |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        val = 'pending' if r['validation_score'] is None else f"{r['validation_score']:.4%}"
        lines.append(f"| {r['id']} | {r['phase']} | {r['calls']} | {r['new_calls']} | ${r['new_usd']:.6f} | {r['best_train_score']:.4%} | {val} | {r['controller_attempts']} / {r['controller_updates_accepted']} |")
    lines += ['', 'Scores are reductions in extra computation over unlimited cache, relative to LRU—not total prefill savings. Milestone files freeze training-selected policies at 12, 25, 50 and 100 total API attempts when reached. Adaptive-stage attempts include both policy and controller proposals. Costs use configured uncached rates; provider cache discounts are not deducted. Unknown responses retain their full reservation. Per-run spending ceilings can stop trials before 100 calls.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    return value


def tick(state, output, project_root, transport=None):
    checkpoint = output / 'checkpoint.json'
    check_budget(state)
    if state['active_wave']:
        jobs = [j for r in state['runs'] for j in r['pending']]
        if state['backend'] == 'mock':
            mock = Mock(output / 'mock-transport.json')
            results = {j['id']: {'program': mock.generate(j['role'], j['prompt'], j['context']), 'error': None,
                       'usage': {'prompt_tokens': 0, 'completion_tokens': 0}, 'explicit_failure': False} for j in jobs}
        else:
            transport = transport or RealtimeTransport(project_root, output / 'requests', workers=4)
            results = transport.poll(state['wave'], jobs)
        updated = copy.deepcopy(state)
        for run in updated['runs']:
            if run['pending']:
                search.apply_results(run, results)
                record_milestones(run)
        updated['active_wave'] = False
        check_budget(updated)
        write_json(checkpoint, updated)
        state.clear(); state.update(updated)
    for run in state['runs']:
        advance(run)
        check_budget(state)
        write_json(checkpoint, state)
    if any(r['pending'] for r in state['runs']):
        state.update(wave=state['wave'] + 1, active_wave=True, status='searching')
    else:
        # Freeze every training winner before reading ANY validation outcomes.
        freeze = output / 'frozen-selection.json'
        if not freeze.exists():
            write_json(freeze, {'source_hashes': state['source_hashes'],
                'prepared_manifest_sha256': state['plan']['prepared_manifest_sha256'],
                'selections': [{'id': r['id'], 'calls': len(r['records']), 'policy_features': r['arm'],
                                'candidate': r['best']['candidate']} for r in state['runs']]})
        state['status'] = 'validating'
        write_json(checkpoint, state)
        for run in state['runs']:
            if run['phase'] == 'done': continue
            if 'validation_baselines' not in run:
                # All suites are identical; no timing-dependent selection uses these baseline times.
                previous = next((r for r in state['runs'] if 'validation_baselines' in r), None)
                run['validation_baselines'] = (copy.deepcopy(previous['validation_baselines']) if previous else
                                              search.evaluate(run['config'], validation=True))
            if not run['validation_baselines']['valid']:
                raise ValueError('Validation baselines failed')
            run['validation'] = search.evaluate(run['config'], run['best']['candidate'], validation=True,
                                                baselines=run['validation_baselines']['baselines'])
            if not run['validation']['valid']:
                raise ValueError('Validation failed')
            run.update(phase='done', finished_at=time.time())
            write_json(checkpoint, state)
            report(state, output)
        state['status'] = 'completed'
    write_json(checkpoint, state)
    return report(state, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, default=Path('runs/mixed-task-v1-pilot-bounded'))
    parser.add_argument('--prepared', type=Path, default=Path('data/mixed/task-v1'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-new-usd', type=float, default=4.0)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--total-calls', type=int, default=100)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--continue-from', type=Path, help='Stopped fixed-controller run to continue with dreams under its existing budget')
    parser.add_argument('--controller-revisions', type=int, default=2)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.resume and args.continue_from:
        parser.error('--resume and --continue-from are mutually exclusive')
    rootdir = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    state = None
    with (output / 'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if (output / 'superseded.json').exists():
                raise ValueError('This run was superseded; resume its continuation instead')
            if args.continue_from:
                state = continue_with_dreams(args.continue_from, output, rootdir, args.controller_revisions)
            elif args.resume:
                state = json.loads((output / 'checkpoint.json').read_text())
                if state['source_hashes'] != search.source_hashes(rootdir):
                    raise ValueError('Source changed since preparation')
                prepared = Path(state['plan']['prepared'])
                if digest(prepared / 'manifest.json') != state['plan']['prepared_manifest_sha256']:
                    raise ValueError('Prepared manifest changed')
                verify_prepared(prepared)
                for run in state['runs']: prepare_config(run['config'], rootdir)
            else:
                state = initialize(args.pilot, args.prepared, rootdir, trials=args.trials,
                                   total_calls=args.total_calls, max_new_usd=args.max_new_usd)
                write_json(output / 'plan.json', state['plan'])
                for rel in state['source_hashes']:
                    path = output / 'source' / rel; path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes((rootdir / rel).read_bytes())
            state.pop('error', None)
            write_json(output / 'checkpoint.json', state)
            while state['status'] != 'completed':
                if args.prepare_only and state['active_wave']: break
                result = tick(state, output, rootdir)
                print(json.dumps({'status': result['status'], 'wave': result['wave'],
                    'new_usd': result['new_usd'], 'new_calls': sum(r['new_calls'] for r in result['runs'])}), flush=True)
            report(state, output)
        except Exception as exc:
            if state is not None:
                state.update(status='needs_attention', error=f'{type(exc).__name__}: {exc}')
                write_json(output / 'checkpoint.json', state)
                report(state, output)
            raise


if __name__ == '__main__':
    main()
