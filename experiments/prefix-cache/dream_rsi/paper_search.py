"""Executable-program Dream-RSI following the paper's Section 3 algorithm.

Cache workload, model, budgets and coefficients are declared task adaptations.
The incompatible appendix AUC objective is deliberately not implemented here.
"""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import time

from .backend import write_json
from .batch_compare import source_hashes
from .batch_transport import cost
from .code_controller import CodeController, INITIAL_CONTROLLER_SPEC, evaluate_controller_source
from .code_policy import evaluate_code_suite
from .code_prompts import discovery_messages, controller_messages
from .code_transport import CodeTransport, request_body, source_allowance
from .evaluation import digest, isolated_evaluate
from .history import observation
from .mixed import verify_prepared
from .runner import prepare_config

LRU_CODE = {'name': 'initial-lru-code', 'source': '''class CachePolicy:
    def choose(self, eligible, now):
        return min(eligible, key=lambda entry: (entry.last_access, entry.access_order)).block_id
''', 'rationale': 'Fixed initial LRU workspace; each new search tree starts here.'}


def read_json(path):
    return json.loads(Path(path).read_text())


def feedback_record(result):
    """The complete authoritative score.json presented for each proposal.

    Bulk simulator counters remain in evaluation.json; no workload content is exposed.
    """
    keys = ('valid', 'score', 'error', 'scoring', 'total_extra_computed_tokens',
            'total_lru_extra_computed_tokens', 'saved_prompt_tokens', 'score_denominator_tokens')
    value = {k: result[k] for k in keys if k in result}
    value['scenarios'] = [{k: row[k] for k in ('name', 'extra_computed_tokens', 'lru_extra_computed_tokens',
         'improvement', 'policy_us_per_request', 'score_contribution') if k in row} for row in result.get('runs', [])]
    return value


def evaluate_policy(config, program, baselines, *, validation=False):
    try:
        return evaluate_code_suite(config['validation' if validation else 'train'], config['block_size'],
            candidate=program, baselines=baselines['baselines'], scoring=config['scoring'],
            policy_features=config['policy_features'], max_policy_us_per_request=config['max_policy_us_per_request'])
    except Exception as exc:
        return {'valid': False, 'score': -1e6, 'error': f'{type(exc).__name__}: {exc}'}


def run_cost(run):
    return sum(record['charged_estimate_usd'] for record in run['records'])


def spent(state):
    return sum(run_cost(run) for run in state['runs'])


def check_budget(state):
    if sum(r['usd_cap'] for r in state['runs']) > state['plan']['max_usd'] + 1e-9:
        raise ValueError('Arm budgets exceed experiment cap')
    if spent(state) > state['plan']['max_usd'] + 1e-9:
        raise ValueError('Experiment spending cap exceeded; no further requests permitted')


def initialize(predecessor, root, *, max_usd=10, cycles=5, online_rounds=11,
               controller_revisions=8, workers=2):
    prior = read_json(predecessor / 'checkpoint.json')
    if prior['status'] != 'paused_by_user':
        raise ValueError('Predecessor must remain stopped before a fresh experiment')
    prepared = Path(prior['plan']['prepared'])
    verify_prepared(prepared)
    if digest(prepared/'manifest.json') != prior['plan']['prepared_manifest_sha256']:
        raise ValueError('Prepared dataset changed')
    template = next(r for r in prior['runs'] if r['arm'] == 'task-v1')
    config = copy.deepcopy(template['config'])
    config.update(cycles=cycles, online_rounds=online_rounds, replay_rounds=online_rounds,
                  max_depth=online_rounds, workers=workers, controller_revisions=controller_revisions,
                  beta_calls=.001, beta_parallel=.00025, revise_after_final_cycle=False)
    config['api'].update(max_completion_tokens=8192, max_calls=200, max_usd=max_usd/2)
    config = prepare_config(config, root)
    baselines = copy.deepcopy(template['baselines'])
    initial_result = evaluate_policy(config, LRU_CODE, baselines)
    if not initial_result['valid'] or abs(initial_result['score']) > 1e-12:
        raise ValueError(f'Executable LRU failed baseline parity: {initial_result.get("error") or initial_result.get("score")}')
    initial = {'id': 0, 'parent': None, 'depth': 0, 'round': 0, 'score': 0., 'valid': True,
               'stagnation': 0, 'failure_streak': 0, 'candidate': LRU_CODE,
               'feedback': feedback_record(initial_result)}
    suite_summary = [{'name': item['name'], 'capacity_blocks': item['capacity_blocks'],
                      'lru_extra_tokens': b['policies']['lru']['extra_computed_tokens']}
                     for item,b in zip(config['train'],baselines['baselines'])]
    runs = []
    for arm in ('fixed', 'dream'):
        runs.append({'id': arm, 'config': copy.deepcopy(config), 'usd_cap': max_usd/2,
            'phase': 'discovery', 'cycle': 1, 'round': 1, 'controller': copy.deepcopy(INITIAL_CONTROLLER_SPEC),
            'rollout_controller': copy.deepcopy(INITIAL_CONTROLLER_SPEC), 'tree': [copy.deepcopy(initial)],
            'histories': [], 'rollouts': [], 'decisions': [], 'revisions': [], 'records': [], 'pending': [],
            'initial': copy.deepcopy(initial), 'best': copy.deepcopy(initial), 'baselines': baselines,
            'suite_summary': suite_summary, 'stop_reason': None, 'progress': [],
            'replay_settings': {'workers': workers, 'max_rounds': online_rounds, 'max_depth': online_rounds,
                                'beta_calls': config['beta_calls'], 'beta_parallel': config['beta_parallel']}})
    return {'version': 1, 'status': 'prepared', 'wave': 0, 'active_wave': False, 'runs': runs,
        'started_at': time.time(), 'source_hashes': source_hashes(root),
        'plan': {'specification': 'Dream-RSI arXiv:2609.14858v1 Section 3',
            'paper': 'https://arxiv.org/html/2609.14858v1#S3',
            'max_usd': max_usd, 'budget_scope': 'Fresh allowance authorized by user; earlier runs excluded',
            'predecessor': str(predecessor.resolve()), 'predecessor_checkpoint_sha256': digest(predecessor/'checkpoint.json'),
            'prepared': str(prepared.resolve()), 'prepared_manifest_sha256': digest(prepared/'manifest.json'),
            'cycles': cycles, 'online_rounds': online_rounds, 'replay_rounds': online_rounds,
            'workers': workers, 'controller_proposals_per_phase': controller_revisions,
            'controller_versions_per_phase': controller_revisions + 1,
            'beta_calls': config['beta_calls'], 'beta_parallel': config['beta_parallel'],
            'objective': 'Eq.1, arithmetic mean over every recorded history; appendix AUC not used',
            'revision_rule': 'Develop sequentially from the latest proposed version, then deploy best including incumbent',
            'baseline': 'Executable parallel-refinement controller using Section 3 single-root action interface',
            'context': 'All proposal sources, rationales, authoritative scores and errors; no history truncation',
            'initialization': 'Fresh LRU trees; no old expression-policy history imported',
            'comparison': 'One independent fixed/adaptive pair; matched model, task, limits, initial code, and dollar caps',
            'validation': 'Freeze both winners before validation; validation never enters search prompts',
            'test': 'Unopened', 'source_ambiguities': ['Appendix AUC/beta sweep and grid planner excluded by user choice',
                'Section 3 single-root actions differ from Section 4 simultaneous workspace initialization'],
            'adaptations': ['MiMo-v2.6-pro and prefix-cache task', '2 workers rather than paper model-specific 10/32',
                '5 cycles, 11 live/replay rounds, 8 revisions, explicit task-scaled coefficients',
                'Stateful standalone Python programs, not a shell coding agent or multi-file workspace',
                'Complete score.json diagnostics in context; detailed simulator counters archived separately',
                '950000-byte conservative input allowance; stop rather than omit history'],
            'pricing': {'url': 'https://mimo.mi.com/docs/en-US/price/pay-as-you-go', 'checked': '2026-09-28',
                        'input_per_million': .435, 'output_per_million': .87, 'cache_discounts_deducted': False}},
        'initial_evaluation': initial_result}


def reserve(run, role, messages, context, parent=None):
    api = run['config']['api']
    try:
        allowance = source_allowance(messages)
    except ValueError as exc:
        run['stop_reason'] = f'full_history_context_limit: {exc}'
        return None
    amount = cost(api, allowance, api['max_completion_tokens'])
    if len(run['records']) >= api['max_calls'] or run_cost(run) + amount > run['usd_cap']:
        run['stop_reason'] = 'budget_reached'
        return None
    call = len(run['records'])+1
    ident = f"{run['id']}-call-{call:04d}"
    run['records'].append({'id': ident, 'call': call, 'role': role, 'context': context,
        'status': 'reserved', 'charged_estimate_usd': amount, 'input_token_allowance': allowance})
    return {'id':ident, 'role':role, 'context':context, 'parent':parent,
            'prompt':messages, 'body':request_body(api,messages)}


def live_actions(run):
    # Reconstruct private controller state from its exact previously visible prefixes.
    with CodeController(run['controller']) as policy:
        for decision in run['decisions']:
            actions = policy.select(run['tree'][:decision['visible_count']], decision['round'],
                                    run['config']['workers'], run['config']['max_depth'])
            if actions != decision['actions']:
                raise ValueError('Controller is not deterministic on saved prefixes')
        return policy.select(run['tree'],run['round'],run['config']['workers'],run['config']['max_depth'])


def finish_cycle(run):
    run['histories'].append(copy.deepcopy(run['tree']))
    run['rollouts'].append({'cycle': run['cycle'], 'controller': copy.deepcopy(run['rollout_controller']),
                            'nodes': copy.deepcopy(run['tree']), 'decisions': copy.deepcopy(run['decisions'])})


def next_cycle(run):
    run['cycle'] += 1
    run.update(tree=[copy.deepcopy(run['initial'])], round=1, decisions=[], phase='discovery',
               rollout_controller=copy.deepcopy(run['controller']))


def advance(run):
    config = run['config']
    while not run['pending'] and run['phase'] not in ('search_done','done'):
        if run['phase'] == 'controller':
            if run['revision'] > config['controller_revisions'] or run['stop_reason']:
                run['controller'] = copy.deepcopy(run['dream_best']['controller'])
                run['revisions'].append(copy.deepcopy(run['revision_log']))
                next_cycle(run)
                continue
            try:
                messages = controller_messages(run['development_current'],run['development_feedback'],
                    run['histories'],run['replay_settings'],run['revision_log'],config['scoring'],
                    online_rounds=config['online_rounds'])
            except ValueError as exc:
                if 'input allowance' not in str(exc): raise
                run['stop_reason'] = f'full_history_context_limit: {exc}'
                continue
            job=reserve(run,'controller',messages,{'cycle':run['cycle'],'revision':run['revision']})
            if job: run['pending'].append(job)
            continue
        if run['stop_reason'] or run['cycle'] > config['cycles']:
            if len(run['tree'])>1: finish_cycle(run)
            run['phase']='search_done'
            break
        try:
            actions=live_actions(run) if run['round']<=config['online_rounds'] else []
        except Exception as exc:
            run['stop_reason']=f'controller_execution_failed: {type(exc).__name__}: {exc}'
            continue
        if not actions:
            finish_cycle(run)
            if run['id']=='dream' and run['cycle']<config['cycles']:
                feedback=evaluate_controller_source(run['controller'],run['histories'],**run['replay_settings'])
                if not feedback['valid']:
                    run['stop_reason']=f'controller_replay_failed: {feedback.get("error")}'
                    next_cycle(run)
                    continue
                incumbent={'revision':0,'controller':copy.deepcopy(run['controller']),'evaluation':feedback}
                run.update(phase='controller',revision=1,development_current=copy.deepcopy(run['controller']),
                    development_feedback=feedback,dream_best=copy.deepcopy(incumbent),revision_log=[incumbent])
            else:
                next_cycle(run)
            continue
        scheduled=[]
        for offset,parent_id in enumerate(actions):
            parent=next(n for n in run['tree'] if n['id']==parent_id)
            try:
                messages=discovery_messages(parent,run['tree'],run['suite_summary'],run['histories'],config['scoring'],
                                             policy_features=config['policy_features'],best=run['best'])
            except ValueError as exc:
                if 'input allowance' not in str(exc): raise
                run['stop_reason'] = f'full_history_context_limit: {exc}'
                break
            job=reserve(run,'discovery',messages,{'cycle':run['cycle'],'node':len(run['tree'])+offset},parent_id)
            if job is None: break
            scheduled.append(parent_id);run['pending'].append(job)
        if scheduled:
            # Budget may stop a partial batch. It ends the cycle after these outcomes.
            run['decisions'].append({'round':run['round'],'visible_count':len(run['tree']),
                                     'actions':actions,'submitted_actions':scheduled})


def apply_results(run,results,output):
    jobs=run['pending'][:]
    for job in jobs:
        result=results[job['id']]
        record=next(r for r in run['records'] if r['id']==job['id'])
        usage=result.get('usage',{})
        if all(type(usage.get(k)) is int and usage[k]>=0 for k in ('prompt_tokens','completion_tokens')):
            record.update(status='accounted',**usage,
                charged_estimate_usd=cost(run['config']['api'],usage['prompt_tokens'],usage['completion_tokens']))
        else:
            record['status']='usage_unknown_reservation_retained'
        record['error']=result.get('error');record['finish_reason']=result.get('finish_reason')
        proposal=result.get('program')
        if job['role']=='discovery':
            evaluation=({'valid':False,'score':-1e6,'error':result['error']} if result.get('error') else
                        evaluate_policy(run['config'],proposal,run['baselines']))
            parent=next(n for n in run['tree'] if n['id']==job['parent'])
            node=observation(job['context']['node'],parent,run['round'],proposal,feedback_record(evaluation))
            node['api_call']=record['call'];run['tree'].append(node)
            if node['valid'] and node['score']>run['best']['score']:
                run['best']=copy.deepcopy(node)|{'cycle':run['cycle']}
            run['progress'].append({'call':record['call'],'cycle':run['cycle'],'round':run['round'],
                'depth':node['depth'],'valid':node['valid'],'score':node['score'],'best_train_score':run['best']['score']})
            folder=output/run['id']/f"cycle-{run['cycle']:03d}"/f"attempt-{node['id']:04d}"
            write_json(folder/'proposal.json',proposal);write_json(folder/'score.json',node['feedback'])
            write_json(folder/'evaluation.json',evaluation)
            if isinstance(proposal,dict) and isinstance(proposal.get('source'),str):
                (folder/'policy.py').write_text(proposal['source'])
        else:
            feedback=({'valid':False,'value':None,'error':result['error']} if result.get('error') else
                evaluate_controller_source(proposal,run['histories'],**run['replay_settings']))
            revision={'revision':run['revision'],'controller':proposal,'evaluation':feedback,'api_call':record['call']}
            run['revision_log'].append(revision)
            if feedback['valid'] and feedback['value']>run['dream_best']['evaluation']['value']+1e-12:
                run['dream_best']=copy.deepcopy(revision)
            # Deliberately keep a worse or invalid intermediate as the next development base.
            run['development_current']=copy.deepcopy(proposal)
            run['development_feedback']=feedback
    run['round' if jobs[0]['role']=='discovery' else 'revision']+=1
    run['pending']=[]


def report(state,output):
    rows=[]
    for run in state['runs']:
        folder=output/run['id'];folder.mkdir(exist_ok=True)
        write_json(folder/'config.json',run['config']);write_json(folder/'usage.json',run['records'])
        write_json(folder/'progress.json',run['progress']);write_json(folder/'best-policy.json',run['best'])
        write_json(folder/'controller.json',run['controller']);write_json(folder/'rollouts.json',run['rollouts'])
        logs=run['revisions']+([run['revision_log']] if run['phase']=='controller' else [])
        write_json(folder/'controller-revisions.json',logs)
        for job in run['pending']:
            write_json(folder/'prompts'/(job['id']+'.json'),job['prompt'])
        rows.append({'id':run['id'],'phase':run['phase'],'cycle':run['cycle'],'round':run['round'],
            'calls':len(run['records']),'discovery_calls':sum(r['role']=='discovery' for r in run['records']),
            'controller_calls':sum(r['role']=='controller' for r in run['records']),
            'usd':run_cost(run),'usd_cap':run['usd_cap'],'best_train_score':run['best']['score'],
            'validation_score':run.get('validation',{}).get('score'),'stop_reason':run['stop_reason'],
            'completed_dream_phases':len(run['revisions'])})
    value={'status':state['status'],'wave':state['wave'],'usd_including_reservations':spent(state),
           'usd_cap':state['plan']['max_usd'],'earlier_spending_excluded':True,'test_opened':False,'runs':rows}
    write_json(output/'report.json',value)
    lines=['# Section 3 executable-program search','',
        f"Status: **{state['status']}**. Fresh budget: **${spent(state):.6f} / ${state['plan']['max_usd']:.2f}**, including pending/unknown reservations.",'',
        'Section 3 algorithm with documented cache-task adaptations. Stateful Python policies and controllers; complete proposal history; sequential revisions and best-version deployment. Final test remains unopened.','',
        '| Arm | Phase | Cycle/round | Policy calls | Controller calls | USD | Best training score |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {r['id']} | {r['phase']} | {r['cycle']}/{r['round']} | {r['discovery_calls']} | {r['controller_calls']} | ${r['usd']:.6f} | {r['best_train_score']:.4%} |")
    lines+=['','Task scores measure reductions in extra recomputation relative to LRU, not GPU speedups. See plan.json for the precise protocol and adaptations.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return value


def tick(state,output,root,transport=None):
    check_budget(state)
    if state['active_wave']:
        jobs=[j for r in state['runs'] for j in r['pending']]
        transport=transport or CodeTransport(root,output/'requests',workers=4)
        outcomes=transport.poll(state['wave'],jobs)
        updated=copy.deepcopy(state)
        for run in updated['runs']:
            if run['pending']: apply_results(run,outcomes,output)
        updated['active_wave']=False
        check_budget(updated);write_json(output/'checkpoint.json',updated)
        state.clear();state.update(updated)
    for run in state['runs']:
        advance(run);check_budget(state);write_json(output/'checkpoint.json',state)
    if any(r['pending'] for r in state['runs']):
        state.update(active_wave=True,wave=state['wave']+1,status='searching')
    else:
        path=output/'frozen-selection.json'
        if not path.exists():
            write_json(path,{'source_hashes':state['source_hashes'],'selections':[
                {'id':r['id'],'candidate':r['best']['candidate'],'train_score':r['best']['score']} for r in state['runs']]})
        state['status']='validating';write_json(output/'checkpoint.json',state)
        for run in state['runs']:
            if run['phase']=='done':continue
            previous=next((r for r in state['runs'] if 'validation_baselines' in r),None)
            run['validation_baselines']=(copy.deepcopy(previous['validation_baselines']) if previous else
                isolated_evaluate({'suite':run['config']['validation'],'block_size':run['config']['block_size'],
                    'scoring':run['config']['scoring']},run['config']['evaluation_timeout_seconds']))
            if not run['validation_baselines']['valid']:raise ValueError('Validation baselines failed')
            run['validation']=evaluate_policy(run['config'],run['best']['candidate'],run['validation_baselines'],validation=True)
            write_json(output/run['id']/'validation.json',run['validation'])
            run['phase']='done';write_json(output/'checkpoint.json',state)
        state['status']='completed';state['finished_at']=time.time()
    write_json(output/'checkpoint.json',state)
    return report(state,output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--predecessor',type=Path,default=Path('runs/mixed-task-dream-20260928'))
    parser.add_argument('--max-usd',type=float,default=10.)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=args.resume)
    state=None
    with (output/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if args.resume:
                state=read_json(output/'checkpoint.json')
                if state['source_hashes']!=source_hashes(root):raise ValueError('Source changed since preparation')
                prepared=Path(state['plan']['prepared']);verify_prepared(prepared)
                if digest(prepared/'manifest.json')!=state['plan']['prepared_manifest_sha256']:raise ValueError('Prepared data changed')
                for run in state['runs']:prepare_config(run['config'],root)
            else:
                state=initialize(args.predecessor,root,max_usd=args.max_usd)
                write_json(output/'plan.json',state['plan'])
                write_json(output/'initial-code-evaluation.json',state['initial_evaluation'])
                for rel in state['source_hashes']:
                    path=output/'source'/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((root/rel).read_bytes())
            state.pop('error',None);write_json(output/'checkpoint.json',state)
            while state['status']!='completed':
                if args.prepare_only and state['active_wave']:break
                result=tick(state,output,root)
                print(json.dumps({k:result[k] for k in ('status','wave','usd_including_reservations')}),flush=True)
            report(state,output)
        except Exception as exc:
            if state is not None:
                state.update(status='needs_attention',error=f'{type(exc).__name__}: {exc}')
                write_json(output/'checkpoint.json',state);report(state,output)
            raise


if __name__=='__main__':main()
