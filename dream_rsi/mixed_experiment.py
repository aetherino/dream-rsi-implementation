"""Two equal-budget feature arms, with an explicit one-use final-test gate."""
import argparse
import json
from pathlib import Path

from .batch_compare import source_hashes
from .evaluation import digest, isolated_evaluate, snapshot_suite
from .mixed import verify_prepared, write_json
from .runner import run
from .programs import RetentionPolicy


def freeze_selection(prepared, search, project_root):
    """Select by training only, then record candidates before opening test results."""
    selections = []
    for features in ('block-v1', 'task-v1'):
        directory = search / features
        summary = json.loads((directory / 'summary.json').read_text())
        config = json.loads((directory / 'config.json').read_text())
        if summary['status'] not in {'completed', 'api_budget_reached'} or summary['validation_valid'] is False:
            raise ValueError('Both searches must finish successfully before freezing')
        if config['policy_features'] != features:
            raise ValueError('Feature arm mismatch')
        candidate = summary['best_policy']
        RetentionPolicy(candidate, features)
        selections.append({'policy_features': features, 'candidate': candidate,
                           'summary_sha256': digest(directory / 'summary.json'),
                           'config_sha256': digest(directory / 'config.json')})
    seal = {'version': 1, 'selection_rule': 'Best training score within each independent feature arm',
            'prepared_manifest_sha256': digest(prepared / 'manifest.json'),
            'source_hashes': source_hashes(project_root), 'selections': selections}
    with (search / 'frozen-selection.json').open('x') as stream:
        json.dump(seal, stream, indent=2)
    return seal


def final_test(prepared, search, project_root):
    verify_prepared(prepared)
    seal = json.loads((search / 'frozen-selection.json').read_text())
    if seal['prepared_manifest_sha256'] != digest(prepared / 'manifest.json'):
        raise ValueError('Prepared data changed since selection')
    if seal['source_hashes'] != source_hashes(project_root):
        raise ValueError('Evaluator source changed since selection')
    selections = seal['selections']
    if {s['policy_features'] for s in selections} != {'block-v1', 'task-v1'} or len(selections) != 2:
        raise ValueError('Expected exactly one frozen policy per arm')
    for selection in selections:
        features = selection['policy_features']
        for name in ('summary', 'config'):
            if selection[f'{name}_sha256'] != digest(search / features / f'{name}.json'):
                raise ValueError('Search artifacts changed since selection')
        RetentionPolicy(selection['candidate'], features)
    # An exclusive receipt is created BEFORE evaluation. Failures also consume the opening.
    receipt = search / 'test-opened.json'
    with receipt.open('x') as stream:
        json.dump({'status': 'started', 'selection_sha256': digest(search / 'frozen-selection.json')}, stream)
    try:
        suite = snapshot_suite(json.loads((prepared / 'test/suite.json').read_text()), project_root)
        payload = {'suite': suite, 'block_size': 16}
        baseline = isolated_evaluate(payload, 900)
        if not baseline['valid']:
            raise ValueError(baseline.get('error'))
        report = {'baselines': baseline, 'arms': {}}
        for selection in selections:
            result = isolated_evaluate(payload | {'candidate': selection['candidate'],
                'policy_features': selection['policy_features'], 'baselines': baseline['baselines']}, 900)
            if not result['valid']:
                raise ValueError(result.get('error'))
            report['arms'][selection['policy_features']] = result
        write_json(search / 'final-test.json', report)
        write_json(receipt, {'status': 'completed', 'selection_sha256': digest(search / 'frozen-selection.json'),
                             'report_sha256': digest(search / 'final-test.json')})
    except Exception as exc:
        write_json(receipt, {'status': 'failed', 'error': str(exc),
                            'selection_sha256': digest(search / 'frozen-selection.json')})
        raise
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['baselines', 'search', 'test'])
    p.add_argument('--prepared', type=Path, default=Path('data/mixed/task-v1'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--backend', choices=['mock', 'mimo'], default='mock')
    p.add_argument('--max-usd-per-arm', type=float, help='Lower the prepared per-arm spending ceiling')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    verify_prepared(args.prepared)
    if args.mode == 'test':
        final_test(args.prepared, args.output, root)
        print('Final test opened once; do not feed its results back into this search.')
        return
    args.output.mkdir(parents=True, exist_ok=False)
    if args.mode == 'baselines':
        suite = snapshot_suite(json.loads((args.prepared / 'train/suite.json').read_text()), root)
        report = isolated_evaluate({'suite': suite, 'block_size': 16}, 900)
        write_json(args.output / 'baselines.json', report)
        write_json(args.output / 'provenance.json', {
            'manifest_sha256': digest(args.prepared / 'manifest.json'),
            'source_hashes': source_hashes(root), 'suite': suite, 'api_cost_usd': 0})
        if not report['valid']:
            raise ValueError(report.get('error'))
        print(f'Training baselines complete: {len(suite)} scenarios. Test remains sealed.')
        return
    for features in ('block-v1', 'task-v1'):
        config = json.loads((args.prepared / 'configs' / f'{features}.json').read_text())
        if args.max_usd_per_arm is not None:
            config['api']['max_usd'] = min(config['api']['max_usd'], args.max_usd_per_arm)
        run(config, args.backend, root, args.output / features)
    freeze_selection(args.prepared, args.output, root)
    print('Both arms complete and selection frozen. Final test remains sealed until explicit test command.')


if __name__ == '__main__':
    main()
