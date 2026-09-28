#!/usr/bin/env python3
"""Prepare mixed, split-isolated episodes without tokenizer downloads or API calls."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dream_rsi.mixed import prepare, verify_prepared


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, default=Path('data/tokenized/qwen2.5'))
    p.add_argument('--output', type=Path, default=Path('data/mixed/task-v1'))
    p.add_argument('--episodes', type=int, default=2)
    p.add_argument('--requests', type=int, default=256)
    p.add_argument('--seed', type=int, default=20260928)
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    manifest = verify_prepared(args.output) if args.verify_only else prepare(
        args.source_root, args.output, seed=args.seed, episodes=args.episodes, requests=args.requests)
    print(f"Prepared/verified {len(manifest['files'])} files. API cost $0. Test remains sealed.")


if __name__ == '__main__':
    main()
