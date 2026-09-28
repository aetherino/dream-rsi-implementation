"""Run: python -m cache_sim --capacities 32 64 128 --output runs/demo.json"""
import argparse
import json
from pathlib import Path

from .engine import sweep
from .policies import BASELINES
from .trace import compile_trace, load_trace, save_trace, synthetic_trace


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU prefix-cache replay; defaults to seeded synthetic data")
    parser.add_argument("--trace", type=Path, help="Pretokenized schema-v1 JSON")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sessions", type=int, default=40)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--capacities", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--policies", nargs="+", choices=list(BASELINES), default=list(BASELINES))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-trace", type=Path)
    parser.add_argument("--details", action="store_true", help="Include per-request outcomes in JSON")
    parser.add_argument("--check-invariants", action="store_true", help="Debug capacity/prefix accounting; adds overhead")
    args = parser.parse_args()
    try:
        trace = load_trace(args.trace) if args.trace else synthetic_trace(args.seed, args.sessions)
        compiled = compile_trace(trace, args.block_size)
        report = sweep(compiled, args.capacities, args.policies, details=args.details,
                       check_invariants=args.check_invariants)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f"error: {error}\n")
    report["trace_source"] = str(args.trace.resolve()) if args.trace else f"synthetic:seed={args.seed}:sessions={args.sessions}"
    if args.save_trace:
        save_trace(trace, args.save_trace)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(trace.requests)} requests; block size {args.block_size} tokens")
    print(f"Unlimited-cache prompt computation: {report['unlimited']['computed_prompt_tokens']:,} tokens")
    print(f"{'policy':<8} {'blocks':>7} {'hit %':>8} {'computed':>10} {'extra':>10} {'evicted':>9} {'policy ms':>11}")
    for run in report["runs"]:
        print(f"{run['policy']:<8} {run['capacity_blocks']:>7} {100*run['prompt_hit_ratio']:>8.2f} "
              f"{run['computed_prompt_tokens']:>10,} {run['extra_computed_tokens']:>10,} "
              f"{run['evicted_blocks']:>9,} {run['policy_time_ms']:>11.3f}")
    if args.output:
        print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
