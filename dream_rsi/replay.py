"""Evaluate a controller against saved histories with zero LLM calls."""
import argparse
import json
from pathlib import Path

from .backend import write_json
from .history import evaluate_controller


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--controller", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        config = json.loads((args.run / "config.json").read_text())
        spec = json.loads(args.controller.read_text())
        histories = [json.loads(path.read_text())["nodes"] for path in sorted(args.run.glob("tree-*.json"))]
        if not histories:
            raise ValueError("No discovery histories in this run")
        result = evaluate_controller(spec, histories, workers=config["workers"], max_rounds=config["replay_rounds"],
                                     max_depth=config["max_depth"], beta_calls=config["beta_calls"],
                                     beta_parallel=config["beta_parallel"])
        if args.output:
            # Preserve recorded run data; replay outputs must be new files.
            if args.output.exists():
                raise ValueError("Replay output already exists")
            write_json(args.output, result)
        print(json.dumps(result, indent=2, allow_nan=False))
        if not result["valid"]:
            parser.exit(2)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
