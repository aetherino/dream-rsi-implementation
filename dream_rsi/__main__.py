"""python -m dream_rsi --config configs/dream-smoke.json --backend mock"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from .runner import run


def main():
    parser = argparse.ArgumentParser(description="Dream-RSI cache-policy and controller search")
    parser.add_argument("--config", type=Path, default=Path("configs/dream-smoke.json"))
    parser.add_argument("--backend", choices=["mock", "mimo"], default="mock")
    parser.add_argument("--output", type=Path, help="New run directory; existing directories are never overwritten")
    parser.add_argument("--max-usd", type=float, help="Override estimated API budget, including controller development")
    parser.add_argument("--max-calls", type=int, help="Override total API call limit")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent.parent
    output = args.output or project_root / "runs" / ("dream-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:6])
    try:
        config = json.loads(args.config.read_text())
        config.setdefault("api", {})
        if args.max_usd is not None:
            config["api"]["max_usd"] = args.max_usd
        if args.max_calls is not None:
            config["api"]["max_calls"] = args.max_calls
        summary = run(config, args.backend, project_root, output)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"Run artifacts: {output.resolve()}")
    if summary["validation_valid"] is False:
        parser.exit(2, "Held-out validation failed; inspect validation.json\n")


if __name__ == "__main__":
    main()
