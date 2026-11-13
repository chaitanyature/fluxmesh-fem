import argparse
import json
from pathlib import Path

from .artifacts import audit, bundle
from .cases import CASES
from .solver import run


def main():
    p = argparse.ArgumentParser(
        description="Adaptive triangular finite elements for planar scalar diffusion"
    )
    sub = p.add_subparsers(dest="command", required=True)
    runp = sub.add_parser("solve")
    runp.add_argument("case", choices=CASES)
    runp.add_argument("--initial", type=int, default=4)
    runp.add_argument("--steps", type=int, default=5)
    runp.add_argument("--theta", type=float, default=0.55)
    runp.add_argument("--mode", choices=("adaptive", "uniform"), default="adaptive")
    runp.add_argument("--max-cells", type=int, default=150000)
    runp.add_argument("--out", required=True, type=Path)
    sub.add_parser("audit").add_argument("directory", type=Path)
    sub.add_parser("compare").add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    try:
        if args.command == "audit":
            result = audit(args.directory)
        elif args.command == "compare":
            from .plot import compare

            result = compare(args.out)
        else:
            settings = {
                k: v for k, v in vars(args).items() if k not in ("out", "command")
            }
            solution, history = run(**settings)
            bundle(args.out, solution, history, settings)
            result = solution.metrics
        print(json.dumps(result, indent=2, allow_nan=False))
        return int(result.get("valid") is False)
    except (OSError, ValueError, KeyError) as exc:
        p.exit(2, f"fluxmesh: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
