"""Generate synthetic boundary case manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from freemesh_rl.data import DEFAULT_COUNTS, generate_cases, write_cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(DEFAULT_COUNTS), default="smoke")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cases = generate_cases(args.preset, args.count, args.seed)
    manifest = write_cases(cases, args.out, args.preset, args.seed)
    print(json.dumps({"manifest": str(manifest), "count": len(cases), "preset": args.preset}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
