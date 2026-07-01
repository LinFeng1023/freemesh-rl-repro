"""Run the deterministic quadrilateral baseline on a manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from freemesh_rl.baseline import run_grid_baseline
from freemesh_rl.data import load_manifest
from freemesh_rl.metrics import summarize_metrics
from freemesh_rl.visualization import plot_mesh


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=12)
    parser.add_argument("--max-plots", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    cases = load_manifest(args.cases)
    results = []
    for idx, case in enumerate(cases):
        result = run_grid_baseline(case.id, case.vertices, resolution=args.resolution)
        results.append(result)
        if idx < args.max_plots:
            plot_mesh(case.vertices, result.quads, args.out / "plots" / f"{case.id}.png", title=case.id)
            plot_mesh(
                case.vertices,
                result.quads,
                args.out / "plots" / f"{case.id}_quality.png",
                title=f"{case.id} quality",
                quality_color=True,
            )
    metrics = [r.metrics for r in results]
    summary = summarize_metrics(metrics)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out / "per_case_metrics.json").write_text(
        json.dumps([m.to_json() for m in metrics], indent=2),
        encoding="utf-8",
    )
    (args.out / "baseline_results.json").write_text(
        json.dumps([r.to_json() for r in results], indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
