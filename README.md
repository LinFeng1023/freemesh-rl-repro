# FreeMesh-RL Reproduction Package

Local reproduction engineering for the paper:

> Reinforcement learning for automatic quadrilateral mesh generation: A soft actor-critic approach

Current status: **Level 2 experimental-structure reproduction reached with documented caveats** plus a GPU-ready runbook for a later Level 3 paper-metric reproduction attempt.

This repository does not claim paper-metric reproduction yet. The local phase is intended to make the CPU smoke path, synthetic boundary generation, baseline evaluation, metrics, visualization, SAC-style training entry points, and GPU runbook ready before renting NVIDIA GPU time.

## Scope

- Reproduce the experiment structure: boundary/domain generation, RL environment, policy training smoke test, evaluation, metrics, and visualization.
- Provide baseline quad-mesh generation and measurable outputs for small local domains.
- Prepare scripts and documentation for a future GPU Level 3 run.
- Keep large data, logs, checkpoints, model weights, and paper PDFs out of the public repository.

## Non-goals

- 3D surface meshing.
- Industrial CAD/CAE product completeness.
- LLM-agent wrappers or unrelated multi-agent systems.
- Long GPU training in the current local phase.
- Claims that Level 3 paper metrics have already been reproduced.

## Quick Start

The runnable package is developed under `src/freemesh_rl`. Use Python 3.10-3.12.

```bash
git clone https://github.com/LinFeng1023/freemesh-rl-repro.git
cd freemesh-rl-repro
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

Base smoke commands:

```bash
python -m freemesh_rl.cli.generate_data --preset smoke --count 30 --out data/smoke --seed 123
python -m freemesh_rl.cli.evaluate_baseline --cases data/smoke/manifest.json --out outputs/baseline_smoke
python -m freemesh_rl.cli.train_smoke --config configs/local_smoke.yaml --fallback-random
```

SAC smoke training:

```bash
python -m pip install -e ".[rl]"
python -m freemesh_rl.cli.train_smoke --config configs/local_smoke.yaml
```

## Repository Status

- Public repo created and pushed.
- Paper/reproduction scope and validation matrix written.
- Python package, synthetic data generator, baseline, metrics, visualization, RL environment, and tests implemented.
- Smoke data and demo report committed.
- Local 10k SAC smoke completed on CPU.
- GPU feasibility analysis, GPU runbook, and gate review written.
- Level 3 paper-metric reproduction remains future work.

## Key Docs

- `docs/reproduction-scope.md`
- `docs/paper-validation-matrix.md`
- `docs/data-generation.md`
- `docs/metrics.md`
- `docs/gpu-feasibility.md`
- `docs/gpu-runbook.md`
- `docs/gate-review.md`
- `results/local_smoke_training.md`
- `results/paper_domains_summary.md`
- `examples/demo_report.md`

## Paper Reference

The local paper PDF is stored at `references/papers/RL for automatic quad mesh generation.pdf` for private reference only and is ignored by Git.

Public documentation will cite only bibliographic metadata, DOI/link, and validation notes.
