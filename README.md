# FreeMesh-RL Reproduction Package

Local reproduction engineering for the paper:

> Reinforcement learning for automatic quadrilateral mesh generation: A soft actor-critic approach

Current target: **Level 2 experimental-structure reproduction** plus a GPU-ready runbook for a later Level 3 paper-metric reproduction attempt.

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

The runnable package is developed under `src/freemesh_rl`. The expected local path after clone is:

```bash
git clone <repo-url>
cd freemesh-rl-repro
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
```

After the implementation stages are complete, the main local smoke commands will be:

```bash
python -m freemesh_rl.cli.generate_data --preset smoke --out data/smoke
python -m freemesh_rl.cli.evaluate_baseline --cases data/smoke/manifest.json --out outputs/baseline_smoke
python -m freemesh_rl.cli.train_smoke --config configs/local_smoke.yaml
```

## Repository Status

- Stage 0: project initialization.
- Stage 1: paper and code evidence review pending.
- Stage 2-9: implementation, validation, GPU runbook, and gate review pending.

## Paper Reference

The local paper PDF is stored at `references/papers/RL for automatic quad mesh generation.pdf` for private reference only and is ignored by Git.

Public documentation will cite only bibliographic metadata, DOI/link, and validation notes.
