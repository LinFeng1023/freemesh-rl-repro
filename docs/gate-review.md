# Local Gate Review

## Verdict

Current local status: **Level 2 reached with caveats**.

GPU-ready status: **ready for a short GPU preflight, not ready for a long paper-metric Level 3 claim**.

Recommendation for next goal: **go for GPU preflight; no-go for claiming Level 3 until domain mapping, full geometry semantics, and metric parity are resolved**.

## Evidence

Completed and pushed:

- Public GitHub repo: https://github.com/LinFeng1023/freemesh-rl-repro
- Project scaffold and tests.
- Paper/reproduction scope docs.
- Validation matrix.
- Synthetic smoke data.
- Baseline metrics and visualization.
- Local SAC smoke training result.
- Paper-domain smoke summary.
- GPU feasibility analysis.
- GPU runbook.

Verified local commands:

```bash
python -m pip install -e ".[dev]"
pytest
python -m freemesh_rl.cli.generate_data --preset smoke --count 30 --out data/smoke --seed 123
python -m freemesh_rl.cli.evaluate_baseline --cases data/smoke/manifest.json --out outputs/baseline_smoke --max-plots 6
python -m pip install -e ".[rl]"
python -m freemesh_rl.cli.train_smoke --config configs/local_smoke.yaml
python -m freemesh_rl.cli.evaluate_policy --config configs/local_smoke.yaml --cases data/smoke/manifest.json --random-policy --episodes 2 --out-dir runs/local_smoke_random_eval_override
```

Observed:

- Tests: `9 passed, 1 skipped` after optional RL deps were installed.
- Baseline smoke: 30 cases, mean completion ratio `0.5884`, valid quad ratio `1.0`, zero invalid accepted quads.
- SAC smoke: 10k timesteps completed on CPU in `122.669` seconds.
- Local CUDA: unavailable.

## Why This Counts As Level 2

The repo now runs the experiment structure:

```text
boundary/domain manifest
  -> RL environment
  -> continuous action space
  -> proxy reward and terminal path
  -> SAC smoke training
  -> evaluation
  -> metrics
  -> visualization/reporting
```

This is an experimental-structure reproduction. It is not paper-metric reproduction.

## Caveats

- The RL environment uses a fast Level 2 proxy action/reward loop, not the full paper boundary-cutting element extraction algorithm.
- The deterministic baseline is a contained-grid sanity baseline, not Blossom-Quad, Pave, or the paper's learned policy.
- Exact paper domains `T1` and `D7-D9` are not yet mapped into this repo.
- Hole-containing domains are not fully supported by the local single-loop geometry contract.
- Several paper metrics remain substitutes or unimplemented for parity: singularity, stretch, taper, and exact scaled Jacobian.
- Checkpoint/resume support for long GPU runs should be added before expensive Level 3 training.

## Local Blockers Before GPU

No blocker for a **short GPU preflight**.

Blockers before a **long Level 3 run**:

1. Add checkpoint callback/resume support.
2. Map or recreate paper domains.
3. Decide whether to port upstream reference domains with attribution.
4. Validate metric definitions against paper/upstream/Verdict/Gmsh.
5. Decide whether to implement full boundary-cutting geometry semantics.

## Go / No-Go

Go:

- Rent a modest NVIDIA GPU for setup validation and a short 10k/100k preflight.
- Use the GPU runbook to clone, install, generate data, run tests, train, evaluate, and download artifacts.

No-go:

- Do not claim Table 6 reproduction.
- Do not run an expensive multi-million-step job before checkpointing and metric/domain parity are fixed.
