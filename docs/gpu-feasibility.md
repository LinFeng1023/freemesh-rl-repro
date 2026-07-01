# GPU Feasibility

## Local Hardware Check

Local machine:

```text
Model: iMac Pro style host
CPU: Intel Core Processor 13th Gen, 14 cores
Memory: 32 GB
GPU: AMD Radeon RX 570, 8 GB VRAM
CUDA: unavailable
```

Python/Torch check in the local venv:

```text
python: 3.12.13
torch: 2.2.2
cuda_available: False
mps_available: True
scaffold device report: cpu
```

Conclusion: this machine is suitable for CPU smoke tests, but not for the final paper-scale NVIDIA GPU reproduction.

## Local Timing

Completed SAC smoke command:

```bash
python -m freemesh_rl.cli.train_smoke --config configs/local_smoke.yaml
```

Observed:

```text
timesteps: 10000
elapsed_seconds: 122.669
device: cpu
```

Simple extrapolation from the local CPU run:

| Steps | Estimated local CPU time |
|---:|---:|
| 10k | 2.0 minutes measured |
| 100k | 20.4 minutes |
| 1M | 3.4 hours |
| 1.2M paper-scale Table 1 | 4.1 hours |
| 4M extended curriculum | 13.6 hours |

These estimates cover the current proxy environment, not full paper-parity boundary surgery or large-domain evaluation.

## Recommended GPU

Minimum:

- NVIDIA T4 or RTX 3060 12 GB
- 32 GB system RAM
- 100 GB disk

Recommended:

- NVIDIA L4, A10, RTX 4070+, or better
- 32-64 GB system RAM
- 150 GB disk

CUDA is preferred over MPS/CPU for the Level 3 goal because the paper used an NVIDIA GTX 1080 Ti and because reproducibility tooling around PyTorch/Stable-Baselines3 is most mature on CUDA.

## Go/No-Go For Renting GPU

Go for a short GPU preflight if the goal is to validate setup, dependency installation, data generation, and checkpoint/resume.

Do not yet book a long Level 3 run expecting paper metrics until these items are addressed:

- Map paper domains `T1` and `D7-D9` to exact upstream or recreated domain files.
- Decide whether to implement full boundary-cutting polygon surgery rather than the local proxy environment.
- Validate paper metric definitions for scaled Jacobian, stretch, taper, and singularity.
- Add checkpoint interval/resume validation on a small GPU run.
