# GPU Runbook

This runbook prepares a future NVIDIA GPU run for Level 3 work. The current repository is Level 2 plus GPU-ready scaffolding; do not report paper-metric reproduction until the Level 3 run is complete and validated.

## Cloud Machine

Minimum:

- Ubuntu 22.04 or 24.04
- NVIDIA T4 or RTX 3060 12 GB
- 32 GB RAM
- 100 GB disk
- Python 3.10-3.12
- CUDA-compatible NVIDIA driver

Recommended:

- NVIDIA L4, A10, RTX 4070+, or better
- 32-64 GB RAM
- 150 GB disk

## Clone

```bash
git clone https://github.com/LinFeng1023/freemesh-rl-repro.git
cd freemesh-rl-repro
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,rl]"
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY
pytest
```

Stop if `torch.cuda.is_available()` is false on a rented NVIDIA machine.

## Generate Data

Smoke data:

```bash
python -m freemesh_rl.cli.generate_data --preset smoke --count 30 --out data/smoke --seed 123
```

GPU train/eval data:

```bash
python -m freemesh_rl.cli.generate_data --preset dev --count 1000 --out data/train --seed 356
python -m freemesh_rl.cli.generate_data --preset hard_validation --count 300 --out data/eval_large --seed 999
```

## Baseline Evaluation

```bash
python -m freemesh_rl.cli.evaluate_baseline --cases data/smoke/manifest.json --out outputs/baseline_smoke --max-plots 12
python -m freemesh_rl.cli.evaluate_baseline --cases data/eval_large/manifest.json --out outputs/baseline_eval_large --max-plots 24
```

## Train

Short GPU preflight:

```bash
python -m freemesh_rl.cli.train_smoke --config configs/local_smoke.yaml
```

Level 3-scale proxy config:

```bash
python -m freemesh_rl.cli.train_smoke --config configs/gpu_level3.yaml
```

Convenience script:

```bash
scripts/run_gpu_training.sh configs/gpu_level3.yaml
```

## Evaluate

```bash
python -m freemesh_rl.cli.evaluate_policy \
  --config configs/gpu_level3.yaml \
  --model-path models/gpu_level3_sac.zip \
  --cases data/eval_large/manifest.json \
  --out-dir outputs/gpu_level3_eval
```

`--cases` overrides the manifest path in the config for evaluation.

## Resume Checkpoint

The current smoke trainer writes final Stable-Baselines3 models but does not yet implement periodic checkpoint callbacks. For a long GPU run, add or verify checkpoint callback support first. Until then, treat this as the stop condition:

```bash
test -f models/gpu_level3_sac.zip
```

## Download Artifacts

From local machine:

```bash
rsync -avz user@GPU_HOST:/path/to/freemesh-rl-repro/results/ results/gpu/
rsync -avz user@GPU_HOST:/path/to/freemesh-rl-repro/outputs/gpu_level3_eval/ outputs/gpu_level3_eval/
rsync -avz user@GPU_HOST:/path/to/freemesh-rl-repro/runs/gpu_level3/ runs/gpu_level3/
rsync -avz user@GPU_HOST:/path/to/freemesh-rl-repro/models/gpu_level3_sac.zip models/
```

Do not commit downloaded model weights or large run directories to Git.

## Expected Outputs

- `data/train/manifest.json`
- `data/eval_large/manifest.json`
- `runs/gpu_level3/train_smoke.json`
- `runs/gpu_level3/train_smoke.md`
- `models/gpu_level3_sac.zip`
- `outputs/gpu_level3_eval/evaluate_policy.json`
- `outputs/gpu_level3_eval/evaluate_policy.md`

## Estimated Time And Cost

Local CPU 10k proxy SAC timing was `122.669` seconds. A cloud GPU should be used first for a short preflight, then for larger runs only after confirming CUDA and checkpoint behavior.

Rough planning:

- Short preflight: 10k steps, under 15 minutes including setup.
- 1.2M proxy steps: reserve 4-8 GPU hours until measured on target hardware.
- Extended 4M proxy/curriculum run: reserve 12-24 GPU hours.

Actual cloud cost depends on provider and GPU. Start with the shortest billed session that allows setup plus 10k preflight.

## Stop Conditions

Stop the run if:

- CUDA is unavailable.
- `pytest` fails.
- Data generation fails.
- 10k preflight fails or does not write `runs/local_smoke/train_smoke.json`.
- Training reward/evaluation summary is missing or non-finite.
- Checkpoint/model output is not created after training.
- Metrics code changes are needed to make claims; do not continue into expensive training before metric validity is fixed.
