# Local Smoke Training

## Environment

- Local venv Python: `3.12.13`
- Base install command:

```bash
python -m pip install -e ".[dev]"
```

- Test command:

```bash
pytest
```

Observed test result:

```text
10 passed
```

## Completed Random-Policy Fallback

Command:

```bash
python -m freemesh_rl.cli.train_smoke --config configs/local_smoke.yaml --fallback-random
```

Observed result:

```text
status: completed_random_fallback
elapsed_seconds: 0.638
episodes: 8
mean_reward: 5.771305725403625
success_rate: 1.0
mean_best_quality: 1.0
```

This verifies the local Level 2 RL environment loop: fixed-size observations, continuous actions, reward calculation, terminal handling, result writing, and evaluation reporting.

## Completed SAC Smoke

After installing optional RL dependencies:

```bash
python -m pip install -e ".[rl]"
```

Command:

```bash
python -m freemesh_rl.cli.train_smoke --config configs/local_smoke.yaml
```

Observed result:

```text
status: completed
elapsed_seconds: 122.669
timesteps: 10000
episodes: 8
mean_reward: 5.931200448128681
success_rate: 1.0
mean_best_quality: 0.9716672028112507
device: cpu
```

The saved model path was `models/local_smoke_sac.zip`, which is intentionally ignored by Git.

## Dependency Status

After optional RL dependency installation:

```json
{
  "gymnasium": true,
  "stable_baselines3": true,
  "torch": true,
  "device": "cpu"
}
```

Direct Torch hardware check:

```text
torch: 2.2.2
cuda_available: False
mps_available: True
```

The scaffold currently reports CPU because CUDA is unavailable and the training CLI does not force MPS. That is acceptable for local smoke; Level 3 should run on NVIDIA CUDA.

## 100k Local Run

The 100k local SAC config was not run. Based on the 10k CPU runtime of `122.669` seconds, a direct 100k local run would be roughly `20.4` minutes before overhead and is not needed for the current CPU preflight. The 100k command remains available:

```bash
python -m freemesh_rl.cli.train_smoke --config configs/local_smoke_100k.yaml
```

## Interpretation

The completed 10k SAC run is not paper-scale training. It is a CPU smoke check of the training/evaluation plumbing. A real Level 3 run must install Torch/Stable-Baselines3 on a CUDA machine, run the GPU config, save checkpoints, and compare learning/evaluation metrics on paper-like domains.
