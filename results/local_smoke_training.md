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

## Completed Smoke Command

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

## SAC Dependency Status

At the time of the base local smoke run:

```json
{
  "gymnasium": true,
  "stable_baselines3": false,
  "torch": false,
  "device": "unavailable"
}
```

Actual SAC training requires:

```bash
python -m pip install -e ".[rl]"
python -m freemesh_rl.cli.train_smoke --config configs/local_smoke.yaml
```

The `configs/local_smoke.yaml` command is configured for `10000` timesteps. `configs/local_smoke_100k.yaml` is configured for `100000` timesteps.

## Interpretation

The completed fallback is not paper-scale SAC training. It is a CPU smoke check of the training/evaluation plumbing. A real Level 3 run must install Torch/Stable-Baselines3, run the SAC configs, save checkpoints, and compare learning/evaluation metrics on paper-like domains.
