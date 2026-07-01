# Project Notes

## Goal

Create a GPU-ready FreeMesh-RL reproduction package that reaches Level 2 locally and leaves a clean path to a later Level 3 reproduction run.

## Reproduction Levels

- Level 0: code runs.
- Level 1: functional reproduction with simple boundaries, mesh output, visualization, metrics, and a baseline.
- Level 2: experimental-structure reproduction from boundary/domain to RL environment, SAC/policy smoke training, evaluation, and metrics.
- Level 3: paper-metric reproduction, reserved for a future GPU goal.
- Level 4: extensions after Level 3.

## Operating Constraints

- Do not commit the paper PDF, large datasets, checkpoints, tensorboard logs, training runs, model weights, or raw output dumps.
- Commit and push each verifiable stage.
- Keep the public repo honest about what has and has not been reproduced.
