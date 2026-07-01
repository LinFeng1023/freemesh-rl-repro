# Reproduction Scope

## Target

This project prepares a local, GPU-ready reproduction package for:

Pan, Jie; Huang, Jingwei; Cheng, Gengdong; Zeng, Yong. "Reinforcement learning for automatic quadrilateral mesh generation: a soft actor-critic approach." *Neural Networks* 157 (2023): 288-304. DOI: `10.1016/j.neunet.2022.10.022`. arXiv: `2203.11203`.

The current goal is **Level 2: experimental-structure reproduction**. It does not claim the paper's final quantitative metrics.

## What Level 2 Means Here

The local package must run the full experiment shape:

1. Generate or load 2D polygonal boundary/domain cases.
2. Expose a mesh-generation environment with state, action, reward, and terminal conditions.
3. Run a deterministic baseline.
4. Run a SAC-compatible smoke-training path, or a documented dependency-skipped/random-policy fallback on CPU.
5. Evaluate generated quadrilateral meshes with geometry metrics.
6. Produce plots and concise result reports.
7. Provide GPU instructions for a later Level 3 run.

## Paper-Evidenced Algorithm Structure

The paper formulates quadrilateral mesh generation as a Markov decision process. At each time step, the meshing policy selects or constructs one quadrilateral element from the current boundary, cuts that element off, and repeats until the remaining boundary can close as a final element.

The paper-evidenced chain is:

```text
piecewise-linear boundary
  -> local reference vertex and partial boundary state
  -> action type plus generated vertex coordinates
  -> quadrilateral candidate
  -> reward from validity, element quality, remaining-boundary quality, and density
  -> updated boundary
  -> final mesh metrics and visualization
```

Important paper settings recovered from the PDF:

- RL algorithm compared against PPO, DDPG, and TD3; SAC is selected for stability and learning performance.
- Training hardware: i7-8700 CPU, Nvidia GTX 1080 Ti GPU, 32 GB RAM.
- SAC network setting selected in the paper: hidden layers `[128, 128, 128]`.
- SAC training total steps in Table 1: `1.2e6`.
- Minibatch size: `256`.
- Discount factor: `0.99`.
- Learning rates for Q and policy networks: `3e-4`.
- Paper training domain selection uses candidate domains `T1`, `T2`, and `T3`, with `T1` selected.
- Paper state/action parameter studies include observation ranges, action radius factor, and density reward parameter.

## Local Implementation Scope

This repository implements a compact, modern Python scaffold rather than directly vendoring or patching the public upstream code. The local scaffold is intentionally narrower:

- Single outer-loop polygon cases are the primary supported geometry.
- Hole-containing domains may be generated or referenced for future evaluation, but full multi-loop boundary surgery is not treated as complete in the local Level 2 claim.
- The deterministic baseline is a simple quadrilateral-grid containment baseline used for sanity checks and metrics, not Blossom-Quad, Pave, or the learned paper policy.
- The local reward is a Level 2 proxy unless explicitly marked otherwise. It is designed to exercise the same state/action/reward/evaluation path, not to prove paper reward parity.
- Scaled Jacobian, stretch, and taper metrics may be implemented as local substitutes unless matched against Verdict/Gmsh definitions in a later Level 3 pass.

## Public Reference Code

Public repository found:

- URL: https://github.com/design-zeng/ReinforcementLearning4MeshGeneration
- License: MIT.
- Default branch: `main`.
- Observed HEAD during local evidence pass: `f756fcf16684d2638956357a2978bccc8be5fc2d`.
- Last observed push: 2025-06-27.
- No releases or tags observed.
- Contains implementation-looking files such as `rl/boundary_env.py`, `rl/baselines/RL_Mesh.py`, `general/components.py`, `general/mesh.py`, `Measurement/quality_verdict.py`, and many `ui/domains/*.json` files.

This repository is treated as a reference source, not as a fully verified official paper artifact. It was created after the paper publication date, includes older ANN/RL work as well as the SAC paper citation, and uses local Windows paths in configuration. No trained checkpoints or paper-version release were found.

## Non-goals

- 3D surface or volume meshing.
- Full CAD/CAE product behavior.
- Long local GPU training.
- Reproducing the paper's Table 6 metrics in the current local phase.
- Claiming parity with Blossom-Quad or Pave baselines.
- Publishing the paper PDF, large training data, checkpoints, TensorBoard runs, or model weights.

## Level 3 Work Left For GPU Goal

Level 3 requires a separate GPU run that:

- Confirms exact paper-domain mappings or recreates equivalent domains.
- Runs SAC for paper-scale timesteps, including `1.2e6` or larger planned curricula.
- Records seeds, hardware, wall-clock cost, checkpoints, and learning curves.
- Evaluates against paper-like domains and paper metrics.
- Validates metric definitions against known geometry-quality conventions.
- Clearly separates exact reproductions from approximations.
