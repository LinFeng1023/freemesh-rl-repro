# Paper-Domain Smoke Evaluation

## Current Status

This local phase does **not** yet include exact paper-domain evaluation. The paper uses training/evaluation domains such as `T1` and domains `D7-D9`, including sharp angles, bottlenecks, uneven boundary segment density, and holes.

The current repo covers those concerns structurally through synthetic single-loop categories:

- `simple_convex`
- `l_shape`
- `u_shape`
- `c_shape`
- `strong_concavity`
- `narrow_channel`
- `multiscale`
- `noisy`
- `near_degenerate`

## Smoke Evaluation Run

Command:

```bash
python -m freemesh_rl.cli.evaluate_baseline --cases data/smoke/manifest.json --out outputs/baseline_smoke --max-plots 6
```

Observed summary:

```json
{
  "case_count": 30,
  "completion_rate": 0.0,
  "mean_completion_ratio": 0.588441932511162,
  "mean_valid_quad_ratio": 1.0,
  "mean_element_quality": 0.9999999999999999,
  "total_invalid_count": 0,
  "total_runtime_seconds": 0.560763824993046
}
```

## Upstream Reference Domains

A public reference repository was found at:

```text
https://github.com/design-zeng/ReinforcementLearning4MeshGeneration
```

It contains many `ui/domains/*.json` files. During this local goal, those files were inspected as evidence but not imported into this repository. The mapping between the paper's `T1`, `D7-D9`, and the public JSON files remains unverified.

## Level 3 Follow-up

Before claiming paper-domain evaluation:

1. Map paper domains to upstream JSON files or recreate them from the paper figures.
2. Add a documented importer for selected small domain files if license/attribution checks pass.
3. Add multi-loop/hole support or explicitly exclude hole domains from exact comparison.
4. Evaluate baseline and trained SAC policy on the mapped paper domains.
5. Report paper metrics only after metric definitions are validated.
