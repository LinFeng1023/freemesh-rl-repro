# Metrics

## Implemented Local Metrics

The local scaffold computes metrics for generated quadrilateral meshes. These metrics support Level 2 experiment-structure checks and smoke comparisons. They are not yet a claim of exact paper-metric parity.

| Metric | Implemented field | Notes |
|---|---|---|
| Completion rate | `completion_rate` | Fraction of cases with at least 98% area coverage, valid quads, and nonzero quads |
| Completion ratio | `completion_ratio` | Sum of valid quad areas divided by source polygon area |
| Valid quad ratio | `valid_quad_ratio` | Valid convex quadrilaterals divided by generated elements |
| Invalid count | `invalid_count` | Count of invalid generated elements |
| Element quality | `element_quality_mean`, `element_quality_min` | Uses the paper's Eq. 7 shape-quality structure: edge quality and angle quality |
| Angle quality | `min_angle_deviation_mean`, `max_angle_deviation_mean` | Deviation from 90 degrees |
| Aspect ratio | `aspect_ratio_mean` | Longest edge divided by shortest edge |
| Scaled Jacobian substitute | `scaled_jacobian_substitute_min` | Minimum corner sine diagnostic; named as substitute until validated against Verdict/Gmsh |
| Runtime | `runtime_seconds` | Wall-clock seconds for the local baseline attempt |

## Baseline Interpretation

The deterministic baseline is an axis-aligned contained-grid quadrilateral baseline. It only accepts complete cells inside the polygon. This makes it robust and easy to audit, but conservative on concave or irregular boundaries.

That means:

- `valid_quad_ratio` can be high while `completion_ratio` is partial.
- Concave, noisy, or narrow-channel polygons may show lower area coverage even when all accepted quads are valid.
- This baseline is not Blossom-Quad, Pave, or the learned FreeMesh-RL policy from the paper.

## Paper Metrics Still Requiring Level 3 Validation

The paper reports singularity, element quality, min/max angle deviations, scaled Jacobian, stretch, taper, and triangle count. Before Level 3 claims, these must be aligned with the paper/upstream definitions or a geometry-quality library such as Verdict/Gmsh.

The local scaffold currently marks scaled Jacobian as a substitute and does not yet compute paper-parity singularity, stretch, or taper.

## Smoke Baseline Result

Command:

```bash
python -m freemesh_rl.cli.evaluate_baseline --cases data/smoke/manifest.json --out outputs/baseline_smoke --max-plots 6
```

Observed summary on the local machine:

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

The `completion_rate` is intentionally strict: the contained-grid baseline leaves boundary-adjacent gaps for many non-rectangular shapes, so this result should be read as a baseline sanity check, not a reproduction-quality mesh result.
