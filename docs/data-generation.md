# Data Generation

## Purpose

The local data generator creates deterministic 2D polygon boundaries for Level 2 reproduction checks. The generated cases are not paper-domain replacements; they are a CPU-friendly scaffold for validating boundary loading, environment resets, baseline meshing, metrics, plots, and training smoke commands.

## Presets

| Preset | Default count | Git policy | Purpose |
|---|---:|---|---|
| `smoke` | 30 | committed | Fast install/test/demo path |
| `dev` | 800 | ignored under `data/generated/` or `data/train/` | Larger local iteration set |
| `hard_validation` | 180 | manifest/seed documentation only unless curated | Harder validation before GPU runs |

The generator cycles through these categories:

- `simple_convex`
- `l_shape`
- `u_shape`
- `c_shape`
- `strong_concavity`
- `narrow_channel`
- `multiscale`
- `noisy`
- `near_degenerate`

These categories map to paper concerns such as sharp angles, bottleneck/narrow regions, uneven boundary segment density, and generalizability to varied 2D shapes.

## Commands

Committed smoke set:

```bash
python -m freemesh_rl.cli.generate_data --preset smoke --count 30 --out data/smoke --seed 123
```

Local development set, not committed:

```bash
python -m freemesh_rl.cli.generate_data --preset dev --count 800 --out data/generated/dev --seed 356
```

Hard validation set, not committed by default:

```bash
python -m freemesh_rl.cli.generate_data --preset hard_validation --count 180 --out data/generated/hard_validation --seed 999
```

GPU-scale training/evaluation manifests:

```bash
python -m freemesh_rl.cli.generate_data --preset dev --count 1000 --out data/train --seed 356
python -m freemesh_rl.cli.generate_data --preset hard_validation --count 300 --out data/eval_large --seed 999
```

## Manifest Schema

`manifest.json` contains:

- `preset`
- `seed`
- `count`
- `categories`
- `cases`

Each case contains:

- `id`
- `category`
- `seed`
- `vertices`
- `notes`
- `path`

The case files repeat the same boundary data so individual examples can be inspected without parsing the whole manifest.

## Limitations

- Current cases are single outer-loop polygons. Hole-containing paper domains are documented as Level 3 work unless a robust multi-loop update contract is added.
- Synthetic cases are generated from fixed templates plus small deterministic jitter. They test mechanics and edge cases; they do not claim paper-domain equivalence.
- Large generated data should stay out of Git.
