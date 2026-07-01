#!/usr/bin/env bash
set -euo pipefail

python -m freemesh_rl.cli.generate_data --preset smoke --count 30 --out data/smoke --seed 123
python -m freemesh_rl.cli.evaluate_baseline --cases data/smoke/manifest.json --out outputs/baseline_smoke
python -m freemesh_rl.cli.train_smoke --config configs/local_smoke.yaml --fallback-random
