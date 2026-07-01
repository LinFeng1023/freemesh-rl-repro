#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/gpu_level3.yaml}"

python -m pip install --upgrade pip
python -m pip install -e ".[dev,rl]"
python -m freemesh_rl.cli.generate_data --preset dev --count 1000 --out data/train --seed 356
python -m freemesh_rl.cli.generate_data --preset hard_validation --count 300 --out data/eval_large --seed 999
python -m freemesh_rl.cli.train_smoke --config "$CONFIG"
python -m freemesh_rl.cli.evaluate_policy --config "$CONFIG" --model models/gpu_level3_sac.zip --cases data/eval_large/manifest.json --out outputs/gpu_level3_eval
