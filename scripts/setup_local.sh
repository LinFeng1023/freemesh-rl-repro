#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

cat <<'MSG'
Local setup complete.

Optional RL dependencies:
  python -m pip install -e ".[rl]"
MSG
