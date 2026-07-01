"""Evaluate a saved SAC policy or random policy on the Level 2 environment."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from freemesh_rl.rl import (
    base_result,
    dependency_status,
    evaluate_random_policy,
    evaluate_sb3_model,
    load_config,
    make_env_from_config,
    write_result_files,
)


INSTALL_HINT = 'Install RL extras with: python -m pip install -e ".[rl]"'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--model-path", "--model", dest="model_path", help="Saved Stable-Baselines3 SAC model path")
    parser.add_argument("--random-policy", action="store_true", help="Evaluate random actions instead of a saved model")
    parser.add_argument("--episodes", type=int, help="Override evaluation episode count")
    parser.add_argument("--out-dir", "--out", dest="out_dir", help="Override output directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    start = time.perf_counter()
    config = load_config(args.config)
    outputs = config.get("outputs", {}) or {}
    out_dir = Path(args.out_dir or outputs.get("run_dir", "runs/local_smoke"))
    episodes = int(args.episodes or config.get("eval_episodes", 4))
    deps = dependency_status()
    result = base_result("evaluate_policy", config)

    try:
        env = make_env_from_config(config)
        if args.random_policy:
            result["status"] = "completed_random_policy"
            result["evaluation_summary"] = evaluate_random_policy(env, episodes, seed=int(config.get("seed", 0)))
            return_code = 0
        elif not deps.has_sb3_stack:
            result["status"] = "skipped_missing_dependency"
            result["message"] = INSTALL_HINT
            return_code = 2
        else:
            model_path = Path(args.model_path or outputs.get("model_path", "models/local_smoke_sac.zip"))
            if not model_path.exists():
                result["status"] = "failed_missing_model"
                result["message"] = f"Model not found: {model_path}"
                return_code = 1
            else:
                result["status"] = "completed"
                result["model_path"] = str(model_path)
                result["evaluation_summary"] = evaluate_sb3_model(model_path, env, episodes)
                return_code = 0
    except ModuleNotFoundError as exc:
        result["status"] = "skipped_missing_dependency"
        result["message"] = f"{exc}. {INSTALL_HINT}"
        return_code = 2
    finally:
        result["elapsed_seconds"] = time.perf_counter() - start
        json_path, md_path = write_result_files(out_dir, result, stem="evaluate_policy")
        print(f"wrote {json_path}")
        print(f"wrote {md_path}")
        if result["status"] == "skipped_missing_dependency":
            print(result.get("message", INSTALL_HINT), file=sys.stderr)

    return return_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
