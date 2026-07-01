"""Train or smoke-run the Level 2 FreeMesh-RL proxy environment."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from freemesh_rl.rl import (
    base_result,
    dependency_status,
    evaluate_random_policy,
    load_config,
    make_env_from_config,
    train_sac_smoke,
    write_result_files,
)


INSTALL_HINT = 'Install RL extras with: python -m pip install -e ".[rl]"'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML smoke-training config path")
    parser.add_argument("--fallback-random", action="store_true", help="Run local random-policy smoke mode without SB3")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    start = time.perf_counter()
    config = load_config(args.config)
    outputs = config.get("outputs", {}) or {}
    run_dir = Path(outputs.get("run_dir", "runs/local_smoke"))
    eval_episodes = int(config.get("eval_episodes", 4))
    deps = dependency_status()
    result = base_result("train_smoke", config)

    try:
        if args.fallback_random:
            env = make_env_from_config(config)
            result["status"] = "completed_random_fallback"
            result["timesteps"] = 0
            result["evaluation_summary"] = evaluate_random_policy(env, eval_episodes, seed=int(config.get("seed", 0)))
            return_code = 0
        elif not deps.has_sb3_stack:
            result["status"] = "skipped_missing_dependency"
            result["message"] = INSTALL_HINT
            result["missing_dependencies"] = [
                name for name, present in deps.as_dict().items() if name != "device" and not present
            ]
            return_code = 2
        else:
            model, train_summary = train_sac_smoke(config)
            model_path = Path(outputs.get("model_path", "models/local_smoke_sac.zip"))
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(model_path))
            env = make_env_from_config(config)
            from freemesh_rl.rl import evaluate_sb3_model

            result["status"] = "completed"
            result["model_path"] = str(model_path)
            result["timesteps"] = int(train_summary["total_timesteps"])
            result["evaluation_summary"] = evaluate_sb3_model(model_path, env, eval_episodes)
            return_code = 0
    except ModuleNotFoundError as exc:
        result["status"] = "skipped_missing_dependency"
        result["message"] = f"{exc}. {INSTALL_HINT}"
        return_code = 2
    finally:
        result["elapsed_seconds"] = time.perf_counter() - start
        json_path, md_path = write_result_files(run_dir, result, stem="train_smoke")
        print(f"wrote {json_path}")
        print(f"wrote {md_path}")
        if result["status"] == "skipped_missing_dependency":
            print(result.get("message", INSTALL_HINT), file=sys.stderr)

    return return_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
