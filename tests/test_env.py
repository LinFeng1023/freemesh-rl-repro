from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from freemesh_rl.env import ACTION_SIZE, OBSERVATION_SIZE, FreeMeshEnv
from freemesh_rl.rl import dependency_status


def test_reset_and_step_shapes() -> None:
    env = FreeMeshEnv(max_steps=3)
    obs, info = env.reset(seed=7)
    assert obs.shape == (OBSERVATION_SIZE,)
    assert obs.dtype == np.float32
    assert "case_id" in info

    next_obs, reward, terminated, truncated, step_info = env.step(np.zeros(ACTION_SIZE, dtype=np.float32))
    assert next_obs.shape == (OBSERVATION_SIZE,)
    assert isinstance(float(reward), float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "metrics" in step_info


def test_random_policy_smoke_episode() -> None:
    env = FreeMeshEnv(max_steps=4)
    obs, _ = env.reset(seed=11)
    total_reward = 0.0
    for _ in range(8):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break
    assert obs.shape == (OBSERVATION_SIZE,)
    assert np.isfinite(total_reward)
    assert "best_metrics" in info


def test_train_smoke_fallback_random_writes_result(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "seed: 5",
                "total_timesteps: 16",
                "eval_episodes: 2",
                "max_steps: 3",
                "data:",
                "  cases: missing_manifest.json",
                "outputs:",
                f"  run_dir: {run_dir}",
                f"  model_path: {tmp_path / 'model.zip'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "freemesh_rl.cli.train_smoke",
            "--config",
            str(config_path),
            "--fallback-random",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads((run_dir / "train_smoke.json").read_text(encoding="utf-8"))
    assert result["status"] == "completed_random_fallback"
    assert result["evaluation_summary"]["episodes"] == 2


def test_missing_optional_deps_path_or_installed_stack(tmp_path: Path) -> None:
    if dependency_status().has_sb3_stack:
        pytest.skip("SB3 stack is installed; missing dependency branch is not active")

    run_dir = tmp_path / "run"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(["eval_episodes: 1", "max_steps: 2", "outputs:", f"  run_dir: {run_dir}"]) + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, "-m", "freemesh_rl.cli.train_smoke", "--config", str(config_path)],
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 2
    result = json.loads((run_dir / "train_smoke.json").read_text(encoding="utf-8"))
    assert result["status"] == "skipped_missing_dependency"
    assert "Install RL extras" in result["message"]
