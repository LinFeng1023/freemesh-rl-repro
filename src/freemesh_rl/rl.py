"""RL helpers shared by smoke training and evaluation CLIs."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .env import FreeMeshEnv, load_cases


@dataclass(frozen=True)
class DependencyStatus:
    gymnasium: bool
    stable_baselines3: bool
    torch: bool
    device: str

    @property
    def has_sb3_stack(self) -> bool:
        return self.gymnasium and self.stable_baselines3 and self.torch

    def as_dict(self) -> dict[str, Any]:
        return {
            "gymnasium": self.gymnasium,
            "stable_baselines3": self.stable_baselines3,
            "torch": self.torch,
            "device": self.device,
        }


def dependency_status() -> DependencyStatus:
    torch_available = find_spec("torch") is not None
    device = "unavailable"
    if torch_available:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception as exc:  # pragma: no cover - defensive reporting only.
            device = f"torch_error:{exc.__class__.__name__}"
    return DependencyStatus(
        gymnasium=find_spec("gymnasium") is not None,
        stable_baselines3=find_spec("stable_baselines3") is not None,
        torch=torch_available,
        device=device,
    )


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def make_env_from_config(config: dict[str, Any]) -> FreeMeshEnv:
    data_cfg = config.get("data", {}) or {}
    manifest = data_cfg.get("cases")
    max_steps = int(config.get("max_steps", 8))
    return FreeMeshEnv(cases=load_cases(manifest), max_steps=max_steps)


def evaluate_random_policy(env: FreeMeshEnv, episodes: int, seed: int | None = None) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    episode_rewards: list[float] = []
    successes = 0
    best_quality: list[float] = []

    for episode in range(int(episodes)):
        _, info = env.reset(seed=None if seed is None else seed + episode)
        done = False
        truncated = False
        total_reward = 0.0
        last_info = info
        while not (done or truncated):
            action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
            _, reward, done, truncated, last_info = env.step(action)
            total_reward += float(reward)
        metrics = (last_info.get("best_metrics") or last_info.get("metrics") or {})
        successes += int(bool(metrics.get("valid", False)))
        best_quality.append(float(metrics.get("mean_quality", 0.0)))
        episode_rewards.append(total_reward)

    return {
        "episodes": int(episodes),
        "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "success_rate": float(successes / max(len(episode_rewards), 1)),
        "mean_best_quality": float(np.mean(best_quality)) if best_quality else 0.0,
        "episode_rewards": episode_rewards,
    }


def evaluate_sb3_model(model_path: str | Path, env: FreeMeshEnv, episodes: int) -> dict[str, Any]:
    from stable_baselines3 import SAC

    model = SAC.load(str(model_path), env=env)
    episode_rewards: list[float] = []
    successes = 0
    qualities: list[float] = []
    for episode in range(int(episodes)):
        obs, info = env.reset(seed=episode)
        done = False
        truncated = False
        total_reward = 0.0
        last_info = info
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, last_info = env.step(action)
            total_reward += float(reward)
        metrics = (last_info.get("best_metrics") or last_info.get("metrics") or {})
        successes += int(bool(metrics.get("valid", False)))
        qualities.append(float(metrics.get("mean_quality", 0.0)))
        episode_rewards.append(total_reward)
    return {
        "episodes": int(episodes),
        "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "success_rate": float(successes / max(len(episode_rewards), 1)),
        "mean_best_quality": float(np.mean(qualities)) if qualities else 0.0,
        "episode_rewards": episode_rewards,
    }


def train_sac_smoke(config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    from stable_baselines3 import SAC

    env = make_env_from_config(config)
    training = config.get("training", {}) or {}
    policy_kwargs = {}
    if "net_arch" in training:
        policy_kwargs["net_arch"] = list(training["net_arch"])

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=float(training.get("learning_rate", 3e-4)),
        buffer_size=int(training.get("buffer_size", 50000)),
        batch_size=int(training.get("batch_size", 128)),
        gamma=float(training.get("gamma", 0.99)),
        policy_kwargs=policy_kwargs or None,
        verbose=0,
        seed=int(config.get("seed", 0)),
    )
    total_timesteps = int(config.get("total_timesteps", 1000))
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    eval_summary = evaluate_sb3_model(config.get("outputs", {}).get("model_path", ""), env, 0) if False else {}
    return model, {"total_timesteps": total_timesteps, "eval_summary": eval_summary}


def write_result_files(run_dir: str | Path, result: dict[str, Any], stem: str = "result") -> tuple[Path, Path]:
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_markdown_result(result), encoding="utf-8")
    return json_path, md_path


def _markdown_result(result: dict[str, Any]) -> str:
    lines = [
        f"# FreeMesh-RL {result.get('mode', 'smoke')} Result",
        "",
        f"- status: `{result.get('status')}`",
        f"- device: `{result.get('dependency_status', {}).get('device', 'unknown')}`",
        f"- elapsed_seconds: `{result.get('elapsed_seconds', 0.0):.3f}`",
        f"- timesteps: `{result.get('timesteps', 0)}`",
        f"- python: `{platform.python_version()}`",
        "",
        "## Evaluation",
        "",
    ]
    evaluation = result.get("evaluation_summary") or {}
    for key in ("episodes", "mean_reward", "success_rate", "mean_best_quality"):
        if key in evaluation:
            lines.append(f"- {key}: `{evaluation[key]}`")
    lines.extend(["", "## Dependencies", ""])
    for key, value in (result.get("dependency_status") or {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def base_result(mode: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "mode": mode,
        "status": "started",
        "config": config or {},
        "dependency_status": dependency_status().as_dict(),
        "elapsed_seconds": 0.0,
        "timesteps": int((config or {}).get("total_timesteps", 0)),
        "evaluation_summary": {},
        "started_at_unix": time.time(),
    }
