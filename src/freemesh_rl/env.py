"""Gymnasium-compatible Level 2 proxy environment for FreeMesh-RL.

This module intentionally implements a simplified single-loop quadrilateral
meshing environment. It is suitable for experimental-structure smoke training,
but it is not a Level 3 reproduction of the paper's full meshing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:  # Gymnasium is optional in pyproject's base install.
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:  # pragma: no cover - exercised by CLI dependency checks.
    gym = None
    spaces = None


OBSERVATION_SIZE = 24
ACTION_SIZE = 3


@dataclass(frozen=True)
class MeshCase:
    """A single simple polygon case for Level 2 smoke evaluation."""

    case_id: str
    vertices: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class MeshAttempt:
    """Metrics from one simplified mesh/baseline attempt."""

    valid: bool
    quad_count: int
    valid_quad_ratio: float
    mean_quality: float
    min_quality: float
    area_coverage: float
    invalid_reason: str | None = None


def _require_gymnasium() -> None:
    if gym is None or spaces is None:
        raise ModuleNotFoundError(
            "gymnasium is required for FreeMeshEnv. Install with: "
            'python -m pip install -e ".[rl]"'
        )


def default_cases() -> list[MeshCase]:
    """Return deterministic synthetic cases used when no manifest is available."""

    return [
        MeshCase("unit_square", ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        MeshCase("wide_rect", ((0.0, 0.0), (1.6, 0.0), (1.6, 1.0), (0.0, 1.0))),
        MeshCase("notched_l", ((0.0, 0.0), (1.0, 0.0), (1.0, 0.55), (0.55, 0.55), (0.55, 1.0), (0.0, 1.0))),
        MeshCase("pentagon", ((0.1, 0.0), (0.9, 0.0), (1.0, 0.65), (0.5, 1.0), (0.0, 0.65))),
    ]


def load_cases(manifest_path: str | Path | None = None) -> list[MeshCase]:
    """Load polygon cases from a JSON manifest, falling back to synthetic cases.

    Accepted manifest shapes are intentionally permissive:
    ``{"cases": [{"id": "...", "vertices": [[x, y], ...]}]}`` or a bare list
    of equivalent case objects. Missing files return :func:`default_cases`.
    """

    if manifest_path is None:
        return default_cases()
    path = Path(manifest_path)
    if not path.exists():
        return default_cases()

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_cases = payload.get("cases", payload) if isinstance(payload, dict) else payload
    cases: list[MeshCase] = []
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            continue
        vertices = raw.get("vertices") or raw.get("boundary") or raw.get("polygon")
        if not vertices:
            continue
        case_id = str(raw.get("id") or raw.get("case_id") or raw.get("name") or f"case_{index:04d}")
        parsed = tuple((float(point[0]), float(point[1])) for point in vertices)
        if len(parsed) >= 3:
            cases.append(MeshCase(case_id=case_id, vertices=parsed))
    return cases or default_cases()


def _polygon_area(vertices: np.ndarray) -> float:
    x = vertices[:, 0]
    y = vertices[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _polygon_perimeter(vertices: np.ndarray) -> float:
    return float(np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1).sum())


def _point_in_polygon(x: float, y: float, vertices: np.ndarray) -> bool:
    inside = False
    j = len(vertices) - 1
    for i in range(len(vertices)):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        crosses = (yi > y) != (yj > y)
        if crosses:
            x_at_y = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            if x < x_at_y:
                inside = not inside
        j = i
    return inside


def _normalize_vertices(vertices: np.ndarray, max_vertices: int = 8) -> np.ndarray:
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) * 0.5
    scale = float(max(maxs[0] - mins[0], maxs[1] - mins[1], 1e-6))
    normalized = (vertices - center) / scale
    if len(normalized) >= max_vertices:
        idx = np.linspace(0, len(normalized) - 1, max_vertices).round().astype(int)
        normalized = normalized[idx]
    else:
        pad = np.zeros((max_vertices - len(normalized), 2), dtype=np.float32)
        normalized = np.vstack([normalized, pad])
    return normalized.astype(np.float32).reshape(-1)


def _cell_quality(width: float, height: float, jitter_penalty: float) -> float:
    ratio = min(width, height) / max(width, height, 1e-12)
    return float(np.clip(ratio * (1.0 - jitter_penalty), 0.0, 1.0))


def _local_proxy_mesh(case: MeshCase, action: np.ndarray) -> MeshAttempt:
    vertices = np.asarray(case.vertices, dtype=np.float64)
    if len(vertices) < 3:
        return MeshAttempt(False, 0, 0.0, 0.0, 0.0, 0.0, "not_enough_vertices")

    area = _polygon_area(vertices)
    if not np.isfinite(area) or area <= 1e-10:
        return MeshAttempt(False, 0, 0.0, 0.0, 0.0, 0.0, "degenerate_polygon")

    resolution = int(np.interp(float(action[0]), [-1.0, 1.0], [3.0, 14.0]).round())
    aspect_bias = float(np.interp(float(action[1]), [-1.0, 1.0], [0.65, 1.55]))
    jitter_penalty = float(np.interp(abs(float(action[2])), [0.0, 1.0], [0.0, 0.32]))

    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    width = max(float(maxs[0] - mins[0]), 1e-6)
    height = max(float(maxs[1] - mins[1]), 1e-6)
    nx = max(2, resolution)
    ny = max(2, int(round(resolution * height / width * aspect_bias)))
    dx = width / nx
    dy = height / ny

    qualities: list[float] = []
    accepted_area = 0.0
    for ix in range(nx):
        for iy in range(ny):
            cx = mins[0] + (ix + 0.5) * dx
            cy = mins[1] + (iy + 0.5) * dy
            if _point_in_polygon(cx, cy, vertices):
                qualities.append(_cell_quality(dx, dy, jitter_penalty))
                accepted_area += dx * dy

    quad_count = len(qualities)
    if quad_count == 0:
        return MeshAttempt(False, 0, 0.0, 0.0, 0.0, 0.0, "no_cells_inside_polygon")

    quality = np.asarray(qualities, dtype=np.float64)
    valid_quad_ratio = float(np.mean(quality >= 0.2))
    mean_quality = float(np.mean(quality))
    min_quality = float(np.min(quality))
    area_coverage = float(np.clip(accepted_area / area, 0.0, 1.5))
    valid = valid_quad_ratio >= 0.5 and area_coverage >= 0.5
    return MeshAttempt(valid, quad_count, valid_quad_ratio, mean_quality, min_quality, area_coverage, None if valid else "low_quality_or_coverage")


def _external_mesh_attempt(case: MeshCase, action: np.ndarray) -> MeshAttempt | None:
    """Try the local baseline module without importing it eagerly."""

    try:
        baseline = import_module("freemesh_rl.baseline")
    except ModuleNotFoundError:
        return None
    runner = (
        getattr(baseline, "run_baseline_attempt", None)
        or getattr(baseline, "generate_baseline_mesh", None)
        or getattr(baseline, "run_grid_baseline", None)
    )
    if runner is None:
        return None
    try:
        result = runner(case=case, action=action)
    except TypeError:
        resolution = int(np.interp(float(action[0]), [-1.0, 1.0], [4.0, 18.0]).round())
        offset_x = float(np.interp(float(action[1]), [-1.0, 1.0], [-0.45, 0.45]))
        offset_y = float(np.interp(float(action[2]), [-1.0, 1.0], [-0.45, 0.45]))
        result = runner(case.case_id, case.vertices, resolution=resolution, offset_x=offset_x, offset_y=offset_y)

    if isinstance(result, MeshAttempt):
        return result
    if hasattr(result, "metrics"):
        metrics = result.metrics
        return MeshAttempt(
            valid=bool(getattr(result, "status", "") in {"complete", "partial"} and metrics.quad_count > 0),
            quad_count=int(metrics.quad_count),
            valid_quad_ratio=float(metrics.valid_quad_ratio),
            mean_quality=float(metrics.element_quality_mean),
            min_quality=float(metrics.element_quality_min),
            area_coverage=float(metrics.completion_ratio),
            invalid_reason=None if getattr(result, "status", "") != "failed" else getattr(result, "reason", "failed"),
        )
    if not isinstance(result, dict):
        return None

    return MeshAttempt(
        valid=bool(result.get("valid", result.get("success", False))),
        quad_count=int(result.get("quad_count", result.get("num_quads", 0))),
        valid_quad_ratio=float(result.get("valid_quad_ratio", result.get("quad_ratio", 0.0))),
        mean_quality=float(result.get("mean_quality", result.get("quality", 0.0))),
        min_quality=float(result.get("min_quality", result.get("mean_quality", 0.0))),
        area_coverage=float(result.get("area_coverage", result.get("coverage", 0.0))),
        invalid_reason=result.get("invalid_reason"),
    )


def run_mesh_attempt(case: MeshCase, action: np.ndarray) -> MeshAttempt:
    """Run a real simplified mesh attempt for the current case/action."""

    external = _external_mesh_attempt(case, action)
    return external if external is not None else _local_proxy_mesh(case, action)


def proxy_reward(metrics: MeshAttempt, completed: bool) -> float:
    """Return the Level 2 proxy reward, not paper Eq. 5-9.

    The reward combines valid quad ratio, element quality, coverage, invalid
    penalties, and a completion bonus. It is meant for smoke training and
    experimental structure validation only.
    """

    reward = (
        2.0 * metrics.valid_quad_ratio
        + 2.0 * metrics.mean_quality
        + 0.75 * min(metrics.area_coverage, 1.0)
        + 0.25 * metrics.min_quality
    )
    if not metrics.valid:
        reward -= 2.0
    if completed and metrics.valid:
        reward += 1.0
    return float(reward)


def _case_features(case: MeshCase, step_index: int, max_steps: int, best: MeshAttempt | None) -> np.ndarray:
    vertices = np.asarray(case.vertices, dtype=np.float64)
    area = _polygon_area(vertices)
    perimeter = _polygon_perimeter(vertices)
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    width = max(float(maxs[0] - mins[0]), 1e-6)
    height = max(float(maxs[1] - mins[1]), 1e-6)
    compactness = 4.0 * math.pi * area / max(perimeter * perimeter, 1e-12)
    descriptor = np.asarray(
        [
            np.clip(area / max(width * height, 1e-12), 0.0, 2.0),
            np.clip(width / max(height, 1e-12), 0.0, 4.0) / 4.0,
            np.clip(compactness, 0.0, 1.0),
            min(len(vertices), 12) / 12.0,
            step_index / max(max_steps, 1),
            0.0 if best is None else best.valid_quad_ratio,
            0.0 if best is None else best.mean_quality,
            0.0 if best is None else min(best.area_coverage, 1.0),
        ],
        dtype=np.float32,
    )
    observation = np.concatenate([_normalize_vertices(vertices), descriptor])
    return observation.astype(np.float32)


if gym is not None:

    class FreeMeshEnv(gym.Env):
        """Single-loop Level 2 proxy RL environment for quadrilateral meshing.

        Actions tune a simplified grid-refinement baseline. Rewards are based on
        proxy mesh metrics and are not the exact FreeMesh-RL paper Eq. 5-9.
        """

        metadata = {"render_modes": []}

        def __init__(
            self,
            cases: Iterable[MeshCase] | None = None,
            max_steps: int = 8,
            terminate_on_valid: bool = True,
        ) -> None:
            super().__init__()
            self.cases = list(cases) if cases is not None else default_cases()
            if not self.cases:
                raise ValueError("FreeMeshEnv requires at least one case")
            self.max_steps = int(max_steps)
            self.terminate_on_valid = bool(terminate_on_valid)
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_SIZE,), dtype=np.float32)
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(OBSERVATION_SIZE,),
                dtype=np.float32,
            )
            self._case: MeshCase = self.cases[0]
            self._step_index = 0
            self._best: MeshAttempt | None = None

        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
            super().reset(seed=seed)
            case_index = None if options is None else options.get("case_index")
            if case_index is None:
                case_index = int(self.np_random.integers(0, len(self.cases)))
            self._case = self.cases[int(case_index) % len(self.cases)]
            self._step_index = 0
            self._best = None
            return self._observe(), {"case_id": self._case.case_id}

        def step(self, action):
            action_arr = np.asarray(action, dtype=np.float32)
            if action_arr.shape != (ACTION_SIZE,) or not np.all(np.isfinite(action_arr)):
                metrics = MeshAttempt(False, 0, 0.0, 0.0, 0.0, 0.0, "invalid_action")
                self._step_index += 1
                return self._observe(), proxy_reward(metrics, False), False, True, self._info(metrics)

            clipped = np.clip(action_arr, self.action_space.low, self.action_space.high)
            metrics = run_mesh_attempt(self._case, clipped)
            self._step_index += 1
            if self._best is None or proxy_reward(metrics, metrics.valid) > proxy_reward(self._best, self._best.valid):
                self._best = metrics

            terminated = bool(metrics.valid and self.terminate_on_valid)
            truncated = bool(self._step_index >= self.max_steps and not terminated)
            reward = proxy_reward(metrics, terminated)
            return self._observe(), reward, terminated, truncated, self._info(metrics)

        def _observe(self) -> np.ndarray:
            return _case_features(self._case, self._step_index, self.max_steps, self._best)

        def _info(self, metrics: MeshAttempt) -> dict[str, Any]:
            return {
                "case_id": self._case.case_id,
                "step": self._step_index,
                "metrics": metrics.__dict__.copy(),
                "best_metrics": None if self._best is None else self._best.__dict__.copy(),
            }

else:

    class FreeMeshEnv:  # type: ignore[no-redef]
        """Placeholder that reports the optional Gymnasium dependency clearly."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_gymnasium()


__all__ = [
    "ACTION_SIZE",
    "OBSERVATION_SIZE",
    "FreeMeshEnv",
    "MeshAttempt",
    "MeshCase",
    "default_cases",
    "load_cases",
    "proxy_reward",
    "run_mesh_attempt",
]
