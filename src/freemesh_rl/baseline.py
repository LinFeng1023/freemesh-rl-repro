"""Deterministic quadrilateral baseline for Level 2 smoke evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

from .geometry import Point, Quad, contained_quads, validate_polygon
from .metrics import MeshMetrics, measure_mesh


@dataclass(frozen=True)
class BaselineResult:
    case_id: str
    status: str
    reason: str
    quads: list[Quad]
    metrics: MeshMetrics

    def to_json(self) -> dict:
        return {
            "case_id": self.case_id,
            "status": self.status,
            "reason": self.reason,
            "quads": [[[float(x), float(y)] for x, y in quad] for quad in self.quads],
            "metrics": self.metrics.to_json(),
        }


def run_grid_baseline(
    case_id: str,
    vertices: Sequence[Point],
    resolution: int = 10,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> BaselineResult:
    start = time.perf_counter()
    check = validate_polygon(vertices)
    if not check.valid:
        metrics = measure_mesh(case_id, vertices, [], "invalid_boundary", time.perf_counter() - start)
        return BaselineResult(case_id, "failed", check.reason, [], metrics)
    quads = contained_quads(vertices, resolution=resolution, offset_x=offset_x, offset_y=offset_y)
    elapsed = time.perf_counter() - start
    metrics = measure_mesh(case_id, vertices, quads, "ok", elapsed)
    if not quads:
        status = "failed"
        reason = "no_contained_quads"
    elif metrics.completion_ratio >= 0.98:
        status = "complete"
        reason = "contained_grid_full_coverage"
    else:
        status = "partial"
        reason = "contained_grid_partial_coverage"
    metrics = measure_mesh(case_id, vertices, quads, status, elapsed)
    return BaselineResult(case_id, status, reason, quads, metrics)
