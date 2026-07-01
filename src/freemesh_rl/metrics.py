"""Mesh quality metrics for local reproduction checks."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import mean
from typing import Sequence

import numpy as np

from .geometry import Point, Quad, cross2d, edge_lengths, is_convex_quad, make_polygon, polygon_area, quad_angles, quad_diagonals, quad_polygon


@dataclass(frozen=True)
class QuadMetrics:
    valid: bool
    area: float
    element_quality: float
    min_angle: float
    max_angle: float
    min_angle_deviation: float
    max_angle_deviation: float
    aspect_ratio: float
    scaled_jacobian_substitute: float


@dataclass(frozen=True)
class MeshMetrics:
    case_id: str
    status: str
    completion_ratio: float
    valid_quad_ratio: float
    invalid_count: int
    quad_count: int
    element_quality_mean: float
    element_quality_min: float
    min_angle_deviation_mean: float
    max_angle_deviation_mean: float
    aspect_ratio_mean: float
    scaled_jacobian_substitute_min: float
    runtime_seconds: float

    def to_json(self) -> dict:
        return self.__dict__.copy()


def quad_scaled_jacobian_substitute(quad: Sequence[Point]) -> float:
    """Corner sine minimum used as a scaled-Jacobian-like local diagnostic."""

    pts = np.asarray(quad, dtype=float)
    if len(pts) != 4:
        return 0.0
    signed_area = polygon_area([(float(x), float(y)) for x, y in pts])
    orientation = 1.0 if signed_area >= 0 else -1.0
    values = []
    for i in range(4):
        prev_vec = pts[i] - pts[i - 1]
        next_vec = pts[(i + 1) % 4] - pts[i]
        denom = float(np.linalg.norm(prev_vec) * np.linalg.norm(next_vec))
        if denom <= 1e-12:
            values.append(-1.0)
            continue
        cross = cross2d(prev_vec, next_vec)
        values.append(orientation * cross / denom)
    return float(min(values))


def quad_element_quality(quad: Sequence[Point]) -> float:
    lengths = edge_lengths(quad)
    d0, d1 = quad_diagonals(quad)
    dmax = max(d0, d1, 1e-12)
    qedge = min(1.0, sqrt(2.0) * min(lengths) / dmax)
    angles = quad_angles(quad)
    max_angle = max(max(angles), 1e-12)
    qangle = max(0.0, min(angles) / max_angle)
    return float(max(0.0, min(1.0, sqrt(qedge * qangle))))


def measure_quad(quad: Sequence[Point]) -> QuadMetrics:
    valid = len(quad) == 4 and is_convex_quad(quad) and quad_polygon(quad).is_valid
    area = abs(float(quad_polygon(quad).area)) if len(quad) == 4 else 0.0
    angles = quad_angles(quad) if len(quad) == 4 else [0.0, 0.0, 0.0, 0.0]
    lengths = edge_lengths(quad) if len(quad) == 4 else [0.0]
    min_len = max(min(lengths), 1e-12)
    return QuadMetrics(
        valid=valid,
        area=area,
        element_quality=quad_element_quality(quad) if valid else 0.0,
        min_angle=min(angles),
        max_angle=max(angles),
        min_angle_deviation=abs(min(angles) - 90.0),
        max_angle_deviation=abs(max(angles) - 90.0),
        aspect_ratio=float(max(lengths) / min_len),
        scaled_jacobian_substitute=quad_scaled_jacobian_substitute(quad) if len(quad) == 4 else 0.0,
    )


def measure_mesh(
    case_id: str,
    boundary: Sequence[Point],
    quads: Sequence[Quad],
    status: str,
    runtime_seconds: float = 0.0,
) -> MeshMetrics:
    polygon = make_polygon(boundary)
    polygon_area_value = max(float(polygon.area), 1e-12)
    quad_metrics = [measure_quad(quad) for quad in quads]
    valid = [m for m in quad_metrics if m.valid]
    mesh_area = sum(m.area for m in valid)
    completion_ratio = float(max(0.0, min(1.0, mesh_area / polygon_area_value)))
    valid_ratio = float(len(valid) / len(quad_metrics)) if quad_metrics else 0.0
    qualities = [m.element_quality for m in valid]
    min_devs = [m.min_angle_deviation for m in valid]
    max_devs = [m.max_angle_deviation for m in valid]
    aspects = [m.aspect_ratio for m in valid]
    jacobians = [m.scaled_jacobian_substitute for m in valid]
    return MeshMetrics(
        case_id=case_id,
        status=status,
        completion_ratio=completion_ratio,
        valid_quad_ratio=valid_ratio,
        invalid_count=len(quad_metrics) - len(valid),
        quad_count=len(quad_metrics),
        element_quality_mean=float(mean(qualities)) if qualities else 0.0,
        element_quality_min=float(min(qualities)) if qualities else 0.0,
        min_angle_deviation_mean=float(mean(min_devs)) if min_devs else 0.0,
        max_angle_deviation_mean=float(mean(max_devs)) if max_devs else 0.0,
        aspect_ratio_mean=float(mean(aspects)) if aspects else 0.0,
        scaled_jacobian_substitute_min=float(min(jacobians)) if jacobians else 0.0,
        runtime_seconds=float(runtime_seconds),
    )


def summarize_metrics(metrics: Sequence[MeshMetrics]) -> dict:
    if not metrics:
        return {
            "case_count": 0,
            "completion_rate": 0.0,
            "mean_completion_ratio": 0.0,
            "mean_valid_quad_ratio": 0.0,
            "mean_element_quality": 0.0,
            "total_invalid_count": 0,
            "total_runtime_seconds": 0.0,
        }
    complete = [m for m in metrics if m.completion_ratio >= 0.98 and m.valid_quad_ratio >= 1.0 and m.quad_count > 0]
    return {
        "case_count": len(metrics),
        "completion_rate": float(len(complete) / len(metrics)),
        "mean_completion_ratio": float(mean(m.completion_ratio for m in metrics)),
        "mean_valid_quad_ratio": float(mean(m.valid_quad_ratio for m in metrics)),
        "mean_element_quality": float(mean(m.element_quality_mean for m in metrics)),
        "total_invalid_count": int(sum(m.invalid_count for m in metrics)),
        "total_runtime_seconds": float(sum(m.runtime_seconds for m in metrics)),
    }
