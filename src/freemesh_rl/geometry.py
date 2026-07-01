"""Geometry helpers for the local FreeMesh-RL reproduction scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, atan2, degrees, hypot
from typing import Iterable, Sequence

import numpy as np
from shapely.geometry import Polygon
from shapely.validation import explain_validity

Point = tuple[float, float]
Quad = list[Point]


def cross2d(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


@dataclass(frozen=True)
class PolygonCheck:
    """Validation result for a single-loop polygon boundary."""

    valid: bool
    reason: str
    area: float
    vertex_count: int


def as_points(vertices: Iterable[Sequence[float]]) -> list[Point]:
    return [(float(x), float(y)) for x, y in vertices]


def polygon_area(vertices: Sequence[Point]) -> float:
    pts = np.asarray(vertices, dtype=float)
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def ensure_ccw(vertices: Sequence[Point]) -> list[Point]:
    pts = as_points(vertices)
    return pts if polygon_area(pts) >= 0 else list(reversed(pts))


def polygon_bounds(vertices: Sequence[Point]) -> tuple[float, float, float, float]:
    pts = np.asarray(vertices, dtype=float)
    return (float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max()))


def make_polygon(vertices: Sequence[Point]) -> Polygon:
    return Polygon(ensure_ccw(vertices))


def validate_polygon(vertices: Sequence[Point], min_area: float = 1e-8) -> PolygonCheck:
    pts = as_points(vertices)
    if len(pts) < 3:
        return PolygonCheck(False, "fewer_than_three_vertices", 0.0, len(pts))
    poly = make_polygon(pts)
    area = float(poly.area)
    if area <= min_area:
        return PolygonCheck(False, "area_too_small", area, len(pts))
    if not poly.is_valid:
        return PolygonCheck(False, explain_validity(poly), area, len(pts))
    return PolygonCheck(True, "ok", area, len(pts))


def edge_lengths(vertices: Sequence[Point]) -> list[float]:
    pts = as_points(vertices)
    return [distance(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]


def distance(a: Point, b: Point) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def angle_degrees(prev_pt: Point, pt: Point, next_pt: Point) -> float:
    a = np.asarray(prev_pt, dtype=float) - np.asarray(pt, dtype=float)
    b = np.asarray(next_pt, dtype=float) - np.asarray(pt, dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    cos_theta = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
    return float(degrees(acos(cos_theta)))


def quad_angles(quad: Sequence[Point]) -> list[float]:
    pts = ensure_ccw(quad)
    return [angle_degrees(pts[i - 1], pts[i], pts[(i + 1) % 4]) for i in range(4)]


def quad_diagonals(quad: Sequence[Point]) -> tuple[float, float]:
    pts = as_points(quad)
    return distance(pts[0], pts[2]), distance(pts[1], pts[3])


def is_convex_quad(quad: Sequence[Point], eps: float = 1e-9) -> bool:
    pts = ensure_ccw(quad)
    if len(pts) != 4:
        return False
    signs: list[float] = []
    for i in range(4):
        a = np.asarray(pts[(i + 1) % 4]) - np.asarray(pts[i])
        b = np.asarray(pts[(i + 2) % 4]) - np.asarray(pts[(i + 1) % 4])
        signs.append(cross2d(a, b))
    return all(s > eps for s in signs)


def quad_polygon(quad: Sequence[Point]) -> Polygon:
    return Polygon(ensure_ccw(quad))


def quad_centroid(quad: Sequence[Point]) -> Point:
    pts = np.asarray(quad, dtype=float)
    c = pts.mean(axis=0)
    return (float(c[0]), float(c[1]))


def normalize_vertices(vertices: Sequence[Point]) -> list[Point]:
    """Normalize vertices into roughly [-1, 1] coordinates for observations."""

    pts = np.asarray(vertices, dtype=float)
    center = pts.mean(axis=0)
    span = np.maximum(pts.max(axis=0) - pts.min(axis=0), 1e-9)
    scale = float(max(span))
    out = (pts - center) / scale
    return [(float(x), float(y)) for x, y in out]


def local_polar_features(vertices: Sequence[Point], count: int = 8) -> list[float]:
    """Return fixed-size radial features around the polygon centroid."""

    pts = np.asarray(normalize_vertices(vertices), dtype=float)
    center = pts.mean(axis=0)
    rel = pts - center
    order = np.argsort([atan2(y, x) for x, y in rel])
    rel = rel[order]
    radii = np.linalg.norm(rel, axis=1)
    if len(radii) == 0:
        return [0.0] * count
    sample_x = np.linspace(0, len(radii) - 1, count)
    return [float(np.interp(x, np.arange(len(radii)), radii)) for x in sample_x]


def regular_grid_quads(
    vertices: Sequence[Point],
    resolution: int,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> list[Quad]:
    """Create axis-aligned candidate quads whose cells cover the polygon bounds."""

    resolution = max(1, int(resolution))
    min_x, min_y, max_x, max_y = polygon_bounds(vertices)
    width = max(max_x - min_x, 1e-9)
    height = max(max_y - min_y, 1e-9)
    cell = max(width, height) / resolution
    start_x = min_x + float(offset_x) * cell
    start_y = min_y + float(offset_y) * cell
    quads: list[Quad] = []
    nx = int(np.ceil(width / cell)) + 2
    ny = int(np.ceil(height / cell)) + 2
    for ix in range(-1, nx):
        x0 = start_x + ix * cell
        x1 = x0 + cell
        for iy in range(-1, ny):
            y0 = start_y + iy * cell
            y1 = y0 + cell
            quads.append([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    return quads


def contained_quads(
    vertices: Sequence[Point],
    resolution: int,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    min_area_ratio: float = 0.999,
) -> list[Quad]:
    """Return candidate quads contained in the polygon.

    The containment baseline is intentionally simple. It is a Level 2 sanity
    baseline, not the paper's learned boundary-extraction algorithm.
    """

    poly = make_polygon(vertices)
    out: list[Quad] = []
    for quad in regular_grid_quads(vertices, resolution, offset_x, offset_y):
        qpoly = quad_polygon(quad)
        if not qpoly.is_valid or qpoly.area <= 0:
            continue
        if poly.covers(qpoly) or (poly.intersection(qpoly).area / qpoly.area) >= min_area_ratio:
            out.append(ensure_ccw(quad))
    return out
