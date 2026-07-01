from __future__ import annotations

from freemesh_rl.geometry import contained_quads, validate_polygon
from freemesh_rl.metrics import measure_mesh, measure_quad, quad_element_quality


def test_square_quality_is_high() -> None:
    square = [(0, 0), (1, 0), (1, 1), (0, 1)]
    metrics = measure_quad(square)
    assert metrics.valid
    assert quad_element_quality(square) == 1.0
    assert metrics.aspect_ratio == 1.0
    assert metrics.scaled_jacobian_substitute > 0.99


def test_bad_quad_has_lower_quality() -> None:
    bad = [(0, 0), (2, 0), (1.7, 0.2), (0, 1)]
    metrics = measure_quad(bad)
    assert metrics.valid
    assert metrics.element_quality < 0.8
    assert metrics.aspect_ratio > 1.0


def test_polygon_validation_and_contained_grid() -> None:
    rect = [(0, 0), (2, 0), (2, 1), (0, 1)]
    check = validate_polygon(rect)
    assert check.valid
    quads = contained_quads(rect, resolution=4)
    assert len(quads) > 0
    mesh = measure_mesh("rect", rect, quads, "ok")
    assert mesh.valid_quad_ratio == 1.0
    assert mesh.completion_ratio > 0.95
