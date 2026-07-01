from __future__ import annotations

import json

from freemesh_rl.baseline import run_grid_baseline
from freemesh_rl.data import CATEGORIES, generate_cases, load_manifest, write_cases


def test_generate_cases_covers_categories() -> None:
    cases = generate_cases("smoke", count=len(CATEGORIES), seed=10)
    assert len(cases) == len(CATEGORIES)
    assert {case.category for case in cases} == set(CATEGORIES)
    assert all(len(case.vertices) >= 4 for case in cases)


def test_write_and_load_manifest(tmp_path) -> None:
    cases = generate_cases("smoke", count=3, seed=20)
    manifest = write_cases(cases, tmp_path, "smoke", 20)
    loaded = load_manifest(manifest)
    assert [case.id for case in loaded] == [case.id for case in cases]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["count"] == 3


def test_rectangular_grid_baseline_completes() -> None:
    rect = [(0, 0), (2, 0), (2, 1), (0, 1)]
    result = run_grid_baseline("rect", rect, resolution=4)
    assert result.status == "complete"
    assert result.metrics.valid_quad_ratio == 1.0
    assert result.metrics.completion_ratio > 0.98
