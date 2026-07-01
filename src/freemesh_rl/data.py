"""Synthetic boundary generation for local FreeMesh-RL smoke tests."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .geometry import Point, ensure_ccw, validate_polygon

CATEGORIES = [
    "simple_convex",
    "l_shape",
    "u_shape",
    "c_shape",
    "strong_concavity",
    "narrow_channel",
    "multiscale",
    "noisy",
    "near_degenerate",
]

DEFAULT_COUNTS = {
    "smoke": 30,
    "dev": 800,
    "hard_validation": 180,
}


@dataclass(frozen=True)
class BoundaryCase:
    id: str
    category: str
    seed: int
    vertices: list[Point]
    notes: str = ""

    def to_json(self) -> dict:
        item = asdict(self)
        item["vertices"] = [[float(x), float(y)] for x, y in self.vertices]
        return item


def _jitter(points: Iterable[Point], rng: np.random.Generator, scale: float) -> list[Point]:
    return [(float(x + rng.normal(0, scale)), float(y + rng.normal(0, scale))) for x, y in points]


def _rescale(points: Iterable[Point], rng: np.random.Generator) -> list[Point]:
    pts = np.asarray(list(points), dtype=float)
    factor = float(rng.uniform(0.85, 1.25))
    shift = rng.uniform(-0.15, 0.15, size=2)
    pts = pts * factor + shift
    return [(float(x), float(y)) for x, y in pts]


def _radial_noisy(rng: np.random.Generator, n: int = 14) -> list[Point]:
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radii = 1.0 + rng.normal(0.0, 0.08, size=n)
    radii += 0.12 * np.sin(3 * angles)
    pts = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    pts -= pts.min(axis=0)
    pts *= np.asarray([3.4, 2.6]) / np.maximum(pts.max(axis=0), 1e-9)
    return [(float(x), float(y)) for x, y in pts]


def vertices_for_category(category: str, seed: int) -> list[Point]:
    rng = np.random.default_rng(seed)
    if category == "simple_convex":
        pts = [(0.0, 0.0), (3.0, 0.0), (3.0, 2.0), (0.0, 2.0)]
        pts = _jitter(pts, rng, 0.015)
    elif category == "l_shape":
        pts = [(0, 0), (4, 0), (4, 1.35), (1.45, 1.35), (1.45, 4), (0, 4)]
        pts = _jitter(pts, rng, 0.025)
    elif category == "u_shape":
        pts = [(0, 0), (4.2, 0), (4.2, 4), (3.0, 4), (3.0, 1.25), (1.2, 1.25), (1.2, 4), (0, 4)]
        pts = _jitter(pts, rng, 0.02)
    elif category == "c_shape":
        pts = [(0, 0), (4.5, 0), (4.5, 1.0), (1.3, 1.0), (1.3, 2.7), (4.5, 2.7), (4.5, 3.7), (0, 3.7)]
        pts = _jitter(pts, rng, 0.02)
    elif category == "strong_concavity":
        pts = [(0, 0), (4, 0), (4, 3.5), (2.45, 2.0), (1.3, 3.6), (0, 3.5)]
        pts = _jitter(pts, rng, 0.025)
    elif category == "narrow_channel":
        pts = [(0, 0), (5.5, 0), (5.5, 1.0), (2.2, 1.0), (2.2, 1.55), (5.5, 1.55), (5.5, 2.55), (0, 2.55)]
        pts = _jitter(pts, rng, 0.012)
    elif category == "multiscale":
        pts = [(0, 0), (5.2, 0), (5.2, 0.75), (3.4, 0.82), (3.35, 2.7), (4.75, 2.8), (4.75, 4.3), (0, 4.3)]
        pts = _jitter(pts, rng, 0.02)
    elif category == "noisy":
        pts = _radial_noisy(rng, n=16)
    elif category == "near_degenerate":
        pts = [(0, 0), (4.0, 0), (4.05, 0.25), (4.0, 2.0), (2.05, 2.05), (2.0, 1.95), (0, 2.0)]
        pts = _jitter(pts, rng, 0.006)
    else:
        raise ValueError(f"unknown category: {category}")
    pts = ensure_ccw(_rescale(pts, rng))
    check = validate_polygon(pts)
    if not check.valid:
        raise ValueError(f"generated invalid polygon for {category}/{seed}: {check.reason}")
    return pts


def generate_cases(preset: str, count: int | None = None, seed: int = 123) -> list[BoundaryCase]:
    if preset not in DEFAULT_COUNTS:
        raise ValueError(f"unknown preset {preset!r}; expected one of {sorted(DEFAULT_COUNTS)}")
    total = int(count or DEFAULT_COUNTS[preset])
    categories = CATEGORIES
    cases: list[BoundaryCase] = []
    for i in range(total):
        category = categories[i % len(categories)]
        case_seed = int(seed + i * 9973)
        vertices = vertices_for_category(category, case_seed)
        cases.append(
            BoundaryCase(
                id=f"{preset}_{i:04d}_{category}",
                category=category,
                seed=case_seed,
                vertices=vertices,
                notes=f"Synthetic {category} boundary for {preset} preset.",
            )
        )
    return cases


def write_cases(cases: list[BoundaryCase], out_dir: Path, preset: str, seed: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = out_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    manifest_cases = []
    for case in cases:
        rel = Path("cases") / f"{case.id}.json"
        (out_dir / rel).write_text(json.dumps(case.to_json(), indent=2), encoding="utf-8")
        item = case.to_json()
        item["path"] = str(rel)
        manifest_cases.append(item)
    manifest = {
        "preset": preset,
        "seed": int(seed),
        "count": len(cases),
        "categories": sorted({case.category for case in cases}),
        "cases": manifest_cases,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def load_case(path: Path) -> BoundaryCase:
    data = json.loads(path.read_text(encoding="utf-8"))
    return BoundaryCase(
        id=str(data["id"]),
        category=str(data["category"]),
        seed=int(data["seed"]),
        vertices=[(float(x), float(y)) for x, y in data["vertices"]],
        notes=str(data.get("notes", "")),
    )


def load_manifest(path: Path) -> list[BoundaryCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    cases: list[BoundaryCase] = []
    for item in data.get("cases", []):
        if "path" in item and (base / item["path"]).exists():
            cases.append(load_case(base / item["path"]))
        else:
            cases.append(
                BoundaryCase(
                    id=str(item["id"]),
                    category=str(item["category"]),
                    seed=int(item["seed"]),
                    vertices=[(float(x), float(y)) for x, y in item["vertices"]],
                    notes=str(item.get("notes", "")),
                )
            )
    return cases
