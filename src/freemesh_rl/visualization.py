"""Plotting helpers for boundaries and quadrilateral meshes."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from .geometry import Point, Quad
from .metrics import measure_quad


def plot_mesh(
    boundary: Sequence[Point],
    quads: Sequence[Quad],
    out_path: Path,
    title: str | None = None,
    quality_color: bool = False,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    bx = [p[0] for p in boundary] + [boundary[0][0]]
    by = [p[1] for p in boundary] + [boundary[0][1]]
    ax.plot(bx, by, color="black", linewidth=1.6, label="boundary")
    if quads:
        polys = [[(x, y) for x, y in quad] for quad in quads]
        if quality_color:
            values = [measure_quad(quad).element_quality for quad in quads]
            coll = PolyCollection(polys, array=values, cmap="viridis", edgecolors="#34495e", linewidths=0.35)
            coll.set_clim(0.0, 1.0)
            ax.add_collection(coll)
            fig.colorbar(coll, ax=ax, shrink=0.8, label="element quality")
        else:
            coll = PolyCollection(polys, facecolors="#9fd3ff", edgecolors="#34495e", linewidths=0.35, alpha=0.75)
            ax.add_collection(coll)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or "Boundary and quadrilateral mesh")
    ax.grid(True, linewidth=0.25, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
