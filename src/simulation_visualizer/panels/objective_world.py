"""
Objective world panel — renders the true world state W_t (§III-A.6).

Draws the environment structure E, reference path π, obstacle set O_t
at their current positions, and the AGV footprint at (P_t, θ_t).
Matches the WG visualizer's color palette and rendering conventions.
"""

from __future__ import annotations

from typing import List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from ..reader import TelemetryHeader, TelemetryStep
from .base import Panel

# Shared color palette — matches src/visualization/world.py
COLOR = {
    "wall":       "#4A4A4A",
    "path":       "#1F9E4B",
    "waypoint":   "#F1C40F",
    "agv":        "#8E44AD",
    "static":     "#D46A1A",
    "dynamic":    "#C43D3D",
    "trajectory": "#C43D3D55",
}


# =========================================================================
# Geometry helpers (same logic as visualization/world.py, standalone)
# =========================================================================

def _signed_area(chain: np.ndarray) -> float:
    """Shoelace signed area; positive for CCW, negative for CW."""
    x, y = chain[:, 0], chain[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _build_occupied_geometry(chains: List[np.ndarray]) -> BaseGeometry:
    """
    Reconstruct the occupied shapely geometry from oriented chains.

    CCW = exterior of an occupied region; CW = hole inside that region.
    """
    exteriors: List[np.ndarray] = []
    holes: List[np.ndarray] = []
    for chain in chains:
        if chain.shape[0] < 3:
            continue
        if _signed_area(chain) >= 0:
            exteriors.append(chain)
        else:
            holes.append(chain)
    if not exteriors:
        return unary_union([])

    from shapely.geometry import Point

    exterior_polys = [ShapelyPolygon(e) for e in exteriors]
    sizes = [p.area for p in exterior_polys]
    grouped_holes: List[List[np.ndarray]] = [[] for _ in exterior_polys]
    for hole in holes:
        pt = Point(float(hole[0, 0]), float(hole[0, 1]))
        candidates = [
            (sizes[i], i)
            for i, poly in enumerate(exterior_polys)
            if poly.contains(pt)
        ]
        if not candidates:
            continue
        _, idx = min(candidates)
        grouped_holes[idx].append(hole)
    polys = [
        ShapelyPolygon(ext, holes=grouped_holes[i])
        for i, ext in enumerate(exteriors)
    ]
    return unary_union(polys)


def _polygon_patch(poly: ShapelyPolygon, **kwargs) -> mpatches.PathPatch:
    """Build a matplotlib PathPatch from a shapely Polygon (with holes)."""
    verts: list = []
    codes: list = []
    for ring in [poly.exterior, *poly.interiors]:
        xy = list(ring.coords)
        verts.extend(xy)
        codes.append(MplPath.MOVETO)
        codes.extend([MplPath.LINETO] * (len(xy) - 2))
        codes.append(MplPath.CLOSEPOLY)
    return mpatches.PathPatch(MplPath(verts, codes), **kwargs)


def _add_occupied(ax: plt.Axes, geom: BaseGeometry) -> None:
    """Render occupied geometry as filled patches."""
    if geom.is_empty:
        return
    if isinstance(geom, ShapelyPolygon):
        polys = [geom]
    elif hasattr(geom, "geoms"):
        polys = [g for g in geom.geoms if isinstance(g, ShapelyPolygon)]
    else:
        return
    for poly in polys:
        ax.add_patch(_polygon_patch(
            poly,
            facecolor=COLOR["wall"],
            edgecolor="black",
            linewidth=0.5,
            zorder=1,
        ))


def _agv_footprint(
    length: float, width: float,
    position: np.ndarray, heading: float,
) -> np.ndarray:
    """Compute the AGV footprint rectangle vertices in world frame."""
    l2, w2 = length / 2.0, width / 2.0
    local = np.array([
        [+l2, +w2], [-l2, +w2], [-l2, -w2], [+l2, -w2],
    ], dtype=float)
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s], [s, c]])
    return local @ R.T + np.asarray(position, dtype=float)


# =========================================================================
# Panel
# =========================================================================

class ObjectiveWorldPanel(Panel):
    """
    Renders the true world state W_t at each step (§III-A.6).

    Layers (back-to-front):
      1. Environment structure E (walls as filled polygons)
      2. Reference path π (polyline + waypoints)
      3. Dynamic-obstacle initial trajectories ξ_i (dashed polylines)
      4. Obstacles O_t at their current centroids (circles)
      5. AGV footprint at (P_t, θ_t)
    """

    def setup(self, ax: plt.Axes, header: TelemetryHeader) -> None:
        """Initialise time-invariant geometry and create mutable artists."""
        self._ax = ax
        self._header = header

        X, Y = header.environment_size
        ax.set_xlim(0.0, X)
        ax.set_ylim(0.0, Y)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)", fontsize=8)
        ax.set_ylabel("y (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)
        ax.set_title("Objective World  $W_t$", fontsize=9, fontweight="bold")

        # 1. Walls
        geom = _build_occupied_geometry(header.environment_chains)
        _add_occupied(ax, geom)

        # 2. Reference path
        pos = header.reference_path_positions
        if pos.shape[0] >= 2:
            ax.plot(
                pos[:, 0], pos[:, 1],
                color=COLOR["path"], linewidth=1.5, zorder=2, label="π",
            )
            ax.scatter(
                pos[:, 0], pos[:, 1],
                color=COLOR["waypoint"], s=8, zorder=3,
            )
            ax.scatter(
                pos[0, 0], pos[0, 1], color=COLOR["path"],
                s=60, marker="o", zorder=4, label="start",
            )
            ax.scatter(
                pos[-1, 0], pos[-1, 1], color=COLOR["path"],
                s=60, marker="*", zorder=4, label="goal",
            )

        # 3. Dynamic obstacle trajectories (from header specs)
        for obs in header.obstacles:
            if obs.is_dynamic:
                traj = obs.trajectory_positions
                ax.plot(
                    traj[:, 0], traj[:, 1],
                    color=COLOR["trajectory"], linewidth=0.8,
                    linestyle="--", zorder=2,
                )

        # 4. Obstacle patches (mutable — will be cleared/redrawn each step)
        self._obstacle_patches: List[mpatches.Patch] = []

        # 5. AGV patch (mutable)
        self._agv_patch: Optional[mpatches.Polygon] = None

        ax.legend(loc="upper right", fontsize=6, framealpha=0.7)

    def update(
        self,
        step: TelemetryStep,
        step_index: int,
        total_steps: int,
    ) -> None:
        """Move obstacles and AGV to their positions at this step."""
        ax = self._ax
        header = self._header

        # Remove old mutable artists
        for p in self._obstacle_patches:
            p.remove()
        self._obstacle_patches.clear()
        if self._agv_patch is not None:
            self._agv_patch.remove()
            self._agv_patch = None

        # Obstacles at current centroids
        n_obs = step.obstacle_centroids.shape[0]
        for i in range(n_obs):
            cx, cy = step.obstacle_centroids[i]
            r = float(step.obstacle_radii[i])
            # Color by velocity: dynamic if moving, static otherwise
            vel_mag = float(np.linalg.norm(step.obstacle_velocities[i]))
            color = COLOR["dynamic"] if vel_mag > 1e-6 else COLOR["static"]
            circle = mpatches.Circle(
                (cx, cy), r,
                facecolor=color, edgecolor="black",
                linewidth=0.4, zorder=4, alpha=0.85,
            )
            ax.add_patch(circle)
            self._obstacle_patches.append(circle)

        # AGV footprint
        fp = _agv_footprint(
            header.agv_length, header.agv_width,
            step.agv_position, step.agv_heading,
        )
        self._agv_patch = mpatches.Polygon(
            fp, closed=True,
            facecolor=COLOR["agv"], edgecolor="black",
            linewidth=0.8, zorder=5, alpha=0.9,
        )
        ax.add_patch(self._agv_patch)
