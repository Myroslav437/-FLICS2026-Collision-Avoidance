"""
World visualization.

Renders a WorldState produced by src.world_generator: the environment
structure E, the reference path pi, the obstacle set O_{t=0} (circles
and squares, static and dynamic with their PRM trajectories), and the
AGV spatial extent S_agv at its initial pose (P_0, theta_0).
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from src.world_generator import ObstacleShape, WorldState
from src.world_generator.metrics import environment_occupied_geometry


COLOR = {
    "wall":       "#4A4A4A",
    "path":       "#1F9E4B",
    "waypoint":   "#F1C40F",
    "agv":        "#8E44AD",
    "static":     "#D46A1A",
    "dynamic":    "#C43D3D",
    "trajectory": "#C43D3D55",
}


def _polygon_patch(poly: Polygon, **kwargs) -> patches.PathPatch:
    """
    Build a matplotlib PathPatch from a shapely Polygon (with holes),
    using the even-odd fill rule so holes render as transparent.
    """
    verts = []
    codes = []
    for ring in [poly.exterior, *poly.interiors]:
        xy = list(ring.coords)
        verts.extend(xy)
        codes.append(MplPath.MOVETO)
        codes.extend([MplPath.LINETO] * (len(xy) - 2))
        codes.append(MplPath.CLOSEPOLY)
    return patches.PathPatch(MplPath(verts, codes), **kwargs)


def _add_occupied(ax: plt.Axes, geom: BaseGeometry) -> None:
    if geom.is_empty:
        return
    if isinstance(geom, Polygon):
        polys = [geom]
    elif hasattr(geom, "geoms"):
        polys = [g for g in geom.geoms if isinstance(g, Polygon)]
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


def plot_world(
    world: WorldState,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8.0, 8.0),
    show_path: bool = True,
    show_trajectories: bool = True,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Render `world` to `ax` (or a new figure if ax is None).

    Layers (back-to-front):
      1. Environment structure E (walls as filled polygons with holes)
      2. Reference path pi (polyline + waypoints)
      3. Dynamic-obstacle trajectories xi_i (dashed polylines)
      4. Obstacles at t=0 (circles/squares coloured by static/dynamic)
      5. AGV footprint at (P_0, theta_0)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    X, Y = world.environment.size
    ax.set_xlim(0.0, X)
    ax.set_ylim(0.0, Y)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.2)

    _add_occupied(ax, environment_occupied_geometry(world))

    if show_path and world.reference_path.positions.shape[0] >= 2:
        pos = world.reference_path.positions
        ax.plot(
            pos[:, 0], pos[:, 1],
            color=COLOR["path"], linewidth=2.0, zorder=2, label="pi",
        )
        ax.scatter(
            pos[:, 0], pos[:, 1],
            color=COLOR["waypoint"], s=12, zorder=3,
        )
        ax.scatter(pos[0, 0], pos[0, 1], color=COLOR["path"],
                   s=80, marker="o", zorder=4, label="start")
        ax.scatter(pos[-1, 0], pos[-1, 1], color=COLOR["path"],
                   s=80, marker="*", zorder=4, label="goal")

    if show_trajectories:
        for obs in world.dynamic_obstacles:
            traj = obs.trajectory.positions
            ax.plot(
                traj[:, 0], traj[:, 1],
                color=COLOR["trajectory"], linewidth=1.0,
                linestyle="--", zorder=2,
            )

    for obs in world.obstacles:
        color = COLOR["dynamic"] if obs.is_dynamic else COLOR["static"]
        centroid = obs.initial_centroid()
        r = obs.spatial.inscribed_radius
        if obs.spatial.shape == ObstacleShape.CIRCLE:
            patch = patches.Circle(
                (float(centroid[0]), float(centroid[1])),
                r, facecolor=color, edgecolor="black",
                linewidth=0.5, zorder=4,
            )
        else:
            patch = patches.Rectangle(
                (float(centroid[0] - r), float(centroid[1] - r)),
                2 * r, 2 * r,
                facecolor=color, edgecolor="black",
                linewidth=0.5, zorder=4,
            )
        ax.add_patch(patch)

    agv_poly = world.agv_extent.footprint_world_frame(
        world.agv_position, world.agv_heading
    )
    ax.add_patch(patches.Polygon(
        agv_poly, closed=True, facecolor=COLOR["agv"],
        edgecolor="black", linewidth=1.0, zorder=5, alpha=0.9,
    ))

    if title is None:
        title = (
            f"W_0 (world {world.world_id}, seed {world.seed}) "
            f"| static={len(world.static_obstacles)} "
            f"dynamic={len(world.dynamic_obstacles)}"
        )
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    return ax


def save_world_figure(
    world: WorldState,
    filepath: str,
    dpi: int = 150,
    **plot_kwargs,
) -> None:
    """Render `world` to a PNG at `filepath`."""
    fig, ax = plt.subplots(figsize=plot_kwargs.pop("figsize", (8.0, 8.0)))
    plot_world(world, ax=ax, **plot_kwargs)
    fig.tight_layout()
    fig.savefig(filepath, dpi=dpi)
    plt.close(fig)
