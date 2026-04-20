"""
Pipeline output panel — renders what the pipeline A produced at each step.

Shows:
  - Detection output Ô_t — cluster centroids with radius circles
  - Fusion output Ō_t — tracked obstacles with velocity arrows
  - Avoidance output — selected control action as an arrow from AGV pose
"""

from __future__ import annotations

from typing import List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from ..reader import TelemetryHeader, TelemetryStep
from .base import Panel
from .objective_world import (
    COLOR,
    _agv_footprint,
    _build_occupied_geometry,
    _add_occupied,
)

# Pipeline-specific colours
PIPELINE_COLOR = {
    "detection":  "#E67E22",   # orange for detected clusters
    "fusion":     "#E74C3C",   # red for fused tracks
    "velocity":   "#C0392B",   # dark red for velocity arrows
    "control":    "#27AE60",   # green for control action arrow
    "agv":        COLOR["agv"],
}


class PipelineOutputPanel(Panel):
    """
    Renders the pipeline A's output at each step.

    Detection output (Ô_t) is shown as orange crosses and radius circles.
    Fusion output (Ō_t) is shown as red filled circles with velocity
    arrows (quiver) indicating the estimated obstacle velocity for
    tracked dynamic obstacles (relevant when fusion is KF).
    The avoidance output is rendered as a green arrow from the AGV's
    current pose showing the commanded control action direction.
    """

    def setup(self, ax: plt.Axes, header: TelemetryHeader) -> None:
        """Initialise the panel with environment structure for context."""
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
        ax.set_title("Pipeline Output", fontsize=9, fontweight="bold")

        # Faint environment for spatial context
        geom = _build_occupied_geometry(header.environment_chains)
        if not geom.is_empty:
            from shapely.geometry import Polygon as SP
            if isinstance(geom, SP):
                polys_list = [geom]
            elif hasattr(geom, "geoms"):
                polys_list = [g for g in geom.geoms if isinstance(g, SP)]
            else:
                polys_list = []
            for poly in polys_list:
                from .objective_world import _polygon_patch
                ax.add_patch(_polygon_patch(
                    poly,
                    facecolor=COLOR["wall"],
                    edgecolor="black",
                    linewidth=0.3,
                    zorder=1,
                    alpha=0.25,
                ))

        # Reference path (subtle context)
        pos = header.reference_path_positions
        if pos.shape[0] >= 2:
            ax.plot(
                pos[:, 0], pos[:, 1],
                color=COLOR["path"], linewidth=0.8, zorder=2,
                alpha=0.3, linestyle=":",
            )

        # Mutable artists
        self._detection_patches: List[mpatches.Patch] = []
        self._detection_scatter: Optional[object] = None
        self._fusion_patches: List[mpatches.Patch] = []
        self._velocity_quiver: Optional[object] = None
        self._control_quiver: Optional[object] = None
        self._agv_patch: Optional[mpatches.Polygon] = None

    def update(
        self,
        step: TelemetryStep,
        step_index: int,
        total_steps: int,
    ) -> None:
        """Redraw detections, fused tracks, and control arrow."""
        ax = self._ax
        header = self._header

        # Clear old mutable artists
        for p in self._detection_patches:
            p.remove()
        self._detection_patches.clear()
        if self._detection_scatter is not None:
            self._detection_scatter.remove()
            self._detection_scatter = None
        for p in self._fusion_patches:
            p.remove()
        self._fusion_patches.clear()
        if self._velocity_quiver is not None:
            self._velocity_quiver.remove()
            self._velocity_quiver = None
        if self._control_quiver is not None:
            self._control_quiver.remove()
            self._control_quiver = None
        if self._agv_patch is not None:
            self._agv_patch.remove()
            self._agv_patch = None

        # --- Detection output Ô_t ---
        if step.detections:
            det_x = [d.centroid[0] for d in step.detections]
            det_y = [d.centroid[1] for d in step.detections]
            self._detection_scatter = ax.scatter(
                det_x, det_y,
                c=PIPELINE_COLOR["detection"],
                marker="x", s=40, linewidths=1.5, zorder=4,
                label="Ô_t" if step_index == 0 else None,
            )
            for d in step.detections:
                circle = mpatches.Circle(
                    (d.centroid[0], d.centroid[1]),
                    d.radius,
                    fill=False,
                    edgecolor=PIPELINE_COLOR["detection"],
                    linewidth=0.8, linestyle="--",
                    zorder=3, alpha=0.6,
                )
                ax.add_patch(circle)
                self._detection_patches.append(circle)

        # --- Fusion output Ō_t ---
        if step.fused:
            for f in step.fused:
                circle = mpatches.Circle(
                    (f.position[0], f.position[1]),
                    f.radius,
                    facecolor=PIPELINE_COLOR["fusion"],
                    edgecolor="black",
                    linewidth=0.4, zorder=4, alpha=0.6,
                )
                ax.add_patch(circle)
                self._fusion_patches.append(circle)

            # Velocity arrows for fused obstacles
            fpos = np.array([f.position for f in step.fused])
            fvel = np.array([f.velocity for f in step.fused])
            vel_mag = np.linalg.norm(fvel, axis=1)
            moving = vel_mag > 1e-6
            if np.any(moving):
                self._velocity_quiver = ax.quiver(
                    fpos[moving, 0], fpos[moving, 1],
                    fvel[moving, 0], fvel[moving, 1],
                    color=PIPELINE_COLOR["velocity"],
                    scale=5.0, scale_units="xy",
                    width=0.004, zorder=5, alpha=0.8,
                )

        # --- Avoidance output: control action arrow ---
        pos = step.agv_position
        heading = step.agv_heading
        ctrl = step.control
        # Arrow direction: current heading + angular_velocity * small dt
        # Arrow length proportional to linear_velocity
        arrow_len = ctrl.linear_velocity * 0.5  # scale for visibility
        dx = arrow_len * np.cos(heading)
        dy = arrow_len * np.sin(heading)
        if arrow_len > 1e-6:
            self._control_quiver = ax.quiver(
                pos[0], pos[1], dx, dy,
                color=PIPELINE_COLOR["control"],
                scale=1.0, scale_units="xy",
                width=0.008, zorder=6, alpha=0.9,
            )

        # AGV footprint
        fp = _agv_footprint(
            header.agv_length, header.agv_width,
            step.agv_position, step.agv_heading,
        )
        self._agv_patch = mpatches.Polygon(
            fp, closed=True,
            facecolor=PIPELINE_COLOR["agv"], edgecolor="black",
            linewidth=0.8, zorder=5, alpha=0.9,
        )
        ax.add_patch(self._agv_patch)
