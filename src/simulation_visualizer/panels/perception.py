"""
Perception panel — renders the perceived world W̃_t (§III-A.7, Eq. 12).

Shows what the pipeline actually sees at each step:
  - Prior map M_0 (dashed, lighter colour — visibly distinct from true E)
  - Distorted LiDAR scan L̂_t (ray endpoints)
  - AGV pose (oracle-known, same as objective panel for parity)

The prior map M_0 is reconstructed deterministically from the header's
simulation seed and perception parameters (eta_cpl, sigma_map), using
the same Bernoulli-dropout + Gaussian-jitter logic as the engine.
"""

from __future__ import annotations

from typing import List, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from ..reader import TelemetryHeader, TelemetryStep
from .base import Panel
from .objective_world import _agv_footprint, _build_occupied_geometry, COLOR

# Perception-specific colours
PERCEPTION_COLOR = {
    "prior_map":  "#8E8E8E",   # lighter grey, dashed
    "lidar_hit":  "#2980B9",   # blue dots for valid returns
    "lidar_miss": "#BDC3C7",   # light grey for context
    "agv":        COLOR["agv"],
}


def _reconstruct_prior_map(
    chains: List[np.ndarray],
    eta_cpl: float,
    sigma_map: float,
    seed: int,
) -> List[np.ndarray]:
    """
    Reconstruct M_0 = P(E, eta_cpl, sigma_map) deterministically.

    Uses the same algorithm as ``perception.prior_map.degrade_prior_map``:
    Bernoulli dropout per chain (probability eta_cpl of retention),
    followed by per-vertex isotropic Gaussian jitter of std dev sigma_map.
    """
    rng = np.random.default_rng(int(seed))
    retained: List[np.ndarray] = []
    for chain in chains:
        if rng.random() < eta_cpl:
            jittered = np.asarray(chain, dtype=float).copy()
            if sigma_map > 0:
                jittered = jittered + rng.normal(
                    loc=0.0, scale=sigma_map, size=jittered.shape,
                )
            retained.append(jittered)
    return retained


class PerceptionPanel(Panel):
    """
    Renders the perceived world W̃_t from Eq. (12) (§III-A.7).

    The prior map M_0 is shown as dashed lines in a lighter colour to
    visually distinguish it from the true environment structure. The
    distorted LiDAR scan is shown as scatter points for finite returns.
    Under Nominal conditions the scan should be clean; under Degraded-2
    noise, misses, and ghosts should be visible.
    """

    def setup(self, ax: plt.Axes, header: TelemetryHeader) -> None:
        """Initialise the prior map and create mutable scan artists."""
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
        ax.set_title(
            "Perception  $\\tilde{W}_t$", fontsize=9, fontweight="bold",
        )

        # Reconstruct M_0
        prior_chains = _reconstruct_prior_map(
            header.environment_chains,
            header.eta_cpl,
            header.sigma_map,
            header.simulation_seed,
        )

        # Draw prior map chains as dashed grey lines
        for chain in prior_chains:
            closed = np.vstack([chain, chain[0:1]])
            ax.plot(
                closed[:, 0], closed[:, 1],
                color=PERCEPTION_COLOR["prior_map"],
                linewidth=1.0, linestyle="--", zorder=1, alpha=0.8,
            )

        # Reference path (subtle, for spatial context)
        pos = header.reference_path_positions
        if pos.shape[0] >= 2:
            ax.plot(
                pos[:, 0], pos[:, 1],
                color=COLOR["path"], linewidth=0.8, zorder=2,
                alpha=0.4, linestyle=":",
            )

        # Mutable artists
        self._lidar_scatter: Optional[object] = None
        self._agv_patch: Optional[mpatches.Polygon] = None

    def update(
        self,
        step: TelemetryStep,
        step_index: int,
        total_steps: int,
    ) -> None:
        """Redraw the LiDAR scan and AGV for this step."""
        ax = self._ax
        header = self._header
        lidar = step.lidar

        # Remove old mutable artists
        if self._lidar_scatter is not None:
            self._lidar_scatter.remove()
            self._lidar_scatter = None
        if self._agv_patch is not None:
            self._agv_patch.remove()
            self._agv_patch = None

        # LiDAR: project finite-range rays to world coordinates
        world_pts = lidar.cartesian_world()
        if world_pts.shape[0] > 0:
            self._lidar_scatter = ax.scatter(
                world_pts[:, 0], world_pts[:, 1],
                c=PERCEPTION_COLOR["lidar_hit"],
                s=2, zorder=3, alpha=0.7,
            )

        # AGV footprint
        fp = _agv_footprint(
            header.agv_length, header.agv_width,
            step.agv_position, step.agv_heading,
        )
        self._agv_patch = mpatches.Polygon(
            fp, closed=True,
            facecolor=PERCEPTION_COLOR["agv"], edgecolor="black",
            linewidth=0.8, zorder=5, alpha=0.9,
        )
        ax.add_patch(self._agv_patch)
