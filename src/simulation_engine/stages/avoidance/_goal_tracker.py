"""
Reference-path goal tracker: given a WaypointPath `pi`, emits a
look-ahead carrot point (the waypoint the AGV should aim at next).

Shared by VFH and DWA (both take a reference path and need a "go
this way" target). Utility code.

The tracker advances a waypoint index monotonically: once the AGV is
within a switch radius of w_j, it targets w_{j+1}. This is a classic
"carrot-chasing" tracker.
"""

from __future__ import annotations

import numpy as np

from world_generator.models import WaypointPath


class WaypointGoalTracker:
    """Tracks progress along pi; emits the current look-ahead target."""

    def __init__(
        self,
        reference_path: WaypointPath,
        switch_radius: float = 0.5,
    ) -> None:
        if switch_radius <= 0:
            raise ValueError("switch_radius must be positive")
        self.path = reference_path
        self.switch_radius = float(switch_radius)
        # Start aiming at waypoint index 1 (skip the start position).
        self._target_index = (
            1 if reference_path.positions.shape[0] > 1 else 0
        )

    def reset(self) -> None:
        self._target_index = (
            1 if self.path.positions.shape[0] > 1 else 0
        )

    @property
    def is_finished(self) -> bool:
        return self._target_index >= self.path.positions.shape[0]

    @property
    def goal(self) -> np.ndarray:
        """Final waypoint w_n."""
        return self.path.positions[-1].copy()

    def current_target(self, agv_position: np.ndarray) -> np.ndarray:
        """
        Advance the tracker and return the current look-ahead waypoint.
        If the path is finished, returns the final waypoint.
        """
        pts = self.path.positions
        while (
            self._target_index < pts.shape[0] - 1
            and np.linalg.norm(pts[self._target_index] - agv_position)
                < self.switch_radius
        ):
            self._target_index += 1
        idx = min(self._target_index, pts.shape[0] - 1)
        return pts[idx].copy()
