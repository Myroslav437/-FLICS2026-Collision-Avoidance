"""
Collision detection for mu_col.

The AGV footprint S_agv at (P_t, theta_t) is tested for intersection
with:
  (a) the occupied geometry of the environment E (walls), and
  (b) each obstacle polygon at progress tau_i(t).

Environment chains are oriented (CCW = occupied exterior, CW = hole);
we use `world_generator.metrics.environment_occupied_geometry` to
reconstruct the proper polygons-with-holes shape so that being inside
the free region does not falsely register as a collision.

The reconstructed occupied geometry is cached once per run since E is
time-invariant.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from shapely.geometry import Polygon as _SPoly
from shapely.geometry.base import BaseGeometry

from world_generator.models import WorldState
from world_generator.metrics import environment_occupied_geometry


def build_environment_occupancy(world: WorldState) -> BaseGeometry:
    """
    Pre-compute the shapely geometry of the occupied region of E.

    Call once per run and pass to `agv_collides()` on every step.
    """
    return environment_occupied_geometry(world)


def agv_collides(
    agv_footprint: np.ndarray,
    environment_occupancy: BaseGeometry,
    obstacle_polygons: Iterable[np.ndarray],
) -> bool:
    """
    True iff the AGV footprint intersects the environment occupied region
    or any obstacle polygon.

    Parameters
    ----------
    agv_footprint        : (4, 2) rectangle vertices at the current pose
    environment_occupancy: shapely geometry from `build_environment_occupancy`
    obstacle_polygons    : world-frame obstacle polygons at the current t
    """
    agv = _SPoly(agv_footprint)

    if environment_occupancy.intersects(agv):
        return True

    for obs_poly in obstacle_polygons:
        poly = _SPoly(obs_poly)
        if agv.intersects(poly):
            return True

    return False
