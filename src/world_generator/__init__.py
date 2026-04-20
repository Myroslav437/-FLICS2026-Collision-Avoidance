"""World Generator.

The package implements:
    WG(Omega, s_wg) -> W_0

See `generator.generate_world` for the top-level entry point.
"""

from .config import (
    AGVParams,
    EnvironmentParams,
    ObstacleParams,
    Omega,
    PathParams,
)
from .generator import generate_world
from .models import (
    AGVExtent,
    EnvironmentStructure,
    Obstacle,
    ObstacleShape,
    SpatialDescriptor,
    WaypointPath,
    WorldState,
)

__all__ = [
    "AGVExtent",
    "AGVParams",
    "EnvironmentParams",
    "EnvironmentStructure",
    "Obstacle",
    "ObstacleParams",
    "ObstacleShape",
    "Omega",
    "PathParams",
    "SpatialDescriptor",
    "WaypointPath",
    "WorldState",
    "generate_world",
]
