"""
Data models for the world generator's formal constructs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


# =============================================================================
# Waypoint path
# =============================================================================

@dataclass
class WaypointPath:
    """
    W = (w_0, w_1, ..., w_n ; v_1, v_2, ..., v_n).

    Waypoint positions `positions` (shape (n+1, 2)) define n straight-line
    segments; `speeds` (shape (n,)) give the target speed on each segment.

    The trivial case |W| = 0 (a stationary entity) is represented by a single
    waypoint and zero-length `speeds`.
    """

    positions: np.ndarray  # shape (n+1, 2), n = number of segments
    speeds: np.ndarray     # shape (n,)

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=float)
        self.speeds = np.asarray(self.speeds, dtype=float)
        if self.positions.ndim != 2 or self.positions.shape[1] != 2:
            raise ValueError(
                f"WaypointPath.positions must have shape (n+1, 2); got {self.positions.shape}"
            )
        if self.positions.shape[0] == 0:
            raise ValueError("WaypointPath must contain at least one waypoint")
        expected_speeds = self.positions.shape[0] - 1
        if self.speeds.shape != (expected_speeds,):
            raise ValueError(
                f"WaypointPath.speeds must have shape ({expected_speeds},); "
                f"got {self.speeds.shape}"
            )
        if np.any(self.speeds < 0):
            raise ValueError("WaypointPath.speeds must be non-negative")

    @property
    def num_segments(self) -> int:
        """|W|: number of straight-line segments."""
        return int(self.positions.shape[0] - 1)

    @property
    def is_stationary(self) -> bool:
        """True iff |W| = 0 (trivial stationary case)."""
        return self.num_segments == 0

    def heading(self, segment_index: int) -> float:
        """
        theta_j = atan2(w_j - w_{j-1}).

        `segment_index` is 0-based (0..n-1).
        """
        if not (0 <= segment_index < self.num_segments):
            raise IndexError(
                f"segment_index {segment_index} out of range "
                f"[0, {self.num_segments})"
            )
        delta = self.positions[segment_index + 1] - self.positions[segment_index]
        return float(np.arctan2(delta[1], delta[0]))

    def total_length(self) -> float:
        """Cumulative length of all segments."""
        if self.is_stationary:
            return 0.0
        deltas = np.diff(self.positions, axis=0)
        return float(np.sum(np.linalg.norm(deltas, axis=1)))

    def to_dict(self) -> dict:
        return {
            "positions": self.positions.tolist(),
            "speeds": self.speeds.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WaypointPath":
        return cls(
            positions=np.asarray(data["positions"], dtype=float),
            speeds=np.asarray(data["speeds"], dtype=float),
        )

    @classmethod
    def stationary(cls, position: np.ndarray) -> "WaypointPath":
        """Construct the |W|=0 path at the given position."""
        pos = np.asarray(position, dtype=float).reshape(1, 2)
        return cls(positions=pos, speeds=np.empty(0, dtype=float))


# =============================================================================
# Spatial descriptor S_i
# =============================================================================

class ObstacleShape(Enum):
    """
    Shape primitives available for obstacles.

    The world generator restricts obstacle shapes to circles and squares,
    each parameterised by the inscribed-circle radius r_i.
    """

    CIRCLE = "circle"
    SQUARE = "square"


@dataclass
class SpatialDescriptor:
    """
    S_i subset R^2  -- the (local-frame) spatial extent of an obstacle.

    The boundary dS_i = (b_1, ..., b_{n_i}) is a closed polygon whose
    vertices are expressed in the obstacle's local coordinate frame.
    Here the shape is described canonically by an `ObstacleShape` label
    plus its inscribed-circle radius r_i; the boundary polygon is
    derivable on demand via `boundary_vertices()`.
    """

    shape: ObstacleShape
    inscribed_radius: float  # r_i >= 0, the inscribed-circle radius
    circle_vertex_count: int = 32  # discretisation for rendering / boundary_vertices()

    def __post_init__(self) -> None:
        if self.inscribed_radius <= 0:
            raise ValueError(
                f"inscribed_radius must be positive; got {self.inscribed_radius}"
            )
        if self.circle_vertex_count < 4:
            raise ValueError("circle_vertex_count must be >= 4")

    def boundary_vertices(self) -> np.ndarray:
        """
        Return dS_i as a vertex array (shape (n_i, 2)) in the obstacle's
        local frame, ordered counter-clockwise.
        """
        r = self.inscribed_radius
        if self.shape == ObstacleShape.CIRCLE:
            n = self.circle_vertex_count
            theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
            return np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        if self.shape == ObstacleShape.SQUARE:
            # r_i = half-side length for squares
            return np.array([
                [+r, +r], [-r, +r], [-r, -r], [+r, -r],
            ], dtype=float)
        raise ValueError(f"Unsupported shape: {self.shape}")

    def bounding_radius(self) -> float:
        """
        Circumscribing radius of the shape (useful for clearance tests that
        can be done cheaply with a disk approximation).
        """
        if self.shape == ObstacleShape.CIRCLE:
            return float(self.inscribed_radius)
        if self.shape == ObstacleShape.SQUARE:
            return float(self.inscribed_radius * np.sqrt(2.0))
        raise ValueError(f"Unsupported shape: {self.shape}")

    def to_dict(self) -> dict:
        return {
            "shape": self.shape.value,
            "inscribed_radius": float(self.inscribed_radius),
            "circle_vertex_count": int(self.circle_vertex_count),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SpatialDescriptor":
        return cls(
            shape=ObstacleShape(data["shape"]),
            inscribed_radius=float(data["inscribed_radius"]),
            circle_vertex_count=int(data.get("circle_vertex_count", 32)),
        )


# =============================================================================
# Obstacle
# =============================================================================

@dataclass
class Obstacle:
    """
    o_i = <S_i, xi_i>.

    For static obstacles xi_i is a stationary path (|xi_i| = 0), i.e. a
    single waypoint at the obstacle's fixed centroid.
    """

    spatial: SpatialDescriptor
    trajectory: WaypointPath

    def initial_centroid(self) -> np.ndarray:
        return self.trajectory.positions[0].copy()

    @property
    def is_dynamic(self) -> bool:
        return not self.trajectory.is_stationary

    def boundary_world_frame(self, t: float = 0.0) -> np.ndarray:
        """
        Return the obstacle's boundary polygon (vertex array) at trajectory
        progress t = 0; heading is aligned with the first trajectory segment
        (or 0 for stationary obstacles).

        This helper is used only by the generator/visualizer for t = 0 static
        plots; full runtime evolution is the job of the simulation engine.
        """
        local = self.spatial.boundary_vertices()
        if self.trajectory.is_stationary:
            heading = 0.0
        else:
            heading = self.trajectory.heading(0)
        c, s = np.cos(heading), np.sin(heading)
        R = np.array([[c, -s], [s, c]])
        return local @ R.T + self.initial_centroid()

    def to_dict(self) -> dict:
        return {
            "spatial": self.spatial.to_dict(),
            "trajectory": self.trajectory.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Obstacle":
        return cls(
            spatial=SpatialDescriptor.from_dict(data["spatial"]),
            trajectory=WaypointPath.from_dict(data["trajectory"]),
        )


# =============================================================================
# Environment structure E
# =============================================================================

@dataclass
class EnvironmentStructure:
    """
    E = {e_1, e_2, ..., e_l}.

    Each e_j is a closed polygonal chain in R^2 representing a contiguous
    region of occupied space (walls, columns, etc.). The AGV may treat E
    as a priori known (see the prior map M_0 = P(E, ...) at simulation time).

    Vertices for each chain are ordered; the chain is implicitly closed
    (the last vertex connects to the first).
    """

    chains: List[np.ndarray]  # each entry: shape (k_j, 2)
    size: Tuple[float, float]  # bounding box (X, Y) of the environment

    def __post_init__(self) -> None:
        cleaned = []
        for chain in self.chains:
            arr = np.asarray(chain, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
                raise ValueError(
                    "Each environment chain must be a (k>=3, 2) array; "
                    f"got {arr.shape}"
                )
            cleaned.append(arr)
        self.chains = cleaned
        if len(self.size) != 2 or self.size[0] <= 0 or self.size[1] <= 0:
            raise ValueError(f"EnvironmentStructure.size must be (X>0, Y>0); got {self.size}")

    def to_dict(self) -> dict:
        return {
            "size": list(self.size),
            "chains": [c.tolist() for c in self.chains],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EnvironmentStructure":
        return cls(
            size=tuple(data["size"]),
            chains=[np.asarray(c, dtype=float) for c in data["chains"]],
        )


# =============================================================================
# AGV spatial extent S_agv
# =============================================================================

@dataclass
class AGVExtent:
    """
    S_agv subset R^2  -- the (local-frame) spatial extent of the AGV.

    The AGV is parameterised by body length L_agv and width W_agv; the
    extent is a centred axis-aligned rectangle in the AGV's local frame
    (x axis = forward / heading direction).
    """

    length: float  # L_agv, along local x
    width: float   # W_agv, along local y

    def __post_init__(self) -> None:
        if self.length <= 0 or self.width <= 0:
            raise ValueError(
                f"AGV dimensions must be positive; got L={self.length}, W={self.width}"
            )

    def boundary_vertices(self) -> np.ndarray:
        """CCW-ordered rectangle vertices in the AGV's local frame."""
        l2, w2 = self.length / 2.0, self.width / 2.0
        return np.array([
            [+l2, +w2], [-l2, +w2], [-l2, -w2], [+l2, -w2],
        ], dtype=float)

    def footprint_world_frame(
        self, position: np.ndarray, heading: float
    ) -> np.ndarray:
        """Vertices of S_agv transformed by (position, heading)."""
        local = self.boundary_vertices()
        c, s = np.cos(heading), np.sin(heading)
        R = np.array([[c, -s], [s, c]])
        return local @ R.T + np.asarray(position, dtype=float)

    def to_dict(self) -> dict:
        return {"length": float(self.length), "width": float(self.width)}

    @classmethod
    def from_dict(cls, data: dict) -> "AGVExtent":
        return cls(length=float(data["length"]), width=float(data["width"]))


# =============================================================================
# World state W_0 (restricted to t=0)
# =============================================================================

@dataclass
class WorldState:
    """
    W_0 = <E, O_0, pi, P_0, S_agv, theta_0, v_0> at t=0.

    The world generator returns initial world states only; the simulation
    engine consumes W_0 and evolves all time-varying quantities.

    Fields
    ------
    environment   : E
    obstacles     : O_0  (list of Obstacle; trajectories carry xi_i)
    reference_path: pi   (AGV reference waypoint path)
    agv_extent    : S_agv
    agv_position  : P_0  = first waypoint of pi
    agv_heading   : theta_0 = heading of pi's first segment
    agv_velocity  : v_0 = zero vector (AGV starts at rest)
    seed          : s_wg used for generation (for reproducibility)
    world_id      : corpus index
    parameters    : snapshot of Omega for later audit
    """

    environment: EnvironmentStructure
    obstacles: List[Obstacle]
    reference_path: WaypointPath
    agv_extent: AGVExtent
    agv_position: np.ndarray
    agv_heading: float
    agv_velocity: np.ndarray
    seed: int
    world_id: int
    parameters: Optional[dict] = None

    def __post_init__(self) -> None:
        self.agv_position = np.asarray(self.agv_position, dtype=float).reshape(2)
        self.agv_velocity = np.asarray(self.agv_velocity, dtype=float).reshape(2)

    # Convenience accessors
    @property
    def static_obstacles(self) -> List[Obstacle]:
        return [o for o in self.obstacles if not o.is_dynamic]

    @property
    def dynamic_obstacles(self) -> List[Obstacle]:
        return [o for o in self.obstacles if o.is_dynamic]

    def to_dict(self) -> dict:
        return {
            "world_id": int(self.world_id),
            "seed": int(self.seed),
            "environment": self.environment.to_dict(),
            "obstacles": [o.to_dict() for o in self.obstacles],
            "reference_path": self.reference_path.to_dict(),
            "agv_extent": self.agv_extent.to_dict(),
            "agv_position": self.agv_position.tolist(),
            "agv_heading": float(self.agv_heading),
            "agv_velocity": self.agv_velocity.tolist(),
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorldState":
        return cls(
            world_id=int(data.get("world_id", 0)),
            seed=int(data.get("seed", 0)),
            environment=EnvironmentStructure.from_dict(data["environment"]),
            obstacles=[Obstacle.from_dict(o) for o in data["obstacles"]],
            reference_path=WaypointPath.from_dict(data["reference_path"]),
            agv_extent=AGVExtent.from_dict(data["agv_extent"]),
            agv_position=np.asarray(data["agv_position"], dtype=float),
            agv_heading=float(data["agv_heading"]),
            agv_velocity=np.asarray(data["agv_velocity"], dtype=float),
            parameters=data.get("parameters"),
        )

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "WorldState":
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
