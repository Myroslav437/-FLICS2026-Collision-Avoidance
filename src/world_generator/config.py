"""
Parameter dataclasses for Omega = <Omega_env, Omega_agv, Omega_path, Omega_obs>.

Each dataclass mirrors one subset of the world generator's parameter
set, with attribute names matching the symbols used in the canonical
spec as closely as a Python identifier allows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import yaml


# =============================================================================
# Omega_env -- Environment parameters
# =============================================================================

@dataclass
class EnvironmentParams:
    """
    Parameters controlling BSP environment generation.

      X, Y            : spatial dimensions of the environment
      bsp_depth       : d_bsp, number of recursive subdivision levels
      min_partition   : (s_min_w, s_min_h), smallest leaf-partition dims
      split_ratio     : [rho_min, rho_max], split-ratio range
      room_padding    : [p_min, p_max], room padding range
      min_corridor    : c_min, minimum corridor width between adjacent boundaries (room walls and partition walls).
      passage_width   : w_pass, corridor / passage width
      wall_thickness  : t_wall, wall thickness
    """

    X: float
    Y: float
    bsp_depth: int
    min_partition: Tuple[float, float]
    split_ratio: Tuple[float, float]
    room_padding: Tuple[float, float]
    min_corridor: float
    passage_width: float
    wall_thickness: float

    def __post_init__(self) -> None:
        if self.X <= 0 or self.Y <= 0:
            raise ValueError("X and Y must be positive")
        if self.bsp_depth < 0:
            raise ValueError("bsp_depth must be >= 0")
        if self.min_partition[0] <= 0 or self.min_partition[1] <= 0:
            raise ValueError("min_partition values must be positive")
        if not (0.0 < self.split_ratio[0] <= self.split_ratio[1] < 1.0):
            raise ValueError("split_ratio must satisfy 0 < rho_min <= rho_max < 1")
        if not (0.0 <= self.room_padding[0] <= self.room_padding[1] < 1.0):
            raise ValueError("room_padding must satisfy 0 <= p_min <= p_max < 1")
        if self.min_corridor <= 0:
            raise ValueError("min_corridor must be positive")
        if self.passage_width <= 0:
            raise ValueError("passage_width must be positive")
        if self.wall_thickness <= 0:
            raise ValueError("wall_thickness must be positive")
        if self.passage_width <= self.wall_thickness:
            raise ValueError(
                "passage_width must exceed wall_thickness for passable corridors"
            )


# =============================================================================
# Omega_agv -- AGV parameters
# =============================================================================

@dataclass
class AGVParams:
    """
    AGV body dimensions.

      L_agv : body length (forward axis)
      W_agv : body width
    """

    L_agv: float
    W_agv: float

    def __post_init__(self) -> None:
        if self.L_agv <= 0 or self.W_agv <= 0:
            raise ValueError("L_agv and W_agv must be positive")


# =============================================================================
# Omega_path -- Path parameters
# =============================================================================

@dataclass
class PathParams:
    """
    Parameters controlling PRM reference-path generation.

      prm_samples     : n_samples, random samples in free space
      prm_neighbors   : k_nn, nearest-neighbor connections
      max_edge_length : l_max, maximum roadmap edge length
      clearance       : d_clear, obstacle inflation radius
      min_path_length : l_pi_min, minimum shortest-path length between
                        start and goal on the roadmap; sampled pairs that
                        produce a shorter path are rejected and resampled.
      speed_min       : lower bound of D_v* (segment target speeds)
      speed_max       : upper bound of D_v* (if == speed_min, constant speed)
    """

    prm_samples: int
    prm_neighbors: int
    max_edge_length: float
    clearance: float
    min_path_length: float
    speed_min: float
    speed_max: float

    def __post_init__(self) -> None:
        if self.prm_samples <= 0:
            raise ValueError("prm_samples must be positive")
        if self.prm_neighbors <= 0:
            raise ValueError("prm_neighbors must be positive")
        if self.max_edge_length <= 0:
            raise ValueError("max_edge_length must be positive")
        if self.clearance < 0:
            raise ValueError("clearance must be non-negative")
        if self.min_path_length < 0:
            raise ValueError("min_path_length must be non-negative")
        if self.speed_min < 0 or self.speed_max < self.speed_min:
            raise ValueError("speeds must satisfy 0 <= speed_min <= speed_max")


# =============================================================================
# Omega_obs -- Obstacle parameters
# =============================================================================

@dataclass
class ObstacleParams:
    """
    Parameters controlling obstacle generation.

      n_static        : number of static obstacles
      n_dynamic       : number of dynamic obstacles
      radius_min      : lower bound of D_r (inscribed-circle radius)
      radius_max      : upper bound of D_r
      shape_weights   : {"circle": p_c, "square": p_s}; obstacles are
                        restricted to circles and squares parameterised
                        by r_i
      min_clearance   : d_min, minimum clearance between obstacle boundaries
                        and between obstacles and environment geometry
      path_clearance  : clearance between the AGV reference path pi and
                        obstacle boundaries (respecting clearance
                        constraints relative to boundary geometry and
                        the reference path)
      prm_samples     : n_samples^obs for trajectory PRM
      prm_neighbors   : k_nn^obs
      max_edge_length : l_max^obs
      trajectory_clearance: d_clear^obs
      dyn_speed_min   : lower bound of D_v^obs
      dyn_speed_max   : upper bound of D_v^obs
      max_place_attempts : cap on placement retries when enforcing d_min
                           (does not bias the generator toward success; a
                           warning count is reported on corpus generation)
    """

    n_static: int
    n_dynamic: int
    radius_min: float
    radius_max: float
    shape_weights: dict
    min_clearance: float
    path_clearance: float
    prm_samples: int
    prm_neighbors: int
    max_edge_length: float
    trajectory_clearance: float
    dyn_speed_min: float
    dyn_speed_max: float
    max_place_attempts: int = 200

    def __post_init__(self) -> None:
        if self.n_static < 0 or self.n_dynamic < 0:
            raise ValueError("Obstacle counts must be non-negative")
        if self.radius_min <= 0 or self.radius_max < self.radius_min:
            raise ValueError("0 < radius_min <= radius_max required")
        allowed = {"circle", "square"}
        if not self.shape_weights or not set(self.shape_weights).issubset(allowed):
            raise ValueError(
                f"shape_weights must be over subset of {allowed}; got {self.shape_weights}"
            )
        if any(v < 0 for v in self.shape_weights.values()):
            raise ValueError("shape_weights must be non-negative")
        if sum(self.shape_weights.values()) <= 0:
            raise ValueError("shape_weights must sum to a positive value")
        if self.min_clearance < 0 or self.path_clearance < 0:
            raise ValueError("Clearance values must be non-negative")
        if self.prm_samples <= 0 or self.prm_neighbors <= 0:
            raise ValueError("PRM sample/neighbor counts must be positive")
        if self.max_edge_length <= 0:
            raise ValueError("max_edge_length must be positive")
        if self.trajectory_clearance < 0:
            raise ValueError("trajectory_clearance must be non-negative")
        if self.dyn_speed_min < 0 or self.dyn_speed_max < self.dyn_speed_min:
            raise ValueError(
                "0 <= dyn_speed_min <= dyn_speed_max required"
            )
        if self.max_place_attempts <= 0:
            raise ValueError("max_place_attempts must be positive")


# =============================================================================
# Omega (composite)
# =============================================================================

@dataclass
class Omega:
    """Omega = <Omega_env, Omega_agv, Omega_path, Omega_obs>."""

    env: EnvironmentParams
    agv: AGVParams
    path: PathParams
    obs: ObstacleParams

    def to_dict(self) -> dict:
        return {
            "environment": self.env.__dict__,
            "agv": self.agv.__dict__,
            "path": self.path.__dict__,
            "obstacles": self.obs.__dict__,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Omega":
        env = EnvironmentParams(
            X=float(data["environment"]["X"]),
            Y=float(data["environment"]["Y"]),
            bsp_depth=int(data["environment"]["bsp_depth"]),
            min_partition=tuple(data["environment"]["min_partition"]),
            split_ratio=tuple(data["environment"]["split_ratio"]),
            room_padding=tuple(data["environment"]["room_padding"]),
            min_corridor=float(data["environment"]["min_corridor"]),
            passage_width=float(data["environment"]["passage_width"]),
            wall_thickness=float(data["environment"]["wall_thickness"]),
        )
        agv = AGVParams(
            L_agv=float(data["agv"]["L_agv"]),
            W_agv=float(data["agv"]["W_agv"]),
        )
        path = PathParams(
            prm_samples=int(data["path"]["prm_samples"]),
            prm_neighbors=int(data["path"]["prm_neighbors"]),
            max_edge_length=float(data["path"]["max_edge_length"]),
            clearance=float(data["path"]["clearance"]),
            min_path_length=float(data["path"]["min_path_length"]),
            speed_min=float(data["path"]["speed_min"]),
            speed_max=float(data["path"]["speed_max"]),
        )
        obs = ObstacleParams(
            n_static=int(data["obstacles"]["n_static"]),
            n_dynamic=int(data["obstacles"]["n_dynamic"]),
            radius_min=float(data["obstacles"]["radius_min"]),
            radius_max=float(data["obstacles"]["radius_max"]),
            shape_weights=dict(data["obstacles"]["shape_weights"]),
            min_clearance=float(data["obstacles"]["min_clearance"]),
            path_clearance=float(data["obstacles"]["path_clearance"]),
            prm_samples=int(data["obstacles"]["prm_samples"]),
            prm_neighbors=int(data["obstacles"]["prm_neighbors"]),
            max_edge_length=float(data["obstacles"]["max_edge_length"]),
            trajectory_clearance=float(data["obstacles"]["trajectory_clearance"]),
            dyn_speed_min=float(data["obstacles"]["dyn_speed_min"]),
            dyn_speed_max=float(data["obstacles"]["dyn_speed_max"]),
            max_place_attempts=int(data["obstacles"].get("max_place_attempts", 200)),
        )
        return cls(env=env, agv=agv, path=path, obs=obs)

    @classmethod
    def load(cls, filepath: str) -> "Omega":
        """Load Omega from a YAML file."""
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))
