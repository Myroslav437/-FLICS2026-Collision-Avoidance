"""
Simulation Engine parameter dataclasses Sigma = <Lambda, Phi, Delta>.

The three canonical configurations (Nominal, Degraded-1, Degraded-2)
are provided as module constants and by name via `get_sigma(name)`.

Single source of truth
----------------------
No field carries a default; constructing a configuration programmatically
requires all values, so accidental omissions fail loudly.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Mapping

import math


# =============================================================================
# Lambda_S -- LiDAR geometric parameters
# =============================================================================

@dataclass(frozen=True)
class LiDARGeometry:
    """Lambda_S = <r_min, r_max, theta_fov, delta_theta, f_scan>."""

    r_min: float      # metres
    r_max: float      # metres
    theta_fov: float  # radians, in (0, 2*pi]
    delta_theta: float  # radians per ray
    f_scan: float     # Hz

    def __post_init__(self) -> None:
        if self.r_min <= 0 or self.r_max <= self.r_min:
            raise ValueError(
                f"LiDAR geometry requires 0 < r_min < r_max; got ({self.r_min}, {self.r_max})"
            )
        if not (0.0 < self.theta_fov <= 2.0 * math.pi + 1e-9):
            raise ValueError(
                f"theta_fov must be in (0, 2*pi]; got {self.theta_fov}"
            )
        if self.delta_theta <= 0:
            raise ValueError(f"delta_theta must be positive; got {self.delta_theta}")
        if self.f_scan <= 0:
            raise ValueError(f"f_scan must be positive; got {self.f_scan}")

    @property
    def n_rays(self) -> int:
        """
        N_rays = round(theta_fov / delta_theta).

        Rays are uniformly distributed over [-theta_fov/2, +theta_fov/2]
        relative to the AGV heading.
        """
        return int(round(self.theta_fov / self.delta_theta))


# =============================================================================
# Lambda_D -- LiDAR distortion parameters
# =============================================================================

@dataclass(frozen=True)
class LiDARDistortion:
    """Lambda_D = <sigma_r, p_miss, p_ghost>."""

    sigma_r: float    # range-noise std dev, >= 0
    p_miss: float     # miss probability, in [0, 1)
    p_ghost: float    # ghost probability, in [0, 1)

    def __post_init__(self) -> None:
        if self.sigma_r < 0:
            raise ValueError(f"sigma_r must be >= 0; got {self.sigma_r}")
        if not (0.0 <= self.p_miss < 1.0):
            raise ValueError(f"p_miss must be in [0, 1); got {self.p_miss}")
        if not (0.0 <= self.p_ghost < 1.0):
            raise ValueError(f"p_ghost must be in [0, 1); got {self.p_ghost}")


# =============================================================================
# Lambda = <Lambda_S, Lambda_D>
# =============================================================================

@dataclass(frozen=True)
class LiDARParams:
    """Lambda = <Lambda_S, Lambda_D>."""

    geometry: LiDARGeometry
    distortion: LiDARDistortion


# =============================================================================
# Phi -- perception / fusion parameters
# =============================================================================

@dataclass(frozen=True)
class PerceptionParams:
    """Phi = <eta_cpl, sigma_map>."""

    eta_cpl: float    # map completeness, in [0, 1]
    sigma_map: float  # map positional noise (m), >= 0

    def __post_init__(self) -> None:
        if not (0.0 <= self.eta_cpl <= 1.0):
            raise ValueError(f"eta_cpl must be in [0, 1]; got {self.eta_cpl}")
        if self.sigma_map < 0:
            raise ValueError(f"sigma_map must be >= 0; got {self.sigma_map}")


# =============================================================================
# Delta -- simulation dynamics parameters
# =============================================================================

@dataclass(frozen=True)
class DynamicsParams:
    """Delta = <v_agv_max, a_agv_max, omega_agv_max, T_horizon>."""

    v_agv_max: float      # m/s
    a_agv_max: float      # m/s^2
    omega_agv_max: float  # rad/s
    T_horizon: float      # s

    def __post_init__(self) -> None:
        if self.v_agv_max <= 0:
            raise ValueError(f"v_agv_max must be positive; got {self.v_agv_max}")
        if self.a_agv_max <= 0:
            raise ValueError(f"a_agv_max must be positive; got {self.a_agv_max}")
        if self.omega_agv_max <= 0:
            raise ValueError(
                f"omega_agv_max must be positive; got {self.omega_agv_max}"
            )
        if self.T_horizon <= 0:
            raise ValueError(f"T_horizon must be positive; got {self.T_horizon}")


# =============================================================================
# Sigma = <Lambda, Phi, Delta>
# =============================================================================

@dataclass(frozen=True)
class SimulationConfig:
    """
    Sigma = <Lambda, Phi, Delta>.

    `name` is a human-readable label (used in telemetry and output
    metadata). It does not affect engine behaviour.
    """

    name: str
    lidar: LiDARParams
    perception: PerceptionParams
    dynamics: DynamicsParams

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "lidar": {
                "geometry": asdict(self.lidar.geometry),
                "distortion": asdict(self.lidar.distortion),
            },
            "perception": asdict(self.perception),
            "dynamics": asdict(self.dynamics),
        }


# =============================================================================
# The three canonical configurations
# =============================================================================

_DEG = math.pi / 180.0


NOMINAL = SimulationConfig(
    name="nominal",
    lidar=LiDARParams(
        geometry=LiDARGeometry(
            r_min=0.05, r_max=10.0,
            theta_fov=270.0 * _DEG, delta_theta=0.5 * _DEG,
            f_scan=15.0,
        ),
        distortion=LiDARDistortion(sigma_r=0.0, p_miss=0.0, p_ghost=0.0),
    ),
    perception=PerceptionParams(eta_cpl=1.0, sigma_map=0.0),
    dynamics=DynamicsParams(
        v_agv_max=1.5, a_agv_max=1.0, omega_agv_max=1.0, T_horizon=120.0
    ),
)

DEGRADED_1 = SimulationConfig(
    name="degraded-1",
    lidar=LiDARParams(
        geometry=LiDARGeometry(
            r_min=0.05, r_max=5.0,
            theta_fov=270.0 * _DEG, delta_theta=1.0 * _DEG,
            f_scan=15.0,
        ),
        distortion=LiDARDistortion(sigma_r=0.02, p_miss=0.02, p_ghost=0.01),
    ),
    perception=PerceptionParams(eta_cpl=0.9, sigma_map=0.05),
    dynamics=DynamicsParams(
        v_agv_max=1.5, a_agv_max=1.0, omega_agv_max=1.0, T_horizon=120.0
    ),
)

DEGRADED_2 = SimulationConfig(
    name="degraded-2",
    lidar=LiDARParams(
        geometry=LiDARGeometry(
            r_min=0.05, r_max=5.0,
            theta_fov=180.0 * _DEG, delta_theta=2.0 * _DEG,
            f_scan=10.0,
        ),
        distortion=LiDARDistortion(sigma_r=0.05, p_miss=0.05, p_ghost=0.02),
    ),
    perception=PerceptionParams(eta_cpl=0.7, sigma_map=0.1),
    dynamics=DynamicsParams(
        v_agv_max=1.5, a_agv_max=1.0, omega_agv_max=1.0, T_horizon=120.0
    ),
)


_SIGMA_REGISTRY: Mapping[str, SimulationConfig] = {
    "nominal": NOMINAL,
    "degraded-1": DEGRADED_1,
    "degraded-2": DEGRADED_2,
    # Aliases the CLI commonly sees
    "deg1": DEGRADED_1,
    "deg2": DEGRADED_2,
}


def get_sigma(name: str) -> SimulationConfig:
    """Look up a canonical configuration by name (case-insensitive)."""
    key = name.strip().lower()
    if key not in _SIGMA_REGISTRY:
        known = ", ".join(sorted({k for k in _SIGMA_REGISTRY}))
        raise KeyError(
            f"unknown simulation configuration '{name}'; known: {known}"
        )
    return _SIGMA_REGISTRY[key]


def list_sigmas() -> list:
    """List the canonical Sigma configuration names (no aliases)."""
    return ["nominal", "degraded-1", "degraded-2"]
