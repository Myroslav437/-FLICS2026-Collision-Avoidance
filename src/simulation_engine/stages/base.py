"""
Abstract stage interfaces  A = <A_det, A_fus, A_avoid>.

Each interface prescribes the input and output types:

  DetectionStage : L_hat_t, M_0   ->  O_hat_t
  FusionStage    : O_hat_t        ->  O_bar_t
  AvoidanceStage : O_bar_t, pi,
                   P_t, theta_t,
                   v_t             ->  control action

The simulation core composes the three stages at runtime and never
references a concrete implementation.

Contract
--------
- `reset(seed)`: called once before a run. Stages that keep internal
  state (e.g. Kalman tracks) must clear it here.
- `step(...)`: called once per simulation step with the arguments listed
  on the method signature. Pure function of (input, internal state);
  must not read the ground-truth world.
- Stages must be importable without side effects beyond registering
  themselves in the per-stage registry (see `registry.py`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..config import SimulationConfig
from ..types import (
    AGVState,
    ControlAction,
    DetectedObstacle,
    FusedObstacle,
    LiDARScan,
    PriorMap,
)


# =============================================================================
# A_det  -- detection
# =============================================================================

class DetectionStage(ABC):
    """
    A_det : L_hat_t, M_0, (P_t, theta_t) -> O_hat_t.

    Receives the distorted scan and the prior map, filters points
    matching known boundary geometry, and groups the remainder into
    obstacle candidates.
    """

    #: A human-readable short name (used in telemetry metadata). Concrete
    #: classes must override.
    name: str = ""

    @abstractmethod
    def reset(self, seed: int) -> None:
        """Clear any internal state before a run."""

    @abstractmethod
    def step(
        self,
        scan: LiDARScan,
        prior_map: PriorMap,
        agv_state: AGVState,
    ) -> List[DetectedObstacle]:
        """Produce the per-step O_hat_t."""


# =============================================================================
# A_fus  -- fusion
# =============================================================================

class FusionStage(ABC):
    """
    A_fus : O_hat_t -> O_bar_t.

    Integrates the current detections with historical observations.
    Maintains per-track state if required (e.g. KF).
    """

    name: str = ""

    @abstractmethod
    def reset(self, seed: int) -> None:
        """Clear any internal state before a run."""

    @abstractmethod
    def step(
        self,
        detections: List[DetectedObstacle],
        dt: float,
    ) -> List[FusedObstacle]:
        """Produce the per-step O_bar_t."""


# =============================================================================
# A_avoid  -- avoidance
# =============================================================================

class AvoidanceStage(ABC):
    """
    A_avoid : O_bar_t, pi, (P_t, theta_t, v_t), Sigma -> control action.

    Receives the fused obstacle state and the reference path, and
    produces a target (linear, angular) velocity command.
    """

    name: str = ""

    @abstractmethod
    def reset(self, sigma: SimulationConfig, seed: int) -> None:
        """Cache Sigma and clear any internal state before a run."""

    @abstractmethod
    def step(
        self,
        fused_obstacles: List[FusedObstacle],
        reference_path,  # world_generator.models.WaypointPath
        agv_state: AGVState,
        dt: float,
    ) -> ControlAction:
        """Produce the per-step control action."""
