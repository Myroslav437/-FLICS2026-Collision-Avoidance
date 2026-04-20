"""
Simulation Engine.

Implements SE(W_0, A, Sigma, s_se) -> M with closed-loop dynamics. The
package is organised as:

  config                    Sigma = <Lambda, Phi, Delta>.
  types                     Data classes for perception, stages, metrics.
  perception                The perception function Psi.
  stages                    A = <A_det, A_fus, A_avoid>; registry + concrete
                            implementations.
  core                      The simulation core f + metrics + telemetry.
"""

from .config import (
    DynamicsParams,
    LiDARDistortion,
    LiDARGeometry,
    LiDARParams,
    PerceptionParams,
    SimulationConfig,
)
from .core import SimulationEngine, run_simulation
from .types import (
    AGVState,
    ControlAction,
    DetectedObstacle,
    FusedObstacle,
    LiDARScan,
    PerceivedWorld,
    PriorMap,
    RunMetrics,
    StepTelemetry,
)

__all__ = [
    "AGVState",
    "ControlAction",
    "DetectedObstacle",
    "DynamicsParams",
    "FusedObstacle",
    "LiDARDistortion",
    "LiDARGeometry",
    "LiDARParams",
    "LiDARScan",
    "PerceivedWorld",
    "PerceptionParams",
    "PriorMap",
    "RunMetrics",
    "SimulationConfig",
    "SimulationEngine",
    "StepTelemetry",
    "run_simulation",
]
