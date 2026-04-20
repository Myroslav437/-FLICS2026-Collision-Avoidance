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
    DEGRADED_1,
    DEGRADED_2,
    NOMINAL,
    DynamicsParams,
    LiDARDistortion,
    LiDARGeometry,
    LiDARParams,
    PerceptionParams,
    SimulationConfig,
    get_sigma,
    list_sigmas,
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
    "DEGRADED_1",
    "DEGRADED_2",
    "DetectedObstacle",
    "DynamicsParams",
    "FusedObstacle",
    "LiDARDistortion",
    "LiDARGeometry",
    "LiDARParams",
    "LiDARScan",
    "NOMINAL",
    "PerceivedWorld",
    "PerceptionParams",
    "PriorMap",
    "RunMetrics",
    "SimulationConfig",
    "SimulationEngine",
    "StepTelemetry",
    "get_sigma",
    "list_sigmas",
    "run_simulation",
]
