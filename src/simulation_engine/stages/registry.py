"""
Stage registry  A = <A_det, A_fus, A_avoid>.

Three independent name -> factory mappings, one per stage. A factory is
any zero-argument callable returning a fresh stage instance; the
simulation core invokes the factory when it assembles the pipeline.

Registering a new algorithm
---------------------------
    from .base import DetectionStage
    from .registry import register_detection

    class MyDetector(DetectionStage):
        name = "MY"
        ...

    register_detection("MY", MyDetector)

The core `SimulationEngine` (see `core.engine`) resolves the three
stage names it receives from the CLI through `get_detection`,
`get_fusion`, and `get_avoidance`. No concrete implementation is ever
imported by the core.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from .base import AvoidanceStage, DetectionStage, FusionStage


_DetectionFactory = Callable[[], DetectionStage]
_FusionFactory = Callable[[], FusionStage]
_AvoidanceFactory = Callable[[], AvoidanceStage]


_DETECTION: Dict[str, _DetectionFactory] = {}
_FUSION: Dict[str, _FusionFactory] = {}
_AVOIDANCE: Dict[str, _AvoidanceFactory] = {}


def _key(name: str) -> str:
    return name.strip().upper()


# ----- Detection -------------------------------------------------------------

def register_detection(name: str, factory: _DetectionFactory) -> None:
    """Register a detection-stage factory under the given (case-insensitive) name."""
    k = _key(name)
    if k in _DETECTION:
        raise ValueError(f"detection stage already registered: {name}")
    _DETECTION[k] = factory


def get_detection(name: str) -> DetectionStage:
    """Instantiate the detection stage registered under `name`."""
    k = _key(name)
    if k not in _DETECTION:
        known = ", ".join(sorted(_DETECTION))
        raise KeyError(f"unknown detection stage '{name}'; known: {known}")
    return _DETECTION[k]()


def list_detection() -> List[str]:
    return sorted(_DETECTION)


# ----- Fusion ----------------------------------------------------------------

def register_fusion(name: str, factory: _FusionFactory) -> None:
    k = _key(name)
    if k in _FUSION:
        raise ValueError(f"fusion stage already registered: {name}")
    _FUSION[k] = factory


def get_fusion(name: str) -> FusionStage:
    k = _key(name)
    if k not in _FUSION:
        known = ", ".join(sorted(_FUSION))
        raise KeyError(f"unknown fusion stage '{name}'; known: {known}")
    return _FUSION[k]()


def list_fusion() -> List[str]:
    return sorted(_FUSION)


# ----- Avoidance -------------------------------------------------------------

def register_avoidance(name: str, factory: _AvoidanceFactory) -> None:
    k = _key(name)
    if k in _AVOIDANCE:
        raise ValueError(f"avoidance stage already registered: {name}")
    _AVOIDANCE[k] = factory


def get_avoidance(name: str) -> AvoidanceStage:
    k = _key(name)
    if k not in _AVOIDANCE:
        known = ", ".join(sorted(_AVOIDANCE))
        raise KeyError(f"unknown avoidance stage '{name}'; known: {known}")
    return _AVOIDANCE[k]()


def list_avoidance() -> List[str]:
    return sorted(_AVOIDANCE)
