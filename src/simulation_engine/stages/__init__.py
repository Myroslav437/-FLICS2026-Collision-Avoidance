"""
Pipeline stages and registry.

Three abstract stage interfaces live in `base`; concrete implementations
live in `detection/`, `fusion/`, `avoidance/` submodules. The registry
is an in-memory name -> factory mapping; adding a new algorithm means:

  1. Write a class that subclasses the appropriate base,
  2. Import it in this package's `__init__.py`,
  3. Call the corresponding `register_*` call.

See the README section "How to add a new algorithm" for a walkthrough.
"""

from .base import AvoidanceStage, DetectionStage, FusionStage
from .registry import (
    get_avoidance,
    get_detection,
    get_fusion,
    list_avoidance,
    list_detection,
    list_fusion,
    register_avoidance,
    register_detection,
    register_fusion,
)

# Side-effect imports populate the registry.  Each concrete module
# registers its class; importing them is the mechanism.
from .detection import ec as _ec  # noqa: F401
from .detection import dbscan as _dbscan  # noqa: F401
from .fusion import passthrough as _passthrough  # noqa: F401
from .fusion import kalman as _kalman  # noqa: F401
from .avoidance import vfh as _vfh  # noqa: F401
from .avoidance import dwa as _dwa  # noqa: F401

__all__ = [
    "AvoidanceStage",
    "DetectionStage",
    "FusionStage",
    "get_avoidance",
    "get_detection",
    "get_fusion",
    "list_avoidance",
    "list_detection",
    "list_fusion",
    "register_avoidance",
    "register_detection",
    "register_fusion",
]
