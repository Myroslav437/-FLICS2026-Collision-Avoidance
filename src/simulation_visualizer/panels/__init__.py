"""
Panel modules for the Simulation Visualizer.

Each panel implements the ``Panel`` interface (see ``base.py``) and
renders one category of telemetry data onto a matplotlib Axes.
"""

from .base import Panel
from .objective_world import ObjectiveWorldPanel
from .perception import PerceptionPanel
from .pipeline_output import PipelineOutputPanel
from .metrics import MetricsPanel

__all__ = [
    "Panel",
    "ObjectiveWorldPanel",
    "PerceptionPanel",
    "PipelineOutputPanel",
    "MetricsPanel",
]
