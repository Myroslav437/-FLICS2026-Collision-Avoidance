"""
Visualization utilities.

The namespace hosts all rendering code, separated by what is being
drawn:

  world                Static W_0 rendering (walls, pi, obstacles, AGV
                       footprint at t = 0). Exposes `plot_world` and
                       `save_world_figure`.

A simulation-run visualiser consuming the engine's JSONL telemetry is
planned and will be added as a sibling module in this package.
"""

from .world import plot_world, save_world_figure

__all__ = ["plot_world", "save_world_figure"]
