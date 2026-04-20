"""
Abstract panel interface.

Each panel module implements this interface to render one category
of telemetry data onto a matplotlib Axes. The compositor creates one
panel per Axes cell and drives the update loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt

from ..reader import TelemetryHeader, TelemetryStep


class Panel(ABC):
    """
    Base interface for visualization panels.

    Subclasses must implement ``setup`` (initialise artists once) and
    ``update`` (move / replace artists for each step).
    """

    @abstractmethod
    def setup(self, ax: plt.Axes, header: TelemetryHeader) -> None:
        """
        Initialise the panel on *ax* using run-level metadata from *header*.

        Called exactly once before the first frame. Implementations should
        store *ax* as an instance attribute and create any time-invariant
        artists (walls, reference path, axis limits, etc.) here.
        """

    @abstractmethod
    def update(
        self,
        step: TelemetryStep,
        step_index: int,
        total_steps: int,
    ) -> None:
        """
        Redraw the panel for a single simulation step.

        Called once per animation frame. Implementations should update
        existing artists in-place where possible (using ``set_data``,
        ``set_offsets``, etc.) for performance.
        """
