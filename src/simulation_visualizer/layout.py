"""
Layout compositor — creates the figure and arranges the four panels.

Panel layout (2×2 grid)::

    +--------------------+--------------------+
    |  Objective World   |  Perception (W̃_t)  |
    +--------------------+--------------------+
    |  Pipeline Output   |  Metrics           |
    +--------------------+--------------------+

A suptitle bar shows world ID, sigma config, pipeline, and step info.
"""

from __future__ import annotations

from typing import List, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from .panels import (
    MetricsPanel,
    ObjectiveWorldPanel,
    Panel,
    PerceptionPanel,
    PipelineOutputPanel,
)
from .reader import TelemetryHeader, TelemetryStep


def create_figure(
    header: TelemetryHeader,
    figsize: Tuple[float, float] = (16.0, 12.0),
) -> Tuple[plt.Figure, List[Panel], plt.Text]:
    """
    Build the 2×2 figure and initialise all four panels.

    Returns
    -------
    fig : matplotlib Figure
    panels : list of four panels in layout order
    suptitle : the suptitle Text artist (updated per frame)
    """
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        wspace=0.25, hspace=0.30,
        left=0.05, right=0.95, top=0.92, bottom=0.05,
    )

    ax_world = fig.add_subplot(gs[0, 0])
    ax_perception = fig.add_subplot(gs[0, 1])
    ax_pipeline = fig.add_subplot(gs[1, 0])
    ax_metrics = fig.add_subplot(gs[1, 1])

    panels: List[Panel] = [
        ObjectiveWorldPanel(),
        PerceptionPanel(),
        PipelineOutputPanel(),
        MetricsPanel(),
    ]

    axes = [ax_world, ax_perception, ax_pipeline, ax_metrics]
    for panel, ax in zip(panels, axes):
        panel.setup(ax, header)

    # Suptitle with run metadata
    pipeline_str = (
        f"{header.pipeline_detection} + "
        f"{header.pipeline_fusion} + "
        f"{header.pipeline_avoidance}"
    )
    title_text = (
        f"World {header.world_id}  |  "
        f"Σ = {header.sigma_name}  |  "
        f"Pipeline: {pipeline_str}  |  "
        f"Seed: {header.simulation_seed}"
    )
    suptitle = fig.suptitle(
        title_text, fontsize=11, fontweight="bold", y=0.97,
    )

    return fig, panels, suptitle


def update_frame(
    panels: List[Panel],
    step: TelemetryStep,
    step_index: int,
    total_steps: int,
) -> None:
    """Delegate a frame update to every panel."""
    for panel in panels:
        panel.update(step, step_index, total_steps)
