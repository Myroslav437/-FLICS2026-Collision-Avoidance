"""
Playback controller — drives animation timing for live and file output.

Supports three modes:
  - Live window playback at configurable speed (default: wall-clock rate)
  - Step-through on keypress (speed = 0)
  - Render to file (MP4 via ffmpeg, GIF via Pillow, or PNG directory)

Uses ``matplotlib.animation.FuncAnimation`` as the animation backend.
"""

from __future__ import annotations

import os
from typing import List, Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from .layout import create_figure, update_frame
from .panels import Panel
from .reader import TelemetryRun


def animate(
    run: TelemetryRun,
    speed: float = 1.0,
    render_to: Optional[str] = None,
    dpi: int = 150,
    figsize: tuple = (16.0, 12.0),
) -> None:
    """
    Animate a telemetry run, either to a live window or to a file.

    Parameters
    ----------
    run : TelemetryRun
        The parsed telemetry run (header + steps).
    speed : float
        Playback speed multiplier. 1.0 = wall-clock rate (interval
        between frames matches the simulation dt). 0 = step-through
        on keypress (space bar advances). Values > 1 speed up.
    render_to : str or None
        If provided, render to this file path. Extension determines
        format: ``.mp4`` (requires ffmpeg), ``.gif`` (requires Pillow),
        or a directory name (writes numbered PNGs).
    dpi : int
        Render DPI for file output.
    figsize : tuple
        Figure size in inches.
    """
    header = run.header
    steps = run.steps
    total_steps = len(steps)

    fig, panels, suptitle = create_figure(header, figsize=figsize)

    # Compute frame interval from telemetry timestamps
    if total_steps >= 2:
        dt = steps[1].time - steps[0].time
    else:
        dt = 1.0 / 15.0  # fallback to 15 Hz

    if speed > 0:
        interval_ms = max(1.0, (dt / speed) * 1000.0)
    else:
        # Step-through: large interval; we rely on key-press below
        interval_ms = 1000000.0  # effectively paused

    # Step-through state
    _step_state = {"current": 0, "paused": speed == 0}

    def _on_key(event):
        """Advance one step on space bar (step-through mode)."""
        if event.key == " " and _step_state["paused"]:
            _step_state["current"] += 1
            if _step_state["current"] < total_steps:
                idx = _step_state["current"]
                update_frame(panels, steps[idx], idx, total_steps)
                fig.canvas.draw_idle()

    def _init():
        """FuncAnimation init — draw the first frame."""
        update_frame(panels, steps[0], 0, total_steps)
        return []

    def _update(frame_idx):
        """FuncAnimation update — draw frame *frame_idx*."""
        if frame_idx < total_steps:
            update_frame(panels, steps[frame_idx], frame_idx, total_steps)
        return []

    if render_to is not None:
        # --- File output ---
        anim = animation.FuncAnimation(
            fig, _update, init_func=_init,
            frames=total_steps, interval=interval_ms,
            blit=False, repeat=False,
        )

        ext = os.path.splitext(render_to)[1].lower()

        if ext == ".mp4":
            writer = animation.FFMpegWriter(
                fps=max(1, int(round(1.0 / dt))),
                metadata={"title": f"World {header.world_id}"},
            )
            anim.save(render_to, writer=writer, dpi=dpi)
        elif ext == ".gif":
            writer = animation.PillowWriter(
                fps=max(1, int(round(1.0 / dt))),
            )
            anim.save(render_to, writer=writer, dpi=dpi)
        elif ext == "":
            # Directory of PNGs
            os.makedirs(render_to, exist_ok=True)
            for idx in range(total_steps):
                update_frame(panels, steps[idx], idx, total_steps)
                fig.savefig(
                    os.path.join(render_to, f"frame_{idx:05d}.png"),
                    dpi=dpi,
                )
        else:
            raise ValueError(
                f"unsupported render format '{ext}'; "
                f"use .mp4, .gif, or a directory path for PNGs"
            )

        plt.close(fig)
    else:
        # --- Live window ---
        if speed == 0:
            # Step-through mode: show first frame, advance on keypress
            update_frame(panels, steps[0], 0, total_steps)
            fig.canvas.mpl_connect("key_press_event", _on_key)
            plt.show()
        else:
            # Keep a reference at module level so the GC doesn't collect it
            # before plt.show() finishes rendering.
            animate._anim = animation.FuncAnimation(
                fig, _update, init_func=_init,
                frames=total_steps, interval=interval_ms,
                blit=False, repeat=False,
            )
            plt.show()
