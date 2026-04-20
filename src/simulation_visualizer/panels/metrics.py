"""
Metrics panel — renders the six per-run metrics from §IV-B.

Displays running accumulated values at each step as a text panel:
  μ_col  — binary collision indicator (0 or 1)
  μ_dev  — mean path deviation (m), running mean
  μ_goal — binary goal-reached indicator (0 or 1)
  μ_vel  — velocity RMSE (m/s), running RMSE or "N/A"
  μ_comp — mean per-step CPU time (s), running mean
  μ_lat  — 95th-percentile per-step CPU time (s)

Aggregate metrics are not fabricated per-step. Binary indicators show
their current value; statistical metrics show the running accumulation.
At run end, final values are displayed in bold.
"""

from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from ..reader import TelemetryHeader, TelemetryStep
from .base import Panel


def _min_distance_to_segments(
    point: np.ndarray, segments: np.ndarray,
) -> float:
    """
    Minimum Euclidean distance from *point* to any segment in *segments*.

    Segments has shape (N, 2, 2); each row is [[x1,y1], [x2,y2]].
    Same algorithm as ``core.metrics._min_distance_to_segments``.
    """
    a = segments[:, 0, :]
    b = segments[:, 1, :]
    v = b - a
    p = point - a
    seg_len_sq = np.einsum("ij,ij->i", v, v)
    seg_len_sq = np.where(seg_len_sq < 1e-12, 1e-12, seg_len_sq)
    t = np.clip(np.einsum("ij,ij->i", p, v) / seg_len_sq, 0.0, 1.0)
    closest = a + t[:, None] * v
    d = np.linalg.norm(closest - point, axis=1)
    return float(np.min(d))


class MetricsPanel(Panel):
    """
    Renders the six §IV-B metrics as a live-updating text panel.

    Running values are updated each step. Metrics that are undefined
    mid-run (e.g., μ_goal before run ends) show 'pending'. At run end,
    final values are shown in bold.
    """

    def setup(self, ax: plt.Axes, header: TelemetryHeader) -> None:
        """Set up the text axis (no axes ticks or spines)."""
        self._ax = ax
        self._header = header

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("Metrics  $\\mathcal{M}$", fontsize=9, fontweight="bold")

        # Build reference path segments for μ_dev computation
        pos = header.reference_path_positions
        if pos.shape[0] >= 2:
            self._ref_segments = np.stack(
                [pos[:-1], pos[1:]], axis=1,
            )  # (N, 2, 2)
        else:
            self._ref_segments = np.empty((0, 2, 2), dtype=float)

        # Accumulators (mirror MetricsAccumulator from core.metrics)
        self._collided = False
        self._goal_reached = False
        self._dev_sum: float = 0.0
        self._dev_count: int = 0
        self._vel_sq_err_sum: float = 0.0
        self._vel_sample_count: int = 0
        self._step_times: List[float] = []

        # Text artist
        self._text_artist: Optional[plt.Text] = None

    def update(
        self,
        step: TelemetryStep,
        step_index: int,
        total_steps: int,
    ) -> None:
        """Accumulate metrics and redraw text."""
        is_final = (step_index == total_steps - 1)

        # --- μ_col ---
        if step.collided:
            self._collided = True

        # --- μ_goal ---
        if step.goal_reached:
            self._goal_reached = True

        # --- μ_dev: running mean deviation from π ---
        if self._ref_segments.shape[0] > 0:
            d = _min_distance_to_segments(
                step.agv_position, self._ref_segments,
            )
            self._dev_sum += d
            self._dev_count += 1

        # --- μ_vel: running RMSE of fused vs ground-truth velocities ---
        if (step.obstacle_centroids.shape[0] > 0
                and step.obstacle_velocities.shape[0] > 0
                and step.fused):
            # Identify dynamic obstacles (non-zero velocity)
            gt_vel_mag = np.linalg.norm(step.obstacle_velocities, axis=1)
            dyn_mask = gt_vel_mag > 1e-9
            dyn_centroids = step.obstacle_centroids[dyn_mask]
            dyn_velocities = step.obstacle_velocities[dyn_mask]
            if dyn_centroids.shape[0] > 0:
                for f in step.fused:
                    diff = dyn_centroids - f.position
                    d = np.linalg.norm(diff, axis=1)
                    j = int(np.argmin(d))
                    err = f.velocity - dyn_velocities[j]
                    self._vel_sq_err_sum += float(
                        err[0] ** 2 + err[1] ** 2,
                    )
                    self._vel_sample_count += 1

        # --- μ_comp: running mean step time ---
        self._step_times.append(step.step_wall_time)

        # --- Render text ---
        mu_col = int(self._collided)

        if self._dev_count > 0:
            mu_dev = self._dev_sum / self._dev_count
            mu_dev_str = f"{mu_dev:.4f} m"
        else:
            mu_dev_str = "—"

        if is_final:
            mu_goal = int(self._goal_reached)
            mu_goal_str = str(mu_goal)
        else:
            if self._goal_reached:
                mu_goal_str = "1"
            else:
                mu_goal_str = "pending"

        if self._vel_sample_count > 0:
            mu_vel = float(
                np.sqrt(self._vel_sq_err_sum / self._vel_sample_count),
            )
            mu_vel_str = f"{mu_vel:.4f} m/s"
        else:
            mu_vel_str = "N/A"

        if self._step_times:
            arr = np.asarray(self._step_times, dtype=float)
            mu_comp = float(arr.mean())
            mu_lat = float(np.percentile(arr, 95.0))
            mu_comp_str = f"{mu_comp:.6f} s"
            mu_lat_str = f"{mu_lat:.6f} s"
        else:
            mu_comp_str = "—"
            mu_lat_str = "—"

        weight = "bold" if is_final else "normal"

        lines = [
            f"μ_col   (collision):      {mu_col}",
            f"μ_dev   (path deviation): {mu_dev_str}",
            f"μ_goal  (goal reached):   {mu_goal_str}",
            f"μ_vel   (velocity RMSE):  {mu_vel_str}",
            f"μ_comp  (mean CPU time):  {mu_comp_str}",
            f"μ_lat   (p95 CPU time):   {mu_lat_str}",
            "",
            f"Step {step.step} / {total_steps}   |   t = {step.time:.2f} s",
        ]
        text = "\n".join(lines)

        if self._text_artist is not None:
            self._text_artist.remove()
        self._text_artist = self._ax.text(
            0.05, 0.85, text,
            transform=self._ax.transAxes,
            fontsize=9, fontfamily="monospace",
            fontweight=weight,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="#F8F9FA",
                edgecolor="#DEE2E6",
                alpha=0.9,
            ),
        )
