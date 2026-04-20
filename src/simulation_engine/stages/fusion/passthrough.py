"""
Pass-through fusion stage.

`bar_O_t = hat_O_t` -- no temporal fusion. Per detection becomes its own
fused observation with zero velocity and an anonymous per-step track ID.
Serves as the minimal-computation baseline.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ...types import DetectedObstacle, FusedObstacle
from ..base import FusionStage
from ..registry import register_fusion


class PassThroughFusion(FusionStage):
    """Pass-through fusion."""

    name = "PT"

    def reset(self, seed: int) -> None:
        return None

    def step(
        self,
        detections: List[DetectedObstacle],
        dt: float,
    ) -> List[FusedObstacle]:
        zero = np.zeros(2, dtype=float)
        return [
            FusedObstacle(
                track_id=i,
                position=d.centroid.copy(),
                velocity=zero.copy(),
                radius=d.radius,
            )
            for i, d in enumerate(detections)
        ]


register_fusion("PT", PassThroughFusion)
