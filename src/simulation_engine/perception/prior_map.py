"""
Prior-map degradation  P(E, eta_cpl, sigma_map) -> M_0.

Applies two independent degradations to the environment structure E:

1. Chain dropout: each chain in E is retained with probability eta_cpl.
   For eta_cpl = 1 every chain is retained; for eta_cpl = 0 only an empty
   map remains. The outer perimeter is not treated specially: eta_cpl
   is the fraction of boundary elements retained, with no distinction
   between perimeter and internal walls.
2. Per-vertex isotropic Gaussian jitter of std dev sigma_map applied
   independently to each retained vertex. For sigma_map = 0 the chains
   are returned unchanged in shape.

The prior map is computed once at the start of a run; subsequent steps
reuse it (computed once and does not change).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from world_generator.models import EnvironmentStructure

from ..types import PriorMap


def degrade_prior_map(
    environment: EnvironmentStructure,
    eta_cpl: float,
    sigma_map: float,
    rng: np.random.Generator,
) -> PriorMap:
    """
    Implements  M_0 = P(E, eta_cpl, sigma_map).

    Parameters
    ----------
    environment : E, the ground-truth environment structure
    eta_cpl     : fraction of chains retained (Bernoulli per chain)
    sigma_map   : per-vertex isotropic Gaussian jitter std dev (m)
    rng         : np.random.Generator driving the degradation
    """
    if not (0.0 <= eta_cpl <= 1.0):
        raise ValueError(f"eta_cpl must be in [0, 1]; got {eta_cpl}")
    if sigma_map < 0:
        raise ValueError(f"sigma_map must be >= 0; got {sigma_map}")

    retained: list = []
    for chain in environment.chains:
        if rng.random() < eta_cpl:
            jittered = np.asarray(chain, dtype=float)
            if sigma_map > 0:
                jittered = jittered + rng.normal(
                    loc=0.0, scale=sigma_map, size=jittered.shape
                )
            retained.append(jittered)
    return PriorMap(chains=retained, size=tuple(environment.size))
