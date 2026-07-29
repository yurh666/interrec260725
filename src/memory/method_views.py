"""Method-naming views over belief state and frontier clusters.

These wrappers alias existing runtime objects for paper-aligned narrative
(Bayesian Latent Intent Memory / Transient Intent Slots) without changing
numerics or control flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.belief.belief_state import BeliefState


@dataclass(frozen=True)
class BayesianLatentIntentMemoryView:
    """Read-only view of ``BeliefState`` as Bayesian Latent Intent Memory M_t=(mu, Sigma)."""

    mean: np.ndarray  # alias of belief.mu
    covariance: np.ndarray  # alias of belief.Sigma

    @classmethod
    def from_belief(cls, belief: BeliefState) -> BayesianLatentIntentMemoryView:
        return cls(mean=belief.mu, covariance=belief.Sigma)

    def to_belief_state(self, *, jitter: float = 1e-6) -> BeliefState:
        return BeliefState(self.mean.copy(), self.covariance.copy(), jitter=jitter)


@dataclass(frozen=True)
class TransientIntentSlotView:
    """Read-only view of a frontier cluster as a transient intent slot for the current turn."""

    slot_id: int  # alias of cluster_id in current round
    member_item_ids: tuple[str, ...]
    centroid: np.ndarray  # alias of cluster centroid / mean embedding direction
    role: str  # preserve / boundary / explore

    @classmethod
    def from_frontier_cluster(cls, cluster: Any, role: str) -> TransientIntentSlotView:
        member_ids = tuple(str(x) for x in getattr(cluster, "member_ids", []) or [])
        raw_centroid = getattr(cluster, "mean_embedding_direction", None)
        if raw_centroid is None or len(raw_centroid) == 0:
            centroid = np.zeros(0, dtype=np.float64)
        else:
            centroid = np.asarray(raw_centroid, dtype=np.float64)
        return cls(
            slot_id=int(getattr(cluster, "cluster_id")),
            member_item_ids=member_ids,
            centroid=centroid,
            role=str(role),
        )
