from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.belief.belief_state import BeliefState
from src.memory.adaptive_intent_units import initialize_adaptive_intent_memory
from src.memory.intent_memory_state import IntentMemoryState
from src.utils.math import l2_normalize


@dataclass
class IncrementalHistoryMemoryBuilder:
    """Build an observed-history memory state without changing InterRec belief geometry."""

    dim: int
    sigma0: float = 1.0
    jitter: float = 1e-6
    adaptive_units_enabled: bool = True
    adaptive_units_config: dict[str, Any] = field(default_factory=dict)

    total_weight: float = 0.0
    _weighted_mu: np.ndarray = field(init=False, repr=False)
    _m2: np.ndarray = field(init=False, repr=False)
    _item_ids: list[str] = field(default_factory=list, init=False, repr=False)
    _vectors: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _weights: list[float] = field(default_factory=list, init=False, repr=False)
    construction_trace: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._weighted_mu = np.zeros(self.dim, dtype=float)
        self._m2 = np.zeros((self.dim, self.dim), dtype=float)

    def add_item(self, item_id: str, embedding: np.ndarray, weight: float) -> None:
        vec = np.asarray(embedding, dtype=float)
        if vec.shape[0] != self.dim:
            raise ValueError(f"Expected embedding dim {self.dim}, got {vec.shape[0]}")
        item_weight = max(float(weight), 0.0)
        if item_weight <= 0:
            item_weight = 1.0

        old_weight = self.total_weight
        old_mu = self._weighted_mu.copy()
        new_weight = old_weight + item_weight
        new_mu = (old_weight * old_mu + item_weight * vec) / new_weight

        # Weighted Welford update for future covariance support; compatible mode
        # still reuses original sigma by default.
        if old_weight > 0:
            delta = vec - old_mu
            delta2 = vec - new_mu
            self._m2 += item_weight * np.outer(delta, delta2)

        self.total_weight = new_weight
        self._weighted_mu = new_mu
        self._item_ids.append(str(item_id))
        self._vectors.append(vec.copy())
        self._weights.append(item_weight)
        self.construction_trace.append(
            {
                "step": len(self._item_ids),
                "item_id": str(item_id),
                "item_weight": float(item_weight),
                "total_weight": float(self.total_weight),
                "mu_norm": float(np.linalg.norm(self.equivalent_mu)),
                "used_future_test": False,
                "recency_used": False,
            }
        )

    @property
    def equivalent_mu(self) -> np.ndarray:
        if self.total_weight <= 0:
            return np.zeros(self.dim, dtype=float)
        return l2_normalize(self._weighted_mu.reshape(1, -1))[0]

    def online_covariance(self) -> np.ndarray:
        if self.total_weight <= 0:
            return np.eye(self.dim, dtype=float) * (self.sigma0 ** 2)
        cov = self._m2 / max(self.total_weight, 1e-12)
        cov = 0.5 * (cov + cov.T)
        return cov + np.eye(self.dim, dtype=float) * self.jitter

    def finalize(
        self,
        *,
        original_belief: BeliefState | None = None,
    ) -> tuple[IntentMemoryState, np.ndarray, np.ndarray]:
        equivalent_mu = self.equivalent_mu
        equivalent_sigma = (
            original_belief.Sigma.copy()
            if original_belief is not None
            else np.eye(self.dim, dtype=float) * (self.sigma0 ** 2)
        )

        memory_state = initialize_adaptive_intent_memory(
            observed_history=self._item_ids,
            index=_InMemoryIndex(self._item_ids, np.asarray(self._vectors, dtype=float), self.dim),
            weights=list(self._weights),
            split_dispersion_threshold=float(self.adaptive_units_config.get("split_dispersion_threshold", 0.08)),
            merge_cosine_threshold=float(self.adaptive_units_config.get("merge_cosine_threshold", 0.92)),
            min_support_weight=float(self.adaptive_units_config.get("min_support_weight", 0.5)),
            max_granularity_level=int(self.adaptive_units_config.get("max_granularity_level", 3)),
            max_units=(
                int(self.adaptive_units_config["max_units"])
                if self.adaptive_units_config.get("max_units") is not None
                else None
            ),
            jitter=self.jitter,
            sigma0=self.sigma0,
        ) if self.adaptive_units_enabled else IntentMemoryState(units=[], dim=self.dim)

        memory_state.metadata.update(
            {
                "init_mode": "incremental_history",
                "view_only": bool(self.adaptive_units_config.get("use_as_view_only", True)),
                "history_item_count": len(self._item_ids),
                "total_weight": float(self.total_weight),
                "construction_trace": list(self.construction_trace),
                "equivalent_mu_norm": float(np.linalg.norm(equivalent_mu)),
                "sigma_source": "original_initialize_belief" if original_belief is not None else "isotropic",
                "future_test_used": False,
                "recency_used": False,
            }
        )
        return memory_state, equivalent_mu, equivalent_sigma


class _InMemoryIndex:
    def __init__(self, item_ids: list[str], vectors: np.ndarray, dim: int) -> None:
        self.item_ids = [str(x) for x in item_ids]
        self.embeddings = vectors.reshape((len(item_ids), dim)) if item_ids else np.zeros((0, dim), dtype=float)
        self.item_id_to_index = {item_id: i for i, item_id in enumerate(self.item_ids)}

    def vectors_for(self, item_ids: list[str]) -> np.ndarray:
        idx = [self.item_id_to_index[str(item_id)] for item_id in item_ids if str(item_id) in self.item_id_to_index]
        return self.embeddings[idx]
