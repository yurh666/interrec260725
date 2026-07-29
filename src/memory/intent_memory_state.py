from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class IntentMemoryUnit:
    unit_id: str
    centroid: np.ndarray
    covariance: np.ndarray
    support_items: list[str]
    support_weight: float
    granularity_level: int
    uncertainty: float
    semantic_label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "centroid": self.centroid.tolist(),
            "covariance_trace": float(np.trace(self.covariance)),
            "support_items": list(self.support_items),
            "support_weight": float(self.support_weight),
            "granularity_level": int(self.granularity_level),
            "uncertainty": float(self.uncertainty),
            "semantic_label": self.semantic_label,
        }


@dataclass
class IntentMemoryState:
    units: list[IntentMemoryUnit]
    dim: int
    turn_id: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def unit_count(self) -> int:
        return len(self.units)

    def copy(self) -> "IntentMemoryState":
        copied_units = [
            IntentMemoryUnit(
                unit_id=u.unit_id,
                centroid=u.centroid.copy(),
                covariance=u.covariance.copy(),
                support_items=list(u.support_items),
                support_weight=u.support_weight,
                granularity_level=u.granularity_level,
                uncertainty=u.uncertainty,
                semantic_label=u.semantic_label,
            )
            for u in self.units
        ]
        return IntentMemoryState(
            units=copied_units,
            dim=self.dim,
            turn_id=self.turn_id,
            metadata=dict(self.metadata),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "unit_count": self.unit_count,
            "turn_id": self.turn_id,
            "total_support_weight": float(sum(u.support_weight for u in self.units)),
            "mean_uncertainty": float(np.mean([u.uncertainty for u in self.units])) if self.units else 0.0,
            "granularity_levels": sorted({u.granularity_level for u in self.units}),
        }
