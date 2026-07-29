from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.memory.memory_evidence import evidence_from_choice, option_alignment_scores


@dataclass
class MemoryUpdateConfig:
    create_similarity_threshold: float = 0.35
    merge_similarity_threshold: float = 0.93
    downweight_factor: float = 0.85
    update_ema: float = 0.25
    support_gain: float = 1.0
    min_support_weight: float = 0.1
    allow_create: bool = True
    allow_merge: bool = True
    allow_downweight: bool = True
    enable_competing_downweight: bool = False
    competing_downweight_factor: float = 0.97
    competing_alignment_threshold: float = 0.35


def decide_memory_operation(
    *,
    memory_state,
    option_vectors: np.ndarray,
    selected_index: int | None,
    selected_vector: np.ndarray | None,
    config: MemoryUpdateConfig,
) -> dict[str, Any]:
    centroids = np.stack([u.centroid for u in memory_state.units], axis=0)
    align = option_alignment_scores(centroids, option_vectors)
    evidence = evidence_from_choice(
        alignment_scores=align,
        selected_index=selected_index,
    )

    op = evidence["operation_hint"]
    if selected_index is None:
        op = "downweight"
    elif op == "create" and selected_vector is not None:
        max_sim = float(np.max(align[:, selected_index])) if align.size else 0.0
        if max_sim >= config.create_similarity_threshold:
            op = "update"
    if op == "create" and not config.allow_create:
        op = "update" if selected_vector is not None and memory_state.units else "noop"
        if op == "update" and not evidence.get("target_unit_indices"):
            nearest = int(np.argmax(align[:, selected_index])) if align.size and selected_index is not None else 0
            evidence["target_unit_indices"] = [nearest]
    if op == "downweight" and not config.allow_downweight:
        op = "noop"
    if op == "merge" and not config.allow_merge:
        op = "update" if selected_vector is not None else "noop"

    target_ids = [memory_state.units[i].unit_id for i in evidence.get("target_unit_indices", []) if i >= 0]

    return {
        "operation": op,
        "target_unit_ids": target_ids,
        "target_unit_indices": evidence.get("target_unit_indices", []),
        "selected_alignment": evidence.get("selected_alignment"),
        "confidence": evidence.get("confidence"),
    }
