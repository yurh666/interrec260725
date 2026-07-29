from __future__ import annotations

from typing import Any

import numpy as np

from src.utils.math import cosine


def option_alignment_scores(
    memory_centroids: np.ndarray,
    option_vectors: np.ndarray,
) -> np.ndarray:
    """Return cosine similarity matrix [n_units, n_options]."""

    if memory_centroids.size == 0 or option_vectors.size == 0:
        return np.zeros((len(memory_centroids), len(option_vectors)), dtype=float)
    scores = np.zeros((len(memory_centroids), len(option_vectors)), dtype=float)
    for i, c in enumerate(memory_centroids):
        for j, v in enumerate(option_vectors):
            scores[i, j] = cosine(c, v)
    return scores


def evidence_from_choice(
    *,
    alignment_scores: np.ndarray,
    selected_index: int | None,
    none_index: int | None = None,
) -> dict[str, Any]:
    n_units = alignment_scores.shape[0]
    if selected_index is None:
        return {
            "operation_hint": "downweight",
            "target_unit_indices": list(range(n_units)),
            "selected_alignment": None,
            "confidence": 0.3,
        }

    unit_scores = alignment_scores[:, selected_index]
    best_idx = int(np.argmax(unit_scores)) if n_units else -1
    best_score = float(unit_scores[best_idx]) if best_idx >= 0 else 0.0

    if best_score < 0.35:
        return {
            "operation_hint": "create",
            "target_unit_indices": [],
            "selected_alignment": best_score,
            "confidence": max(best_score, 0.2),
        }

    return {
        "operation_hint": "update",
        "target_unit_indices": [best_idx],
        "selected_alignment": best_score,
        "confidence": best_score,
    }
