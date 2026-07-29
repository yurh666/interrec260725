from __future__ import annotations

from typing import Any

import numpy as np

from src.belief.belief_state import BeliefState
from src.embedding.index import EmbeddingIndex


def find_uncertainty_directions(
    belief: BeliefState,
    K: int = 5,
) -> list[dict[str, Any]]:
    """Return top-K eigenvectors of Sigma_t as uncertainty directions."""
    eigenvalues, eigenvectors = belief.eigendecompose(K)
    return [
        {
            "direction_id": k,
            "lambda": float(eigenvalues[k]),
            "eigenvector": eigenvectors[:, k],
        }
        for k in range(len(eigenvalues))
    ]


def find_anchors_for_direction(
    direction: dict[str, Any],
    belief: BeliefState,
    index: EmbeddingIndex,
    top_k: int = 5,
    exclude: set[str] | None = None,
) -> dict[str, Any]:
    """For one uncertainty direction, find positive and negative anchor items."""
    v = direction["eigenvector"]
    centered = index.embeddings - belief.mu[None, :]
    projections = centered @ v
    order = np.argsort(projections)
    exclude = exclude or set()

    def pick(indices: list[int]) -> list[str]:
        out = []
        for idx in indices:
            item_id = index.item_ids[int(idx)]
            if item_id not in exclude:
                out.append(item_id)
            if len(out) >= top_k:
                break
        return out

    pos_anchors = pick(list(order[::-1]))
    neg_anchors = pick(list(order))

    return {
        "direction_id": direction["direction_id"],
        "lambda": direction["lambda"],
        "eigenvector": v,
        "positive_anchors": pos_anchors,
        "negative_anchors": neg_anchors,
    }


def build_anchor_directions(
    belief: BeliefState,
    index: EmbeddingIndex,
    K: int = 5,
    anchor_top_k: int = 5,
    exclude: set[str] | None = None,
) -> list[dict[str, Any]]:
    directions = find_uncertainty_directions(belief, K)
    return [
        find_anchors_for_direction(d, belief, index, anchor_top_k, exclude)
        for d in directions
    ]
