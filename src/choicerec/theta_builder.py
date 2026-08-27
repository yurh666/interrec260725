from __future__ import annotations

from typing import Any

import numpy as np

from src.embedding.index import EmbeddingIndex
from src.utils.math import weighted_average


def build_choicerec_theta_star(
    *,
    remaining_future_valid: list[str],
    future_test: list[str],
    index: EmbeddingIndex,
    session: dict[str, Any],
    dataset: Any = None,
) -> dict[str, Any]:
    """Turn-specific hidden target: mean(remaining valid + future_test) embeddings."""
    source_items = [str(x) for x in remaining_future_valid] + [str(x) for x in future_test]
    available = [iid for iid in source_items if str(iid) in index.item_id_to_index]
    dim = index.embeddings.shape[1]
    if not available:
        return {
            "theta_star": np.zeros(dim, dtype=float),
            "theta_star_source": "choicerec_dynamic",
            "theta_star_items": [],
            "remaining_valid_count": len(remaining_future_valid),
            "future_test_count": len(future_test),
        }

    vectors = index.vectors_for(available)
    if dataset is not None:
        weights = np.asarray(
            dataset.weights_for_items(session["user_id"], available),
            dtype=float,
        )
    else:
        weights = np.ones(len(available), dtype=float)

    theta = weighted_average(vectors, weights)
    norm = float(np.linalg.norm(theta))
    if norm > 1e-9:
        theta = theta / norm

    return {
        "theta_star": theta.astype(float),
        "theta_star_source": "choicerec_dynamic",
        "theta_star_items": available,
        "remaining_valid_count": len(remaining_future_valid),
        "future_test_count": len(future_test),
    }
