from __future__ import annotations

from typing import Any

import numpy as np

from src.embedding.index import EmbeddingIndex
from src.utils.math import l2_normalize, weighted_average


def build_theta_star(
    session: dict[str, Any],
    index: EmbeddingIndex,
    split: str = "test",
    mode: str = "future_average",
    dataset: Any = None,
) -> dict[str, Any]:
    """Construct the ground-truth preference vector theta* for a user simulation.

    Parameters
    ----------
    session:
        Session dict from sessions.json. Contains observed_history / future_train /
        future_valid / future_test.
    index:
        Pre-built embedding index for all items.
    split:
        One of 'train', 'valid', 'test'. Determines which future split is used to
        build theta*. This is recorded in the run log as theta_star_source.
    mode:
        'future_average' — weighted average of future split embeddings.
    dataset:
        InterRecDataset (optional) — used to retrieve per-item play_count weights.

    Returns
    -------
    dict with keys: theta_star (np.ndarray), theta_star_source (str), theta_star_items (list).
    """
    split_key = {"train": "future_train", "valid": "future_valid", "test": "future_test"}.get(
        split, "future_test"
    )
    source_items: list[str] = session.get(split_key, [])
    if not source_items:
        dim = index.embeddings.shape[1]
        return {
            "theta_star": np.zeros(dim, dtype=float),
            "theta_star_source": split_key,
            "theta_star_items": [],
        }

    available = [item_id for item_id in source_items if str(item_id) in index.item_id_to_index]
    if not available:
        dim = index.embeddings.shape[1]
        return {
            "theta_star": np.zeros(dim, dtype=float),
            "theta_star_source": split_key,
            "theta_star_items": available,
        }

    vectors = index.vectors_for(available)
    if dataset is not None:
        weights = dataset.weights_for_items(session["user_id"], available)
        weights = np.array(weights, dtype=float)
    else:
        weights = np.ones(len(available), dtype=float)

    theta = weighted_average(vectors, weights)
    norm = np.linalg.norm(theta)
    if norm > 1e-9:
        theta = theta / norm

    return {
        "theta_star": theta.astype(float),
        "theta_star_source": split_key,
        "theta_star_items": available,
    }
