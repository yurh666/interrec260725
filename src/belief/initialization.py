from __future__ import annotations

import numpy as np

from src.belief.belief_state import BeliefState
from src.embedding.index import EmbeddingIndex
from src.utils.math import l2_normalize, weighted_average


def initialize_belief(
    observed_history: list[str],
    index: EmbeddingIndex,
    weights: list[float] | None = None,
    init_mode: str = "weighted_average",
    sigma0: float = 1.0,
    jitter: float = 1e-6,
) -> BeliefState:
    dim = index.embeddings.shape[1]
    if init_mode == "cold_start" or not observed_history:
        mu = np.zeros(dim, dtype=float)
    elif init_mode == "weighted_average":
        vectors = index.vectors_for(observed_history)
        if len(vectors) == 0:
            mu = np.zeros(dim, dtype=float)
        else:
            usable_weights = np.asarray(weights[: len(vectors)] if weights else np.ones(len(vectors)), dtype=float)
            mu = weighted_average(vectors, usable_weights)
            mu = l2_normalize(mu.reshape(1, -1))[0]
    else:
        raise ValueError(f"Unsupported belief init mode: {init_mode}")
    Sigma = (sigma0 ** 2) * np.eye(dim, dtype=float)
    return BeliefState(mu=mu, Sigma=Sigma, jitter=jitter)
