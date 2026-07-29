from __future__ import annotations

import numpy as np

from src.belief.belief_state import BeliefState


def belief_scores(belief: BeliefState, item_embeddings: np.ndarray, lambda_explore: float = 0.1) -> np.ndarray:
    mean_score = item_embeddings @ belief.mu
    uncertainty = np.einsum("ij,jk,ik->i", item_embeddings, belief.Sigma, item_embeddings)
    return mean_score + lambda_explore * np.sqrt(np.maximum(uncertainty, 0.0))
