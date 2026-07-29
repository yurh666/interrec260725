from __future__ import annotations

import numpy as np

from src.belief.belief_state import BeliefState
from src.utils.math import l2_normalize


def implicit_positive_update(belief: BeliefState, clicked_vectors: np.ndarray, lr: float = 0.1, shrink: float = 0.98) -> BeliefState:
    if len(clicked_vectors) == 0:
        return belief.copy()
    target = l2_normalize(np.mean(clicked_vectors, axis=0).reshape(1, -1))[0]
    updated = belief.copy()
    updated.mu = l2_normalize(((1 - lr) * updated.mu + lr * target).reshape(1, -1))[0]
    updated.Sigma = updated.Sigma * shrink
    updated.ensure_psd(force=True)
    return updated
