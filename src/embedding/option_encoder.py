from __future__ import annotations

import numpy as np

from src.embedding.index import EmbeddingIndex
from src.utils.math import l2_normalize


def option_vector_from_items(index: EmbeddingIndex, item_ids: list[str]) -> np.ndarray:
    vectors = index.vectors_for(item_ids)
    if len(vectors) == 0:
        return np.zeros(index.embeddings.shape[1], dtype=float)
    return l2_normalize(np.mean(vectors, axis=0).reshape(1, -1))[0]
