from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.utils.io import load_pickle, read_json
from src.utils.math import l2_normalize


class EmbeddingIndex:
    """CPU retrieval from offline ``item_embeddings.npy`` (normalized dot-product, no CUDA)."""

    def __init__(self, embeddings: np.ndarray, item_ids: list[str], item_id_to_index: dict[str, int]) -> None:
        self.embeddings = embeddings.astype(float)
        self.item_ids = [str(x) for x in item_ids]
        self.item_id_to_index = {str(k): int(v) for k, v in item_id_to_index.items()}

    @classmethod
    def load(cls, processed_dir: str | Path, embedding_filename: str = "item_embeddings.npy") -> "EmbeddingIndex":
        p = Path(processed_dir)
        embeddings = np.load(p / embedding_filename)
        item_id_to_index = read_json(p / "item_id_to_index.json")
        item_ids_json = [
            item_id for item_id, _ in sorted(item_id_to_index.items(), key=lambda kv: int(kv[1]))
        ]
        # Prefer deterministic recovery from JSON + .npy (works on CPU-only machines).
        # item_index.pkl may contain sklearn / CUDA artifacts from another host and fail to unpickle here.
        if len(item_ids_json) == embeddings.shape[0]:
            return cls(embeddings, item_ids_json, item_id_to_index)
        meta = load_pickle(p / "item_index.pkl")
        return cls(embeddings, meta["item_ids"], item_id_to_index)

    def vectors_for(self, item_ids: list[str]) -> np.ndarray:
        idx = [self.item_id_to_index[str(item_id)] for item_id in item_ids if str(item_id) in self.item_id_to_index]
        return self.embeddings[idx]

    def vector_for(self, item_id: str) -> np.ndarray:
        return self.embeddings[self.item_id_to_index[str(item_id)]]

    def search(self, query: np.ndarray, top_k: int = 10, exclude: set[str] | None = None) -> list[tuple[str, float]]:
        q = l2_normalize(query.reshape(1, -1))[0]
        scores = self.embeddings @ q
        order = np.argsort(-scores)
        out: list[tuple[str, float]] = []
        exclude = exclude or set()
        for idx in order:
            item_id = self.item_ids[int(idx)]
            if item_id in exclude:
                continue
            out.append((item_id, float(scores[int(idx)])))
            if len(out) >= top_k:
                break
        return out

    def item_meta(self) -> dict[str, Any]:
        return {"n_items": len(self.item_ids), "dim": int(self.embeddings.shape[1])}
