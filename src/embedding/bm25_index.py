from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from src.embedding.item_encoder import build_item_text


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+", str(text).lower())


@dataclass
class BM25Index:
    item_ids: list[str]
    corpus_tokens: list[list[str]]
    engine: object

    @classmethod
    def build(cls, items: pd.DataFrame) -> "BM25Index":
        from rank_bm25 import BM25Okapi

        texts = [build_item_text(row) for _, row in items.fillna("").iterrows()]
        tokens = [tokenize(t) for t in texts]
        return cls(items["item_id"].astype(str).tolist(), tokens, BM25Okapi(tokens))

    def search(self, query: str, top_k: int = 10, exclude: set[str] | None = None) -> list[tuple[str, float]]:
        scores = self.engine.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), str(self.item_ids[i])))
        exclude = exclude or set()
        out = []
        for idx in order:
            item_id = self.item_ids[idx]
            if item_id in exclude:
                continue
            out.append((item_id, float(scores[idx])))
            if len(out) >= top_k:
                break
        return out

    def search_pos_neg(
        self,
        positive_query: str,
        negative_query: str,
        *,
        gamma: float,
        prefetch_k: int = 300,
        exclude: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Rank items by BM25(pos) - gamma * BM25(neg); return up to prefetch_k hits (for re-ranking)."""
        import numpy as np

        pos_s = np.asarray(self.engine.get_scores(tokenize(positive_query)), dtype=float)
        if negative_query and str(negative_query).strip():
            neg_s = np.asarray(self.engine.get_scores(tokenize(negative_query)), dtype=float)
        else:
            neg_s = np.zeros_like(pos_s)
        comb = pos_s - float(gamma) * neg_s
        exclude = exclude or set()
        out: list[tuple[str, float]] = []
        order = sorted(range(len(comb)), key=lambda i: (-float(comb[i]), str(self.item_ids[i])))
        for idx in order:
            iid = self.item_ids[int(idx)]
            if iid in exclude:
                continue
            out.append((iid, float(comb[int(idx)])))
            if len(out) >= prefetch_k:
                break
        return out
