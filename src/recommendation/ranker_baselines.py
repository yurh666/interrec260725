from __future__ import annotations

import pandas as pd

from src.embedding.bm25_index import BM25Index
from src.embedding.item_encoder import build_item_text


def build_profile_query(observed_item_ids: list[str], items: pd.DataFrame) -> str:
    item_set = {str(x) for x in observed_item_ids}
    rows = items[items["item_id"].astype(str).isin(item_set)]
    return "\n".join(build_item_text(row) for _, row in rows.fillna("").iterrows())


class BM25BaselineRecommender:
    implementation_mode = "reproduction"

    def __init__(self, items: pd.DataFrame) -> None:
        self.items = items
        self.index = BM25Index.build(items)

    def recommend(self, observed_item_ids: list[str], top_k: int = 10) -> list[tuple[str, float]]:
        query = build_profile_query(observed_item_ids, self.items)
        return self.index.search(query, top_k=top_k, exclude=set(observed_item_ids))
