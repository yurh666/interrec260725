from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import read_json


@dataclass
class InterRecDataset:
    interactions: pd.DataFrame
    items: pd.DataFrame
    user_splits: list[dict[str, Any]]
    sessions: list[dict[str, Any]]

    @classmethod
    def load(cls, processed_dir: str | Path) -> "InterRecDataset":
        p = Path(processed_dir)
        return cls(
            interactions=pd.read_csv(p / "interactions.csv"),
            items=pd.read_csv(p / "items.csv").fillna(""),
            user_splits=read_json(p / "user_splits.json"),
            sessions=read_json(p / "sessions.json"),
        )

    def item_texts(self) -> list[str]:
        from src.embedding.item_encoder import build_item_text

        return [build_item_text(row) for _, row in self.items.iterrows()]

    def item_ids(self) -> list[str]:
        return self.items["item_id"].astype(str).tolist()

    @property
    def _item_texts_cache(self) -> dict[str, str]:
        """Lazy-built dict: item_id -> concatenated text string (for keyword fallback).

        Used by hypothesis_vectorizer when BM25 is not available.
        """
        if not hasattr(self, "_item_texts_cache_"):
            from src.embedding.item_encoder import build_item_text
            self._item_texts_cache_ = {
                str(row["item_id"]): build_item_text(row)
                for _, row in self.items.iterrows()
            }
        return self._item_texts_cache_

    def item_lookup(self) -> dict[str, dict[str, Any]]:
        """Return dict: item_id -> metadata row dict (for LLM prompt enrichment)."""
        return {
            str(row["item_id"]): row.to_dict()
            for _, row in self.items.iterrows()
        }

    def weights_for_items(self, user_id: str, item_ids: list[str]) -> list[float]:
        rows = self.interactions[
            (self.interactions["user_id"].astype(str) == str(user_id))
            & (self.interactions["item_id"].astype(str).isin(item_ids))
        ]
        weight_col = None
        for col in ["play_count", "rating", "behavior"]:
            if col in rows.columns:
                weight_col = col
                break
        if weight_col is None:
            return [1.0 for _ in item_ids]
        weight_map = rows.groupby("item_id")[weight_col].sum().to_dict()
        return [math.log1p(float(weight_map.get(item_id, 1.0))) for item_id in item_ids]
