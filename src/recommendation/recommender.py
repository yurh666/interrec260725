from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.belief.belief_state import BeliefState
from src.embedding.index import EmbeddingIndex
from src.recommendation.scorer import belief_scores

if TYPE_CHECKING:
    from src.recommendation.csc_readout import CscReadoutOverlay


class BeliefRecommender:
    def __init__(self, index: EmbeddingIndex, lambda_explore: float = 0.1) -> None:
        self.index = index
        self.lambda_explore = lambda_explore
        self._csc_overlay: CscReadoutOverlay | None = None

    def set_csc_overlay(self, overlay: CscReadoutOverlay | None) -> None:
        self._csc_overlay = overlay

    def clear_csc_overlay(self) -> None:
        self._csc_overlay = None

    @property
    def csc_overlay_active(self) -> bool:
        return self._csc_overlay is not None

    def _scores_for_ranking(self, belief: BeliefState) -> np.ndarray:
        if self._csc_overlay is not None:
            return self._csc_overlay.adjusted_scores
        return belief_scores(belief, self.index.embeddings, self.lambda_explore)

    def recommend(self, belief: BeliefState, top_k: int = 10, exclude: set[str] | None = None) -> list[tuple[str, float]]:
        ranked = self.rank_all(belief, exclude=exclude)
        return ranked[:top_k]

    def rank_all(self, belief: BeliefState, exclude: set[str] | None = None) -> list[tuple[str, float]]:
        scores = self._scores_for_ranking(belief)
        order = np.argsort(-scores)
        exclude = exclude or set()
        out: list[tuple[str, float]] = []
        for idx in order:
            item_id = self.index.item_ids[int(idx)]
            if item_id in exclude:
                continue
            out.append((item_id, float(scores[int(idx)])))
        return out
