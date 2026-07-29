from __future__ import annotations


class BGEReranker:
    """Placeholder for the later baseline milestone.

    The current InterRec milestone intentionally avoids running heavy BGE reranking
    while baseline experiments are active.
    """

    implementation_mode = "not_implemented"

    def rerank(self, query: str, candidates: list[str]) -> list[str]:
        raise NotImplementedError("BGE reranker baseline is out of scope for the current milestone.")
