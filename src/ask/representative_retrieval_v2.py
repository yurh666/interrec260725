"""Representative retrieval v2: positive-negative BM25 and option overlap penalty."""

from __future__ import annotations

import hashlib
from typing import Any


def mean_pairwise_jaccard_overlap(rep_sets: list[set[str]]) -> float:
    """Mean Jaccard similarity between representative sets (high ⇒ more overlap)."""
    m = len(rep_sets)
    if m < 2:
        return 0.0
    vals: list[float] = []
    for i in range(m):
        for j in range(i + 1, m):
            a, b = rep_sets[i], rep_sets[j]
            if not a and not b:
                continue
            inter = len(a & b)
            union = len(a | b) or 1
            vals.append(inter / union)
    return sum(vals) / len(vals) if vals else 0.0


def centroid_vectors_hash(rep_ids: list[str], vector_norm: float) -> str:
    """Stable short hash for logging (not cryptographic)."""

    payload = "|".join(str(x) for x in rep_ids) + f"|{vector_norm:.6f}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def prefetch_pos_neg(
    bm25_index: Any,
    *,
    positive_query: str,
    negative_query: str,
    gamma: float,
    prefetch_k: int,
) -> list[tuple[str, float]]:
    if bm25_search := getattr(bm25_index, "search_pos_neg", None):
        return list(
            bm25_search(
                str(positive_query),
                str(negative_query or ""),
                gamma=float(gamma),
                prefetch_k=int(prefetch_k),
            )
        )
    return list(bm25_index.search(str(positive_query), top_k=int(prefetch_k)))


def apply_overlap_penalty(
    prefetched: list[list[tuple[str, float]]],
    *,
    rep_k: int,
    coef: float,
) -> list[list[tuple[str, float]]]:
    """Penalty for items that appear in other options' provisional top-`rep_k` sets."""
    if not prefetched:
        return prefetched
    initial_sets: list[set[str]] = []
    for cand in prefetched:
        initial_sets.append(set(item_id for item_id, _ in cand[:rep_k]))

    adjusted: list[list[tuple[str, float]]] = []
    for idx, cand in enumerate(prefetched):
        row: list[tuple[str, float]] = []
        for item_id, sc in cand:
            overlap_hits = sum(
                1 for j, s in enumerate(initial_sets) if j != idx and item_id in s
            )
            row.append((item_id, float(sc) - float(coef) * overlap_hits))
        row.sort(key=lambda x: (-x[1], str(x[0])))
        adjusted.append(row)
    return adjusted

