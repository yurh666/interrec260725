from __future__ import annotations

import math


def hit_rate(recommended: list[str], relevant: set[str], k: int) -> float:
    return float(any(item in relevant for item in recommended[:k]))


def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return float(dcg / idcg) if idcg > 0 else 0.0


def mrr_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """Recall@k = |rec[:k] ∩ relevant| / |relevant| (0 if relevant empty)."""
    if not relevant:
        return 0.0
    inter = len(set(recommended[:k]) & relevant)
    return float(inter / len(relevant))


def strict_ranking_metrics(
    recommended: list[str],
    relevant: set[str],
    ks: tuple[int, ...] = (5, 10),
) -> dict[str, float]:
    """Standard Hit/Recall/NDCG/MRR at each k using only ``relevant`` (= future_test)."""
    if not relevant:
        return {}
    out: dict[str, float] = {}
    for k in ks:
        out[f"HitRate@{k}"] = hit_rate(recommended, relevant, k)
        out[f"Recall@{k}"] = recall_at_k(recommended, relevant, k)
        out[f"NDCG@{k}"] = ndcg_at_k(recommended, relevant, k)
        out[f"MRR@{k}"] = mrr_at_k(recommended, relevant, k)
    return out


def ranking_metrics_multik(recommended: list[str], relevant: set[str], ks: tuple[int, ...] = (5, 10)) -> dict[str, float]:
    """Hit / NDCG / Recall at each k, plus MRR@5 and MRR@10."""
    out: dict[str, float] = {}
    for k in ks:
        out[f"HitRate@{k}"] = hit_rate(recommended, relevant, k)
        out[f"NDCG@{k}"] = ndcg_at_k(recommended, relevant, k)
        out[f"Recall@{k}"] = recall_at_k(recommended, relevant, k)
        out[f"MRR@{k}"] = mrr_at_k(recommended, relevant, k)
    return out


def ranking_metrics(recommended: list[str], relevant: set[str], k: int = 10) -> dict[str, float]:
    return {
        f"HitRate@{k}": hit_rate(recommended, relevant, k),
        f"NDCG@{k}": ndcg_at_k(recommended, relevant, k),
        f"MRR@{k}": mrr_at_k(recommended, relevant, k),
    }
