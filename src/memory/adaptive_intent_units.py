from __future__ import annotations

from typing import Any

import numpy as np

from src.belief.belief_state import BeliefState
from src.embedding.index import EmbeddingIndex
from src.memory.intent_granularity import merge_near_duplicate_units, split_unit_if_needed
from src.memory.intent_memory_state import IntentMemoryState, IntentMemoryUnit
from src.utils.math import l2_normalize, weighted_average


def _normalize_weights(raw_weights: list[float] | None, n: int) -> np.ndarray:
    if not raw_weights:
        return np.ones(n, dtype=float)
    w = np.asarray(raw_weights[:n], dtype=float)
    w = np.maximum(w, 0.0)
    if float(np.sum(w)) <= 0:
        return np.ones(n, dtype=float)
    return w


def initialize_adaptive_intent_memory(
    observed_history: list[str],
    index: EmbeddingIndex,
    weights: list[float] | None = None,
    *,
    split_dispersion_threshold: float = 0.08,
    merge_cosine_threshold: float = 0.92,
    min_support_weight: float = 0.5,
    max_granularity_level: int = 3,
    max_units: int | None = None,
    jitter: float = 1e-6,
    sigma0: float = 1.0,
) -> IntentMemoryState:
    """Build adaptive-granularity intent memory units from observed history only."""

    item_ids = [str(x) for x in observed_history]
    if not item_ids:
        dim = index.embeddings.shape[1]
        unit = IntentMemoryUnit(
            unit_id="unit_0",
            centroid=np.zeros(dim, dtype=float),
            covariance=np.eye(dim, dtype=float) * (sigma0 ** 2),
            support_items=[],
            support_weight=0.0,
            granularity_level=0,
            uncertainty=float(sigma0 ** 2),
        )
        return IntentMemoryState(units=[unit], dim=dim, metadata={"init_mode": "adaptive_empty"})

    vectors = index.vectors_for(item_ids)
    w = _normalize_weights(weights, len(vectors))
    dim = vectors.shape[1]

    counter = {"n": 0}

    def next_unit_id() -> str:
        uid = f"unit_{counter['n']}"
        counter["n"] += 1
        return uid

    clusters = split_unit_if_needed(
        item_ids,
        vectors,
        w,
        split_dispersion_threshold=split_dispersion_threshold,
        max_granularity_level=max_granularity_level,
        granularity_level=0,
        jitter=jitter,
        next_unit_id_fn=next_unit_id,
    )
    clusters = merge_near_duplicate_units(
        clusters,
        merge_cosine_threshold=merge_cosine_threshold,
        min_support_weight=min_support_weight,
    )
    if max_units is not None and max_units > 0 and len(clusters) > max_units:
        clusters = _coarsen_clusters_to_max_units(clusters, max_units=max_units)

    units: list[IntentMemoryUnit] = []
    for ids, vecs, unit_w, level in clusters:
        centroid = weighted_average(vecs, unit_w)
        centroid = l2_normalize(centroid.reshape(1, -1))[0]
        cov = np.cov(vecs.T, aweights=unit_w) if len(vecs) > 1 else np.eye(dim) * jitter
        if cov.ndim == 0:
            cov = np.eye(dim, dtype=float) * float(cov)
        cov = cov + np.eye(dim, dtype=float) * jitter
        support_weight = float(np.sum(unit_w))
        uncertainty = float(np.trace(cov))
        units.append(
            IntentMemoryUnit(
                unit_id=next_unit_id(),
                centroid=centroid,
                covariance=cov,
                support_items=list(ids),
                support_weight=support_weight,
                granularity_level=level,
                uncertainty=uncertainty,
            )
        )

    if not units:
        centroid = l2_normalize(weighted_average(vectors, w).reshape(1, -1))[0]
        units = [
            IntentMemoryUnit(
                unit_id="unit_0",
                centroid=centroid,
                covariance=np.eye(dim, dtype=float) * (sigma0 ** 2),
                support_items=item_ids,
                support_weight=float(np.sum(w)),
                granularity_level=0,
                uncertainty=float(sigma0 ** 2),
            )
        ]

    return IntentMemoryState(
        units=units,
        dim=dim,
        metadata={
            "init_mode": "adaptive",
            "split_dispersion_threshold": split_dispersion_threshold,
            "merge_cosine_threshold": merge_cosine_threshold,
            "max_units": max_units,
            "history_item_count": len(item_ids),
        },
    )


def _coarsen_clusters_to_max_units(
    clusters: list[tuple[list[str], np.ndarray, np.ndarray, int]],
    *,
    max_units: int,
) -> list[tuple[list[str], np.ndarray, np.ndarray, int]]:
    """Merge low-support clusters into the nearest retained centroid."""

    if len(clusters) <= max_units:
        return clusters
    supports = [float(np.sum(c[2])) for c in clusters]
    keep_indices = sorted(range(len(clusters)), key=lambda i: supports[i], reverse=True)[:max_units]
    keep_set = set(keep_indices)
    kept = [
        (
            list(clusters[i][0]),
            np.asarray(clusters[i][1], dtype=float).copy(),
            np.asarray(clusters[i][2], dtype=float).copy(),
            int(clusters[i][3]),
        )
        for i in keep_indices
    ]

    def centroid(vecs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return l2_normalize(weighted_average(vecs, weights).reshape(1, -1))[0]

    for i, (ids, vecs, weights, level) in enumerate(clusters):
        if i in keep_set:
            continue
        src_centroid = centroid(vecs, weights)
        dst_idx = max(
            range(len(kept)),
            key=lambda j: float(np.dot(src_centroid, centroid(kept[j][1], kept[j][2]))),
        )
        dst_ids, dst_vecs, dst_weights, dst_level = kept[dst_idx]
        kept[dst_idx] = (
            dst_ids + list(ids),
            np.concatenate([dst_vecs, np.asarray(vecs, dtype=float)], axis=0),
            np.concatenate([dst_weights, np.asarray(weights, dtype=float)], axis=0),
            min(dst_level, int(level)),
        )
    return kept


def adaptive_memory_to_belief(
    memory_state: IntentMemoryState,
    *,
    sigma0: float = 1.0,
    jitter: float = 1e-6,
) -> BeliefState:
    """Convert intent memory units into a Gaussian belief (mixture moment matching)."""

    if not memory_state.units:
        dim = memory_state.dim
        return BeliefState(
            mu=np.zeros(dim, dtype=float),
            Sigma=np.eye(dim, dtype=float) * (sigma0 ** 2),
            jitter=jitter,
        )

    weights = np.array([max(u.support_weight, 1e-8) for u in memory_state.units], dtype=float)
    weights = weights / float(np.sum(weights))
    centroids = np.stack([u.centroid for u in memory_state.units], axis=0)
    mu = weighted_average(centroids, weights)
    mu = l2_normalize(mu.reshape(1, -1))[0]

    dim = memory_state.dim
    Sigma = np.zeros((dim, dim), dtype=float)
    for w, unit in zip(weights, memory_state.units):
        diff = unit.centroid - mu
        Sigma += w * (unit.covariance + np.outer(diff, diff))
    Sigma += np.eye(dim, dtype=float) * max(jitter, (sigma0 ** 2) * 0.01)

    target_trace = float(dim) * (sigma0 ** 2)
    cur_trace = float(np.trace(Sigma))
    if cur_trace > 0 and abs(cur_trace - target_trace) / max(target_trace, 1e-12) > 0.05:
        Sigma = Sigma * (target_trace / cur_trace)

    iso = np.eye(dim, dtype=float) * (sigma0 ** 2)
    ref_entropy = 0.5 * (dim * np.log(2 * np.pi * np.e) + dim * np.log(sigma0 ** 2))
    for _ in range(24):
        trial = BeliefState(mu=mu, Sigma=Sigma, jitter=jitter)
        if trial.entropy() >= ref_entropy * 0.999:
            break
        Sigma = 0.80 * Sigma + 0.20 * iso

    belief = BeliefState(mu=mu, Sigma=Sigma, jitter=jitter)
    belief.ensure_psd(force=True)
    return belief


def memory_units_for_logging(memory_state: IntentMemoryState) -> dict[str, Any]:
    return {
        **memory_state.summary(),
        "units": [u.to_dict() for u in memory_state.units],
    }
