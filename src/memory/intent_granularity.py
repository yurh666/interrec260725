from __future__ import annotations

import numpy as np

from src.utils.math import cosine, l2_normalize, weighted_average


def _weighted_dispersion(vectors: np.ndarray, weights: np.ndarray) -> float:
    if len(vectors) <= 1:
        return 0.0
    center = weighted_average(vectors, weights)
    diffs = vectors - center
    w = weights / max(float(np.sum(weights)), 1e-12)
    return float(np.sum(w * np.sum(diffs * diffs, axis=1)))


def _unit_covariance(vectors: np.ndarray, weights: np.ndarray, jitter: float) -> np.ndarray:
    dim = vectors.shape[1]
    if len(vectors) <= 1:
        return np.eye(dim, dtype=float) * jitter
    center = weighted_average(vectors, weights)
    diffs = vectors - center
    w = weights / max(float(np.sum(weights)), 1e-12)
    cov = np.zeros((dim, dim), dtype=float)
    for i in range(len(vectors)):
        cov += w[i] * np.outer(diffs[i], diffs[i])
    cov += np.eye(dim, dtype=float) * jitter
    return cov


def _unit_uncertainty(covariance: np.ndarray) -> float:
    try:
        eigvals = np.linalg.eigvalsh(covariance)
        return float(np.max(eigvals))
    except Exception:
        return float(np.trace(covariance))


def split_unit_if_needed(
    item_ids: list[str],
    vectors: np.ndarray,
    weights: np.ndarray,
    *,
    split_dispersion_threshold: float,
    max_granularity_level: int,
    granularity_level: int,
    jitter: float,
    next_unit_id_fn,
) -> list[tuple[list[str], np.ndarray, np.ndarray, int]]:
    if len(item_ids) <= 1 or granularity_level >= max_granularity_level:
        return [(item_ids, vectors, weights, granularity_level)]

    dispersion = _weighted_dispersion(vectors, weights)
    if dispersion < split_dispersion_threshold:
        return [(item_ids, vectors, weights, granularity_level)]

    # Two-way split along the principal axis of weighted covariance.
    center = weighted_average(vectors, weights)
    cov = _unit_covariance(vectors, weights, jitter)
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, int(np.argmax(eigvals))]
    except Exception:
        axis = vectors[0] - center

    projections = vectors @ axis
    median = float(np.median(projections))
    left_mask = projections <= median
    right_mask = ~left_mask
    if not left_mask.any() or not right_mask.any():
        return [(item_ids, vectors, weights, granularity_level)]

    out: list[tuple[list[str], np.ndarray, np.ndarray, int]] = []
    for mask in (left_mask, right_mask):
        sub_ids = [item_ids[i] for i in range(len(item_ids)) if mask[i]]
        sub_vecs = vectors[mask]
        sub_w = weights[mask]
        out.extend(
            split_unit_if_needed(
                sub_ids,
                sub_vecs,
                sub_w,
                split_dispersion_threshold=split_dispersion_threshold,
                max_granularity_level=max_granularity_level,
                granularity_level=granularity_level + 1,
                jitter=jitter,
                next_unit_id_fn=next_unit_id_fn,
            )
        )
    return out


def merge_near_duplicate_units(
    clusters: list[tuple[list[str], np.ndarray, np.ndarray, int]],
    *,
    merge_cosine_threshold: float,
    min_support_weight: float,
) -> list[tuple[list[str], np.ndarray, np.ndarray, int]]:
    if len(clusters) <= 1:
        return clusters

    merged: list[tuple[list[str], np.ndarray, np.ndarray, int]] = []
    used = [False] * len(clusters)

    centroids = []
    for _, vecs, w, _ in clusters:
        c = weighted_average(vecs, w)
        centroids.append(l2_normalize(c.reshape(1, -1))[0])

    for i in range(len(clusters)):
        if used[i]:
            continue
        ids_i, vecs_i, w_i, level_i = clusters[i]
        cur_ids = list(ids_i)
        cur_vecs = vecs_i
        cur_w = w_i
        cur_level = level_i
        used[i] = True

        for j in range(i + 1, len(clusters)):
            if used[j]:
                continue
            sim = cosine(centroids[i], centroids[j])
            w_j = float(np.sum(clusters[j][2]))
            if sim >= merge_cosine_threshold and min(float(np.sum(cur_w)), w_j) >= min_support_weight:
                ids_j, vecs_j, w_j_arr, level_j = clusters[j]
                cur_ids.extend(ids_j)
                cur_vecs = np.vstack([cur_vecs, vecs_j])
                cur_w = np.concatenate([cur_w, w_j_arr])
                cur_level = min(cur_level, level_j)
                used[j] = True

        merged.append((cur_ids, cur_vecs, cur_w, cur_level))

    return merged
