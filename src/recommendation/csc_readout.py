"""CS-C competing-item gain cap — readout-only scoring overlay (no belief update)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.ask.frontier_clustered_v32_rank_safe_mcq import _cosine
from src.embedding.index import EmbeddingIndex
from src.recommendation.scorer import belief_scores

MASK_DECAY: dict[str, float] = {"light": 0.5, "medium": 0.3}
DEFAULT_CAP_QUANTILE = 0.90
DEFAULT_MASK_STRENGTH = "light"


def csc_enabled(fc32_cfg: dict[str, Any]) -> bool:
    return bool(fc32_cfg.get("enable_csc_competing_gain_cap"))


def csc_config_from_fc32(fc32_cfg: dict[str, Any]) -> dict[str, Any]:
    strength = str(fc32_cfg.get("csc_cluster_mask_strength") or DEFAULT_MASK_STRENGTH).lower()
    if strength not in MASK_DECAY:
        strength = DEFAULT_MASK_STRENGTH
    return {
        "cap_quantile": float(fc32_cfg.get("csc_cap_quantile", DEFAULT_CAP_QUANTILE)),
        "cluster_mask_strength": strength,
    }


def cluster_centroid_from_summary(cl: dict[str, Any], index: EmbeddingIndex) -> np.ndarray | None:
    names = [str(x) for x in (cl.get("representative_names") or [])]
    vecs = []
    for nm in names:
        if nm in index.item_id_to_index:
            vecs.append(index.embeddings[int(index.item_id_to_index[nm])].astype(float))
    if vecs:
        return np.mean(np.stack(vecs, axis=0), axis=0)
    return None


def cluster_centroids_from_generation(
    gen: dict[str, Any],
    index: EmbeddingIndex,
    *,
    logged_hypotheses: list[dict[str, Any]] | None = None,
) -> dict[int, np.ndarray]:
    """Build cluster centroids from logged hypothesis item IDs (not artist-name previews)."""
    hyp_to_cluster: dict[str, int] = {}
    cluster_from_parsed: dict[int, int] = {}
    for h in gen.get("parsed_hypotheses") or []:
        if not isinstance(h, dict) or h.get("positive_cluster_id") is None:
            continue
        cid = int(h["positive_cluster_id"])
        hid = str(h.get("hypothesis_id") or h.get("option_id") or "")
        if hid:
            hyp_to_cluster[hid] = cid
        cluster_from_parsed[cid] = cid

    cluster_vecs: dict[int, list[np.ndarray]] = {cid: [] for cid in cluster_from_parsed}

    def _add_reps(cluster_id: int, reps: list[str]) -> None:
        for rid in reps:
            if rid in index.item_id_to_index:
                cluster_vecs.setdefault(cluster_id, []).append(
                    index.embeddings[int(index.item_id_to_index[rid])].astype(float)
                )

    for h in logged_hypotheses or []:
        if not isinstance(h, dict):
            continue
        hid = str(h.get("hypothesis_id") or "")
        cid = hyp_to_cluster.get(hid)
        if cid is None:
            pc = h.get("positive_cluster_id")
            if pc is not None:
                cid = int(pc)
        if cid is None:
            continue
        _add_reps(cid, [str(x) for x in (h.get("representative_items") or [])])

    if not any(cluster_vecs.values()):
        mapping = gen.get("option_to_cluster_mapping") or {}
        for opt_id, meta in mapping.items():
            if not isinstance(meta, dict):
                continue
            pc = meta.get("positive_cluster_id")
            if pc is None:
                continue
            cid = int(pc)
            for h in gen.get("parsed_hypotheses") or []:
                if isinstance(h, dict) and str(h.get("option_id")) == str(opt_id):
                    reps = [str(x) for x in (h.get("representative_items") or [])]
                    _add_reps(cid, reps)

    if not any(cluster_vecs.values()):
        for cl in gen.get("cluster_summaries") or []:
            if not isinstance(cl, dict) or cl.get("cluster_id") is None:
                continue
            cid = int(cl["cluster_id"])
            _add_reps(cid, [str(x) for x in (cl.get("representative_names") or [])])

    out: dict[int, np.ndarray] = {}
    for cid, vecs in cluster_vecs.items():
        if vecs:
            out[cid] = np.mean(np.stack(vecs, axis=0), axis=0)
    return out


def bound_cluster_for_option(gen: dict[str, Any], option_id: str) -> int | None:
    for h in gen.get("parsed_hypotheses") or []:
        if isinstance(h, dict) and str(h.get("option_id")) == option_id:
            pc = h.get("positive_cluster_id")
            if pc is not None:
                return int(pc)
    mapping = gen.get("option_to_cluster_mapping") or {}
    clusters = mapping.get(option_id)
    if isinstance(clusters, list) and clusters:
        return int(clusters[0])
    if isinstance(clusters, int):
        return clusters
    if isinstance(clusters, dict):
        pc = clusters.get("positive_cluster_id")
        if pc is not None:
            return int(pc)
    return None


def item_nearest_cluster(vec: np.ndarray, centroids: dict[int, np.ndarray]) -> int | None:
    best_id: int | None = None
    best_sim = -2.0
    for cid, cvec in centroids.items():
        sim = _cosine(vec, cvec)
        if sim > best_sim:
            best_sim = sim
            best_id = cid
    return best_id


def selected_vector_from_hyp_vectorized(hyp_vectorized: list[dict[str, Any]], selected_option_id: str) -> np.ndarray | None:
    for h in hyp_vectorized:
        if str(h.get("hypothesis_id")) == str(selected_option_id) or str(h.get("option_id")) == str(selected_option_id):
            vec = h.get("vector")
            if vec is not None:
                return np.asarray(vec, dtype=float)
    return None


def apply_csc_adjusted_scores(
    score_pre: np.ndarray,
    raw_delta: np.ndarray,
    index: EmbeddingIndex,
    seen: set[str],
    selected_vec: np.ndarray,
    selected_cluster: int | None,
    centroids: dict[int, np.ndarray],
    *,
    cap_quantile: float,
    mask_strength: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    decay = MASK_DECAY.get(mask_strength, MASK_DECAY[DEFAULT_MASK_STRENGTH])
    valid_idx = [i for i, iid in enumerate(index.item_ids) if str(iid) not in seen]
    sims = np.array([_cosine(selected_vec, index.embeddings[i]) for i in valid_idx]) if valid_idx else np.array([])
    tau_sim = float(np.quantile(sims, cap_quantile)) if len(sims) else 1.0

    pos = raw_delta[raw_delta > 0]
    gain_cap = float(np.quantile(pos, cap_quantile)) if len(pos) else np.inf

    adjusted = raw_delta.copy()
    num_capped = 0
    num_cluster_masked = 0
    capped_deltas: list[float] = []

    for i in valid_idx:
        if raw_delta[i] <= 0:
            continue
        gvec = index.embeddings[i]
        cid = item_nearest_cluster(gvec, centroids)
        sim_i = _cosine(selected_vec, gvec)
        if selected_cluster is not None and cid is not None and cid != selected_cluster and sim_i >= tau_sim:
            adjusted[i] *= decay
            num_cluster_masked += 1
        if adjusted[i] > gain_cap:
            adjusted[i] = gain_cap
            num_capped += 1
        if adjusted[i] != raw_delta[i]:
            capped_deltas.append(float(adjusted[i] - raw_delta[i]))

    cs_scores = score_pre + adjusted
    summary = {
        "tau_sim": round(tau_sim, 6),
        "gain_cap": round(gain_cap, 6) if np.isfinite(gain_cap) else None,
        "num_cluster_masked": num_cluster_masked,
        "num_gain_capped": num_capped,
        "num_items_capped": num_cluster_masked + num_capped,
        "capped_score_delta_mean": round(float(np.mean(capped_deltas)), 6) if capped_deltas else 0.0,
        "capped_score_delta_min": round(float(np.min(capped_deltas)), 6) if capped_deltas else 0.0,
        "capped_score_delta_max": round(float(np.max(capped_deltas)), 6) if capped_deltas else 0.0,
    }
    return cs_scores, summary


def top_gainers_from_delta(
    delta: np.ndarray,
    index: EmbeddingIndex,
    seen: set[str],
    *,
    k: int = 5,
) -> list[dict[str, Any]]:
    gains = []
    for i, iid in enumerate(index.item_ids):
        if str(iid) in seen:
            continue
        g = float(delta[i])
        if g > 0:
            gains.append({"item_id": str(iid), "score_delta": round(g, 6)})
    gains.sort(key=lambda x: -x["score_delta"])
    return gains[:k]


def competing_items_gain_more_count(
    delta: np.ndarray,
    index: EmbeddingIndex,
    seen: set[str],
    target_item_id: str | None,
) -> int | None:
    if not target_item_id or target_item_id not in index.item_id_to_index:
        return None
    ti = int(index.item_id_to_index[target_item_id])
    target_delta = float(delta[ti])
    count = 0
    for i, iid in enumerate(index.item_ids):
        if str(iid) in seen:
            continue
        if float(delta[i]) > target_delta:
            count += 1
    return count


@dataclass
class CscReadoutOverlay:
    """Fixed adjusted score vector for ranking until next v32 ask overwrites it."""

    adjusted_scores: np.ndarray
    telemetry: dict[str, Any] = field(default_factory=dict)


def try_build_csc_overlay(
    *,
    index: EmbeddingIndex,
    belief_before: Any,
    belief_after: Any,
    lambda_explore: float,
    seen: set[str],
    fc32_gen: dict[str, Any],
    selected_option_id: str,
    hyp_vectorized: list[dict[str, Any]],
    csc_cfg: dict[str, Any],
    audit_target_item_id: str | None = None,
    logged_hypotheses: list[dict[str, Any]] | None = None,
) -> tuple[CscReadoutOverlay | None, dict[str, Any]]:
    """Build CS-C overlay after Laplace update. Returns (overlay, telemetry dict)."""
    cap_q = float(csc_cfg.get("cap_quantile", DEFAULT_CAP_QUANTILE))
    mask = str(csc_cfg.get("cluster_mask_strength", DEFAULT_MASK_STRENGTH))

    tel: dict[str, Any] = {
        "csc_enabled": True,
        "csc_applied": False,
        "csc_config": {"cap_quantile": cap_q, "cluster_mask_strength": mask},
        "selected_cluster_id": None,
        "bound_cluster_id": None,
        "csc_fallback_reason": None,
        "num_items_capped": 0,
        "capped_score_delta_summary": {},
        "top_gainers_before_csc": [],
        "top_gainers_after_csc": [],
        "competing_items_gain_more_before": None,
        "competing_items_gain_more_after": None,
        "target_rank_before": None,
        "target_rank_after": None,
        "used_target_for_runtime_decision": False,
    }

    if str(selected_option_id) in {"", "none"}:
        tel["csc_fallback_reason"] = "no_substantive_option_selected"
        return None, tel

    sel_vec = selected_vector_from_hyp_vectorized(hyp_vectorized, selected_option_id)
    if sel_vec is None:
        tel["csc_fallback_reason"] = "missing_selected_option_vector"
        return None, tel

    bound_cluster = bound_cluster_for_option(fc32_gen, selected_option_id)
    centroids = cluster_centroids_from_generation(
        fc32_gen, index, logged_hypotheses=logged_hypotheses
    )
    if bound_cluster is None:
        tel["csc_fallback_reason"] = "missing_bound_cluster_id"
        return None, tel
    if not centroids:
        tel["csc_fallback_reason"] = "missing_cluster_centroids"
        return None, tel
    if bound_cluster not in centroids:
        tel["csc_fallback_reason"] = "bound_cluster_not_in_centroids"
        return None, tel

    emb = index.embeddings
    score_pre = belief_scores(belief_before, emb, lambda_explore)
    score_post = belief_scores(belief_after, emb, lambda_explore)
    raw_delta = score_post - score_pre

    tel["top_gainers_before_csc"] = top_gainers_from_delta(raw_delta, index, seen)
    tel["competing_items_gain_more_before"] = competing_items_gain_more_count(
        raw_delta, index, seen, audit_target_item_id
    )

    cs_scores, cap_summary = apply_csc_adjusted_scores(
        score_pre,
        raw_delta,
        index,
        seen,
        sel_vec,
        bound_cluster,
        centroids,
        cap_quantile=cap_q,
        mask_strength=mask,
    )
    cs_delta = cs_scores - score_pre
    tel["top_gainers_after_csc"] = top_gainers_from_delta(cs_delta, index, seen)
    tel["competing_items_gain_more_after"] = competing_items_gain_more_count(
        cs_delta, index, seen, audit_target_item_id
    )

    tel["csc_applied"] = True
    tel["selected_cluster_id"] = bound_cluster
    tel["bound_cluster_id"] = bound_cluster
    tel["num_items_capped"] = cap_summary.get("num_items_capped", 0)
    tel["capped_score_delta_summary"] = cap_summary

    if audit_target_item_id and audit_target_item_id in index.item_id_to_index:
        ti = int(index.item_id_to_index[audit_target_item_id])
        order_pre = np.argsort(-score_pre)
        order_post = np.argsort(-cs_scores)
        rank_before = rank_after = None
        r = 1
        for idx in order_pre:
            if str(index.item_ids[idx]) in seen:
                continue
            if idx == ti:
                rank_before = r
                break
            r += 1
        r = 1
        for idx in order_post:
            if str(index.item_ids[idx]) in seen:
                continue
            if idx == ti:
                rank_after = r
                break
            r += 1
        tel["target_rank_before"] = rank_before
        tel["target_rank_after"] = rank_after

    overlay = CscReadoutOverlay(adjusted_scores=cs_scores, telemetry=tel)
    return overlay, tel
