"""MOVIELENS_ONLY_READOUT_RERANK — runtime-safe v2 belief+itemknn readout rerank within candidate pool."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.ask.movielens_candidate_recall import (
    _load_global_stats,
    _minmax,
    build_movielens_candidate_pool,
)
from src.ask.movielens_speed_optimization import movielens_speed_opt_cfg, movielens_speed_opt_enabled
from src.belief.belief_state import BeliefState
from src.embedding.index import EmbeddingIndex
from src.recommendation.scorer import belief_scores


def movielens_readout_rerank_enabled(cfg: dict[str, Any], dataset_name: str) -> bool:
    block = cfg.get("movielens_readout_rerank") or {}
    if not bool(block.get("enabled", False)):
        return False
    dom = str(block.get("domain_only", "movielens")).strip().lower()
    return dom == "movielens" and str(dataset_name).strip().lower() == "movielens"


def _genre_overlap_vec(
    *,
    n: int,
    item_ids: list[str],
    item_to_idx: dict[str, int],
    cat_map: dict[str, str],
    tag_map: dict[str, set[str]],
    observed: list[str],
) -> np.ndarray:
    hist_cats = Counter(cat_map.get(i, "") for i in observed if cat_map.get(i))
    hist_tags: set[str] = set()
    for i in observed:
        hist_tags |= tag_map.get(i, set())
    out = np.zeros(n, dtype=float)
    for iid in item_ids:
        idx = item_to_idx.get(iid)
        if idx is None:
            continue
        c = cat_map.get(iid, "")
        out[idx] = (1.0 if c in hist_cats else 0.0) + 0.25 * len(hist_tags & tag_map.get(iid, set()))
    return out


def compute_movielens_readout_feature_vectors(
    *,
    belief: BeliefState,
    index: EmbeddingIndex,
    processed_dir: Path,
    observed_history: list[str],
    seen_items: set[str],
    lambda_explore: float,
    pool: set[str],
) -> dict[str, np.ndarray]:
    """Runtime feature vectors (catalog-sized arrays) for items in candidate pool scope."""
    emb = index.embeddings.astype(float)
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    n = emb.shape[0]
    item_to_idx = index.item_id_to_index

    belief_raw = belief_scores(belief, emb, lambda_explore=lambda_explore)
    obs = [str(x) for x in observed_history if str(x) in item_to_idx]
    sem = np.zeros(n)
    if obs:
        mu = emb[[item_to_idx[i] for i in obs]].mean(axis=0)
        mu = mu / max(float(np.linalg.norm(mu)), 1e-12)
        sem = emb @ mu

    stats = _load_global_stats(processed_dir)
    iknn = np.zeros(n)
    pop_sqrt = stats["pop_sqrt"]
    co = stats["co_counts"]
    iti = stats["item_to_idx"]
    for iid in obs:
        if iid not in iti:
            continue
        a = iti[iid]
        da = pop_sqrt[a]
        for b, cnt in co.get(a, {}).items():
            iknn[b] += cnt / max(da * pop_sqrt[b], 1e-9)

    cat_map: dict[str, str] = {}
    tag_map: dict[str, set[str]] = {}
    for row in csv.DictReader((processed_dir / "items.csv").open(newline="", encoding="utf-8")):
        iid = str(row["item_id"])
        cat_map[iid] = row.get("category") or ""
        tag_map[iid] = set(t for t in (row.get("tags") or "").lower().split("|") if t)

    genre = _genre_overlap_vec(
        n=n,
        item_ids=index.item_ids,
        item_to_idx=item_to_idx,
        cat_map=cat_map,
        tag_map=tag_map,
        observed=obs,
    )
    pop = np.log1p(stats["pop_arr"])

    pool_rank = np.full(n, 1e9)
    pool_sorted = sorted(pool, key=lambda x: -iknn[item_to_idx[x]] if x in item_to_idx else -1.0)
    for r, iid in enumerate(pool_sorted):
        if iid in item_to_idx:
            pool_rank[item_to_idx[iid]] = r + 1

    hist_max = np.zeros(n)
    hist_mean = np.zeros(n)
    if obs:
        obs_idx = [item_to_idx[i] for i in obs]
        sims = emb @ emb[obs_idx].T
        hist_max = sims.max(axis=1)
        hist_mean = sims.mean(axis=1)

    return {
        "belief_score_raw": belief_raw,
        "belief_score": _minmax(belief_raw),
        "semantic_score": _minmax(sem),
        "itemknn_score_raw": iknn,
        "itemknn_score": _minmax(iknn),
        "genre_overlap_score": _minmax(genre),
        "popularity_score": _minmax(pop),
        "candidate_pool_rank_score": _minmax(1.0 / np.maximum(pool_rank, 1)),
        "history_max_similarity": _minmax(hist_max),
        "history_mean_similarity": _minmax(hist_mean),
    }


def apply_movielens_readout_rerank(
    *,
    belief: BeliefState,
    index: EmbeddingIndex,
    cfg: dict[str, Any],
    observed_history: list[str],
    seen_items: set[str],
    top_k: int,
    lambda_explore: float,
    baseline_recs: list[tuple[str, float]],
    session_id: str | None = None,
    session_speed_stats: dict[str, Any] | None = None,
) -> tuple[list[tuple[str, float]], dict[str, Any]]:
    """MOVIELENS_ONLY_READOUT_RERANK: rerank within candidate pool; no future target."""
    rr_cfg = dict(cfg.get("movielens_readout_rerank") or {})
    if bool(rr_cfg.get("use_future_target", False)):
        raise ValueError("MOVIELENS_ONLY_READOUT_RERANK: use_future_target must be false")

    processed_dir = Path(str((cfg.get("dataset") or {}).get("processed_dir", "")))
    mcr_cfg = dict(cfg.get("movielens_candidate_recall") or {})
    pool_size = int(rr_cfg.get("candidate_pool_size") or mcr_cfg.get("candidate_pool_size") or 500)

    # Build or reuse candidate pool (ItemKNN, no future target).
    pool_cfg = {
        **mcr_cfg,
        "enabled": True,
        "domain_only": "movielens",
        "retriever": "itemknn_cooccurrence",
        "candidate_pool_size": pool_size,
        "use_future_target": False,
    }
    speed_opt = movielens_speed_opt_cfg(cfg)
    pool, pool_audit = build_movielens_candidate_pool(
        cfg=pool_cfg,
        processed_dir=processed_dir,
        embeddings=index.embeddings,
        item_ids=list(index.item_ids),
        item_id_to_index=dict(index.item_id_to_index),
        observed_history=list(observed_history),
        seen_items=set(seen_items),
        session_id=session_id,
        session_pool_cache=bool(
            movielens_speed_opt_enabled(cfg) and speed_opt.get("itemknn_pool_session_cache", False)
        ),
        session_speed_stats=session_speed_stats,
    )

    feats = compute_movielens_readout_feature_vectors(
        belief=belief,
        index=index,
        processed_dir=processed_dir,
        observed_history=observed_history,
        seen_items=seen_items,
        lambda_explore=lambda_explore,
        pool=pool,
    )

    weights = dict(rr_cfg.get("weights") or {})
    w_ik = float(weights.get("itemknn", 0.5))
    w_bel = float(weights.get("belief", 0.4))
    w_gen = float(weights.get("genre", 0.1))
    w_pop = float(weights.get("popularity", 0.0))

    n = len(index.item_ids)
    combined = (
        w_ik * feats["itemknn_score"]
        + w_bel * feats["belief_score"]
        + w_gen * feats["genre_overlap_score"]
        + w_pop * feats["popularity_score"]
    )

    item_to_idx = index.item_id_to_index
    mask = np.full(n, -np.inf)
    for iid in pool:
        if iid in item_to_idx and str(iid) not in seen_items:
            mask[item_to_idx[iid]] = combined[item_to_idx[iid]]

    order = np.argsort(-mask)
    reranked: list[tuple[str, float]] = []
    for idx in order:
        if not np.isfinite(mask[idx]) or mask[idx] <= -1e30:
            continue
        iid = index.item_ids[int(idx)]
        reranked.append((iid, float(combined[int(idx)])))
        if len(reranked) >= top_k:
            break

    baseline_ids = [x for x, _ in baseline_recs[:top_k]]
    rerank_ids = [x for x, _ in reranked[:top_k]]
    miss_to_hit = sum(1 for iid in rerank_ids if iid not in baseline_ids)
    hit_to_miss = sum(1 for iid in baseline_ids if iid not in rerank_ids)

    audit = {
        "enabled": True,
        "domain_only": str(rr_cfg.get("domain_only", "movielens")),
        "variant": str(rr_cfg.get("variant", "v2_belief_itemknn_balanced")),
        "weights": {"itemknn": w_ik, "belief": w_bel, "genre": w_gen, "popularity": w_pop},
        "apply_scope": str(rr_cfg.get("apply_scope", "candidate_pool")),
        "candidate_pool_size": pool_size,
        "pool_allowed_count": len(pool),
        "use_future_target": False,
        "baseline_top10": baseline_ids,
        "reranked_top10": rerank_ids,
        "miss_to_hit_items": [i for i in rerank_ids if i not in baseline_ids],
        "hit_to_miss_items": [i for i in baseline_ids if i not in rerank_ids],
        "miss_to_hit_count": miss_to_hit,
        "hit_to_miss_count": hit_to_miss,
        "belief_uses_runtime_mu_sigma": True,
        "pool_audit": pool_audit,
    }
    return reranked, audit
