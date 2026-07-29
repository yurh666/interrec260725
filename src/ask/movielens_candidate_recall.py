"""MOVIELENS_ONLY_CANDIDATE_RECALL — ItemKNN/hybrid candidate pools for runtime frontier (MovieLens-only)."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_CO_CACHE: dict[str, dict[str, Any]] = {}
_SESSION_POOL_CACHE: dict[str, tuple[set[str], dict[str, Any]]] = {}


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 1]


def _minmax(scores: np.ndarray) -> np.ndarray:
    finite = np.isfinite(scores)
    out = np.zeros_like(scores, dtype=float)
    if not finite.any():
        return out
    lo = float(np.min(scores[finite]))
    hi = float(np.max(scores[finite]))
    if hi <= lo:
        return out
    out[finite] = (scores[finite] - lo) / (hi - lo)
    return out


def _load_global_stats(processed_dir: Path) -> dict[str, Any]:
    key = str(processed_dir.resolve())
    if key in _CO_CACHE:
        return _CO_CACHE[key]

    with (processed_dir / "item_id_to_index.json").open(encoding="utf-8") as f:
        item_to_idx = {str(k): int(v) for k, v in json.load(f).items()}
    n_items = len(item_to_idx)
    item_categories = [""] * n_items
    item_tag_token_sets: list[set[str]] = [set() for _ in range(n_items)]
    cat_to_indices: dict[str, list[int]] = defaultdict(list)
    term_to_indices: dict[str, list[int]] = defaultdict(list)

    with (processed_dir / "items.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            iid = str(row.get("item_id", ""))
            if iid not in item_to_idx:
                continue
            idx = item_to_idx[iid]
            cat = row.get("category", "") or ""
            toks = set(_tokenize(row.get("tags", "")) + _tokenize(cat))
            item_categories[idx] = cat
            item_tag_token_sets[idx] = toks
            if cat:
                cat_to_indices[cat].append(idx)
            for t in toks:
                term_to_indices[t].append(idx)

    sessions = json.loads((processed_dir / "sessions.json").read_text(encoding="utf-8"))
    train_lists: list[list[str]] = []
    pop_counts: Counter[str] = Counter()
    for s in sessions:
        seen = [str(x) for x in s.get("observed_history", []) if str(x) in item_to_idx]
        train = [str(x) for x in s.get("future_train", []) if str(x) in item_to_idx]
        vals = list(dict.fromkeys(seen + train))
        if vals:
            train_lists.append(vals)
            pop_counts.update(vals)

    pop_arr = np.zeros(n_items, dtype=float)
    for iid, c in pop_counts.items():
        pop_arr[item_to_idx[iid]] = float(c)
    pop_sqrt = np.sqrt(np.maximum(pop_arr, 1.0))

    co_counts: dict[int, Counter[int]] = defaultdict(Counter)
    for vals in train_lists:
        idxs = [item_to_idx[i] for i in vals]
        if len(idxs) > 120:
            idxs = idxs[:120]
        uniq = list(dict.fromkeys(idxs))
        for i, a in enumerate(uniq):
            for b in uniq[i + 1 :]:
                co_counts[a][b] += 1
                co_counts[b][a] += 1

    out = {
        "item_to_idx": item_to_idx,
        "n_items": n_items,
        "pop_arr": pop_arr,
        "pop_sqrt": pop_sqrt,
        "co_counts": co_counts,
        "item_categories": item_categories,
        "item_tag_token_sets": item_tag_token_sets,
        "cat_to_indices": cat_to_indices,
        "term_to_indices": term_to_indices,
    }
    _CO_CACHE[key] = out
    return out


def movielens_candidate_recall_enabled(cfg: dict[str, Any], dataset_name: str) -> bool:
    if not bool(cfg.get("enabled", False)):
        return False
    dom = str(cfg.get("domain_only", "movielens")).strip().lower()
    return dom == "movielens" and str(dataset_name).strip().lower() == "movielens"


def clear_movielens_session_pool_cache() -> None:
    _SESSION_POOL_CACHE.clear()


def _session_pool_cache_key(
    *,
    session_id: str,
    cfg: dict[str, Any],
    observed_history: list[str],
) -> str:
    hist_key = ",".join(sorted(str(x) for x in observed_history[:40]))
    return "|".join(
        [
            str(session_id),
            str(cfg.get("domain_only", "movielens")),
            str(cfg.get("retriever", "itemknn_cooccurrence")),
            str(cfg.get("candidate_pool_size", 500)),
            hist_key,
        ]
    )


def build_movielens_candidate_pool(
    *,
    cfg: dict[str, Any],
    processed_dir: Path,
    embeddings: np.ndarray,
    item_ids: list[str],
    item_id_to_index: dict[str, int],
    observed_history: list[str],
    seen_items: set[str],
    session_id: str | None = None,
    session_pool_cache: bool = False,
    session_speed_stats: dict[str, Any] | None = None,
) -> tuple[set[str], dict[str, Any]]:
    if session_pool_cache and session_id:
        key = _session_pool_cache_key(
            session_id=session_id, cfg=cfg, observed_history=observed_history
        )
        if key in _SESSION_POOL_CACHE:
            pool, audit = _SESSION_POOL_CACHE[key]
            out_audit = {**audit, "session_cache_hit": True, "session_cache_miss": False}
            if session_speed_stats is not None:
                session_speed_stats["itemknn_pool_cache_hits"] = int(
                    session_speed_stats.get("itemknn_pool_cache_hits", 0)
                ) + 1
            return set(pool), out_audit
    if bool(cfg.get("use_future_target", False)):
        raise ValueError("MOVIELENS_ONLY_CANDIDATE_RECALL: use_future_target must be false")

    stats = _load_global_stats(processed_dir)
    n_items = stats["n_items"]
    pop_arr = stats["pop_arr"]
    pop_sqrt = stats["pop_sqrt"]
    co_counts = stats["co_counts"]
    item_categories = stats["item_categories"]
    item_tag_token_sets = stats["item_tag_token_sets"]
    cat_to_indices = stats["cat_to_indices"]
    term_to_indices = stats["term_to_indices"]
    item_to_idx = stats["item_to_idx"]

    emb = embeddings.astype(np.float64)
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)

    obs_ids = [str(x) for x in observed_history if str(x) in item_to_idx]
    obs_idxs = [item_to_idx[i] for i in obs_ids]
    exclude = (
        set(str(x) for x in observed_history)
        if session_pool_cache
        else set(str(x) for x in seen_items)
    )

    def semantic_scores() -> np.ndarray:
        if not obs_idxs:
            return np.zeros(n_items, dtype=float)
        mu = emb[obs_idxs].mean(axis=0)
        mu = mu / max(float(np.linalg.norm(mu)), 1e-12)
        return emb @ mu

    def itemknn_scores() -> np.ndarray:
        scores = np.zeros(n_items, dtype=float)
        for iid in obs_ids:
            a = item_to_idx[iid]
            denom_a = pop_sqrt[a]
            for b, cnt in co_counts.get(a, {}).items():
                scores[b] += cnt / max(denom_a * pop_sqrt[b], 1e-9)
        return scores

    def category_scores(sem: np.ndarray) -> np.ndarray:
        cats = Counter()
        tag_terms = Counter()
        for iid in obs_ids:
            idx = item_to_idx[iid]
            cat = item_categories[idx]
            if cat:
                cats[cat] += 1
            for t in item_tag_token_sets[idx]:
                tag_terms[t] += 1
        scores = 0.05 * sem
        for cat, w in cats.items():
            if cat in cat_to_indices:
                scores[cat_to_indices[cat]] += w
        for term, w in tag_terms.items():
            if term in term_to_indices:
                scores[term_to_indices[term]] += 0.25 * w
        return scores

    retriever = str(cfg.get("retriever", "itemknn_cooccurrence")).strip().lower()
    weights = dict(cfg.get("weights") or {})
    sem = semantic_scores()
    iknn = itemknn_scores()
    cat = category_scores(sem)
    pop = np.log1p(pop_arr)

    if retriever in ("hybrid", "hybrid_semantic_itemknn_category_popularity"):
        w_sem = float(weights.get("semantic", 0.2))
        w_iknn = float(weights.get("itemknn", 0.6))
        w_cat = float(weights.get("category", 0.1))
        w_pop = float(weights.get("popularity", 0.1))
        combined = (
            w_sem * _minmax(sem)
            + w_iknn * _minmax(iknn)
            + w_cat * _minmax(cat)
            + w_pop * _minmax(pop)
        )
        score_vec = combined
        retriever_label = "hybrid"
    elif retriever in ("itemknn_cooccurrence", "itemknn"):
        score_vec = iknn
        retriever_label = "itemknn_cooccurrence"
    elif retriever in ("popularity", "pop"):
        score_vec = pop
        retriever_label = "popularity"
    else:
        score_vec = sem
        retriever_label = "current_semantic_ranker"

    pool_size = max(50, int(cfg.get("candidate_pool_size", 500)))
    for iid in exclude:
        if iid in item_to_idx:
            score_vec[item_to_idx[iid]] = -np.inf

    order = np.argsort(-score_vec)
    allowed: set[str] = set()
    idx_to_item = {v: k for k, v in item_to_idx.items()}
    for idx in order:
        if not np.isfinite(score_vec[idx]) or score_vec[idx] <= -1e30:
            continue
        iid = idx_to_item.get(int(idx))
        if not iid or iid in exclude:
            continue
        allowed.add(iid)
        if len(allowed) >= pool_size:
            break

    audit = {
        "enabled": True,
        "domain_only": str(cfg.get("domain_only", "movielens")),
        "retriever": retriever_label,
        "candidate_pool_size": pool_size,
        "allowed_count": len(allowed),
        "use_future_target": False,
        "observed_history_len": len(obs_ids),
        "fallback_to_semantic": bool(cfg.get("fallback_to_semantic", True)),
        "session_cache_hit": False,
        "session_cache_miss": bool(session_pool_cache and session_id),
    }
    if session_pool_cache and session_id:
        key = _session_pool_cache_key(
            session_id=session_id, cfg=cfg, observed_history=observed_history
        )
        _SESSION_POOL_CACHE[key] = (set(allowed), dict(audit))
        if session_speed_stats is not None:
            session_speed_stats["itemknn_pool_cache_misses"] = int(
                session_speed_stats.get("itemknn_pool_cache_misses", 0)
            ) + 1
    return allowed, audit
