from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.ask.representative_retrieval_v2 import (
    apply_overlap_penalty,
    centroid_vectors_hash,
    mean_pairwise_jaccard_overlap,
    prefetch_pos_neg,
)
from src.embedding.index import EmbeddingIndex
from src.utils.math import l2_normalize

SUPPORTED_REPRESENTATIVE_RETRIEVAL_MODES = frozenset(
    {"legacy", "contrastive_v2", "representative_topk", "frontier_aware", "cluster_centroid"}
)


def _rep_embedding_diversity(vectors: np.ndarray) -> float:
    """1 - mean pairwise cosine similarity (higher ⇒ more spread reps)."""
    if vectors.shape[0] <= 1:
        return 1.0
    sim = vectors @ vectors.T
    n = sim.shape[0]
    off = [float(sim[i, j]) for i in range(n) for j in range(i + 1, n)]
    return float(1.0 - (sum(off) / len(off) if off else 0.0))


def _extract_prebound_rep_ids(hypothesis: dict[str, Any]) -> list[str]:
    """Explicit representative IDs from cluster-bound v2 (before BM25 overwrite)."""
    for key in ("representative_items", "representative_item_ids"):
        raw = hypothesis.get(key)
        if isinstance(raw, list) and raw:
            return [str(x) for x in raw if str(x)][:8]
    return []


def vectorize_from_prebound_representatives(
    hypothesis: dict[str, Any],
    index: EmbeddingIndex,
) -> dict[str, Any] | None:
    """Use parsed cluster-bound representative item IDs directly (skip BM25 re-retrieval)."""
    rep_ids = _extract_prebound_rep_ids(hypothesis)
    if not rep_ids:
        return None
    reps_ok = [r for r in rep_ids if str(r) in index.item_id_to_index]
    if not reps_ok:
        return None

    dim = index.embeddings.shape[1]
    vectors = index.vectors_for(reps_ok)
    if vectors.shape[0] == 0:
        return None
    avg = l2_normalize(np.mean(vectors, axis=0).reshape(1, -1))[0]
    div = _rep_embedding_diversity(l2_normalize(vectors))
    partial = len(reps_ok) < len(rep_ids)
    return {
        **hypothesis,
        "representative_items": reps_ok,
        "representative_items_scores": [],
        "vector": avg,
        "vector_norm": float(np.linalg.norm(avg)),
        "representative_vector_source": "cluster_bound_prebound_reps",
        "vector_binding_source": "parsed_representative_items",
        "representative_diversity": div,
        "centroid_ids": list(reps_ok),
        "centroid_vectors_hash": centroid_vectors_hash(reps_ok, float(np.linalg.norm(avg))),
        "parsed_representative_items": rep_ids,
        "runtime_representative_items": reps_ok,
        "reps_match": (not partial) and (set(rep_ids) == set(reps_ok)),
        "vector_binding_fallback": partial,
        "fallback_reason": "partial_reps_missing_from_index" if partial else None,
    }


def build_cluster_bound_vector_binding_audit(
    parsed_hypotheses: list[dict[str, Any]] | None,
    hyp_vectorized: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare parsed vs runtime representative binding for telemetry."""
    generic_sig = frozenset({"2", "6", "7", "8", "9"})
    parsed_by_id = {
        str(h.get("hypothesis_id") or h.get("option_id") or ""): h
        for h in (parsed_hypotheses or [])
        if isinstance(h, dict)
    }
    rows: list[dict[str, Any]] = []
    match_n = generic_n = tied_groups = 0
    opt_vectors: list[np.ndarray] = []

    for hv in hyp_vectorized:
        hid = str(hv.get("hypothesis_id") or "")
        ph = parsed_by_id.get(hid) or {}
        parsed_reps = _extract_prebound_rep_ids(ph) or _extract_prebound_rep_ids(hv)
        runtime_reps = [str(x) for x in (hv.get("representative_items") or [])]
        reps_match = parsed_reps == runtime_reps if parsed_reps else None
        if reps_match is True:
            match_n += 1
        generic_used = generic_sig.issubset(set(runtime_reps)) and len(runtime_reps) >= 5
        if generic_used:
            generic_n += 1
        vec = hv.get("vector")
        if vec is not None:
            opt_vectors.append(np.asarray(vec, dtype=float))
        rows.append(
            {
                "hypothesis_id": hid,
                "parsed_bound_cluster_id": ph.get("bound_cluster_id"),
                "parsed_representative_items": parsed_reps,
                "runtime_representative_items": runtime_reps,
                "reps_match": reps_match,
                "generic_reps_used": generic_used,
                "vector_binding_source": hv.get("vector_binding_source")
                or hv.get("representative_vector_source"),
                "vector_binding_fallback": bool(hv.get("vector_binding_fallback")),
                "fallback_reason": hv.get("fallback_reason"),
            }
        )

    h_pairwise_max = None
    if len(opt_vectors) >= 2:
        sims = []
        for i in range(len(opt_vectors)):
            for j in range(i + 1, len(opt_vectors)):
                sims.append(float(np.dot(opt_vectors[i], opt_vectors[j])))
        if sims:
            h_pairwise_max = round(max(sims), 6)
            if max(sims) - min(sims) < 1e-6:
                tied_groups = 1

    n = max(len(rows), 1)
    return {
        "parsed_runtime_reps_match_rate": round(match_n / n, 4),
        "generic_reps_fallback_count": generic_n,
        "h_options_all_tied": tied_groups > 0,
        "h_option_pairwise_sim_max": h_pairwise_max,
        "per_hypothesis": rows,
    }


def vectorize_hypothesis(
    hypothesis: dict[str, Any],
    index: EmbeddingIndex,
    representative_top_k: int = 5,
    item_texts: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Retrieve representative items for a hypothesis and average their embeddings.

    Implements: v_h = (1/K) * sum_{j=1..K} e_{i_j}
    where {i_j} are the items whose text best matches the feature_signature.

    Parameters
    ----------
    item_texts : dict mapping item_id -> concatenated text (artist_name + tags + ...).
                 Used for keyword fallback when BM25 is not available.
                 Without this, fallback is essentially random for numeric IDs.
    """
    feature_tokens = hypothesis.get("feature_signature", [])
    query = " ".join(str(t) for t in feature_tokens)

    matched: list[tuple[str, float]] = _keyword_match(
        query, index, top_k=representative_top_k, item_texts=item_texts
    )

    if not matched:
        dim = index.embeddings.shape[1]
        return {
            **hypothesis,
            "representative_items": [],
            "vector": np.zeros(dim, dtype=float),
            "vector_norm": 0.0,
            "representative_items_scores": [],
            "representative_vector_source": "keyword_empty",
            "representative_diversity": 0.0,
            "centroid_ids": [],
            "centroid_vectors_hash": "",
        }

    rep_ids = [item_id for item_id, _ in matched]
    vectors = index.vectors_for(rep_ids)
    avg = np.mean(vectors, axis=0)
    normalized = l2_normalize(avg.reshape(1, -1))[0]
    div = _rep_embedding_diversity(l2_normalize(vectors))
    return {
        **hypothesis,
        "representative_items": rep_ids,
        "vector": normalized,
        "vector_norm": float(np.linalg.norm(normalized)),
        "representative_items_scores": [float(s) for _, s in matched],
        "representative_vector_source": "keyword",
        "representative_diversity": div,
        "centroid_ids": list(rep_ids),
        "centroid_vectors_hash": centroid_vectors_hash(rep_ids, float(np.linalg.norm(normalized))),
    }


def vectorize_hypotheses(
    hypotheses: list[dict[str, Any]],
    index: EmbeddingIndex,
    representative_top_k: int = 5,
    bm25_index: Any = None,
    item_texts: dict[str, str] | None = None,
    *,
    timing_out: dict[str, float] | None = None,
    representative_retrieval: dict[str, Any] | None = None,
    diagnostics_out: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Vectorize all hypotheses.

    Priority (legacy):
    1. BM25 search on item text (preferred – retrieves items by semantic content)
    2. Keyword overlap against item_texts dict (fallback when BM25 not available)

    Representative retrieval modes:
    - legacy / representative_topk: BM25 or keyword top-k representatives.
    - contrastive_v2: positive-negative BM25 + optional overlap penalty.
    - frontier_aware / cluster_centroid: prefer prebound cluster representatives, with BM25 fallback.

    item_texts : dict mapping item_id -> full text string built from item metadata.

    timing_out
        When provided, accumulates seconds spent in BM25, dense vector gather,
        and keyword-fallback paths (profiling only).
    """
    representative_retrieval = representative_retrieval or {}
    mode = str(representative_retrieval.get("mode", "legacy"))
    if mode not in SUPPORTED_REPRESENTATIVE_RETRIEVAL_MODES:
        mode = "legacy"
    if (
        mode == "contrastive_v2"
        and bm25_index is not None
        and representative_retrieval.get("use_positive_negative_queries", True)
    ):
        return _vectorize_hypotheses_contrastive_v2(
            hypotheses,
            index,
            representative_top_k,
            bm25_index,
            item_texts,
            representative_retrieval,
            timing_out=timing_out,
            diagnostics_out=diagnostics_out,
        )

    bm25_sec = 0.0
    dense_sec = 0.0
    keyword_sec = 0.0
    out = []
    for h in hypotheses:
        prebound = vectorize_from_prebound_representatives(h, index)
        if prebound is not None:
            out.append(prebound)
            continue
        if bm25_index is not None:
            query = " ".join(str(t) for t in h.get("feature_signature", []))
            t0 = time.perf_counter()
            matched = bm25_index.search(query, top_k=representative_top_k)
            bm25_sec += time.perf_counter() - t0
            rep_ids = [item_id for item_id, _ in matched]
            scores = [float(s) for _, s in matched]
            if rep_ids:
                t1 = time.perf_counter()
                vectors = index.vectors_for(rep_ids)
                avg = l2_normalize(np.mean(vectors, axis=0).reshape(1, -1))[0]
                dense_sec += time.perf_counter() - t1
                div = _rep_embedding_diversity(l2_normalize(vectors))
                out.append(
                    {
                        **h,
                        "representative_items": rep_ids,
                        "representative_items_scores": scores,
                        "vector": avg,
                        "vector_norm": float(np.linalg.norm(avg)),
                        "representative_vector_source": "bm25_legacy",
                        "representative_diversity": div,
                        "centroid_ids": list(rep_ids),
                        "centroid_vectors_hash": centroid_vectors_hash(
                            rep_ids, float(np.linalg.norm(avg))
                        ),
                    }
                )
                continue
        t2 = time.perf_counter()
        out.append(vectorize_hypothesis(h, index, representative_top_k, item_texts=item_texts))
        keyword_sec += time.perf_counter() - t2

    if diagnostics_out is not None:
        diagnostics_out["option_overlap_score"] = mean_pairwise_jaccard_overlap(
            [set(x.get("representative_items", []) or []) for x in out]
        )

    if timing_out is not None:
        timing_out["bm25_lexical_sec"] = timing_out.get("bm25_lexical_sec", 0.0) + bm25_sec
        timing_out["hypothesis_dense_vectors_sec"] = (
            timing_out.get("hypothesis_dense_vectors_sec", 0.0) + dense_sec
        )
        timing_out["hypothesis_keyword_fallback_sec"] = (
            timing_out.get("hypothesis_keyword_fallback_sec", 0.0) + keyword_sec
        )
    return out


def _vectorize_hypotheses_contrastive_v2(
    hypotheses: list[dict[str, Any]],
    index: EmbeddingIndex,
    representative_top_k: int,
    bm25_index: Any,
    item_texts: dict[str, str] | None,
    rep_cfg: dict[str, Any],
    *,
    timing_out: dict[str, float] | None,
    diagnostics_out: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    gamma = float(rep_cfg.get("gamma_negative_bm25", 0.5))
    prefetch_k = int(rep_cfg.get("prefetch_k", 400))
    overlap_coef = float(rep_cfg.get("overlap_penalty_coef", 0.35))
    use_penalty = bool(rep_cfg.get("use_overlap_penalty", True))
    log_scores = bool(rep_cfg.get("log_scores", True))

    bm25_sec = 0.0
    dense_sec = 0.0
    keyword_sec = 0.0

    prefetched: list[list[tuple[str, float]]] = []
    for h in hypotheses:
        pos_q = str(
            h.get("positive_retrieval_query")
            or " ".join(str(t) for t in h.get("feature_signature", []))
        ).strip()
        neg_q = str(h.get("negative_retrieval_query") or "").strip()
        t0 = time.perf_counter()
        try:
            cand = prefetch_pos_neg(
                bm25_index,
                positive_query=pos_q,
                negative_query=neg_q,
                gamma=gamma,
                prefetch_k=prefetch_k,
            )
        except Exception:
            cand = []
        bm25_sec += time.perf_counter() - t0
        prefetched.append(cand)

    if use_penalty and len(prefetched) > 1:
        prefetched = apply_overlap_penalty(
            prefetched, rep_k=representative_top_k, coef=overlap_coef
        )

    out: list[dict[str, Any]] = []
    for h, cand in zip(hypotheses, prefetched):
        prebound = vectorize_from_prebound_representatives(h, index)
        if prebound is not None:
            out.append(prebound)
            continue
        picked = cand[:representative_top_k] if cand else []
        rep_ids = [item_id for item_id, _ in picked]
        scores = [float(s) for _, s in picked] if log_scores else []

        if not rep_ids:
            t2 = time.perf_counter()
            fb = vectorize_hypothesis(h, index, representative_top_k, item_texts=item_texts)
            keyword_sec += time.perf_counter() - t2
            fb["representative_vector_source"] = "contrastive_v2_fallback_keyword"
            out.append(fb)
            continue

        t1 = time.perf_counter()
        vectors = index.vectors_for(rep_ids)
        avg = l2_normalize(np.mean(vectors, axis=0).reshape(1, -1))[0]
        dense_sec += time.perf_counter() - t1
        div = _rep_embedding_diversity(l2_normalize(vectors))
        src = "bm25_pos_neg_overlap" if use_penalty else "bm25_pos_neg"
        row = {
            **h,
            "representative_items": rep_ids,
            "representative_items_scores": scores,
            "vector": avg,
            "vector_norm": float(np.linalg.norm(avg)),
            "representative_vector_source": src,
            "representative_diversity": div,
            "centroid_ids": list(rep_ids),
            "centroid_vectors_hash": centroid_vectors_hash(
                rep_ids, float(np.linalg.norm(avg))
            ),
        }
        out.append(row)

    if diagnostics_out is not None:
        diagnostics_out["option_overlap_score"] = mean_pairwise_jaccard_overlap(
            [set(x.get("representative_items", []) or []) for x in out]
        )

    if timing_out is not None:
        timing_out["bm25_lexical_sec"] = timing_out.get("bm25_lexical_sec", 0.0) + bm25_sec
        timing_out["hypothesis_dense_vectors_sec"] = (
            timing_out.get("hypothesis_dense_vectors_sec", 0.0) + dense_sec
        )
        timing_out["hypothesis_keyword_fallback_sec"] = (
            timing_out.get("hypothesis_keyword_fallback_sec", 0.0) + keyword_sec
        )
    return out


def _keyword_match(
    query: str,
    index: EmbeddingIndex,
    top_k: int,
    item_texts: dict[str, str] | None = None,
) -> list[tuple[str, float]]:
    """Keyword overlap fallback.

    Matches query tokens against item_texts (artist_name + tags + description).
    If item_texts is not provided, falls back to item_id string match (weak).
    """
    tokens = set(query.lower().split())
    scores = []
    for item_id in index.item_ids:
        if item_texts:
            text = item_texts.get(str(item_id), str(item_id))
        else:
            # Last-resort: match against item_id string (unreliable for numeric IDs)
            text = str(item_id).lower().replace("::", " ")
        item_tokens = set(text.lower().split())
        overlap = len(tokens & item_tokens)
        scores.append((item_id, float(overlap)))
    scores.sort(key=lambda x: (-x[1], str(x[0])))
    return scores[:top_k]
