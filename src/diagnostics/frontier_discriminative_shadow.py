"""Frontier-discriminative v3.1 shadow proxies (OBSERVATIONAL ONLY).

These scores are appended to ``full_log`` when ``logging.frontier_discriminative_shadow`` is true.
Must not import or use oracle labels / future_* slices / relevance / ranking metrics modules.
Inputs: BeliefState, EmbeddingIndex embeddings, unseen mask, hypotheses, selection ids, reco scores only.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.belief.belief_state import BeliefState
from src.embedding.index import EmbeddingIndex
from src.recommendation.scorer import belief_scores

_EPS = 1e-9
_RANK_DAMAGE_ENTROPY_COEFF = 1.0


def _jf(x: float | None) -> float | None:
    """Serialize-friendly float (NaN / inf -> None for JSON-safe logs)."""

    if x is None:
        return None
    if isinstance(x, (float, np.floating)):
        xf = float(x)
        if not math.isfinite(xf):
            return None
        return xf
    return float(x)


def _normalize_unit(vec: np.ndarray) -> tuple[np.ndarray, bool]:
    n = float(np.linalg.norm(vec))
    if n < _EPS:
        return vec.astype(np.float64, copy=False), False
    return (vec.astype(np.float64, copy=False) / n).reshape(-1), True


def _top10_softmax_entropy(rec_scores_topk: Sequence[float]) -> float:
    if len(rec_scores_topk) < 2:
        return 0.0
    x = np.asarray(rec_scores_topk[:10], dtype=np.float64)
    x = x - float(np.max(x))
    p = np.exp(x)
    s = float(np.sum(p)) + 1e-15
    p = p / s
    return float(-np.sum(p * np.log(p + 1e-15)))


def _selected_hypothesis_unit_vector(
    hyp_vectorized: Sequence[Mapping[str, Any]],
    selected_hypothesis_ids: Sequence[str],
    missing_flags: list[str],
) -> np.ndarray | None:
    ids = [str(x) for x in selected_hypothesis_ids if str(x)]
    by_id = {str(h["hypothesis_id"]): h for h in hyp_vectorized if "hypothesis_id" in h}
    if not ids:
        missing_flags.append("no_selected_hypothesis_ids")
        return None

    mats: list[np.ndarray] = []
    for hid in ids:
        h = by_id.get(str(hid))
        if not h:
            missing_flags.append("selected_hypothesis_id_not_in_hyp_vectorized")
            return None
        v = np.asarray(h.get("vector"), dtype=np.float64).reshape(-1)
        vn, ok = _normalize_unit(v)
        if not ok:
            missing_flags.append("zero_norm_hypothesis_vector")
            return None
        mats.append(vn)

    if not mats:
        return None

    pooled = np.mean(np.stack(mats, axis=0), axis=0)
    pooled_n, ok = _normalize_unit(pooled)
    if not ok:
        missing_flags.append("zero_norm_pooled_hypothesis_vector")
        return None
    return pooled_n


def _gather_frontier(
    belief: BeliefState,
    index: EmbeddingIndex,
    *,
    seen_items: set[str],
    frontier_k: int,
    lambda_explore: float,
    missing_flags: list[str],
) -> tuple[list[str], np.ndarray, np.ndarray] | tuple[None, None, None]:
    scores = belief_scores(belief, index.embeddings, lambda_explore)
    order = np.argsort(-scores)
    frontier_ids: list[str] = []
    frontier_idx: list[int] = []
    for ix in order:
        iix = int(ix)
        iid = str(index.item_ids[iix])
        if iid in seen_items:
            continue
        frontier_ids.append(iid)
        frontier_idx.append(iix)
        if len(frontier_ids) >= frontier_k:
            break

    if len(frontier_ids) == 0:
        missing_flags.append("empty_frontier")
        return None, None, None

    ef = np.stack([index.embeddings[i] for i in frontier_idx], axis=0)
    s_frontier = scores[frontier_idx]
    return frontier_ids, ef.astype(np.float64, copy=False), s_frontier


def frontier_topk_hash(frontier_ids: Iterable[str]) -> str:
    body = ",".join(sorted(str(x) for x in frontier_ids))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:24]


def compute_frontier_discriminative_shadow_bundle(
    belief: BeliefState,
    index: EmbeddingIndex,
    *,
    seen_items: set[str],
    hyp_vectorized: Sequence[Mapping[str, Any]],
    selected_hypothesis_ids: Sequence[str],
    recommendation_scores_topk: Sequence[float],
    lambda_explore: float,
    frontier_k: int = 50,
) -> dict[str, Any]:
    """Pure observation bundle for one turn — never consults oracle / labels / offline metrics."""

    t0 = time.perf_counter()
    missing_flags: list[str] = []

    frontier_ids_tuple = _gather_frontier(
        belief, index, seen_items=seen_items, frontier_k=frontier_k, lambda_explore=lambda_explore, missing_flags=missing_flags
    )
    frontier_ids, ef, frontier_belief_scores = frontier_ids_tuple  # type: ignore[misc]

    v_h = _selected_hypothesis_unit_vector(hyp_vectorized, selected_hypothesis_ids, missing_flags)

    defaults: dict[str, Any] = {
        "enabled": True,
        "frontier_k": int(frontier_k),
        "frontier_topk_hash": None,
        "P1_mean_F50_dot": None,
        "P1_cos_mu": None,
        "D3_uncertainty_weighted_frontier_contrast": None,
        "D4_anti_collapse_score": None,
        "D5_boundary_challenge_score": None,
        "boundary_challenge_score": None,
        "frontier_discrimination_proxy": None,
        "frontier_score_mean": None,
        "frontier_score_std": None,
        "rank_damage_risk_proxy": None,
        "missing_flags": sorted(set(missing_flags)),
        "compute_ms": round((time.perf_counter() - t0) * 1000.0, 6),
    }

    if frontier_ids is None or ef is None or frontier_belief_scores is None:
        defaults["missing_flags"] = sorted(set(missing_flags))
        return defaults

    defaults["frontier_topk_hash"] = frontier_topk_hash(frontier_ids)

    if v_h is None:
        defaults["missing_flags"] = sorted(set(missing_flags))
        defaults["compute_ms"] = round((time.perf_counter() - t0) * 1000.0, 6)
        return defaults

    d = ef @ v_h.astype(np.float64)
    mu_d = float(np.mean(d))
    sig_d = float(np.std(d, ddof=0)) if d.size >= 2 else 0.0

    defaults["P1_mean_F50_dot"] = _jf(mu_d)
    defaults["frontier_score_mean"] = _jf(mu_d)
    defaults["frontier_score_std"] = _jf(sig_d)

    defaults["P1_cos_mu"] = None
    mu_b = belief.mu.astype(np.float64).reshape(-1)
    mun, ok_mu = _normalize_unit(mu_b)
    if ok_mu:
        defaults["P1_cos_mu"] = _jf(float(np.dot(v_h, mun)))

    defaults["D4_anti_collapse_score"] = _jf(sig_d / (mu_d + _EPS))

    # D5: frontier belief_score rank partitions — top belief-scored 10 vs next 40 (or remainder)
    d5_val = float("nan")
    if len(frontier_ids) >= 50:
        ord_ff = np.argsort(-np.asarray(frontier_belief_scores, dtype=np.float64))
        idx_top = ord_ff[:10]
        idx_deep = ord_ff[10:50]
        top_e = np.mean(ef[idx_top], axis=0)
        deep_e = np.mean(ef[idx_deep], axis=0)
        d5_val = float(np.dot(v_h, top_e) - np.dot(v_h, deep_e))
    elif len(frontier_ids) >= 20:
        ord_ff = np.argsort(-np.asarray(frontier_belief_scores, dtype=np.float64))
        idx_top = ord_ff[:10]
        idx_deep = ord_ff[10:]
        if idx_deep.size == 0:
            missing_flags.append("boundary_challenge_short_frontier_deep_empty")
        else:
            top_e = np.mean(ef[idx_top], axis=0)
            deep_e = np.mean(ef[idx_deep], axis=0)
            d5_val = float(np.dot(v_h, top_e) - np.dot(v_h, deep_e))
    else:
        missing_flags.append("boundary_challenge_short_frontier")

    if len(frontier_ids) >= 20 and math.isfinite(d5_val):
        defaults["D5_boundary_challenge_score"] = _jf(d5_val)
        defaults["boundary_challenge_score"] = _jf(d5_val)
    else:
        defaults["D5_boundary_challenge_score"] = None
        defaults["boundary_challenge_score"] = None

    # D3: uncertainty weighted contrast high-dot vs low-dot by median(d) on frontier
    try:
        u_i = np.einsum("ij,jk,ik->i", ef, belief.Sigma.astype(np.float64), ef)
        u_i = np.sqrt(np.maximum(u_i.astype(np.float64), 0.0))
        wsum = float(np.sum(u_i))
        if wsum > _EPS and d.size > 1:
            w = u_i / wsum
            med = float(np.median(d))
            hi = d >= med
            lo = d < med
            if np.any(hi) and np.any(lo):
                m_hi = float(np.sum(w[hi] * d[hi]) / (np.sum(w[hi]) + _EPS))
                m_lo = float(np.sum(w[lo] * d[lo]) / (np.sum(w[lo]) + _EPS))
                defaults["D3_uncertainty_weighted_frontier_contrast"] = _jf(abs(m_hi - m_lo))
            else:
                missing_flags.append("d3_median_partition_empty_half")
        else:
            missing_flags.append("d3_uncertainty_mass_zero_or_single_item")
    except Exception:
        missing_flags.append("d3_uncertainty_projection_failed")

    # Fused proxy: per-turn naive mean of finite proxies (NOT global-z; offline studies use pooled z).
    comps: list[float] = []
    for key in ("D4_anti_collapse_score", "D3_uncertainty_weighted_frontier_contrast", "D5_boundary_challenge_score"):
        v = defaults.get(key)
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            comps.append(float(v))
    defaults["frontier_discrimination_proxy"] = _jf(float(np.mean(comps))) if comps else None

    margin = (
        float(recommendation_scores_topk[0] - recommendation_scores_topk[1])
        if len(recommendation_scores_topk) >= 2
        else None
    )
    ent10 = _top10_softmax_entropy(recommendation_scores_topk)
    if margin is not None:
        defaults["rank_damage_risk_proxy"] = _jf(
            (-margin + _RANK_DAMAGE_ENTROPY_COEFF * ent10),
        )
    else:
        defaults["rank_damage_risk_proxy"] = _jf(_RANK_DAMAGE_ENTROPY_COEFF * ent10)
        missing_flags.append("rank_damage_margin_missing")

    defaults["missing_flags"] = sorted(set(missing_flags))
    defaults["compute_ms"] = round((time.perf_counter() - t0) * 1000.0, 6)
    return defaults


# Back-compat / plan naming (session simulator uses ``safe_*`` helper).
compute_frontier_discriminative_shadow = compute_frontier_discriminative_shadow_bundle


def safe_frontier_discriminative_shadow_turn_dict(
    belief: BeliefState,
    index: EmbeddingIndex,
    *,
    seen_items: set[str],
    hyp_vectorized: Sequence[Mapping[str, Any]],
    selected_hypothesis_ids: Sequence[str],
    recommendation_scores_topk: Sequence[float],
    lambda_explore: float,
    frontier_k: int = 50,
) -> dict[str, Any]:
    """Catch-all wrapper so session simulator never crashes on diagnostics."""

    t0_wrap = time.perf_counter()
    try:
        return compute_frontier_discriminative_shadow_bundle(
            belief,
            index,
            seen_items=seen_items,
            hyp_vectorized=hyp_vectorized,
            selected_hypothesis_ids=selected_hypothesis_ids,
            recommendation_scores_topk=recommendation_scores_topk,
            lambda_explore=lambda_explore,
            frontier_k=frontier_k,
        )
    except Exception as exc:  # noqa: BLE001 — shadow must never break session
        return {
            "enabled": True,
            "frontier_k": int(frontier_k),
            "frontier_topk_hash": None,
            "P1_mean_F50_dot": None,
            "P1_cos_mu": None,
            "D3_uncertainty_weighted_frontier_contrast": None,
            "D4_anti_collapse_score": None,
            "D5_boundary_challenge_score": None,
            "boundary_challenge_score": None,
            "frontier_discrimination_proxy": None,
            "frontier_score_mean": None,
            "frontier_score_std": None,
            "rank_damage_risk_proxy": None,
            "missing_flags": sorted({"shadow_bundle_exception"}),
            "shadow_error": repr(exc),
            "compute_ms": round((time.perf_counter() - t0_wrap) * 1000.0, 6),
        }
