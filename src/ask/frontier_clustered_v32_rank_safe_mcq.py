"""Rank-safe MCQ roles + embedding proxies (runtime-safe, no oracle labels)."""

from __future__ import annotations

import copy
import re
from typing import Any

import numpy as np

from src.embedding.index import EmbeddingIndex

NOT_SURE_OPTION_TEXT = "Not sure — either leaning could work; skip narrowing for now."

_SOFT_AGGRESSIVE_TOKENS = ("experimental", "noisy", "abrasive", "push", "venture", "uncover hidden")

EXPLORATORY_PHRASES = (
    "experimental",
    "adventurous",
    "eclectic",
    "underground",
    "raw, energetic",
    "edgier",
    "less familiar",
    "explore",
    "curious about",
    "push outside",
    "bold new",
)

CONSERVATIVE_PHRASES = (
    "current top",
    "keep listening",
    "stay with",
    "stick with",
    "similar to what",
    "already in your",
    "favorites you",
    "preserve",
    "maintain",
)


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v
    return v / n


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = _l2_normalize(np.asarray(a, dtype=float).reshape(-1))
    b = _l2_normalize(np.asarray(b, dtype=float).reshape(-1))
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def top10_centroid(index: EmbeddingIndex, top10_ids: list[str]) -> np.ndarray | None:
    vecs: list[np.ndarray] = []
    for iid in top10_ids[:10]:
        s = str(iid)
        if s in index.item_id_to_index:
            vecs.append(index.embeddings[int(index.item_id_to_index[s])].astype(float))
    if not vecs:
        return None
    return _l2_normalize(np.mean(np.stack(vecs, axis=0), axis=0))


def vector_from_hypothesis_logged(h: dict[str, Any], index: EmbeddingIndex) -> np.ndarray | None:
    rep_ids = [str(x) for x in (h.get("representative_items") or [])[:8]]
    if not rep_ids:
        return None
    vecs = []
    for iid in rep_ids:
        if iid in index.item_id_to_index:
            vecs.append(index.embeddings[int(index.item_id_to_index[iid])].astype(float))
    if not vecs:
        return None
    return _l2_normalize(np.mean(np.stack(vecs, axis=0), axis=0))


def vector_from_hyp_vectorized(h: dict[str, Any]) -> np.ndarray | None:
    v = h.get("vector")
    if v is None:
        return None
    try:
        arr = np.asarray(v, dtype=float).reshape(-1)
        if arr.size == 0:
            return None
        return _l2_normalize(arr)
    except (TypeError, ValueError):
        return None


def _exploratory_text_score(text: str) -> int:
    low = (text or "").lower()
    return sum(1 for p in EXPLORATORY_PHRASES if p in low)


def _infer_role_from_geometry(
    cos_to_centroid: float,
    top10_overlap_frac: float,
) -> tuple[str, str]:
    if cos_to_centroid >= 0.72 or top10_overlap_frac >= 0.35:
        return "current_top_preserve", "preserve"
    if cos_to_centroid >= 0.45:
        return "boundary_challenge", "challenge"
    return "explore_uncertain", "explore"


def _hypothesis_has_item_embedding(h: dict[str, Any], index: EmbeddingIndex | None) -> bool:
    if index is None:
        return False
    for iid in (h.get("representative_items") or [])[:8]:
        if str(iid) in index.item_id_to_index:
            return True
    return False


def is_v32_substantive_generation_block(frontier_gen_block: dict[str, Any] | None) -> bool:
    """True when this turn has a v32 MCQ bundle with ≥3 parsed hypotheses."""

    if not isinstance(frontier_gen_block, dict):
        return False
    ph = frontier_gen_block.get("parsed_hypotheses")
    if not isinstance(ph, list) or len(ph) < 3:
        return False
    return any(isinstance(h, dict) for h in ph)


def validate_rank_safety_role_triplet(hypotheses: list[dict[str, Any]]) -> list[str]:
    """Return schema warnings when substantive hypotheses lack the required role mix."""

    warnings: list[str] = []
    hyps = [h for h in hypotheses if isinstance(h, dict)]
    roles = [str(h.get("rank_safety_role") or "") for h in hyps]
    c = {r: roles.count(r) for r in roles if r}
    if c.get("current_top_preserve", 0) != 1:
        warnings.append(f"role_count_current_top_preserve={c.get('current_top_preserve', 0)}")
    if c.get("boundary_challenge", 0) < 1:
        warnings.append("role_missing_boundary_challenge")
    if c.get("explore_uncertain", 0) < 1:
        warnings.append("role_missing_explore_uncertain")
    if len(set(roles)) < len([r for r in roles if r]):
        warnings.append("role_duplicate_detected")
    if roles and all(r == "explore_uncertain" for r in roles if r):
        warnings.append("all_roles_explore_uncertain")
    return warnings


def assign_rank_safety_to_hypotheses(
    hypotheses: list[dict[str, Any]],
    *,
    top10_ids: list[str],
    index: EmbeddingIndex | None,
    cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Annotate parsed hypotheses with rank_safety_role metadata (mock / offline)."""

    cfg = cfg or {}
    if not bool(cfg.get("rank_safe_mcq_roles_enabled", True)):
        return hypotheses

    centroid = top10_centroid(index, top10_ids) if index is not None else None
    top_set = {str(x) for x in top10_ids[:10]}

    for h in hypotheses:
        if not isinstance(h, dict):
            continue
        hid = str(h.get("hypothesis_id") or "")
        text = " ".join(
            str(h.get(k) or "")
            for k in (
                "intent_text",
                "positive_intent",
                "short_user_facing_option_text",
                "text_description",
                "positive_style_phrase",
            )
        )
        reps = [str(x) for x in (h.get("representative_items") or [])]
        covered = [r for r in reps if r in top_set]
        overlap_frac = len(covered) / max(1, min(10, len(top_set)))

        cos_c = 0.0
        has_vec = False
        if centroid is not None and index is not None:
            hv = vector_from_hypothesis_logged(h, index)
            if hv is not None:
                has_vec = True
                cos_c = _cosine(hv, centroid)

        geometry_reliable = bool(has_vec and centroid is not None)
        geometry_role, effect = _infer_role_from_geometry(cos_c, overlap_frac)

        llm_role_raw = h.get("llm_declared_rank_safety_role") or h.get("rank_safety_role")
        llm_role = str(llm_role_raw).strip() if llm_role_raw is not None else ""
        llm_role = llm_role if llm_role in ("current_top_preserve", "boundary_challenge", "explore_uncertain") else ""

        has_cur_ev = bool(h.get("current_top_evidence_items"))
        has_bnd_ev = bool(h.get("boundary_evidence_items"))
        has_unc_ev = bool(h.get("uncertain_evidence_items"))

        def _llm_evidence_supports(role: str) -> bool:
            if role == "current_top_preserve":
                return has_cur_ev
            if role == "boundary_challenge":
                return has_bnd_ev
            if role == "explore_uncertain":
                return has_unc_ev
            return False

        final_role = geometry_role
        conflict = False
        reason_parts: list[str] = [f"geom={geometry_role}"]

        if llm_role:
            reason_parts.append(f"llm={llm_role}")
            if not geometry_reliable:
                final_role = llm_role
                reason_parts.append("trust_llm_geometry_unreliable")
            elif llm_role == geometry_role:
                final_role = llm_role
                reason_parts.append("llm_agrees")
            else:
                conflict = True
                if _llm_evidence_supports(llm_role):
                    final_role = llm_role
                    reason_parts.append("keep_llm_with_role_evidence")
                elif llm_role == "current_top_preserve" and overlap_frac >= 0.2:
                    final_role = llm_role
                    reason_parts.append("keep_llm_preserve_overlap")
                elif llm_role == "boundary_challenge" and cos_c >= 0.35:
                    final_role = llm_role
                    reason_parts.append("keep_llm_boundary_cos")
                else:
                    final_role = geometry_role
                    reason_parts.append("fallback_to_geometry")
        else:
            reason_parts.append("no_llm_role")

        role = final_role
        if role == "current_top_preserve":
            effect = "preserve"
        elif role == "boundary_challenge":
            effect = "challenge"
        else:
            effect = "explore"

        if not llm_role:
            if _exploratory_text_score(text) >= 2 and role != "current_top_preserve":
                role = "explore_uncertain"
                effect = "explore"
            if any(p in text.lower() for p in CONSERVATIVE_PHRASES):
                role = "current_top_preserve"
                effect = "preserve"

        h["llm_declared_rank_safety_role"] = llm_role or None
        h["geometry_rank_safety_role"] = geometry_role
        h["final_rank_safety_role"] = role
        h["role_assignment_conflict"] = bool(conflict and llm_role != role)
        h["role_assignment_reason"] = "+".join(reason_parts)[:220]

        h["rank_safety_role"] = role
        h["expected_rank_effect"] = effect
        h["covered_current_top_items"] = covered[:6]
        h["risk_note"] = (
            f"cos_to_top10_centroid={cos_c:.3f};overlap={overlap_frac:.2f};hyp={hid}"
        )[:120]

    role_warnings = validate_rank_safety_role_triplet(hypotheses)
    if role_warnings:
        for h in hypotheses:
            if isinstance(h, dict):
                h.setdefault("rank_safety_role_schema_warnings", list(role_warnings))

    return hypotheses


def selected_option_similarity_to_current_top_centroid(
    *,
    index: EmbeddingIndex | None,
    current_top10: list[str],
    hyp_vectorized: list[dict[str, Any]] | None,
    parsed_hypotheses: list[dict[str, Any]] | None,
    selected_hypothesis_id: str | None,
) -> float | None:
    """Runtime-safe proxy: cosine(selected hyp vector, current top10 centroid)."""

    if not selected_hypothesis_id or index is None:
        return None
    centroid = top10_centroid(index, [str(x) for x in (current_top10 or [])[:10]])
    if centroid is None:
        return None
    hid = str(selected_hypothesis_id)
    vec = None
    for hv in hyp_vectorized or []:
        if str(hv.get("hypothesis_id") or "") == hid:
            vec = vector_from_hyp_vectorized(hv)
            break
    if vec is None:
        for h in parsed_hypotheses or []:
            if isinstance(h, dict) and str(h.get("hypothesis_id") or "") == hid:
                vec = vector_from_hypothesis_logged(h, index)
                break
    if vec is None:
        return None
    return round(_cosine(vec, centroid), 4)


def compute_v2_option_features(
    *,
    index: EmbeddingIndex | None,
    current_top10: list[str],
    hyp_vectorized: list[dict[str, Any]] | None,
    parsed_hypotheses: list[dict[str, Any]] | None,
    recommendation_scores: list[float] | None = None,
    selected_hypothesis_id: str | None = None,
) -> dict[str, Any]:
    """Embedding-based rank-safe proxies (no belief update, no oracle)."""

    top = [str(x) for x in (current_top10 or [])[:10]]
    hyps = list(parsed_hypotheses or []) or []
    hvecs = list(hyp_vectorized or []) or []

    centroid = top10_centroid(index, top) if index is not None else None
    top_embs: list[np.ndarray] = []
    if index is not None:
        for iid in top:
            if iid in index.item_id_to_index:
                top_embs.append(
                    index.embeddings[int(index.item_id_to_index[iid])].astype(float),
                )

    coverages: list[float] = []
    displacements: list[float] = []
    preservation_scores: list[float] = []
    roles: list[str] = []

    id_to_vec: dict[str, np.ndarray] = {}
    for hv in hvecs:
        hid = str(hv.get("hypothesis_id") or "")
        vv = vector_from_hyp_vectorized(hv)
        if hid and vv is not None:
            id_to_vec[hid] = vv

    for h in hyps:
        if not isinstance(h, dict):
            continue
        hid = str(h.get("hypothesis_id") or "")
        role = str(h.get("rank_safety_role") or "")
        if role:
            roles.append(role)

        vec = id_to_vec.get(hid)
        if vec is None and index is not None:
            vec = vector_from_hypothesis_logged(h, index)
        if vec is None:
            continue

        if centroid is not None:
            preservation_scores.append(_cosine(vec, centroid))
            displacements.append(max(0.0, 1.0 - _cosine(vec, centroid)))

        if top_embs:
            dots = [_cosine(vec, te) for te in top_embs]
            coverages.append(float(max(dots)))

    # Lightweight perturbation proxy: shift mass toward hyp direction vs centroid on top10 slots.
    perturb_disp = 0.0
    if centroid is not None and top_embs and hvecs:
        for hv in hvecs[:5]:
            vv = vector_from_hyp_vectorized(hv)
            if vv is None:
                continue
            deltas = [max(0.0, _cosine(vv, te) - _cosine(centroid, te)) for te in top_embs]
            perturb_disp = max(perturb_disp, float(np.mean(deltas)) if deltas else 0.0)

    preservation = float(np.mean(preservation_scores)) if preservation_scores else 0.0
    cov_max = float(max(coverages)) if coverages else 0.0
    cov_min = float(min(coverages)) if coverages else 0.0
    disp_proxy = float(max(displacements + [perturb_disp])) if (displacements or perturb_disp) else 0.0

    conservative_available = "current_top_preserve" in roles
    if not conservative_available and hyps:
        conservative_available = any(
            str(h.get("rank_safety_role") or "") == "current_top_preserve" for h in hyps if isinstance(h, dict)
        )

    explore_hits = sum(
        1
        for h in hyps
        if isinstance(h, dict)
        and (
            str(h.get("rank_safety_role") or "") == "explore_uncertain"
            or _exploratory_text_score(
                " ".join(
                    str(h.get(k) or "")
                    for k in ("intent_text", "positive_intent", "short_user_facing_option_text")
                ),
            )
            >= 2
        )
    )
    n_substantive = max(1, len([h for h in hyps if isinstance(h, dict)]))
    all_exploratory = bool(explore_hits >= n_substantive and not conservative_available)

    has_preserve = conservative_available
    has_boundary = any(str(r) == "boundary_challenge" for r in roles)
    has_explore = any(str(r) == "explore_uncertain" for r in roles) or explore_hits > 0

    sel_sim = selected_option_similarity_to_current_top_centroid(
        index=index,
        current_top10=top,
        hyp_vectorized=hvecs,
        parsed_hypotheses=hyps,
        selected_hypothesis_id=selected_hypothesis_id,
    )

    return {
        "current_top_preservation_score": round(preservation, 4),
        "option_current_top_coverage_max": round(cov_max, 4),
        "option_current_top_coverage_min": round(cov_min, 4),
        "expected_top10_displacement_proxy": round(disp_proxy, 4),
        "conservative_option_available": bool(conservative_available),
        "all_options_exploratory_risk": bool(all_exploratory),
        "has_current_top_preserve_option": bool(has_preserve),
        "has_boundary_challenge_option": bool(has_boundary),
        "has_explore_uncertain_option": bool(has_explore),
        "selected_option_current_top_similarity": sel_sim,
        "selected_option_similarity_to_current_top_centroid": sel_sim,
    }


def apply_conservative_mcq_to_generation_block(
    frontier_block: dict[str, Any],
    *,
    conservative_hypothesis_id: str | None,
) -> dict[str, Any]:
    """Restrict frontier MCQ bundle to conservative + unsure options only."""

    if not conservative_hypothesis_id or not isinstance(frontier_block, dict):
        return frontier_block

    hyps = frontier_block.get("parsed_hypotheses")
    if isinstance(hyps, list):
        frontier_block["parsed_hypotheses"] = [
            h
            for h in hyps
            if isinstance(h, dict)
            and (
                str(h.get("hypothesis_id")) == str(conservative_hypothesis_id)
                or str(h.get("rank_safety_role") or "") == "current_top_preserve"
            )
        ] or hyps

    cq = frontier_block.get("choice_question")
    if not isinstance(cq, dict):
        return frontier_block
    opts = cq.get("options")
    if not isinstance(opts, list):
        return frontier_block

    role_by_hid: dict[str, str] = {}
    if isinstance(hyps, list):
        for h in hyps:
            if isinstance(h, dict):
                role_by_hid[str(h.get("hypothesis_id") or "")] = str(h.get("rank_safety_role") or "")

    kept: list[dict[str, Any]] = []
    for o in opts:
        if not isinstance(o, dict):
            continue
        hid = str(o.get("maps_to_hypothesis_id") or o.get("mapped_hypothesis_id") or "")
        if hid in ("__none__", "none", ""):
            kept.append(o)
            continue
        if hid == str(conservative_hypothesis_id):
            kept.append(o)
            continue
        if role_by_hid.get(hid) == "current_top_preserve":
            kept.append(o)
            continue
        if str(o.get("rank_safety_role") or "") == "current_top_preserve":
            kept.append(o)
    if not any(
        str(o.get("maps_to_hypothesis_id") or "") == str(conservative_hypothesis_id) for o in kept if isinstance(o, dict)
    ):
        for o in opts:
            if not isinstance(o, dict):
                continue
            hid = str(o.get("maps_to_hypothesis_id") or "")
            if hid == str(conservative_hypothesis_id):
                kept.append(o)
                break
    if kept:
        cq["options"] = kept
        frontier_block["choice_question"] = cq
        frontier_block["rank_safe_conservative_filter_applied"] = True
        frontier_block["conservative_rewrite_mode"] = "hard"
    return frontier_block


def resolve_conservative_rewrite_mode(fc32_cfg: dict[str, Any] | None) -> str:
    """Return ``hard`` (default) or ``soft_balanced`` for conservative MCQ rewrite."""
    cfg = fc32_cfg or {}
    mode = str(cfg.get("conservative_rewrite_mode") or cfg.get("rank_safe_conservative_rewrite_mode") or "").strip().lower()
    if mode in {"soft_balanced", "soft-balanced", "soft_balanced_a", "soft"}:
        return "soft_balanced"
    if bool(cfg.get("rank_safe_soft_rewrite_enabled", False)):
        tmpl = str(cfg.get("rank_safe_soft_rewrite_template") or "A").strip().upper()
        if tmpl in {"A", "BALANCED", "BALANCED_SOFT"}:
            return "soft_balanced"
    return "hard"


def _soften_explore_text_balanced(text: str) -> str:
    t = str(text or "").strip()
    for tok in _SOFT_AGGRESSIVE_TOKENS:
        t = re.sub(re.escape(tok), "gentler", t, flags=re.I)
    if "smaller step" not in t.lower():
        t = f"{t} (smaller step — stay closer to current favorites.)"
    return t[:320]


def _hyp_by_role(hyps: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for h in hyps:
        if not isinstance(h, dict):
            continue
        role = str(h.get("rank_safety_role") or h.get("final_rank_safety_role") or "")
        if role:
            out[role] = h
    return out


def _extract_question_stem(cq: dict[str, Any]) -> str:
    qtext = str(cq.get("question_text") or cq.get("final_question_text") or "").strip()
    if "\n\nA)" in qtext:
        qtext = qtext.split("\n\nA)")[0].strip()
    elif "\nA)" in qtext:
        qtext = qtext.split("\nA)")[0].strip()
    if not qtext:
        qtext = "Which stylistic leaning should personalization favor next?"
    return qtext


def apply_soft_balanced_conservative_mcq_to_generation_block(
    frontier_block: dict[str, Any],
    *,
    conservative_hypothesis_id: str | None,
) -> dict[str, Any]:
    """Template A soft rewrite: preserve + boundary + softened explore + not-sure."""

    if not isinstance(frontier_block, dict):
        return frontier_block

    hyps_raw = frontier_block.get("parsed_hypotheses")
    if not isinstance(hyps_raw, list):
        return frontier_block

    hyps = [copy.deepcopy(h) for h in hyps_raw if isinstance(h, dict)]
    by_role = _hyp_by_role(hyps)
    explore = by_role.get("explore_uncertain")
    if explore:
        src = str(explore.get("short_user_facing_option_text") or explore.get("intent_text") or "")
        softened = _soften_explore_text_balanced(src)
        explore["short_user_facing_option_text"] = softened
        if explore.get("intent_text"):
            explore["intent_text"] = softened
        explore["soft_rewrite_applied"] = True
        explore["soft_rewrite_template"] = "A"
        explore["explore_softening_level"] = "light"

    frontier_block["parsed_hypotheses"] = hyps

    cq_in = frontier_block.get("choice_question")
    cq_seed = copy.deepcopy(cq_in) if isinstance(cq_in, dict) else {}
    qtext = _extract_question_stem(cq_seed)

    role_labels = [
        ("A", "current_top_preserve"),
        ("B", "boundary_challenge"),
        ("C", "explore_uncertain"),
    ]
    opts: list[dict[str, Any]] = []
    for oid, role in role_labels:
        h = by_role.get(role)
        if not h:
            continue
        hid = str(h.get("hypothesis_id") or h.get("option_id") or "")
        text = str(h.get("short_user_facing_option_text") or h.get("intent_text") or "")
        opt: dict[str, Any] = {
            "option_id": oid,
            "text": text[:400],
            "maps_to_hypothesis_id": hid,
            "mapped_hypothesis_id": hid,
            "rank_safety_role": role,
        }
        cr = h.get("contrast_ref")
        if isinstance(cr, dict):
            opt["contrast_ref"] = copy.deepcopy(cr)
        if role == "explore_uncertain":
            opt["expected_rank_effect"] = "soft_explore"
        opts.append(opt)

    if conservative_hypothesis_id:
        pres = by_role.get("current_top_preserve")
        if pres and str(pres.get("hypothesis_id") or "") != str(conservative_hypothesis_id):
            for o in opts:
                if o.get("rank_safety_role") == "current_top_preserve":
                    o["maps_to_hypothesis_id"] = str(conservative_hypothesis_id)
                    o["mapped_hypothesis_id"] = str(conservative_hypothesis_id)

    opts.append(
        {
            "option_id": "D",
            "text": NOT_SURE_OPTION_TEXT,
            "maps_to_hypothesis_id": "__none__",
            "mapped_hypothesis_id": "__none__",
        }
    )

    cq_out = dict(cq_seed)
    cq_out["question_text"] = qtext
    cq_out["final_question_text"] = qtext
    cq_out["options"] = opts
    frontier_block["choice_question"] = cq_out
    frontier_block["rank_safe_conservative_filter_applied"] = True
    frontier_block["rank_safe_soft_rewrite_applied"] = True
    frontier_block["conservative_rewrite_mode"] = "soft_balanced"
    _revalidate_generation_block_schema(frontier_block)
    return frontier_block


def _planned_pairs_from_block(frontier_block: dict[str, Any]) -> set[tuple[int, int]]:
    from src.ask.frontier_clustered_v32_schema import planned_contrast_pairs_from_diag

    return planned_contrast_pairs_from_diag(frontier_block.get("planned_cluster_contrasts") or [])


def _revalidate_generation_block_schema(frontier_block: dict[str, Any]) -> dict[str, Any]:
    import os

    from src.ask.frontier_clustered_v32_schema import validate_generation_bundle

    rolefix_mode = str(os.environ.get("V41_PROMPT_ROLE_FIX") or "").strip().lower() in ("1", "true", "yes", "on")
    hyps = frontier_block.get("parsed_hypotheses") or []
    cq = frontier_block.get("choice_question") or {}
    val = validate_generation_bundle(
        hyps if isinstance(hyps, list) else [],
        cq if isinstance(cq, dict) else {},
        _planned_pairs_from_block(frontier_block),
        hypothesis_count_band=(3, 3) if rolefix_mode else None,
        rolefix_mode=rolefix_mode,
    )
    frontier_block["schema_validation_result"] = val
    return val


def apply_conservative_rewrite_to_generation_block(
    frontier_block: dict[str, Any],
    *,
    conservative_hypothesis_id: str | None,
    rewrite_mode: str = "hard",
) -> dict[str, Any]:
    if rewrite_mode == "soft_balanced":
        return apply_soft_balanced_conservative_mcq_to_generation_block(
            frontier_block,
            conservative_hypothesis_id=conservative_hypothesis_id,
        )
    return apply_conservative_mcq_to_generation_block(
        frontier_block,
        conservative_hypothesis_id=conservative_hypothesis_id,
    )


def pick_conservative_hypothesis_id(
    parsed_hypotheses: list[dict[str, Any]] | None,
    hyp_vectorized: list[dict[str, Any]] | None,
    *,
    index: EmbeddingIndex | None,
    current_top10: list[str],
) -> str | None:
    hyps = [h for h in (parsed_hypotheses or []) if isinstance(h, dict)]
    if not hyps:
        return None
    for h in hyps:
        if str(h.get("rank_safety_role") or "") == "current_top_preserve":
            return str(h.get("hypothesis_id") or "") or None

    centroid = top10_centroid(index, current_top10) if index else None
    best_id: str | None = None
    best_cos = -1.0
    for h in hyps:
        hid = str(h.get("hypothesis_id") or "")
        vec = None
        if hyp_vectorized:
            for hv in hyp_vectorized:
                if str(hv.get("hypothesis_id")) == hid:
                    vec = vector_from_hyp_vectorized(hv)
                    break
        if vec is None and index is not None:
            vec = vector_from_hypothesis_logged(h, index)
        if vec is None or centroid is None:
            continue
        c = _cosine(vec, centroid)
        if c > best_cos:
            best_cos = c
            best_id = hid
    return best_id
