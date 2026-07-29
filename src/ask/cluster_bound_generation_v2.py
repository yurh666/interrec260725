"""Cluster-bound generation v2 — runtime prompt/schema/self_check for frontier_clustered_v32.

Does not read target/future/hit_delta. On failure after one retry, caller falls back to legacy v32 generation.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import Any

import numpy as np

from src.ask.frontier_clustered_v32 import (
    FrontierCluster,
    ClusterContrast,
    sanitize_user_question_copy,
    sanitize_item_name_for_prompt,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_MD = (
    REPO_ROOT
    / "experiments/ranking_aware_voi_gap/frontier_clustered_v32_oos_skip_analysis"
    / "question_option_alignment_next"
    / "CLUSTER_BOUND_GENERATION_V2_PROMPT.md"
)

ROLE_TO_RANK_SAFETY = {
    "preserve": "current_top_preserve",
    "boundary": "boundary_challenge",
    "explore": "explore_uncertain",
}

GENERIC_PHRASES = (
    "adventurous direction",
    "something different",
    "explore more",
    "broader taste",
    "new styles",
    "less familiar",
    "for variety",
    "new listening experience",
    "slightly edgier",
)

DOMAIN_TERMS = {
    "lastfm_music": {
        "rock", "pop", "indie", "alternative", "folk", "soul", "jazz", "funk", "electronic",
        "guitar", "synth", "britpop", "punk", "hip", "hop", "lyrical", "rhythmic", "raw", "acoustic",
    },
    "movielens_movie": {"drama", "thriller", "comedy", "suspense", "surreal", "visual", "pacing", "character"},
    "yelp_business": {"casual", "dining", "noodle", "rice", "cozy", "takeout", "upscale", "dinner"},
    "amazon_book": {"fiction", "mystery", "historical", "speculative", "world", "character", "reflective", "accessible"},
}


def cluster_bound_generation_v2_enabled(fc32_cfg: dict[str, Any]) -> bool:
    return bool(fc32_cfg.get("cluster_bound_generation_v2_enabled"))


def resolve_domain(cfg: dict[str, Any], fc32_cfg: dict[str, Any] | None = None) -> str:
    fc32 = fc32_cfg or {}
    dom = str(fc32.get("cluster_bound_domain") or "").strip()
    if dom in DOMAIN_TERMS:
        return dom
    ds = str((cfg.get("dataset") or {}).get("name") or cfg.get("dataset_name") or "").lower()
    if "movie" in ds:
        return "movielens_movie"
    if "yelp" in ds:
        return "yelp_business"
    if "book" in ds or "amazon" in ds:
        return "amazon_book"
    return "lastfm_music"


def load_cluster_bound_v2_system_prompt() -> str:
    if PROMPT_MD.is_file():
        body = PROMPT_MD.read_text(encoding="utf-8")
        if "## System Instruction" in body:
            chunk = body.split("## System Instruction", 1)[1]
            chunk = chunk.split("## Input Schema", 1)[0].strip()
            if chunk:
                return chunk
    return dedent(
        """\
        You generate preference hypotheses for an interactive recommender.
        Every option must remain grounded in the provided catalog cluster with representative items.
        Return valid JSON only matching the cluster-bound v2 output schema."""
    )


def _squish(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip())


def _cluster_phrase_from_row(row: dict[str, Any], domain: str) -> str:
    cand = _squish(str(row.get("cluster_phrase_candidate") or row.get("cluster_phrase") or ""))
    if cand and len(cand) > 3:
        return cand[:120]
    text = str(row.get("option_text") or "")
    tokens = re.findall(r"[a-z0-9&']+", text.lower())
    hits = [t for t in tokens if t in DOMAIN_TERMS.get(domain, set())]
    if hits:
        return " ".join(dict.fromkeys(hits[:4]))
    kws = [str(x) for x in (row.get("cluster_keywords") or []) if str(x).strip()][:4]
    if kws:
        return " ".join(kws)
    role = str(row.get("role") or "preference")
    return f"{role} catalog cluster"


def _generic_fail(text: str, names: list[str]) -> bool:
    lo = text.lower()
    has_anchor = any(n.lower() in lo for n in names if n)
    for p in GENERIC_PHRASES:
        if p in lo and not has_anchor:
            return True
    if not has_anchor and len(lo.split()) < 8:
        vague = ("explore", "different", "variety", "adventurous", "broader")
        if any(v in lo for v in vague):
            return True
    return False


PRESERVE_CUES = (
    "keep",
    "stay",
    "stick",
    "still",
    "prefer",
    "maintain",
    "retain",
    "continue",
    "lean into",
    "hold on",
    "preserve",
)
SHIFT_CUES = (
    "shift",
    "toward",
    "towards",
    "move toward",
    "moving toward",
    "but",
    "while",
    "yet",
    "explore",
    "try",
    "pivot",
    "lean into",
    "branch",
    "transition",
    "instead",
)


def _text_has_cue(text: str, cues: tuple[str, ...]) -> bool:
    lo = text.lower()
    return any(c in lo for c in cues)


def _boundary_keyword_check_legacy(text: str) -> bool:
    lo = text.lower()
    return "still" in lo and ("shift" in lo or "but" in lo)


def _boundary_semantic_check(option: dict[str, Any]) -> dict[str, Any]:
    """Field-first boundary validation; keyword check is advisory only."""
    text = str(option.get("option_text") or "")
    lo = text.lower()
    preserve_feat = _squish(str(option.get("boundary_preserve_feature") or ""))
    shift_feat = _squish(str(option.get("boundary_shift_feature") or ""))
    relation = _squish(str(option.get("boundary_relation") or ""))
    names = [str(x) for x in (option.get("representative_item_names") or []) if str(x).strip()]
    phrase = _squish(str(option.get("cluster_phrase") or ""))

    has_preserve_field = len(preserve_feat) >= 3
    has_shift_field = len(shift_feat) >= 3
    has_relation = len(relation) >= 8

    text_preserve = _text_has_cue(text, PRESERVE_CUES) or _text_has_cue(preserve_feat, PRESERVE_CUES)
    text_shift = _text_has_cue(text, SHIFT_CUES) or _text_has_cue(shift_feat, SHIFT_CUES) or _text_has_cue(relation, SHIFT_CUES)
    has_anchor = any(n.lower() in lo for n in names) or bool(phrase)
    not_generic = not _generic_fail(text, names)

    sidecar_ok = has_preserve_field and has_shift_field
    semantic_ok = (text_preserve and text_shift and has_anchor and not_generic) or (
        sidecar_ok and has_anchor and not_generic
    )
    keyword_legacy = _boundary_keyword_check_legacy(text)
    pass_with_warning = bool(sidecar_ok and semantic_ok and not keyword_legacy)

    return {
        "has_preserve_field": has_preserve_field,
        "has_shift_field": has_shift_field,
        "has_boundary_relation": has_relation,
        "text_has_preserve_semantics": text_preserve,
        "text_has_shift_semantics": text_shift,
        "sidecar_preserve_shift_present": sidecar_ok,
        "boundary_text_keyword_missing": bool(semantic_ok and not keyword_legacy),
        "boundary_semantic_pass": semantic_ok,
        "boundary_keyword_legacy_pass": keyword_legacy,
        "boundary_pass_with_warning": pass_with_warning,
    }


def programmatic_self_check_legacy(option: dict[str, Any], domain: str) -> dict[str, bool]:
    """Pre-fix checker (still/shift keywords) for replay comparison."""
    role = str(option.get("role") or "")
    text = str(option.get("option_text") or "")
    names = [str(x) for x in (option.get("representative_item_names") or []) if str(x).strip()]
    lo = text.lower()
    has_anchor = any(n.lower() in lo for n in names)
    bound_ok = str(option.get("bound_cluster_id") or "").strip() != ""
    phrase_ok = bool(_squish(str(option.get("cluster_phrase") or "")))
    dist_ok = bool(_squish(str(option.get("distinction_from_other_options") or "")))
    generic_bad = _generic_fail(text, names)
    boundary_ok = role != "boundary" or _boundary_keyword_check_legacy(text)
    explore_ok = role != "explore" or (has_anchor and len(lo.split()) >= 6)
    domain_ok = True
    if domain == "lastfm_music":
        domain_ok = not any(w in lo for w in ("restaurant", "book title", "movie title"))
    elif domain == "yelp_business":
        domain_ok = not any(w in lo for w in ("tracks", "vocals", "movie"))
    return {
        "all_options_have_bound_cluster_id": bound_ok,
        "all_options_use_representative_items": has_anchor and len(names) >= 1,
        "no_generic_option_without_anchor": not generic_bad,
        "boundary_options_specify_preserve_and_shift": boundary_ok,
        "explore_options_bind_to_high_uncertainty_cluster": explore_ok,
        "options_are_semantically_distinct": dist_ok,
        "cluster_phrase_present": phrase_ok,
        "domain_language_valid": domain_ok,
    }


def programmatic_self_check(option: dict[str, Any], domain: str) -> dict[str, bool]:
    role = str(option.get("role") or "")
    text = str(option.get("option_text") or "")
    names = [str(x) for x in (option.get("representative_item_names") or []) if str(x).strip()]
    lo = text.lower()
    has_anchor = any(n.lower() in lo for n in names)
    bound_ok = str(option.get("bound_cluster_id") or "").strip() != ""
    phrase_ok = bool(_squish(str(option.get("cluster_phrase") or "")))
    dist_ok = bool(_squish(str(option.get("distinction_from_other_options") or "")))
    generic_bad = _generic_fail(text, names)
    if role == "boundary":
        bnd = _boundary_semantic_check(option)
        boundary_ok = bool(bnd["boundary_semantic_pass"])
    else:
        boundary_ok = True
    explore_ok = role != "explore" or (has_anchor and len(lo.split()) >= 6)
    domain_ok = True
    if domain == "lastfm_music":
        domain_ok = not any(w in lo for w in ("restaurant", "book title", "movie title"))
    elif domain == "yelp_business":
        domain_ok = not any(w in lo for w in ("tracks", "vocals", "movie"))
    return {
        "all_options_have_bound_cluster_id": bound_ok,
        "all_options_use_representative_items": has_anchor and len(names) >= 1,
        "no_generic_option_without_anchor": not generic_bad,
        "boundary_options_specify_preserve_and_shift": boundary_ok,
        "explore_options_bind_to_high_uncertainty_cluster": explore_ok,
        "options_are_semantically_distinct": dist_ok,
        "cluster_phrase_present": phrase_ok,
        "domain_language_valid": domain_ok,
    }


def validate_cluster_bound_v2_schema(parsed: dict[str, Any]) -> dict[str, Any]:
    hard: list[str] = []
    soft: list[str] = []
    opts = parsed.get("options") if isinstance(parsed.get("options"), list) else []
    if not str(parsed.get("question_text") or "").strip():
        hard.append("empty_question_text")
    if len(opts) != 3:
        hard.append(f"option_count_not_3:{len(opts)}")
    roles_seen: set[str] = set()
    for i, o in enumerate(opts):
        if not isinstance(o, dict):
            hard.append(f"option_{i}_not_dict")
            continue
        for req in (
            "option_id",
            "role",
            "bound_cluster_id",
            "representative_item_ids",
            "representative_item_names",
            "cluster_phrase",
            "option_text",
            "distinction_from_other_options",
        ):
            if req not in o or not str(o.get(req) or "").strip():
                if req in ("representative_item_ids", "representative_item_names"):
                    arr = o.get(req)
                    if not isinstance(arr, list) or len(arr) < 1:
                        hard.append(f"option_{i}_missing_{req}")
                else:
                    hard.append(f"option_{i}_missing_{req}")
        role = str(o.get("role") or "")
        if role not in ("preserve", "boundary", "explore"):
            hard.append(f"option_{i}_bad_role:{role}")
        else:
            roles_seen.add(role)
        if role == "boundary":
            for bf in ("boundary_preserve_feature", "boundary_shift_feature"):
                if not _squish(str(o.get(bf) or "")):
                    hard.append(f"option_{i}_missing_{bf}")
            if not _squish(str(o.get("boundary_relation") or "")):
                soft.append(f"option_{i}_missing_boundary_relation_recommended")
    if roles_seen != {"preserve", "boundary", "explore"}:
        hard.append(f"role_mix_incomplete:{sorted(roles_seen)}")
    sc = parsed.get("self_check")
    if not isinstance(sc, dict):
        hard.append("missing_self_check_object")
    hard_pass = len(hard) == 0
    return {
        "schema_hard_pass": hard_pass,
        "schema_soft_warnings": soft,
        "hard_errors_detail": hard,
        "fallback_reason": None if hard_pass else "; ".join(hard[:12]),
    }


def validate_cluster_bound_v2_self_check(
    parsed: dict[str, Any],
    *,
    domain: str,
    require_llm_self_check: bool = True,
    use_legacy_boundary_checker: bool = False,
) -> dict[str, Any]:
    opts = [o for o in (parsed.get("options") or []) if isinstance(o, dict)]
    per_option: list[dict[str, Any]] = []
    warnings: list[str] = []
    all_prog_pass = True
    checker = programmatic_self_check_legacy if use_legacy_boundary_checker else programmatic_self_check
    for o in opts:
        prog = checker(o, domain)
        row: dict[str, Any] = {"option_id": o.get("option_id"), "programmatic": prog}
        if str(o.get("role") or "") == "boundary":
            row["boundary_detail"] = _boundary_semantic_check(o)
            if row["boundary_detail"].get("boundary_pass_with_warning"):
                warnings.append(f"boundary_text_keyword_missing:{o.get('option_id')}")
            if row["boundary_detail"].get("sidecar_preserve_shift_present"):
                warnings.append(f"sidecar_preserve_shift_present:{o.get('option_id')}")
        per_option.append(row)
        if not all(prog.values()):
            all_prog_pass = False
    llm_sc = parsed.get("self_check") if isinstance(parsed.get("self_check"), dict) else {}
    llm_pass = True
    if require_llm_self_check:
        for k, v in llm_sc.items():
            if k.startswith("options_are") and v is False:
                llm_pass = False
            elif v is False:
                llm_pass = False
    pass_ok = all_prog_pass and (llm_pass or not require_llm_self_check)
    return {
        "self_check_pass": pass_ok,
        "programmatic_all_pass": all_prog_pass,
        "llm_self_check_pass": llm_pass,
        "per_option": per_option,
        "llm_self_check": llm_sc,
        "self_check_warnings": warnings,
        "fallback_reason": None if pass_ok else "self_check_failed",
    }


def repairable_self_check_soft_accept(
    parsed: dict[str, Any],
    schema_val: dict[str, Any],
    sc_val: dict[str, Any],
    *,
    fc32_cfg: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """Classify Book-style v2 self-check failures that are safe to deliver.

    This is config-gated and intentionally narrow: schema must hard-pass, three
    bound options must exist, and each option must keep representative evidence.
    It never reads target/future data.
    """

    enabled = bool(fc32_cfg.get("self_check_soft_accept_repairable", False))
    audit: dict[str, Any] = {
        "enabled": enabled,
        "applied": False,
        "reason": None,
        "failed_programmatic_checks": [],
    }
    if not enabled:
        audit["reason"] = "disabled"
        return False, audit
    if not schema_val.get("schema_hard_pass"):
        audit["reason"] = "schema_not_hard_pass"
        return False, audit

    opts = [o for o in (parsed.get("options") or []) if isinstance(o, dict)]
    if len(opts) != 3:
        audit["reason"] = f"option_count_not_3:{len(opts)}"
        return False, audit

    allowed_soft = set(
        fc32_cfg.get(
            "self_check_soft_accept_allowed_failures",
            [
                "all_options_use_representative_items",
                "explore_options_bind_to_high_uncertainty_cluster",
                "no_generic_option_without_anchor",
            ],
        )
        or []
    )
    hard_failures: list[str] = []
    soft_failures: list[str] = []
    for opt_row in sc_val.get("per_option") or []:
        oid = str(opt_row.get("option_id") or "")
        prog = opt_row.get("programmatic") or {}
        for key, val in prog.items():
            if val is not False:
                continue
            label = f"{oid}:{key}"
            if key in allowed_soft:
                soft_failures.append(label)
            else:
                hard_failures.append(label)
    audit["failed_programmatic_checks"] = soft_failures + hard_failures
    if hard_failures:
        audit["reason"] = "hard_programmatic_failures:" + ",".join(hard_failures[:8])
        return False, audit

    for o in opts:
        if not str(o.get("bound_cluster_id") or "").strip():
            audit["reason"] = "missing_bound_cluster_id"
            return False, audit
        if not str(o.get("option_text") or "").strip():
            audit["reason"] = "missing_option_text"
            return False, audit
        names = [str(x) for x in (o.get("representative_item_names") or []) if str(x).strip()]
        ids = [str(x) for x in (o.get("representative_item_ids") or []) if str(x).strip()]
        phrase = _squish(str(o.get("cluster_phrase") or ""))
        if not names and not ids and not phrase:
            audit["reason"] = "missing_all_evidence"
            return False, audit

    audit["applied"] = True
    audit["reason"] = "schema_pass_repairable_self_check_soft_fail"
    return True, audit


def _pick_three_candidate_clusters(
    clusters: list[FrontierCluster],
    contrasts: list[ClusterContrast],
) -> list[tuple[FrontierCluster, str, ClusterContrast | None]]:
    if len(clusters) < 2:
        return []
    by_id = {int(c.cluster_id): c for c in clusters}
    preserve = max(clusters, key=lambda c: (c.overlap_top10_frac, -(c.mean_rank or 999)))
    uncertain = max(
        clusters,
        key=lambda c: (
            float(c.mean_uncertainty) if math.isfinite(c.mean_uncertainty) else -1.0,
            -c.overlap_top10_frac,
        ),
    )
    boundary_cl = None
    boundary_ct: ClusterContrast | None = None
    for ct in contrasts:
        ref = int(ct.safe_or_reference_cluster_id)
        alt = int(ct.contrasting_cluster_id)
        if ref in by_id and alt in by_id and ref != preserve.cluster_id:
            boundary_cl = by_id.get(alt) or by_id.get(ref)
            boundary_ct = ct
            break
    if boundary_cl is None:
        mids = sorted(clusters, key=lambda c: c.overlap_top10_frac)[len(clusters) // 2]
        boundary_cl = mids
    out: list[tuple[FrontierCluster, str, ClusterContrast | None]] = [
        (preserve, "preserve", None),
        (boundary_cl, "boundary", boundary_ct),
        (uncertain, "explore", None),
    ]
    return out


def build_cluster_bound_v2_input(
    *,
    domain: str,
    clusters: list[FrontierCluster],
    contrasts: list[ClusterContrast],
    history_summary: str,
    anchor_names: list[str],
    item_lookup: dict[str, dict[str, Any]],
    current_top_names: list[str] | None = None,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    letters = ["A", "B", "C"]
    picks = _pick_three_candidate_clusters(clusters, contrasts)
    for letter, (cl, role, ct) in zip(letters, picks):
        rep_ids = list(cl.member_ids)[:5]
        rep_names: list[str] = []
        rep_meta: list[dict[str, Any]] = []
        for rid in rep_ids:
            row = item_lookup.get(str(rid), {})
            nm_raw = row.get("name") or row.get("title") or row.get("item_name") or str(rid)
            nm, _tag = sanitize_item_name_for_prompt(str(nm_raw))
            if nm:
                rep_names.append(nm[:74])
            rep_meta.append(
                {
                    "item_id": str(rid),
                    "name": nm or str(rid)[:48],
                    "genre": row.get("genre"),
                    "artist": row.get("artist"),
                }
            )
        kws = re.findall(r"[a-z0-9&']+", _squish(cl.semantic_summary).lower())[:6]
        row: dict[str, Any] = {
            "option_id": letter,
            "role": role,
            "bound_cluster_id": str(cl.cluster_id),
            "representative_item_ids": [str(x) for x in rep_ids[:5]],
            "representative_item_names": rep_names[:5],
            "representative_item_metadata": rep_meta[:5],
            "cluster_keywords": kws,
            "cluster_phrase_candidate": _squish(cl.semantic_summary)[:80],
            "high_uncertainty_score": float(cl.mean_uncertainty) if math.isfinite(cl.mean_uncertainty) else 0.0,
        }
        if role == "boundary" and ct is not None:
            ref_cl = next((c for c in clusters if c.cluster_id == ct.safe_or_reference_cluster_id), None)
            alt_cl = next((c for c in clusters if c.cluster_id == ct.contrasting_cluster_id), None)
            if ref_cl and alt_cl:
                preserve_feat = _squish(ref_cl.semantic_summary)[:60] or "current top preference"
                shift_feat = _squish(alt_cl.semantic_summary)[:60] or phrase
                row["boundary_preserve_feature"] = preserve_feat
                row["boundary_shift_feature"] = shift_feat
                row["boundary_relation"] = (
                    f"preserve {preserve_feat} but shift toward {shift_feat}"
                )
                row["contrast_cluster_id"] = str(ct.contrasting_cluster_id)
                row["contrast_cluster_keywords"] = re.findall(
                    r"[a-z0-9&']+",
                    _squish(alt_cl.semantic_summary).lower(),
                )[:6]
        elif role == "boundary":
            cp = _squish(cl.semantic_summary)[:60] or "bound cluster style"
            row["boundary_preserve_feature"] = "current preference"
            row["boundary_shift_feature"] = cp
            row["boundary_relation"] = f"preserve current preference while shifting toward {cp}"
        if role == "explore":
            row["high_uncertainty_score"] = max(
                float(cl.mean_uncertainty) if math.isfinite(cl.mean_uncertainty) else 0.0,
                0.5,
            )
        candidates.append(row)

    return {
        "domain": domain,
        "question_context": {
            "current_user_preference_summary": history_summary[:400] if history_summary else None,
            "current_top_items": (current_top_names or anchor_names)[:6],
            "current_top_cluster_keywords": anchor_names[:6],
        },
        "candidate_options": candidates,
    }


def _deterministic_v2_option_text(cand: dict[str, Any], domain: str) -> str:
    role = str(cand.get("role") or "preserve")
    names = [str(x) for x in (cand.get("representative_item_names") or []) if str(x).strip()]
    joined = ", ".join(names[:2]) if len(names) >= 2 else (names[0] if names else "the representative items")
    phrase = _cluster_phrase_from_row(cand, domain)
    if role == "preserve":
        return f"I prefer items similar to {joined}, especially {phrase}."
    if role == "boundary":
        preserve = _squish(str(cand.get("boundary_preserve_feature") or "the current preference"))
        shift = _squish(str(cand.get("boundary_shift_feature") or phrase))
        return f"I want to keep {preserve}, while moving toward {shift} like {joined}."
    return f"I want to explore music in the style of {joined}, characterized by {phrase}."


def generate_cluster_bound_v2_mock(input_payload: dict[str, Any]) -> dict[str, Any]:
    domain = str(input_payload.get("domain") or "lastfm_music")
    options_out: list[dict[str, Any]] = []
    for cand in input_payload.get("candidate_options") or []:
        if not isinstance(cand, dict):
            continue
        text = _deterministic_v2_option_text(cand, domain)
        phrase = _cluster_phrase_from_row(cand, domain)
        row_out: dict[str, Any] = {
            "option_id": cand.get("option_id"),
            "role": cand.get("role"),
            "bound_cluster_id": cand.get("bound_cluster_id"),
            "representative_item_ids": cand.get("representative_item_ids") or [],
            "representative_item_names": cand.get("representative_item_names") or [],
            "cluster_phrase": phrase,
            "option_text": text,
            "distinction_from_other_options": (
                f"This {cand.get('role')} option stays anchored to cluster {cand.get('bound_cluster_id')} "
                f"via {', '.join((cand.get('representative_item_names') or [])[:2])}."
            ),
            "why_cluster_bound": f"Grounded in bound_cluster_id={cand.get('bound_cluster_id')}.",
            "generic_terms_used": [],
            "confidence": 0.85,
        }
        if str(cand.get("role") or "") == "boundary":
            br = _squish(str(cand.get("boundary_relation") or ""))
            row_out["boundary_preserve_feature"] = _squish(
                str(cand.get("boundary_preserve_feature") or "the current preference"),
            )
            row_out["boundary_shift_feature"] = _squish(str(cand.get("boundary_shift_feature") or phrase))
            row_out["boundary_relation"] = br or (
                f"preserve {row_out['boundary_preserve_feature']} while shifting toward {phrase}"
            )
        options_out.append(row_out)
    self_checks = [programmatic_self_check(o, domain) for o in options_out]
    merged_sc = {k: all(d.get(k, False) for d in self_checks) for k in (self_checks[0] if self_checks else {})}
    return {
        "question_text": "Which direction best matches your preference right now?",
        "options": options_out,
        "self_check": merged_sc,
    }


def generate_cluster_bound_v2_via_llm(
    input_payload: dict[str, Any],
    llm_client: Any,
    *,
    max_output_tokens: int | None = None,
    session_id: str | None = None,
    turn_id: int | None = None,
) -> tuple[str, dict[str, Any]]:
    if getattr(llm_client, "mode", "mock") == "mock":
        parsed = generate_cluster_bound_v2_mock(input_payload)
        return json.dumps(parsed, ensure_ascii=False, indent=2), parsed

    payload_json = json.dumps(input_payload, ensure_ascii=False, indent=2)
    if len(payload_json) > 24_000:
        payload_json = payload_json[:24_000] + "\n…[truncated]"

    schema_rules = dedent(
        """\
        CLUSTER_BOUND_GENERATION_V2 (strict JSON only):

        Return a single JSON object with keys: question_text, options (exactly 3), self_check.

        Each option MUST contain:
          option_id, role (preserve|boundary|explore), bound_cluster_id,
          representative_item_ids, representative_item_names, cluster_phrase,
          option_text, distinction_from_other_options, why_cluster_bound,
          generic_terms_used (array), confidence (0-1).

        Each option SHOULD also contain (optional for parser; recommended):
          positive_item_anchors (2-4 names from representative_item_names),
          negative_item_anchors (0-2 contrast names if provided in input).

        boundary role MUST ALSO contain:
          boundary_preserve_feature, boundary_shift_feature, boundary_relation
        (preserve/explore roles must NOT include boundary_* fields).

        boundary option_text must express preserve + shift semantics (not only still/shift keywords).
        explore role: MUST bind to high-uncertainty grouping with representative anchors.
        Do NOT use internal words (cluster, frontier, rank, boundary, top 10) in option_text.

        self_check may include booleans such as:
          all_options_have_bound_cluster_id, all_options_use_representative_items,
          all_options_have_positive_anchors, negative_anchors_are_contrastive_if_used,
          no_generic_option_without_anchor, boundary_options_specify_preserve_and_shift,
          explore_options_bind_to_high_uncertainty_cluster, options_are_semantically_distinct,
          domain_language_is_correct.

        Use domain vocabulary from the input domain field. No markdown fences."""
    )

    system_blob = load_cluster_bound_v2_system_prompt()
    msgs = [
        {"role": "system", "content": system_blob + "\n\n" + schema_rules},
        {
            "role": "user",
            "content": "Generate cluster-bound MCQ JSON for this input:\n" + payload_json,
        },
    ]
    import os

    temp_default = 0.0
    temp_env = os.environ.get("INTERREC_CB_V2_GENERATION_TEMPERATURE", "").strip()
    try:
        temp = float(temp_env) if temp_env else temp_default
    except ValueError:
        temp = temp_default
    chat_kw: dict[str, Any] = {"temperature": temp, "top_p": 1.0}
    if max_output_tokens is not None:
        chat_kw["max_tokens"] = int(max_output_tokens)

    if getattr(llm_client, "use_stable_cache_keys", False):
        from src.agents.stable_cache_key import stable_cluster_bound_generation_payload

        cache_payload = stable_cluster_bound_generation_payload(
            input_payload, session_id=session_id, turn_id=turn_id
        )
    else:
        cache_payload = input_payload

    parsed_root = llm_client.structured_json(
        msgs,
        cache_namespace="cluster_bound_generation_v2",
        cache_payload=cache_payload,
        profile_component="cluster_bound_generation_v2",
        **chat_kw,
    )
    if isinstance(parsed_root, dict) and "options" in parsed_root:
        parsed = parsed_root
    elif isinstance(parsed_root, dict):
        parsed = parsed_root
    else:
        parsed = {"question_text": "", "options": [], "self_check": {}}
    raw = json.dumps(parsed, ensure_ascii=False, indent=2)
    return raw, parsed


@dataclass
class ClusterBoundV2Result:
    success: bool
    raw_text: str = ""
    parsed: dict[str, Any] = field(default_factory=dict)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    choice_question: dict[str, Any] = field(default_factory=dict)
    option_to_cluster_mapping: dict[str, Any] = field(default_factory=dict)
    schema_validation: dict[str, Any] = field(default_factory=dict)
    self_check_validation: dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    fallback_reason: str | None = None
    telemetry: dict[str, Any] = field(default_factory=dict)
    api_call_count: int = 0
    token_usage: dict[str, Any] | None = None


def cluster_bound_v2_to_hypotheses(
    parsed: dict[str, Any],
    *,
    contrasts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Map v2 options → v32 parsed_hypotheses + choice_question + omap."""

    hyps: list[dict[str, Any]] = []
    opts: list[dict[str, Any]] = []
    omap: dict[str, Any] = {}
    planned = {(int(c["reference_cluster_id"]), int(c["contrasting_cluster_id"])) for c in contrasts if isinstance(c, dict) and "reference_cluster_id" in c}
    if not planned and contrasts:
        for c in contrasts:
            try:
                a = int(c.get("safe_or_reference_cluster_id", c.get("reference_cluster_id")))
                b = int(c.get("contrasting_cluster_id"))
                planned.add((a, b))
                planned.add((b, a))
            except Exception:
                pass

    for o in parsed.get("options") or []:
        if not isinstance(o, dict):
            continue
        oid = str(o.get("option_id") or "")
        hid = f"h{oid}" if oid else ""
        role = str(o.get("role") or "")
        try:
            bound = int(str(o.get("bound_cluster_id")).replace("cluster_", ""))
        except Exception:
            try:
                bound = int(o.get("bound_cluster_id"))
            except Exception:
                bound = -1
        rep_ids = [str(x) for x in (o.get("representative_item_ids") or [])[:5]]
        rep_names = [str(x) for x in (o.get("representative_item_names") or [])[:5]]
        text = _squish(str(o.get("option_text") or ""))
        contrast_ref = None
        for a, b in planned:
            if bound in (a, b):
                contrast_ref = {"reference_cluster_id": a, "contrasting_cluster_id": b}
                break
        if contrast_ref is None and bound >= 0:
            contrast_ref = {"reference_cluster_id": bound, "contrasting_cluster_id": bound}

        rs_role = ROLE_TO_RANK_SAFETY.get(role, "boundary_challenge")
        hyps.append(
            {
                "hypothesis_id": hid,
                "option_id": hid,
                "rank_safety_role": rs_role,
                "role": role,
                "bound_cluster_id": bound,
                "positive_cluster_id": bound,
                "contrast_cluster_id": contrast_ref["contrasting_cluster_id"] if contrast_ref else bound,
                "contrast_ref": contrast_ref,
                "representative_item_ids": rep_ids,
                "representative_item_names": rep_names,
                "representative_items": rep_ids,
                "cluster_phrase": _squish(str(o.get("cluster_phrase") or "")),
                "boundary_preserve_feature": _squish(str(o.get("boundary_preserve_feature") or "")),
                "boundary_shift_feature": _squish(str(o.get("boundary_shift_feature") or "")),
                "boundary_relation": _squish(str(o.get("boundary_relation") or "")),
                "distinction_from_other_options": _squish(str(o.get("distinction_from_other_options") or "")),
                "why_cluster_bound": _squish(str(o.get("why_cluster_bound") or "")),
                "intent_text": text,
                "positive_intent": text,
                "short_user_facing_option_text": text,
                "text_description": text,
                "negative_contrast": "n/a",
                "distinguishing_features": [_squish(str(o.get("cluster_phrase") or ""))[:40]],
                "feature_signature": [f"cb_{bound}", role],
                "generic_terms_used": list(o.get("generic_terms_used") or []),
                "confidence": float(o.get("confidence") or 0.0),
                "positive_item_anchors": [str(x) for x in (o.get("positive_item_anchors") or [])[:6]],
                "negative_item_anchors": [str(x) for x in (o.get("negative_item_anchors") or [])[:4]],
            }
        )
        opt_text = sanitize_user_question_copy(text)
        opts.append(
            {
                "option_id": oid,
                "text": opt_text,
                "maps_to_hypothesis_id": hid,
                "mapped_hypothesis_id": hid,
                "role": role,
                "bound_cluster_id": bound,
                "representative_item_ids": rep_ids,
                "representative_item_names": rep_names,
                "cluster_phrase": o.get("cluster_phrase"),
                "distinction_from_other_options": o.get("distinction_from_other_options"),
                "contrast_ref": contrast_ref,
            }
        )
        omap[oid] = {
            "positive_cluster_id": bound,
            "contrast_cluster_id": contrast_ref["contrasting_cluster_id"] if contrast_ref else bound,
            "bound_cluster_id": bound,
        }

    qtext = sanitize_user_question_copy(
        str(parsed.get("question_text") or "Which direction best matches your preference right now?"),
    )
    cq = {
        "final_question_text": qtext,
        "question_text": qtext,
        "options": opts,
        "extra": {"cluster_bound_generation_v2": True},
    }
    return hyps, cq, opts, omap


def run_cluster_bound_generation_v2(
    *,
    clusters: list[FrontierCluster],
    contrasts: list[ClusterContrast],
    contrasts_ser: list[dict[str, Any]],
    history_summary: str,
    anchor_names: list[str],
    item_lookup: dict[str, Any],
    llm_client: Any,
    gen_mode: str,
    cfg: dict[str, Any],
    fc32_cfg: dict[str, Any],
    current_top_names: list[str] | None = None,
    max_output_tokens: int | None = None,
    session_id: str | None = None,
    turn_id: int | None = None,
) -> ClusterBoundV2Result:
    """Try v2 generation with at most one retry; never uses target/future."""

    domain = resolve_domain(cfg, fc32_cfg)
    input_payload = build_cluster_bound_v2_input(
        domain=domain,
        clusters=clusters,
        contrasts=contrasts,
        history_summary=history_summary,
        anchor_names=anchor_names,
        item_lookup=item_lookup,
        current_top_names=current_top_names,
    )
    tel: dict[str, Any] = {
        "cluster_bound_generation_v2_enabled": True,
        "domain": domain,
        "runtime_used_target": False,
        "runtime_used_future": False,
        "runtime_used_hit_delta": False,
        "retry_count": 0,
        "schema_pass": False,
        "self_check_pass": False,
    }

    retry_count = 0
    last_fb: str | None = None
    raw_text = ""
    parsed: dict[str, Any] = {}

    for attempt in range(2):
        if str(gen_mode).lower() == "llm":
            raw_text, parsed = generate_cluster_bound_v2_via_llm(
                input_payload,
                llm_client,
                max_output_tokens=max_output_tokens,
                session_id=session_id,
                turn_id=turn_id,
            )
            lau = getattr(llm_client, "last_structured_audit", {}) or {}
            for ck in (
                "cache_hit",
                "cache_miss",
                "cache_key_sha256",
                "cache_key_version",
                "cache_mode",
                "payload_hash",
                "prompt_hash",
                "cache_miss_reason",
            ):
                if ck in lau:
                    tel[ck] = lau[ck]
            tel["api_call_count"] = int(lau.get("n_api_calls") or 0) + int(tel.get("api_call_count") or 0)
            tu = getattr(llm_client, "_last_token_usage", None)
            if isinstance(tu, dict):
                tel["token_usage"] = dict(tu)
        else:
            parsed = generate_cluster_bound_v2_mock(input_payload)
            raw_text = json.dumps(parsed, ensure_ascii=False, indent=2)

        schema_val = validate_cluster_bound_v2_schema(parsed)
        sc_val = validate_cluster_bound_v2_self_check(parsed, domain=domain)
        soft_accept, soft_accept_audit = repairable_self_check_soft_accept(
            parsed,
            schema_val,
            sc_val,
            fc32_cfg=fc32_cfg,
        )
        tel["schema_validation"] = schema_val
        tel["self_check_validation"] = sc_val
        tel["self_check_soft_accept"] = soft_accept_audit
        tel["schema_pass"] = bool(schema_val.get("schema_hard_pass"))
        tel["self_check_pass"] = bool(sc_val.get("self_check_pass") or soft_accept)

        if tel["schema_pass"] and tel["self_check_pass"]:
            hyps, cq, _opts, omap = cluster_bound_v2_to_hypotheses(parsed, contrasts=contrasts_ser)
            tel["retry_count"] = retry_count
            tel["v2_last_parsed"] = parsed
            if soft_accept:
                sc_val = dict(sc_val)
                sc_val["self_check_pass"] = True
                sc_val["soft_accept_applied"] = True
                sc_val["original_fallback_reason"] = sc_val.get("fallback_reason")
                sc_val["fallback_reason"] = None
            return ClusterBoundV2Result(
                success=True,
                raw_text=raw_text,
                parsed=parsed,
                hypotheses=hyps,
                choice_question=cq,
                option_to_cluster_mapping=omap,
                schema_validation=schema_val,
                self_check_validation=sc_val,
                retry_count=retry_count,
                telemetry=tel,
                api_call_count=int(tel.get("api_call_count") or 0),
                token_usage=tel.get("token_usage"),
            )

        last_fb = sc_val.get("fallback_reason") or schema_val.get("fallback_reason") or "cluster_bound_v2_validation_failed"
        if attempt == 0:
            retry_count = 1
            tel["retry_count"] = 1
            continue
        break

    tel["retry_count"] = retry_count
    tel["fallback_reason"] = last_fb
    tel["v2_last_parsed"] = parsed
    return ClusterBoundV2Result(
        success=False,
        raw_text=raw_text,
        parsed=parsed,
        retry_count=retry_count,
        fallback_reason=last_fb,
        schema_validation=tel.get("schema_validation") or {},
        self_check_validation=tel.get("self_check_validation") or {},
        telemetry=tel,
        api_call_count=int(tel.get("api_call_count") or 0),
        token_usage=tel.get("token_usage"),
    )


# Run-wide counter for quickstart forced-delivery smoke (not for performance experiments).
_CB_V2_FORCE_DELIVERY_RUN_COUNT = 0


def reset_cb_v2_force_delivery_run_counter() -> None:
    """Reset per-run forced delivery budget (call once at experiment start)."""
    global _CB_V2_FORCE_DELIVERY_RUN_COUNT
    _CB_V2_FORCE_DELIVERY_RUN_COUNT = 0


def cb_v2_force_delivery_run_count() -> int:
    return int(_CB_V2_FORCE_DELIVERY_RUN_COUNT)


def force_delivery_smoke_enabled(fc32_cfg: dict[str, Any]) -> bool:
    return bool(fc32_cfg.get("cluster_bound_v2_force_delivery_smoke"))


def cb_v2_generation_succeeded(frontier_gen_block: dict[str, Any] | None) -> bool:
    if not isinstance(frontier_gen_block, dict):
        return False
    cb = frontier_gen_block.get("cluster_bound_generation_v2") or {}
    return bool(cb.get("success"))


def try_cb_v2_force_delivery_smoke(
    fc32_cfg: dict[str, Any],
    *,
    frontier_gen_block: dict[str, Any] | None,
    policy_natural_should_ask: bool,
    oos_skip_applied: bool,
    session_forced_count: int,
    original_voi: float,
) -> tuple[bool, dict[str, Any] | None]:
    """Diagnostic-only: force one ask after successful v2 gen when natural VOI gate says no."""
    global _CB_V2_FORCE_DELIVERY_RUN_COUNT

    if not force_delivery_smoke_enabled(fc32_cfg):
        return False, None
    if policy_natural_should_ask or oos_skip_applied:
        return False, None
    if not cb_v2_generation_succeeded(frontier_gen_block):
        return False, None

    max_sess = max(0, int(fc32_cfg.get("max_forced_delivery_per_session", 1) or 1))
    max_run = max(0, int(fc32_cfg.get("max_forced_delivery_total", 2) or 2))
    if session_forced_count >= max_sess:
        return False, None
    if _CB_V2_FORCE_DELIVERY_RUN_COUNT >= max_run:
        return False, None

    _CB_V2_FORCE_DELIVERY_RUN_COUNT += 1
    audit: dict[str, Any] = {
        "forced_delivery_smoke": True,
        "original_question_selection_ask": False,
        "original_voi": float(original_voi),
        "force_reason": "cluster_bound_v2_delivery_smoke",
        "used_for_performance": False,
        "forced_run_index": int(_CB_V2_FORCE_DELIVERY_RUN_COUNT),
        "forced_session_index": int(session_forced_count) + 1,
        "max_forced_delivery_per_session": max_sess,
        "max_forced_delivery_total": max_run,
    }
    return True, audit


# Run-wide counter for entropy-capped ask gate (n5 smoke / pilot).
_CB_V2_ENTROPY_GATE_RUN_COUNT = 0


def reset_cb_v2_entropy_gate_run_counter() -> None:
    global _CB_V2_ENTROPY_GATE_RUN_COUNT
    _CB_V2_ENTROPY_GATE_RUN_COUNT = 0


def entropy_ask_gate_enabled(fc32_cfg: dict[str, Any]) -> bool:
    return bool(fc32_cfg.get("enable_cluster_bound_v2_entropy_ask_gate"))


def resolve_entropy_ask_gate_threshold(
    fc32_cfg: dict[str, Any],
    *,
    belief_entropy: float | None,
    session_entropy_samples: list[float],
) -> float | None:
    """Resolve belief-entropy threshold (fixed config or session-relative median)."""
    mode = str(fc32_cfg.get("entropy_ask_gate_threshold_mode") or "fixed").strip().lower()
    if mode == "session_median" and session_entropy_samples:
        return float(statistics.median(session_entropy_samples))
    raw = fc32_cfg.get("entropy_ask_gate_min_belief_entropy")
    if raw is not None:
        return float(raw)
    if belief_entropy is not None and session_entropy_samples:
        return float(statistics.median(session_entropy_samples))
    return None


def try_cb_v2_entropy_ask_gate(
    fc32_cfg: dict[str, Any],
    *,
    frontier_gen_block: dict[str, Any] | None,
    policy_natural_should_ask: bool,
    oos_skip_applied: bool,
    session_gate_count: int,
    belief_entropy: float | None,
    session_entropy_samples: list[float],
    original_voi: float,
) -> tuple[bool, dict[str, Any] | None]:
    """Supplemental ask when v2 succeeded but VOI gate declined and entropy is high."""
    global _CB_V2_ENTROPY_GATE_RUN_COUNT

    if not entropy_ask_gate_enabled(fc32_cfg):
        return False, None
    if policy_natural_should_ask or oos_skip_applied:
        return False, None
    if not cb_v2_generation_succeeded(frontier_gen_block):
        return False, None
    if belief_entropy is None:
        return False, None

    max_sess = max(0, int(fc32_cfg.get("max_entropy_gate_asks_per_session", 1) or 1))
    max_run = max(0, int(fc32_cfg.get("max_entropy_gate_asks_per_run", 5) or 5))
    if session_gate_count >= max_sess:
        return False, None
    if _CB_V2_ENTROPY_GATE_RUN_COUNT >= max_run:
        return False, None

    threshold = resolve_entropy_ask_gate_threshold(
        fc32_cfg,
        belief_entropy=belief_entropy,
        session_entropy_samples=session_entropy_samples,
    )
    if threshold is None or float(belief_entropy) < float(threshold):
        return False, None

    _CB_V2_ENTROPY_GATE_RUN_COUNT += 1
    audit: dict[str, Any] = {
        "entropy_ask_gate_enabled": True,
        "entropy_ask_gate_triggered": True,
        "original_question_selection_ask": False,
        "original_voi": float(original_voi),
        "belief_entropy": float(belief_entropy),
        "entropy_threshold": float(threshold),
        "per_session_entropy_gate_count": int(session_gate_count) + 1,
        "force_reason": "cluster_bound_v2_entropy_gate",
        "used_target_for_runtime_decision": False,
        "used_future_for_runtime_decision": False,
        "used_for_performance": False,
        "entropy_gate_run_index": int(_CB_V2_ENTROPY_GATE_RUN_COUNT),
        "max_entropy_gate_asks_per_session": max_sess,
        "max_entropy_gate_asks_per_run": max_run,
    }
    return True, audit
