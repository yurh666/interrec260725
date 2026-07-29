"""Runtime-safe rank-safe guard for frontier_clustered_v32 asks (default OFF).

Uses only online signals: scores, top10 lists, belief summaries, ask counters,
hypothesis/option embedding proxies (no oracle labels or ranking metrics).
"""

from __future__ import annotations

import math
from typing import Any

from src.ask.frontier_clustered_v32_pregate import top10_softmax_entropy
from src.ask.frontier_clustered_v32_rank_safe_mcq import (
    compute_v2_option_features,
    pick_conservative_hypothesis_id,
)
from src.embedding.index import EmbeddingIndex


def _f(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _resolve_ternary_action(
    *,
    risk: float,
    conservative_thr: float,
    skip_thr: float,
    allow_conservative: bool,
    conservative_available: bool,
    buffer_mode: bool = False,
    unavailable_fallback: str = "allow_normal",
) -> str:
    """Map risk score to ternary action.

    buffer_mode=True (tradeoff Policy A/B): medium risk → conservative_only when enabled;
    never skip solely because preserve-role option is missing.
    """
    if risk >= skip_thr:
        return "skip_ask"
    if risk >= conservative_thr and allow_conservative:
        if buffer_mode or conservative_available:
            return "conservative_only"
        if unavailable_fallback == "skip_ask":
            return "skip_ask"
        return "allow_normal"
    return "allow_normal"


def _parse_ask_effect_memory(ask_history: dict[str, Any]) -> dict[str, Any]:
    """Summarize recent ask outcomes from session memory (runtime-safe)."""

    hist = ask_history or {}
    ring = hist.get("ask_effect_memory_ring")
    if not isinstance(ring, list):
        ring = []

    recent_nochange = 0
    recent_high_disp = 0
    for entry in ring[-2:]:
        if not isinstance(entry, dict):
            continue
        if bool(entry.get("top10_unchanged")):
            recent_nochange += 1
        if float(entry.get("approximate_displacement") or 0.0) >= 0.35:
            recent_high_disp += 1

    adjustment = "none"
    if recent_nochange >= 2 or (recent_nochange >= 1 and recent_high_disp >= 1):
        adjustment = "more_conservative"
    elif recent_high_disp >= 2:
        adjustment = "more_conservative"
    elif recent_nochange == 0 and recent_high_disp == 0 and ring:
        adjustment = "allow_normal"

    return {
        "ask_effect_memory_enabled": True,
        "recent_ask_nochange_count": recent_nochange,
        "recent_ask_high_displacement_count": recent_high_disp,
        "memory_adjusted_action": adjustment,
        "ask_effect_memory_ring_size": len(ring),
    }


def _resolve_v41_role_aware_action(
    *,
    risk: float,
    conservative_thr: float,
    skip_thr: float,
    allow_conservative: bool,
    conservative_available: bool,
    all_exploratory: bool,
    pres: float,
    disp: float,
    selected_sim: float | None,
    consecutive_ask_count: int,
    memory: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[str, str]:
    """Role-aware v4.1 routing: preserve option + proxies, skip only on compound high risk."""

    pres_allow = float(cfg.get("rank_safe_guard_v41_preservation_allow_min") or 0.50)
    disp_allow = float(cfg.get("rank_safe_guard_v41_displacement_allow_max") or 0.35)
    disp_cons = float(cfg.get("rank_safe_guard_v41_displacement_conservative_min") or 0.28)
    disp_high = float(cfg.get("rank_safe_guard_v41_displacement_high_min") or 0.45)
    sel_sim_low = float(cfg.get("rank_safe_guard_v41_selected_sim_low") or 0.42)
    sel_sim_allow = float(cfg.get("rank_safe_guard_v41_selected_sim_allow_min") or 0.50)

    mem_adj = str(memory.get("memory_adjusted_action") or "none")
    recent_nochange = int(memory.get("recent_ask_nochange_count") or 0)
    recent_high_disp = int(memory.get("recent_ask_high_displacement_count") or 0)

    allow_reasons: list[str] = []
    if conservative_available:
        allow_reasons.append("preserve_option")
    if pres >= pres_allow:
        allow_reasons.append("high_preservation")
    if not all_exploratory:
        allow_reasons.append("not_all_exploratory")
    if disp <= disp_allow:
        allow_reasons.append("low_displacement")
    if selected_sim is None or selected_sim >= sel_sim_allow:
        allow_reasons.append("selected_near_top")

    role_allow_pass = (
        conservative_available
        and pres >= pres_allow
        and not all_exploratory
        and disp <= disp_allow
        and (selected_sim is None or selected_sim >= sel_sim_allow)
        and mem_adj != "more_conservative"
    )

    if role_allow_pass and risk < skip_thr:
        return "allow_normal", "role_allow:" + "+".join(allow_reasons)

    cons_reasons: list[str] = []
    if risk >= conservative_thr:
        cons_reasons.append("medium_risk")
    if all_exploratory:
        cons_reasons.append("all_exploratory")
    if not conservative_available:
        cons_reasons.append("no_preserve_option")
    if selected_sim is not None and selected_sim < sel_sim_low:
        cons_reasons.append("low_selected_sim")
    if disp >= disp_cons:
        cons_reasons.append("displacement_proxy")
    if mem_adj == "more_conservative":
        cons_reasons.append("ask_memory_conservative")

    skip_compound = (
        risk >= skip_thr
        and not conservative_available
        and (all_exploratory or disp >= disp_high)
        and (
            consecutive_ask_count >= 2
            or recent_nochange >= 1
            or recent_high_disp >= 1
        )
    )
    if skip_compound:
        parts = ["high_risk", "no_preserve", "explore_or_high_disp", "compound"]
        if recent_nochange:
            parts.append("recent_nochange")
        if recent_high_disp:
            parts.append("recent_high_disp")
        return "skip_ask", "skip:" + "+".join(parts)

    if risk >= skip_thr + 0.06 and not conservative_available and disp >= disp_high:
        return "skip_ask", "skip:extreme_risk_no_preserve"

    if cons_reasons and allow_conservative:
        return "conservative_only", "conservative:" + "+".join(cons_reasons)

    buffer_mode = bool(cfg.get("rank_safe_guard_conservative_buffer_mode", True))
    if risk >= conservative_thr and allow_conservative and buffer_mode:
        return "conservative_only", "conservative:buffer_risk"

    return "allow_normal", "default_allow"


def _relaxed_explore_guard_enabled(cfg: dict[str, Any]) -> bool:
    policy = str(cfg.get("rank_safe_guard_policy") or "").strip().lower()
    if policy == "v41_relaxed_roleaware_v2":
        return True
    return bool(cfg.get("relaxed_explore_guard_enabled", False))


def _preserve_evidence_score(
    parsed_hypotheses: list[dict[str, Any]] | None,
    pres: float,
    cov_max: float,
) -> float:
    """Proxy for how well options anchor current-top evidence (runtime-safe)."""

    if not isinstance(parsed_hypotheses, list):
        return float(pres)
    has_preserve = any(
        isinstance(h, dict) and str(h.get("rank_safety_role") or "") == "current_top_preserve"
        for h in parsed_hypotheses
    )
    if has_preserve:
        return max(float(pres), float(cov_max), 0.45)
    return float(min(pres, cov_max))


def _top_preserve_similarity_risk(pres: float, cov_max: float) -> float:
    return float(max(0.0, 1.0 - pres, 1.0 - cov_max))


def _role_entropy(role_counts: dict[str, int]) -> float:
    total = sum(max(0, int(v)) for v in role_counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for v in role_counts.values():
        if v <= 0:
            continue
        p = float(v) / float(total)
        ent -= p * math.log(p)
    return float(ent / math.log(max(2, len(role_counts))))


def extract_option_set_rank_safety_features(
    parsed_hypotheses: list[dict[str, Any]] | None,
    generation_block: dict[str, Any] | None = None,
    telemetry: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract MCQ/option-set rank-safety features available before user answer."""

    hyps = [h for h in (parsed_hypotheses or []) if isinstance(h, dict)]
    tel = telemetry or {}
    role_counts = {
        "n_preserve": 0,
        "n_boundary": 0,
        "n_explore": 0,
        "n_unsure": 0,
        "n_unknown": 0,
    }

    evidence_counts = {
        "preserve": 0,
        "boundary": 0,
        "explore": 0,
    }
    declared_counts = {
        "current_top_preserve": 0,
        "boundary_challenge": 0,
        "explore_uncertain": 0,
        "unknown": 0,
    }
    option_texts: list[str] = []
    role_values: list[str] = []
    for h in hyps:
        role = str(h.get("rank_safety_role") or h.get("final_rank_safety_role") or "").strip()
        if role == "current_top_preserve":
            role_counts["n_preserve"] += 1
        elif role == "boundary_challenge":
            role_counts["n_boundary"] += 1
        elif role == "explore_uncertain":
            role_counts["n_explore"] += 1
        elif role == "unsure":
            role_counts["n_unsure"] += 1
        else:
            role_counts["n_unknown"] += 1
            role = "unknown"
        role_values.append(role)

        declared = str(h.get("llm_declared_rank_safety_role") or "").strip()
        if declared in ("current_top_preserve", "boundary_challenge", "explore_uncertain"):
            declared_counts[declared] += 1
        else:
            declared_counts["unknown"] += 1

        if h.get("current_top_evidence_items"):
            evidence_counts["preserve"] += 1
        if h.get("boundary_evidence_items"):
            evidence_counts["boundary"] += 1
        if h.get("uncertain_evidence_items"):
            evidence_counts["explore"] += 1
        option_texts.append(
            str(
                h.get("short_user_facing_option_text")
                or h.get("intent_text")
                or h.get("positive_intent")
                or ""
            )
        )

    option_count = len(hyps)
    has_preserve = role_counts["n_preserve"] > 0
    has_boundary = role_counts["n_boundary"] > 0
    has_explore = role_counts["n_explore"] > 0
    has_all_three = bool(has_preserve and has_boundary and has_explore)
    all_exploratory = bool(option_count > 0 and role_counts["n_explore"] == option_count)
    all_preserve = bool(option_count > 0 and role_counts["n_preserve"] == option_count)
    no_preserve = not has_preserve
    no_boundary = not has_boundary

    pres = float(tel.get("current_top_preservation_score") or 0.0)
    cov_max = float(tel.get("option_current_top_coverage_max") or 0.0)
    disp = float(tel.get("expected_top10_displacement_proxy") or 0.0)
    risk = float(tel.get("rank_damage_risk_proxy") or 0.0)
    selected_sim_raw = tel.get("selected_option_current_top_similarity")
    selected_sim = float(selected_sim_raw) if selected_sim_raw is not None else None
    mem_adj = str(tel.get("memory_adjusted_action") or "none")
    recent_high_disp = int(tel.get("recent_ask_high_displacement_count") or 0)
    consecutive_ask = int(tel.get("consecutive_ask_count") or 0)

    preserve_evidence_score = max(
        pres,
        cov_max,
        min(1.0, evidence_counts["preserve"] / max(1, option_count)),
        0.45 if has_preserve else 0.0,
    )
    boundary_evidence_score = min(1.0, evidence_counts["boundary"] / max(1, option_count))
    explore_evidence_score = min(1.0, evidence_counts["explore"] / max(1, option_count))
    top_preserve_risk = _top_preserve_similarity_risk(pres, cov_max)
    conservative_anchor_score = max(preserve_evidence_score, cov_max if has_preserve else 0.0)

    role_count_for_entropy = {
        "preserve": role_counts["n_preserve"],
        "boundary": role_counts["n_boundary"],
        "explore": role_counts["n_explore"],
        "unsure": role_counts["n_unsure"] + role_counts["n_unknown"],
    }
    entropy = _role_entropy(role_count_for_entropy)
    role_balance_score = min(
        1.0,
        (
            (1.0 if has_preserve else 0.0)
            + (1.0 if has_boundary else 0.0)
            + (1.0 if has_explore else 0.0)
        )
        / 3.0,
    )
    non_empty_texts = [t for t in option_texts if t]
    text_diversity = len({t.lower()[:60] for t in non_empty_texts}) / max(1, len(non_empty_texts))

    max_memory_risk = risk
    if mem_adj == "more_conservative":
        max_memory_risk = max(max_memory_risk, 0.60)
    if recent_high_disp >= 2:
        max_memory_risk = max(max_memory_risk, 0.60)

    role_collapse = bool(option_count > 0 and (all_exploratory or all_preserve or role_balance_score <= 0.34))
    max_explore_risk = risk if has_explore else 0.0
    max_boundary_risk = max(risk * 0.75, disp) if has_boundary else 0.0
    max_preserve_risk = max(risk * 0.5, top_preserve_risk if has_preserve and disp >= 0.45 else 0.0)

    return {
        "option_count": option_count,
        "role_counts": role_counts,
        "declared_role_counts": declared_counts,
        "has_preserve": has_preserve,
        "has_boundary": has_boundary,
        "has_explore": has_explore,
        "has_all_three_roles": has_all_three,
        "all_exploratory": all_exploratory,
        "all_preserve": all_preserve,
        "no_boundary": no_boundary,
        "no_preserve": no_preserve,
        "no_conservative_option": no_preserve,
        "max_explore_risk": round(float(max_explore_risk), 4),
        "mean_explore_risk": round(float(max_explore_risk) / max(1, role_counts["n_explore"]), 4)
        if has_explore
        else 0.0,
        "max_boundary_risk": round(float(max_boundary_risk), 4),
        "max_preserve_risk": round(float(max_preserve_risk), 4),
        "max_expected_displacement": round(disp, 4),
        "mean_expected_displacement": round(disp, 4),
        "max_top_preserve_similarity_risk": round(float(top_preserve_risk), 4),
        "min_top_centroid_similarity": selected_sim,
        "max_memory_risk": round(float(max_memory_risk), 4),
        "memory_adjusted_action": mem_adj,
        "recent_ask_high_displacement_count": recent_high_disp,
        "consecutive_ask_count": consecutive_ask,
        "recent_conservative_count": tel.get("recent_conservative_count"),
        "recent_harmful_like_count": tel.get("recent_harmful_like_count"),
        "role_entropy": round(entropy, 4),
        "option_semantic_diversity": round(float(text_diversity), 4),
        "role_balance_score": round(float(role_balance_score), 4),
        "has_role_collapse": role_collapse,
        "preserve_evidence_score": round(float(preserve_evidence_score), 4),
        "boundary_evidence_score": round(float(boundary_evidence_score), 4),
        "explore_evidence_score": round(float(explore_evidence_score), 4),
        "conservative_anchor_score": round(float(conservative_anchor_score), 4),
        "voi_selected_hypothesis_id": tel.get("voi_selected_hypothesis_id"),
        "voi_selected_role": tel.get("voi_selected_role"),
        "selected_fields_audit_only": True,
        "missing_fields": {
            "selected_option_current_top_similarity": selected_sim_raw is None,
        },
    }


def _resolve_v42_option_set_action(
    *,
    features: dict[str, Any],
    cfg: dict[str, Any],
    allow_conservative: bool,
    schema_unstable: bool = False,
) -> tuple[str, str]:
    """Option-set level guard: route by full MCQ risk profile, not selected role."""

    enable_skip = bool(cfg.get("rank_safe_guard_enable_skip", False))
    option_count = int(features.get("option_count") or 0)
    if enable_skip and (schema_unstable or option_count <= 0):
        return "skip_ask", "option_set_schema_invalid_skip"

    if not allow_conservative:
        return "allow_normal", "option_set_unknown_fallback"

    mem_adj = str(features.get("memory_adjusted_action") or "none")
    recent_high = int(features.get("recent_ask_high_displacement_count") or 0)
    max_memory_risk = float(features.get("max_memory_risk") or 0.0)
    max_explore_risk = float(features.get("max_explore_risk") or 0.0)
    max_disp = float(features.get("max_expected_displacement") or 0.0)
    anchor = float(features.get("conservative_anchor_score") or 0.0)
    preserve_ev = float(features.get("preserve_evidence_score") or 0.0)
    role_entropy = float(features.get("role_entropy") or 0.0)
    boundary_ev = float(features.get("boundary_evidence_score") or 0.0)

    memory_risk_min = float(cfg.get("v42_option_set_memory_risk_min") or 0.60)
    explore_risk_min = float(cfg.get("v42_option_set_explore_risk_min") or 0.60)
    disp_min = float(cfg.get("v42_option_set_displacement_min") or 0.55)
    anchor_min = float(cfg.get("v42_option_set_anchor_min") or 0.45)
    weak_preserve_max = float(cfg.get("v42_option_set_weak_preserve_max") or 0.45)
    entropy_low = float(cfg.get("v42_option_set_role_entropy_low") or 0.35)

    if (
        features.get("all_exploratory")
        and mem_adj == "more_conservative"
        and (max_memory_risk >= memory_risk_min or recent_high >= 2)
    ):
        return "conservative_only", "option_set_all_exploratory_high_memory_risk"

    if (
        features.get("has_explore")
        and (features.get("no_preserve") or anchor <= anchor_min)
        and max_explore_risk >= explore_risk_min
    ):
        return "conservative_only", "option_set_explore_without_preserve_anchor"

    if (
        features.get("has_explore")
        and max_disp >= disp_min
        and max_explore_risk >= explore_risk_min
        and preserve_ev <= weak_preserve_max
    ):
        return "conservative_only", "option_set_high_displacement_explore"

    if (
        features.get("has_explore")
        and mem_adj == "more_conservative"
        and max_memory_risk >= memory_risk_min
        and recent_high >= 2
    ):
        return "conservative_only", "option_set_role_unbalanced_high_risk"

    if (
        (role_entropy <= entropy_low or features.get("has_role_collapse"))
        and max_disp >= disp_min
        and mem_adj == "more_conservative"
    ):
        return "conservative_only", "option_set_role_unbalanced_high_risk"

    if (
        features.get("no_boundary")
        and boundary_ev >= float(cfg.get("v42_option_set_boundary_needed_min") or 0.50)
        and max_disp >= disp_min
    ):
        return "conservative_only", "option_set_no_boundary_when_needed"

    if (
        features.get("has_all_three_roles")
        and anchor >= anchor_min
        and max_disp < disp_min
        and max_explore_risk < explore_risk_min
        and mem_adj != "more_conservative"
    ):
        return "allow_normal", "option_set_allow_balanced_low_risk"

    return "allow_normal", "option_set_allow_balanced_low_risk"


def _resolve_v42_1_option_set_action(
    *,
    features: dict[str, Any],
    cfg: dict[str, Any],
    allow_conservative: bool,
    schema_unstable: bool = False,
) -> tuple[str, str]:
    """V42.1: v42 option-set guard plus a narrow high-displacement tightening."""

    action, reason = _resolve_v42_option_set_action(
        features=features,
        cfg=cfg,
        allow_conservative=allow_conservative,
        schema_unstable=schema_unstable,
    )
    if action != "allow_normal" or not allow_conservative:
        return action, reason

    max_disp = float(features.get("max_expected_displacement") or 0.0)
    anchor = float(features.get("conservative_anchor_score") or 0.0)
    has_explore = bool(features.get("has_explore"))
    has_preserve = bool(features.get("has_preserve"))
    has_boundary = bool(features.get("has_boundary"))
    disp_min = float(cfg.get("v42_1_high_displacement_min") or 0.58)
    anchor_max = float(cfg.get("v42_1_moderate_anchor_max") or 0.60)

    if has_explore and has_preserve and has_boundary and max_disp >= disp_min and anchor <= anchor_max:
        return "conservative_only", "option_set_high_displacement_moderate_anchor"

    return action, reason


def _explore_harmful_risk_v2(
    *,
    disp: float,
    pres: float,
    cov_max: float,
    selected_sim: float | None,
    risk: float,
    conservative_thr: float,
    memory: dict[str, Any],
    cfg: dict[str, Any],
    parsed_hypotheses: list[dict[str, Any]] | None = None,
    consecutive_ask_count: int = 0,
    all_exploratory: bool = False,
    conservative_available: bool = False,
) -> tuple[bool, str | None]:
    """Composite explore harmful-risk (no oracle labels)."""

    disp_min = float(cfg.get("relaxed_explore_disp_min") or 0.45)
    sim_max = float(cfg.get("relaxed_explore_top_sim_max") or 0.35)
    pres_ev_max = float(cfg.get("relaxed_explore_preserve_evidence_max") or 0.30)
    pres_risk_min = float(cfg.get("relaxed_explore_preserve_risk_min") or 0.45)
    mem_high_disp_min = int(cfg.get("relaxed_explore_memory_high_disp_min") or 2)
    mem_risk_min = float(cfg.get("relaxed_explore_memory_risk_min") or 0.60)
    consec_ask_min = int(cfg.get("relaxed_explore_consecutive_ask_min") or 1)
    pres_evidence = _preserve_evidence_score(parsed_hypotheses, pres, cov_max)
    top_pres_risk = _top_preserve_similarity_risk(pres, cov_max)
    sim_low = selected_sim is None or float(selected_sim) <= sim_max
    mem_adj = str(memory.get("memory_adjusted_action") or "none")
    recent_high_disp = int(memory.get("recent_ask_high_displacement_count") or 0)

    if disp >= disp_min and sim_low and pres_evidence <= pres_ev_max:
        return True, "explore_high_displacement_low_top_similarity"
    if disp >= disp_min and top_pres_risk >= pres_risk_min:
        return True, "explore_high_displacement_weak_preserve"
    if selected_sim is None and disp >= disp_min and top_pres_risk >= pres_risk_min:
        return True, "explore_high_displacement_weak_preserve"
    if (
        disp < disp_min
        and all_exploratory
        and not conservative_available
        and mem_adj == "more_conservative"
        and recent_high_disp >= mem_high_disp_min
        and risk >= mem_risk_min
        and consecutive_ask_count >= consec_ask_min
    ):
        return True, "explore_high_displacement_weak_preserve"
    return False, None


def _selected_hypothesis_role(
    parsed_hypotheses: list[dict[str, Any]] | None,
    selected_hypothesis_id: str | None,
) -> str:
    if not selected_hypothesis_id or not isinstance(parsed_hypotheses, list):
        return ""
    for h in parsed_hypotheses:
        if isinstance(h, dict) and str(h.get("hypothesis_id") or "") == str(selected_hypothesis_id):
            return str(h.get("rank_safety_role") or h.get("final_rank_safety_role") or "")
    return ""


def _resolve_v41_relaxed_roleaware_action(
    *,
    risk: float,
    conservative_thr: float,
    skip_thr: float,
    allow_conservative: bool,
    conservative_available: bool,
    all_exploratory: bool,
    pres: float,
    disp: float,
    selected_sim: float | None,
    selected_role: str,
    consecutive_ask_count: int,
    memory: dict[str, Any],
    cfg: dict[str, Any],
    schema_unstable: bool = False,
    cov_max: float = 0.0,
    v2_explore_guard: bool = False,
    parsed_hypotheses: list[dict[str, Any]] | None = None,
) -> tuple[str, str]:
    """Relaxed v4.1 routing: role-first defaults; conservative only on clear high-risk signals."""

    enable_skip = bool(cfg.get("rank_safe_guard_enable_skip", False))
    allow_boundary = bool(cfg.get("relaxed_allow_boundary_default", True))
    allow_explore = bool(cfg.get("relaxed_allow_explore_default", True))
    preserve_high_risk_only = bool(cfg.get("relaxed_preserve_requires_high_risk", True))

    disp_high = float(cfg.get("rank_safe_guard_v41_displacement_high_min") or 0.45)
    disp_cons = float(cfg.get("rank_safe_guard_v41_displacement_conservative_min") or 0.28)
    sel_sim_low = float(cfg.get("rank_safe_guard_v41_selected_sim_low") or 0.42)
    mem_adj = str(memory.get("memory_adjusted_action") or "none")
    recent_high_disp = int(memory.get("recent_ask_high_displacement_count") or 0)

    role = (selected_role or "").strip().lower()
    if not role and all_exploratory:
        role = "explore_uncertain"

    disp_preserve_min = float(cfg.get("relaxed_preserve_disp_min") or 0.55)
    preserve_risk_margin = float(cfg.get("relaxed_preserve_risk_margin") or 0.12)

    high_risk = risk >= skip_thr - 0.12 or (
        risk >= conservative_thr + preserve_risk_margin and disp >= disp_high
    )
    medium_risk = conservative_thr <= risk < skip_thr - 0.12
    preserve_risk_high = (
        risk >= conservative_thr + preserve_risk_margin + 0.08
        or (
            risk >= conservative_thr + preserve_risk_margin
            and disp >= disp_preserve_min
            and (selected_sim is None or selected_sim < sel_sim_low)
        )
        or (
            mem_adj == "more_conservative"
            and risk >= conservative_thr
            and disp >= disp_preserve_min
        )
    )

    if enable_skip:
        skip_compound = (
            schema_unstable
            or (
                risk >= skip_thr
                and not conservative_available
                and all_exploratory
                and consecutive_ask_count >= 2
            )
        )
        if skip_compound:
            return "skip_ask", "relaxed_skip:schema_or_compound_high_risk"

    if role == "boundary_challenge":
        if allow_boundary and not high_risk:
            return "allow_normal", "relaxed_allow:boundary_default"
        if disp >= disp_high and risk >= conservative_thr and allow_conservative:
            return "conservative_only", "relaxed_conservative:boundary_high_disp"
        return "allow_normal", "relaxed_allow:boundary_medium_risk"

    if role == "explore_uncertain":
        if v2_explore_guard:
            harmful, harm_code = _explore_harmful_risk_v2(
                disp=disp,
                pres=pres,
                cov_max=cov_max,
                selected_sim=selected_sim,
                risk=risk,
                conservative_thr=conservative_thr,
                memory=memory,
                cfg=cfg,
                parsed_hypotheses=parsed_hypotheses,
                consecutive_ask_count=consecutive_ask_count,
                all_exploratory=all_exploratory,
                conservative_available=conservative_available,
            )
            if harmful and allow_conservative:
                code = harm_code or "explore_high_displacement_weak_preserve"
                return "conservative_only", f"relaxed_v2_conservative:{code}"
            return "allow_normal", "explore_allow_default"

        if allow_explore and not (high_risk and recent_high_disp >= 1 and disp >= disp_high):
            if medium_risk and allow_conservative and conservative_available and disp >= disp_cons:
                return "allow_normal", "relaxed_allow:explore_light_risk"
            return "allow_normal", "relaxed_allow:explore_default"
        if high_risk and disp >= disp_high and allow_conservative:
            return "conservative_only", "relaxed_conservative:explore_high_uncertainty_disp"
        return "allow_normal", "relaxed_allow:explore_no_compound"

    if role == "current_top_preserve":
        if preserve_high_risk_only:
            if preserve_risk_high and allow_conservative:
                return "conservative_only", "relaxed_conservative:preserve_high_risk"
            return "allow_normal", "relaxed_allow:preserve_low_or_medium_risk"
        if medium_risk and allow_conservative:
            return "conservative_only", "relaxed_conservative:preserve_medium_risk"
        return "allow_normal", "relaxed_allow:preserve_default"

    if role in ("unsure", "unknown", ""):
        if schema_unstable and allow_conservative and conservative_available:
            return "conservative_only", "relaxed_conservative:unsure_schema_unstable"
        if not conservative_available and risk >= conservative_thr and allow_conservative:
            return "conservative_only", "relaxed_conservative:unsure_no_preserve_option"
        if high_risk and allow_conservative and conservative_available:
            return "conservative_only", "relaxed_conservative:unsure_high_risk"
        return "allow_normal", "relaxed_allow:unsure_default"

    if high_risk and allow_conservative and conservative_available:
        return "conservative_only", f"relaxed_conservative:role_{role or 'other'}_high_risk"
    return "allow_normal", f"relaxed_allow:role_{role or 'other'}_default"


def build_ask_effect_memory_entry(
    *,
    pre_ask_top10: list[str] | tuple[str, ...],
    post_ask_top10: list[str] | tuple[str, ...],
    approximate_displacement: float | None = None,
) -> dict[str, Any]:
    """Record one ask outcome for lightweight session memory (runtime-safe)."""

    prev = [str(x) for x in pre_ask_top10[:10]]
    post = [str(x) for x in post_ask_top10[:10]]
    top10_unchanged = bool(prev and post and prev == post)
    jaccard = 0.0
    if prev and post:
        sa, sb = set(prev), set(post)
        union = len(sa | sb) or 1
        jaccard = len(sa & sb) / union
    disp = approximate_displacement
    if disp is None:
        disp = 0.0 if top10_unchanged else max(0.0, 1.0 - jaccard)
    return {
        "top10_unchanged": top10_unchanged,
        "top10_stability_jaccard": round(jaccard, 4),
        "approximate_displacement": round(float(disp), 4),
    }


def rank_safe_guard_scope_v32_only(cfg: dict[str, Any] | None) -> bool:
    scope = str((cfg or {}).get("rank_safe_guard_scope") or "").strip().lower()
    return scope in ("v32_only", "v32_generation_turn", "v32_generation_turns")


def build_out_of_scope_non_v32_guard_audit(
    *,
    cfg: dict[str, Any] | None,
    reason: str,
    turn: int | None = None,
) -> dict[str, Any]:
    """Telemetry when v41 guard does not apply (contrastive_v2 / lazy cooldown path)."""

    enabled = bool((cfg or {}).get("rank_safe_guard_enabled", False))
    return {
        "rank_safe_guard_enabled": enabled,
        "rank_safe_guard_log_only": bool((cfg or {}).get("rank_safe_guard_log_only", True)),
        "rank_safe_guard_scope": "out_of_scope_non_v32",
        "rank_safe_guard_scope_reason": reason,
        "v41_action_applied": False,
        "non_v32_original_question_preserved": True,
        "rank_safe_action": "out_of_scope_non_v32",
        "rank_safe_action_v41": None,
        "should_block_ask": False,
        "should_conservative_only": False,
        "would_block_ask": False,
        "would_conservative_only": False,
        "guard_reason": f"out_of_scope:{reason}",
        "role_aware_reason": None,
        "frontier_generation_exists": False,
        "turn": turn,
        "rank_safe_guard_version": str((cfg or {}).get("rank_safe_guard_version", "v41")),
        "v41_role_aware_enabled": True,
    }


def compute_rank_safe_guard(
    *,
    turn_log_context: dict[str, Any] | None,
    recommendation_scores: list[float],
    previous_top10: list[str] | tuple[str, ...] | None,
    current_top10: list[str] | tuple[str, ...] | None,
    belief_state: dict[str, Any] | None,
    ask_history: dict[str, Any] | None,
    config: dict[str, Any] | None,
    frontier_generation_exists: bool = False,
    hyp_vectorized: list[dict[str, Any]] | None = None,
    parsed_hypotheses: list[dict[str, Any]] | None = None,
    embedding_index: EmbeddingIndex | None = None,
) -> dict[str, Any]:
    """Return telemetry + ternary action for a would-be v32 ask."""

    cfg = config or {}
    ctx = turn_log_context or {}
    hist = ask_history or {}
    belief = belief_state or {}

    enabled = bool(cfg.get("rank_safe_guard_enabled", False))
    log_only = bool(cfg.get("rank_safe_guard_log_only", True))
    action_mode = str(cfg.get("rank_safe_guard_action_mode", "binary"))
    legacy_mode = str(cfg.get("rank_safe_guard_mode", "skip_ask"))
    allow_conservative = bool(cfg.get("rank_safe_guard_allow_conservative_question", True))
    guard_version = str(cfg.get("rank_safe_guard_version", "v4")).lower()
    v41_enabled = guard_version in ("v41", "4.1") or bool(cfg.get("rank_safe_guard_v41_role_aware_enabled", False))
    ask_memory_enabled = bool(cfg.get("rank_safe_guard_ask_effect_memory_enabled", v41_enabled))

    selected_hid = ctx.get("selected_hypothesis_id")
    if not selected_hid and isinstance(ctx.get("selected_hypothesis_ids"), list) and ctx["selected_hypothesis_ids"]:
        selected_hid = str(ctx["selected_hypothesis_ids"][0])

    scores = [float(s) for s in (recommendation_scores or [])[:10]]
    prev = [str(x) for x in (previous_top10 or [])[:10]]
    cur = [str(x) for x in (current_top10 or [])[:10]]

    margin12 = scores[0] - scores[1] if len(scores) >= 2 else 0.0
    margin10 = scores[0] - scores[9] if len(scores) >= 10 else margin12
    entropy10 = float(top10_softmax_entropy(scores)) if scores else 0.0

    prev_margin12_raw = hist.get("prev_margin12", hist.get("prev_turn_margin12"))
    prev_margin12 = float(prev_margin12_raw) if prev_margin12_raw is not None else None

    top10_unchanged = bool(prev and cur and prev == cur)
    jaccard = 0.0
    if prev and cur:
        sa, sb = set(prev), set(cur)
        union = len(sa | sb) or 1
        jaccard = len(sa & sb) / union
    top10_stability_score = 1.0 if top10_unchanged else jaccard

    recent_ask_count = int(hist.get("ask_count_so_far", hist.get("recent_ask_count", 0)) or 0)
    consecutive_ask_count = int(
        hist.get("consecutive_ask_no_rank_change", hist.get("consecutive_ask_count", 0)) or 0
    )
    consec_limit = int(cfg.get("rank_safe_guard_consecutive_ask_limit", 2) or 2)

    belief_entropy = _f(belief.get("entropy")) if belief.get("entropy") is not None else None

    margin_thr = float(cfg.get("rank_safe_guard_margin_threshold") or 0.045)
    prev_margin_thr = float(cfg.get("rank_safe_guard_prev_margin_threshold") or 0.024)
    stab_thr = float(cfg.get("rank_safe_guard_stability_threshold") or 0.95)
    skip_thr = float(cfg.get("rank_safe_guard_skip_threshold") or cfg.get("rank_safe_guard_risk_threshold") or 0.88)
    conservative_thr = float(
        cfg.get("rank_safe_guard_conservative_threshold") or max(0.35, skip_thr - 0.35),
    )
    buffer_mode = bool(cfg.get("rank_safe_guard_conservative_buffer_mode", True))
    unavailable_fallback = str(cfg.get("rank_safe_guard_unavailable_conservative_fallback", "allow_normal"))
    tradeoff_policy = str(cfg.get("rank_safe_guard_tradeoff_policy", "B"))
    risk_scale = float(cfg.get("rank_safe_guard_risk_scale", 1.0))
    recent_soft_cap = int(cfg.get("rank_safe_guard_recent_ask_soft_cap", 3) or 3)

    risk = 0.0
    reasons: list[str] = []

    if prev_margin12 is not None and prev_margin12 <= prev_margin_thr and recent_ask_count >= recent_soft_cap:
        risk += 0.28
        reasons.append(f"low_prev_margin<={prev_margin_thr:.4f}")
    if margin12 <= margin_thr and recent_ask_count >= max(2, recent_soft_cap - 1):
        risk += 0.18
        reasons.append(f"low_current_margin<={margin_thr:.4f}")
    if top10_stability_score >= stab_thr:
        risk += 0.14
        reasons.append(f"stable_top10>={stab_thr:.2f}")
    if consecutive_ask_count >= consec_limit and top10_stability_score >= 0.8:
        risk += 0.12
        reasons.append(f"consecutive_no_rank_change>={consec_limit}")
    if not top10_unchanged and jaccard < 0.55 and recent_ask_count >= 2:
        w_vol = 0.06 if tradeoff_policy in ("A", "B") else 0.1
        risk += w_vol
        reasons.append("volatile_top10_after_asks")
    if entropy10 < float(cfg.get("rank_safe_guard_max_entropy", 2.0) or 2.0):
        risk += 0.06
        reasons.append("low_top10_entropy")

    ph = parsed_hypotheses
    if ph is None and isinstance(ctx.get("parsed_hypotheses"), list):
        ph = ctx["parsed_hypotheses"]

    v2 = compute_v2_option_features(
        index=embedding_index,
        current_top10=cur,
        hyp_vectorized=hyp_vectorized,
        parsed_hypotheses=ph,
        recommendation_scores=scores,
        selected_hypothesis_id=str(selected_hid) if selected_hid else None,
    )

    pres = float(v2.get("current_top_preservation_score") or 0.0)
    cov_max = float(v2.get("option_current_top_coverage_max") or 0.0)
    disp = float(v2.get("expected_top10_displacement_proxy") or 0.0)
    conservative_available = bool(v2.get("conservative_option_available"))
    all_exploratory = bool(v2.get("all_options_exploratory_risk"))
    selected_sim_raw = v2.get("selected_option_current_top_similarity")
    selected_sim = float(selected_sim_raw) if selected_sim_raw is not None else None

    memory_info: dict[str, Any] = {}
    if ask_memory_enabled:
        memory_info = _parse_ask_effect_memory(hist)
    else:
        memory_info = {
            "ask_effect_memory_enabled": False,
            "recent_ask_nochange_count": 0,
            "recent_ask_high_displacement_count": 0,
            "memory_adjusted_action": "none",
        }

    pres_thr = float(cfg.get("rank_safe_guard_preservation_min", 0.55))
    cov_thr = float(cfg.get("rank_safe_guard_coverage_min", 0.42))
    disp_thr = float(cfg.get("rank_safe_guard_displacement_max", 0.38))

    if pres < pres_thr and recent_ask_count >= 2:
        risk += 0.22
        reasons.append(f"low_preservation<{pres_thr:.2f}")
    if cov_max < cov_thr:
        risk += 0.16
        reasons.append(f"low_top_coverage<{cov_thr:.2f}")
    if disp > disp_thr:
        risk += 0.18
        reasons.append(f"high_displacement>{disp_thr:.2f}")
    if not conservative_available and recent_ask_count >= 2:
        w_nc = 0.04 if buffer_mode else 0.14
        risk += w_nc
        reasons.append("no_conservative_option")
    if all_exploratory:
        w_ex = 0.06 if buffer_mode else 0.12
        risk += w_ex
        reasons.append("all_options_exploratory")

    risk = float(min(1.0, risk * risk_scale))

    role_aware_reason = ""
    rank_safe_action_v41: str | None = None
    guard_policy = str(cfg.get("rank_safe_guard_policy", "") or "").strip().lower()
    selected_role = _selected_hypothesis_role(ph, str(selected_hid) if selected_hid else None)
    option_set_features = extract_option_set_rank_safety_features(
        ph,
        ctx.get("frontier_clustered_v32_generation")
        if isinstance(ctx.get("frontier_clustered_v32_generation"), dict)
        else None,
        {
            **v2,
            **memory_info,
            "rank_damage_risk_proxy": risk,
            "consecutive_ask_count": consecutive_ask_count,
            "voi_selected_hypothesis_id": str(selected_hid) if selected_hid else None,
            "voi_selected_role": selected_role or None,
        },
        cfg,
    )
    schema_unstable = False
    if isinstance(ph, list):
        for h in ph:
            if isinstance(h, dict) and (h.get("role_assignment_conflict") or h.get("rank_safety_role_schema_warnings")):
                schema_unstable = True
                break

    if action_mode == "ternary" and v41_enabled:
        if guard_policy == "v42_option_set_guard":
            rank_safe_action, role_aware_reason = _resolve_v42_option_set_action(
                features=option_set_features,
                cfg=cfg,
                allow_conservative=allow_conservative,
                schema_unstable=schema_unstable,
            )
        elif guard_policy == "v42_1_option_set_guard":
            rank_safe_action, role_aware_reason = _resolve_v42_1_option_set_action(
                features=option_set_features,
                cfg=cfg,
                allow_conservative=allow_conservative,
                schema_unstable=schema_unstable,
            )
        elif guard_policy in ("v41_relaxed_roleaware", "v41_relaxed_roleaware_v2"):
            rank_safe_action, role_aware_reason = _resolve_v41_relaxed_roleaware_action(
                risk=risk,
                conservative_thr=conservative_thr,
                skip_thr=skip_thr,
                allow_conservative=allow_conservative,
                conservative_available=conservative_available,
                all_exploratory=all_exploratory,
                pres=pres,
                disp=disp,
                selected_sim=selected_sim,
                selected_role=selected_role,
                consecutive_ask_count=consecutive_ask_count,
                memory=memory_info,
                cfg=cfg,
                schema_unstable=schema_unstable,
                cov_max=cov_max,
                v2_explore_guard=_relaxed_explore_guard_enabled(cfg),
                parsed_hypotheses=ph,
            )
        else:
            rank_safe_action, role_aware_reason = _resolve_v41_role_aware_action(
                risk=risk,
                conservative_thr=conservative_thr,
                skip_thr=skip_thr,
                allow_conservative=allow_conservative,
                conservative_available=conservative_available,
                all_exploratory=all_exploratory,
                pres=pres,
                disp=disp,
                selected_sim=selected_sim,
                consecutive_ask_count=consecutive_ask_count,
                memory=memory_info,
                cfg=cfg,
            )
        if not bool(cfg.get("rank_safe_guard_enable_skip", False)) and rank_safe_action == "skip_ask":
            rank_safe_action = (
                "conservative_only"
                if allow_conservative and conservative_available
                else "allow_normal"
            )
            role_aware_reason = f"relaxed_skip_disabled→{rank_safe_action}"
        rank_safe_action_v41 = rank_safe_action
    elif action_mode == "ternary":
        rank_safe_action = _resolve_ternary_action(
            risk=risk,
            conservative_thr=conservative_thr,
            skip_thr=skip_thr,
            allow_conservative=allow_conservative,
            conservative_available=conservative_available,
            buffer_mode=buffer_mode,
            unavailable_fallback=unavailable_fallback,
        )
    else:
        rank_safe_action = "skip_ask" if risk >= skip_thr else "allow_normal"

    would_block = bool(enabled and rank_safe_action == "skip_ask")
    would_conservative = bool(enabled and rank_safe_action == "conservative_only")

    if not enabled:
        should_block_ask = False
        should_conservative_only = False
        guard_reason = "disabled"
        fallback_action = "allow"
        rank_safe_action = "allow_normal"
    elif log_only:
        should_block_ask = False
        should_conservative_only = False
        guard_reason = f"log_only:{rank_safe_action}:" + "+".join(reasons) if reasons else f"log_only:{rank_safe_action}"
        fallback_action = "allow"
    else:
        should_block_ask = rank_safe_action == "skip_ask"
        should_conservative_only = rank_safe_action == "conservative_only"
        guard_reason = rank_safe_action + ":" + "+".join(reasons) if reasons else rank_safe_action
        if role_aware_reason:
            guard_reason = f"{role_aware_reason}|risk={risk:.3f}"
        if rank_safe_action == "skip_ask":
            fallback_action = "skip_ask"
        elif rank_safe_action == "conservative_only":
            fallback_action = "conservative_only"
        else:
            fallback_action = "allow"

    conservative_hid = pick_conservative_hypothesis_id(
        ph,
        hyp_vectorized,
        index=embedding_index,
        current_top10=cur,
    )

    would_block_ask = bool(
        enabled
        and (
            rank_safe_action == "skip_ask"
            or (action_mode != "ternary" and risk >= skip_thr)
        )
    )

    return {
        "rank_safe_guard_enabled": enabled,
        "rank_safe_guard_log_only": log_only,
        "rank_safe_guard_action_mode": action_mode,
        "rank_safe_action": rank_safe_action,
        "should_block_ask": bool(should_block_ask and not log_only),
        "should_conservative_only": bool(should_conservative_only and not log_only),
        "would_block_ask": would_block or (enabled and risk >= skip_thr and action_mode != "ternary"),
        "would_conservative_only": would_conservative,
        "conservative_hypothesis_id": conservative_hid,
        "guard_reason": guard_reason,
        "rank_damage_risk_proxy": round(risk, 4),
        "top10_stability_score": round(top10_stability_score, 4),
        "score_margin_at_10": round(float(margin10), 6),
        "score_margin_top1_minus_top2": round(float(margin12), 6),
        "prev_turn_margin_top1_minus_top2": (
            round(float(prev_margin12), 6) if prev_margin12 is not None else None
        ),
        "top10_softmax_entropy": round(entropy10, 4),
        "top10_unchanged_vs_previous": top10_unchanged,
        "recent_ask_count": recent_ask_count,
        "consecutive_ask_count": consecutive_ask_count,
        "belief_entropy": belief_entropy,
        "frontier_generation_exists": bool(frontier_generation_exists),
        "fallback_action": fallback_action,
        "turn": ctx.get("turn"),
        "skip_threshold_used": skip_thr,
        "conservative_threshold_used": conservative_thr,
        "margin_threshold_used": margin_thr,
        "prev_margin_threshold_used": prev_margin_thr,
        "stability_threshold_used": stab_thr,
        "conservative_buffer_mode": buffer_mode,
        "tradeoff_policy": tradeoff_policy,
        "rank_safe_guard_version": guard_version,
        "v41_role_aware_enabled": bool(v41_enabled),
        "rank_safe_action_v41": rank_safe_action_v41,
        "role_aware_reason": role_aware_reason or None,
        "rank_safe_guard_policy": guard_policy or None,
        "selected_rank_safety_role": selected_role or None,
        "reason_code": (role_aware_reason.split(":")[0] if role_aware_reason else None),
        "option_set_rank_safety_features": option_set_features,
        "option_set_guard_core_signal": bool(guard_policy == "v42_option_set_guard"),
        **memory_info,
        **v2,
    }
