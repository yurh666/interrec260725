"""Unified strict InterRec cascade when-to-ask gate (domain-agnostic).

When ``gate_forward`` is enabled, the cascade pre-gate runs **before** expensive
LLM question generation (direction_translator / frontier hypotheses / option_writer).

Uses only session-so-far online signals — no ``future_*`` / oracle targets.
"""

from __future__ import annotations

from typing import Any

from src.ask.frontier_clustered_v32_pregate import (
    build_budget_state,
    evaluate_frontier_clustered_v32_pre_gate,
)


def strict_cascade_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    ask_cfg = cfg.get("ask") or {}
    return dict(
        ask_cfg.get("strict_cascade_when_to_ask")
        or cfg.get("strict_cascade_when_to_ask")
        or {}
    )


def strict_cascade_enabled(cfg: dict[str, Any]) -> bool:
    sc = strict_cascade_cfg(cfg)
    return bool(sc.get("enabled", False))


def strict_cascade_gate_forward_enabled(cfg: dict[str, Any]) -> bool:
    sc = strict_cascade_cfg(cfg)
    return bool(sc.get("enabled", False)) and bool(sc.get("gate_forward", False))


def merge_fc32_gate_params(fc32_cfg: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Merge fc32 YAML with strict_cascade defaults for pre-gate evaluation."""

    sc = strict_cascade_cfg(cfg)
    merged = dict(fc32_cfg)
    for key in (
        "ask_cooldown_turns",
        "disable_generation_after_consecutive_no_rank_change",
        "max_v32_asks_per_session",
        "max_v32_generations_per_session",
        "pre_gate_max_top10_margin",
        "pre_gate_min_uncertainty",
    ):
        if key not in merged and key in sc:
            merged[key] = sc[key]
    if "ask_cooldown_turns" not in merged:
        merged.setdefault("ask_cooldown_turns", 1)
    if "disable_generation_after_consecutive_no_rank_change" not in merged:
        merged.setdefault("disable_generation_after_consecutive_no_rank_change", 3)
    return merged


def evaluate_strict_cascade_gate(
    *,
    cfg: dict[str, Any],
    fc32_cfg: dict[str, Any],
    turn_idx: int,
    ask_count_so_far: int,
    v32_generation_count_so_far: int,
    last_v32_generation_turn: int,
    consecutive_ask_rank_unchanged: int,
    belief_entropy: float | None,
    sigma_trace: float | None,
    margin_top12: float | None,
    rec_top10_ids: tuple[str, ...],
    softmax_entropy_top10: float | None,
) -> tuple[bool, str, dict[str, Any]]:
    """Return ``(should_enter_llm_pipeline, reason, audit)``."""

    gate_fc32 = merge_fc32_gate_params(fc32_cfg, cfg)
    gate_pass, gate_reason, feats = evaluate_frontier_clustered_v32_pre_gate(
        fc32_cfg=gate_fc32,
        turn_idx=turn_idx,
        ask_count_so_far=ask_count_so_far,
        v32_generation_count_so_far=v32_generation_count_so_far,
        last_v32_generation_turn=last_v32_generation_turn,
        cooldown_turns=int(gate_fc32.get("ask_cooldown_turns", 0) or 0),
        consecutive_ask_rank_unchanged=consecutive_ask_rank_unchanged,
        belief_entropy=belief_entropy,
        sigma_trace=sigma_trace,
        margin_top12=margin_top12,
        rec_top10_ids=rec_top10_ids,
        softmax_entropy_top10=softmax_entropy_top10,
    )
    budget = build_budget_state(
        fc32_cfg=gate_fc32,
        current_turn_idx=turn_idx,
        ask_count_so_far=ask_count_so_far,
        v32_generation_count_so_far=v32_generation_count_so_far,
        last_v32_generation_turn=last_v32_generation_turn,
        cooldown_turns=int(gate_fc32.get("ask_cooldown_turns", 0) or 0),
        consecutive_ask_rank_unchanged=consecutive_ask_rank_unchanged,
        gate_pass=gate_pass,
        gate_reason=gate_reason,
    )
    sc = strict_cascade_cfg(cfg)
    audit: dict[str, Any] = {
        "strict_cascade_enabled": True,
        "gate_forward": bool(sc.get("gate_forward", False)),
        "gate_pass": bool(gate_pass),
        "gate_reason": str(gate_reason),
        "skip_reason": None if gate_pass else str(gate_reason),
        "ask_reason": str(gate_reason) if gate_pass else None,
        "should_enter_llm_pipeline": bool(gate_pass),
        "llm_pipeline_skipped": not bool(gate_pass),
        "features": feats,
        "budget_state": budget,
        "candidate_scope_adaptive": bool(sc.get("candidate_scope_adaptive", False)),
        "question_granularity_adaptive": bool(sc.get("question_granularity_adaptive", False)),
    }
    return bool(gate_pass), str(gate_reason), audit


def build_gate_forward_turn_summary(
    *,
    audit: dict[str, Any],
    do_ask: bool,
    llm_called: bool,
) -> dict[str, Any]:
    """Attach final turn outcome to gate audit for logging."""

    out = dict(audit)
    out["final_do_ask"] = bool(do_ask)
    out["llm_called"] = bool(llm_called)
    if do_ask:
        out["ask_reason"] = out.get("ask_reason") or "pipeline_completed_do_ask"
    elif out.get("llm_pipeline_skipped"):
        out["skip_reason"] = out.get("skip_reason") or out.get("gate_reason")
    else:
        out["skip_reason"] = "pipeline_completed_no_ask"
    return out
