"""Runtime-safe strict routing repair for frontier_clustered_v32 lazy ask path."""

from __future__ import annotations

from typing import Any

RESCUE_ENTROPY_MIN_DEFAULT = 175.0
RESCUE_MARGIN_MAX_DEFAULT = 0.015
COOLDOWN_ENTROPY_MIN_DEFAULT = 178.0
COOLDOWN_MARGIN_MAX_DEFAULT = 0.015


def repair_flags(fc32_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "near_miss_enabled": bool(fc32_cfg.get("enable_near_miss_rescue_ask_strict", False)),
        "cooldown_reroute_enabled": bool(fc32_cfg.get("enable_cooldown_high_risk_reroute_strict", False)),
        "max_rescue_per_session": int(fc32_cfg.get("near_miss_rescue_max_per_session", 1) or 1),
        "max_cooldown_reroute_per_session": int(fc32_cfg.get("cooldown_reroute_max_per_session", 1) or 1),
        "max_total_repair_per_session": int(fc32_cfg.get("max_total_repair_actions_per_session", 2) or 2),
        "rescue_entropy_min": float(fc32_cfg.get("near_miss_rescue_entropy_min", RESCUE_ENTROPY_MIN_DEFAULT)),
        "rescue_margin_max": float(fc32_cfg.get("near_miss_rescue_margin_max", RESCUE_MARGIN_MAX_DEFAULT)),
        "cooldown_entropy_min": float(fc32_cfg.get("cooldown_reroute_entropy_min", COOLDOWN_ENTROPY_MIN_DEFAULT)),
        "cooldown_margin_max": float(fc32_cfg.get("cooldown_reroute_margin_max", COOLDOWN_MARGIN_MAX_DEFAULT)),
        "rescue_min_turn": int(fc32_cfg.get("near_miss_rescue_min_turn", 3) or 3),
        "rescue_max_ask_count": int(fc32_cfg.get("near_miss_rescue_max_ask_count_so_far", 1) or 1),
    }


def can_apply_repair(state: dict[str, Any], flags: dict[str, Any]) -> bool:
    return int(state.get("repair_total_count", 0) or 0) < int(flags["max_total_repair_per_session"])


def high_uncertainty_proxy(
    *,
    belief_entropy: float | None,
    margin_top12: float | None,
    entropy_min: float,
    margin_max: float,
) -> bool:
    if belief_entropy is None or margin_top12 is None:
        return False
    return float(belief_entropy) >= entropy_min and float(margin_top12) <= margin_max


def evaluate_cooldown_reroute_strict(
    *,
    fc32_cfg: dict[str, Any],
    state: dict[str, Any],
    gate_pass: bool,
    gate_reason: str,
    ask_count_so_far: int,
    belief_entropy: float | None,
    margin_top12: float | None,
) -> tuple[bool, list[str]]:
    flags = repair_flags(fc32_cfg)
    if not flags["cooldown_reroute_enabled"]:
        return False, []
    if gate_pass or gate_reason != "cooldown_active":
        return False, []
    if int(state.get("cooldown_reroute_count", 0) or 0) >= flags["max_cooldown_reroute_per_session"]:
        return False, ["cooldown_reroute_cap_reached"]
    if not can_apply_repair(state, flags):
        return False, ["max_total_repair_reached"]
    if ask_count_so_far != 0:
        return False, ["not_first_ask_in_session"]
    if not high_uncertainty_proxy(
        belief_entropy=belief_entropy,
        margin_top12=margin_top12,
        entropy_min=flags["cooldown_entropy_min"],
        margin_max=flags["cooldown_margin_max"],
    ):
        return False, ["high_risk_proxy_false"]
    return True, [
        "cooldown_active",
        "contrastive_v2_would_fallback",
        "first_ask_cooldown_oos",
        "high_risk_proxy",
    ]


def evaluate_near_miss_rescue_strict(
    *,
    fc32_cfg: dict[str, Any],
    state: dict[str, Any],
    turn_idx: int,
    ask_count_so_far: int,
    do_ask: bool,
    gate_pass: bool | None,
    gate_reason: str | None,
    voi_eval: float,
    original_should_ask: bool,
    belief_entropy: float | None,
    margin_top12: float | None,
) -> tuple[bool, list[str]]:
    flags = repair_flags(fc32_cfg)
    if not flags["near_miss_enabled"]:
        return False, []
    if do_ask:
        return False, []
    if int(state.get("near_miss_rescue_count", 0) or 0) >= flags["max_rescue_per_session"]:
        return False, ["rescue_cap_reached"]
    if not can_apply_repair(state, flags):
        return False, ["max_total_repair_reached"]
    if ask_count_so_far > flags["rescue_max_ask_count"]:
        return False, ["session_ask_count_gt_cap"]
    if turn_idx < flags["rescue_min_turn"]:
        return False, ["turn_lt_min"]
    if gate_pass is False or (gate_reason not in (None, "pass")):
        return False, [f"pre_gate_block:{gate_reason}"]
    voi_skip = voi_eval < 0 or not original_should_ask
    if not voi_skip:
        return False, ["voi_not_negative_and_not_midask_skip"]
    if not high_uncertainty_proxy(
        belief_entropy=belief_entropy,
        margin_top12=margin_top12,
        entropy_min=flags["rescue_entropy_min"],
        margin_max=flags["rescue_margin_max"],
    ):
        return False, ["uncertainty_proxy_low"]
    return True, [
        "session_ask_count_lte_cap",
        "turn_gte_min",
        "pre_gate_pass",
        "voi_negative_or_midask_skip",
        "high_uncertainty_proxy",
    ]


def build_cooldown_reroute_audit(
    *,
    turn_idx: int,
    reasons: list[str],
    state: dict[str, Any],
    gate_reason: str,
    belief_entropy: float | None,
    margin_top12: float | None,
    ask_count_so_far: int,
) -> dict[str, Any]:
    rescue_n = int(state.get("near_miss_rescue_count", 0) or 0)
    cool_n = int(state.get("cooldown_reroute_count", 0) or 0) + 1
    total = rescue_n + cool_n
    return {
        "repair_action": "cooldown_reroute",
        "repair_reason": reasons,
        "repair_used_count": {
            "near_miss_rescue": rescue_n,
            "cooldown_reroute": cool_n,
            "total": total,
        },
        "turn": int(turn_idx),
        "cooldown_active": True,
        "original_path": "contrastive_v2",
        "final_path": "v32_generation",
        "original_decision": "contrastive_v2_fallback",
        "final_decision": "v32_generation_reroute",
        "pre_gate_reason": gate_reason,
        "runtime_signals": {
            "ask_count_so_far": ask_count_so_far,
            "belief_entropy": belief_entropy,
            "margin_top1_minus_top2": margin_top12,
        },
        "uses_target_future_hit_delta": False,
    }


def build_near_miss_rescue_audit(
    *,
    turn_idx: int,
    reasons: list[str],
    state: dict[str, Any],
    original_decision: bool,
    final_decision: bool,
    voi_eval: float,
    belief_entropy: float | None,
    margin_top12: float | None,
    ask_count_so_far: int,
) -> dict[str, Any]:
    rescue_n = int(state.get("near_miss_rescue_count", 0) or 0) + 1
    cool_n = int(state.get("cooldown_reroute_count", 0) or 0)
    total = rescue_n + cool_n
    return {
        "repair_action": "near_miss_rescue",
        "repair_reason": reasons,
        "repair_used_count": {
            "near_miss_rescue": rescue_n,
            "cooldown_reroute": cool_n,
            "total": total,
        },
        "turn": int(turn_idx),
        "original_decision": "skip_ask" if not original_decision else "ask",
        "final_decision": "ask" if final_decision else "skip_ask",
        "midask_voi_eval": float(voi_eval),
        "runtime_signals": {
            "ask_count_so_far": ask_count_so_far,
            "belief_entropy": belief_entropy,
            "margin_top1_minus_top2": margin_top12,
        },
        "uses_target_future_hit_delta": False,
    }


def new_session_state() -> dict[str, Any]:
    return {
        "near_miss_rescue_count": 0,
        "cooldown_reroute_count": 0,
        "repair_total_count": 0,
    }


def record_cooldown_reroute(state: dict[str, Any]) -> None:
    state["cooldown_reroute_count"] = int(state.get("cooldown_reroute_count", 0) or 0) + 1
    state["repair_total_count"] = int(state.get("repair_total_count", 0) or 0) + 1


def record_near_miss_rescue(state: dict[str, Any]) -> None:
    state["near_miss_rescue_count"] = int(state.get("near_miss_rescue_count", 0) or 0) + 1
    state["repair_total_count"] = int(state.get("repair_total_count", 0) or 0) + 1
