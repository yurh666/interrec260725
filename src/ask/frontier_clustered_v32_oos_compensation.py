"""Low-cost v32-only compensation when OOS skip would suppress an ask (label-free)."""

from __future__ import annotations

from typing import Any


def oos_skip_v32_compensation_enabled(fc32_cfg: dict[str, Any]) -> bool:
    return bool(fc32_cfg.get("enable_oos_skip_v32_compensation", False))


def frontier_uncertainty_high(
    *,
    belief_entropy: float | None,
    margin_top12: float | None,
    entropy_min: float,
    margin_max: float,
) -> bool:
    if not isinstance(belief_entropy, (int, float)) or not isinstance(margin_top12, (int, float)):
        return False
    return float(belief_entropy) >= float(entropy_min) and float(margin_top12) <= float(margin_max)


def should_apply_oos_v32_compensation(
    *,
    fc32_cfg: dict[str, Any],
    turn_idx: int,
    oos_skip_count_so_far: int,
    v32_ask_count_so_far: int,
    compensation_used: bool,
    belief_entropy: float | None,
    margin_top12: float | None,
) -> tuple[bool, str]:
    """Return (apply, reason_code). Uses runtime-safe signals only — no target/future/hit."""
    if not oos_skip_v32_compensation_enabled(fc32_cfg):
        return False, "compensation_disabled"
    if compensation_used:
        return False, "compensation_already_used"
    turn_min = int(fc32_cfg.get("oos_skip_v32_compensation_turn_min", 3) or 3)
    if turn_idx < turn_min:
        return False, "turn_below_min"
    oos_min = int(fc32_cfg.get("oos_skip_v32_compensation_oos_skip_min", 2) or 2)
    if oos_skip_count_so_far < oos_min:
        return False, "oos_skip_count_below_min"
    v32_max = int(fc32_cfg.get("oos_skip_v32_compensation_v32_ask_max", 2) or 2)
    if v32_ask_count_so_far > v32_max:
        return False, "v32_ask_count_above_max"
    ent_min = float(fc32_cfg.get("oos_skip_v32_compensation_entropy_min", 175.0) or 175.0)
    margin_max = float(fc32_cfg.get("oos_skip_v32_compensation_margin_max", 0.02) or 0.02)
    if not frontier_uncertainty_high(
        belief_entropy=belief_entropy,
        margin_top12=margin_top12,
        entropy_min=ent_min,
        margin_max=margin_max,
    ):
        return False, "frontier_uncertainty_not_high"
    return True, "high_skip_low_v32_high_uncertainty"


def build_compensation_audit(
    *,
    trigger_reason: str,
    trigger_turn: int,
    oos_skip_count: int,
    v32_ask_count: int,
    gate_reason: str,
    belief_entropy: float | None,
    margin_top12: float | None,
) -> dict[str, Any]:
    return {
        "compensation_applied": True,
        "compensation_action": "oos_skip_v32_compensation",
        "compensation_reason": trigger_reason,
        "trigger_turn": int(trigger_turn),
        "oos_skip_count_at_trigger": int(oos_skip_count),
        "v32_ask_count_at_trigger": int(v32_ask_count),
        "original_oos_reason": str(gate_reason),
        "original_fallback_path": "contrastive_v2",
        "belief_entropy_at_trigger": belief_entropy,
        "margin_top1_minus_top2_at_trigger": margin_top12,
        "skipped_due_to_oos_policy": False,
        "uses_target_future_hit_delta": False,
    }
