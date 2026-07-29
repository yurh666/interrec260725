"""Skip contrastive_v2 fallback when frontier_clustered_v32 pre-gate fails (OOS policy)."""

from __future__ import annotations

from typing import Any


def oos_contrastive_v2_skip_enabled(fc32_cfg: dict[str, Any]) -> bool:
    return bool(fc32_cfg.get("enable_oos_contrastive_v2_skip", False))


def should_apply_oos_contrastive_v2_skip(
    *,
    fc32_cfg: dict[str, Any],
    gate_pass: bool,
    reroute_ok: bool,
) -> bool:
    """True when pre_gate failed, routing repair did not reroute, and OOS skip policy is on."""
    if gate_pass or reroute_ok:
        return False
    return oos_contrastive_v2_skip_enabled(fc32_cfg)


def build_oos_skip_audit(*, gate_reason: str) -> dict[str, Any]:
    return {
        "oos_skip_applied": True,
        "original_oos_reason": str(gate_reason),
        "original_fallback_path": "contrastive_v2",
        "skipped_due_to_oos_policy": True,
    }
