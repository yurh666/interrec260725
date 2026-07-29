"""Rank-safe guard policy registry — defaults and full-rank run validation."""

from __future__ import annotations

from typing import Any

CURRENT_CANDIDATE_POLICY = "v42_option_set_guard"
EXPLORATORY_FAILED_POLICIES = frozenset({"v42_1_option_set_guard"})


def resolve_rank_safe_policy(cfg: dict[str, Any]) -> str:
    fc32 = ((cfg.get("ask") or {}).get("hypothesis_generation") or {}).get("frontier_clustered_v32") or {}
    return str(fc32.get("rank_safe_guard_policy") or "").strip().lower()


def light_full_rank_telemetry_enabled(cfg: dict[str, Any]) -> bool:
    opt = cfg.get("movielens_speed_optimization") or {}
    return bool(opt.get("enabled")) and bool(opt.get("light_full_rank_telemetry", False))


def full_rank_telemetry_enabled(cfg: dict[str, Any]) -> bool:
    if light_full_rank_telemetry_enabled(cfg):
        return False
    tel = cfg.get("telemetry") or {}
    if "full_rank_telemetry_enabled" in tel:
        return bool(tel.get("full_rank_telemetry_enabled"))
    fc32 = ((cfg.get("ask") or {}).get("hypothesis_generation") or {}).get("frontier_clustered_v32") or {}
    return bool(fc32.get("full_rank_telemetry_enabled", False))


def validate_fullrank_block_smoke_config(cfg: dict[str, Any]) -> dict[str, Any]:
    policy = resolve_rank_safe_policy(cfg)
    fc32 = ((cfg.get("ask") or {}).get("hypothesis_generation") or {}).get("frontier_clustered_v32") or {}
    issues: list[str] = []
    if policy != CURRENT_CANDIDATE_POLICY:
        issues.append(f"policy_must_be_{CURRENT_CANDIDATE_POLICY}_not_{policy or 'unset'}")
    if policy in EXPLORATORY_FAILED_POLICIES:
        issues.append(f"exploratory_failed_policy_blocked:{policy}")
    if not full_rank_telemetry_enabled(cfg):
        issues.append("full_rank_telemetry_enabled_must_be_true")
    if bool(fc32.get("rank_safe_guard_log_only", True)):
        issues.append("block_execution_requires_log_only_false")
    if not bool(fc32.get("rank_safe_guard_block_execution", False)):
        issues.append("rank_safe_guard_block_execution_must_be_true")
    return {
        "policy": policy,
        "current_candidate_policy": CURRENT_CANDIDATE_POLICY,
        "full_rank_telemetry_enabled": full_rank_telemetry_enabled(cfg),
        "block_execution": not bool(fc32.get("rank_safe_guard_log_only", True)),
        "conservative_rewrite_execution": bool(fc32.get("rank_safe_guard_conservative_rewrite_execution", False)),
        "conservative_rewrite_mode": str(
            fc32.get("conservative_rewrite_mode") or fc32.get("rank_safe_conservative_rewrite_mode") or "hard"
        ),
        "guard_uses_target_labels": False,
        "issues": issues,
        "ok": not issues,
    }
