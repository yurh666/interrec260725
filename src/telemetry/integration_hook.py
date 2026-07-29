"""Attach audit-only full-rank telemetry to turn logs (never guard input)."""

from __future__ import annotations

from typing import Any

from src.guard.policy_registry import full_rank_telemetry_enabled, light_full_rank_telemetry_enabled
from src.telemetry.full_rank_effect import compute_full_rank_snapshot, _target_items_from_context


def attach_light_rank_audit(
    *,
    rec_top10: list[str],
    rec_scores: list[float],
    movielens_rr_audit: dict[str, Any] | None,
    target_items: list[str],
) -> dict[str, Any]:
    """Light telemetry: Top10 + pool rank + readout scores only (no full-catalog rank_all)."""
    targets = [str(x) for x in target_items]
    pool_rank = None
    readout_enabled = False
    if movielens_rr_audit:
        readout_enabled = bool(movielens_rr_audit.get("enabled"))
        pool_rank = movielens_rr_audit.get("target_pool_rank")
    best_target_rank = None
    for t in targets:
        if t in rec_top10:
            best_target_rank = rec_top10.index(t) + 1
            break
    return {
        "light_telemetry": True,
        "catalog_size_ranked": None,
        "best_target_rank": best_target_rank,
        "target_pool_rank": pool_rank,
        "readout_rerank_enabled": readout_enabled,
        "top10_ids": list(rec_top10[:10]),
        "top10_scores_head": [float(x) for x in rec_scores[:10]],
        "hit_at_10": bool(best_target_rank is not None and best_target_rank <= 10),
        "audit_only": True,
        "used_by_guard_decision": False,
        "target_item_ids": targets,
    }


def attach_full_rank_audit_if_enabled(
    *,
    cfg: dict[str, Any],
    recommender: Any,
    belief: Any,
    seen_items: set[str],
    theta_star_items: list[str] | None,
    future_test_items: list[str] | None,
    rec_top10: list[str] | None = None,
    rec_scores: list[float] | None = None,
    movielens_rr_audit: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    targets = _target_items_from_context(
        theta_star_items=theta_star_items,
        future_test_items=future_test_items,
    )
    if light_full_rank_telemetry_enabled(cfg):
        return attach_light_rank_audit(
            rec_top10=list(rec_top10 or []),
            rec_scores=list(rec_scores or []),
            movielens_rr_audit=movielens_rr_audit,
            target_items=targets,
        )
    if not full_rank_telemetry_enabled(cfg):
        return None
    snap = compute_full_rank_snapshot(
        recommender,
        belief,
        exclude=set(seen_items),
        target_items=targets,
    )
    return {
        **snap,
        "audit_only": True,
        "used_by_guard_decision": False,
        "target_item_ids": targets,
    }
