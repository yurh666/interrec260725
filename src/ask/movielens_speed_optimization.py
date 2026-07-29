"""MOVIELENS_ONLY speed optimizations — config-gated, no future-target runtime decisions."""

from __future__ import annotations

from typing import Any

from src.ask.frontier_clustered_v32_oos_skip import should_apply_oos_contrastive_v2_skip
from src.ask.frontier_clustered_v32_pregate import evaluate_frontier_clustered_v32_pre_gate


def movielens_speed_opt_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg.get("movielens_speed_optimization") or {})


def movielens_speed_opt_enabled(cfg: dict[str, Any]) -> bool:
    opt = movielens_speed_opt_cfg(cfg)
    if not bool(opt.get("enabled", False)):
        return False
    dom = str(opt.get("domain_only", "movielens")).strip().lower()
    dataset = str((cfg.get("dataset") or {}).get("name", "")).strip().lower()
    return dom == "movielens" and dataset == "movielens"


def fc32_cfg_for_skip_eval(fc32_cfg: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Pre-gate thresholds for skip *decision only* — does not alter main generation path."""
    out = dict(fc32_cfg)
    if movielens_speed_opt_enabled(cfg):
        opt = movielens_speed_opt_cfg(cfg)
        if opt.get("pre_gate_max_top10_margin") is not None:
            out["pre_gate_max_top10_margin"] = opt.get("pre_gate_max_top10_margin")
        if opt.get("pre_gate_min_uncertainty") is not None:
            out["pre_gate_min_uncertainty"] = opt.get("pre_gate_min_uncertainty")
        if out.get("pre_gate_max_top10_margin") is None and out.get("pre_gate_min_uncertainty") is None:
            out["pre_gate_max_top10_margin"] = 0.08
            out["pre_gate_min_uncertainty"] = 40.0
    return out


_SAFE_SKIP_REASONS = frozenset(
    {
        "high_margin_and_low_entropy_skip",
        "stagnation_consecutive_ask_without_top10_change",
        "budget_max_v32_generations_reached",
        "budget_max_v32_asks_reached",
    }
)


def evaluate_early_pre_gate(
    *,
    fc32_cfg: dict[str, Any],
    turn_idx: int,
    ask_count_so_far: int,
    v32_generation_count_so_far: int,
    last_v32_generation_turn: int,
    cooldown_turns: int,
    consecutive_ask_rank_unchanged: int,
    belief_entropy: float | None,
    sigma_trace: float | None,
    margin_top12: float | None,
    rec_top10_ids: tuple[str, ...],
    softmax_entropy_top10: float | None,
) -> tuple[bool, str, dict[str, Any]]:
    return evaluate_frontier_clustered_v32_pre_gate(
        fc32_cfg=fc32_cfg,
        turn_idx=turn_idx,
        ask_count_so_far=ask_count_so_far,
        v32_generation_count_so_far=v32_generation_count_so_far,
        last_v32_generation_turn=last_v32_generation_turn,
        cooldown_turns=cooldown_turns,
        consecutive_ask_rank_unchanged=consecutive_ask_rank_unchanged,
        belief_entropy=belief_entropy,
        sigma_trace=sigma_trace,
        margin_top12=margin_top12,
        rec_top10_ids=rec_top10_ids,
        softmax_entropy_top10=softmax_entropy_top10,
    )


def should_skip_expensive_generation(
    *,
    cfg: dict[str, Any],
    trace_skip_llm: bool,
    pre_gate_pass: bool,
    pre_gate_reason: str,
    belief_entropy: float | None,
    margin_top12: float | None,
    ask_count: int,
    fc32_cfg: dict[str, Any],
    midask_cfg: dict[str, Any],
    entropy_gate_session_count: int,
) -> tuple[bool, dict[str, Any]]:
    """Return (skip, audit). Conservative: only skip when confident no ask / OOS path."""
    opt = movielens_speed_opt_cfg(cfg)
    audit: dict[str, Any] = {
        "movielens_speed_optimization": True,
        "noask_skip_expensive_generation": bool(opt.get("noask_skip_expensive_generation")),
    }
    if not movielens_speed_opt_enabled(cfg) or trace_skip_llm:
        audit["skip_reason"] = "disabled_or_trace_replay"
        return False, audit
    if not bool(opt.get("noask_skip_expensive_generation", False)):
        audit["skip_reason"] = "flag_off"
        return False, audit

    max_ent = int(fc32_cfg.get("max_entropy_gate_asks_per_session", 1) or 1)
    ent_min = float(fc32_cfg.get("entropy_ask_gate_min_belief_entropy", 181.0) or 181.0)
    if (
        bool(fc32_cfg.get("enable_cluster_bound_v2_entropy_ask_gate"))
        and entropy_gate_session_count < max_ent
        and belief_entropy is not None
        and float(belief_entropy) >= ent_min
    ):
        audit["skip_reason"] = "entropy_gate_may_fire"
        return False, audit

    if bool(midask_cfg.get("enabled", False)):
        ask_min = int(midask_cfg.get("ask_count_min", 3))
        ask_max = int(midask_cfg.get("ask_count_max", 4))
        if ask_min <= ask_count <= ask_max:
            audit["skip_reason"] = "midask_window_active"
            return False, audit

    if bool(fc32_cfg.get("force_ask_for_sanity")):
        audit["skip_reason"] = "force_ask_for_sanity"
        return False, audit

    if not pre_gate_pass and str(pre_gate_reason) in _SAFE_SKIP_REASONS:
        audit["skip_reason"] = f"pre_gate_fail:{pre_gate_reason}"
        audit["skip_cluster_bound_when_noask"] = bool(opt.get("skip_cluster_bound_when_noask", True))
        audit["skip_frontier_hypotheses_when_noask"] = bool(opt.get("skip_frontier_hypotheses_when_noask", True))
        return True, audit

    audit["skip_reason"] = "conservative_full_pipeline"
    return False, audit


def should_apply_oos_on_speed_skip(
    *,
    fc32_cfg: dict[str, Any],
    gate_pass: bool,
) -> bool:
    return should_apply_oos_contrastive_v2_skip(
        fc32_cfg=fc32_cfg,
        gate_pass=bool(gate_pass),
        reroute_ok=False,
    )


def new_session_speed_stats() -> dict[str, Any]:
    return {
        "direction_translator_cache_hits": 0,
        "direction_translator_cache_misses": 0,
        "itemknn_pool_cache_hits": 0,
        "itemknn_pool_cache_misses": 0,
    }
