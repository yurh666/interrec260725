"""Runtime Intent Memory wiring for session_simulator (6A / memory_active paths)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.belief.belief_state import BeliefState
from src.belief.initialization import initialize_belief
from src.belief.laplace_update import laplace_choice_update
from src.choicerec.subexperiment_runtime import belief_update_ablation_mode
from src.memory.adaptive_intent_units import adaptive_memory_to_belief
from src.memory.incremental_history_builder import IncrementalHistoryMemoryBuilder
from src.memory.incremental_memory_update import update_intent_memory_with_choice
from src.memory.intent_memory_state import IntentMemoryState
from src.memory.memory_to_belief import project_memory_to_belief, rank_safe_fallback_if_needed
from src.memory.memory_update_policy import MemoryUpdateConfig
from src.recommendation.recommender import BeliefRecommender


def _normalize_mode(mode: str) -> str:
    m = str(mode or "original").strip().lower()
    if m in {"compatibility", "compatible", "memory_compatible"}:
        return "memory_compatible"
    if m in {"memory_active", "active"}:
        return "memory_active"
    return m


def merge_memory_cfg(cfg: dict[str, Any], belief_cfg: dict[str, Any]) -> dict[str, Any]:
    legacy = dict(belief_cfg.get("memory") or {})
    root = dict(cfg.get("memory") or {})
    out = {**legacy, **root}
    for key in ("adaptive_units", "projection", "rank_safe_blend"):
        if isinstance(legacy.get(key), dict) or isinstance(root.get(key), dict):
            out[key] = {**(legacy.get(key) or {}), **(root.get(key) or {})}
    return out


@dataclass
class MemorySessionContext:
    enabled: bool
    mode: str
    belief_update_mode: str
    memory_cfg: dict[str, Any]
    belief_update_ablation: str | None = None
    intent_memory_state: IntentMemoryState | None = None
    memory_update_traces: list[dict[str, Any]] = field(default_factory=list)
    memory_init_trace: dict[str, Any] = field(default_factory=dict)


def resolve_memory_session_context(cfg: dict[str, Any], belief_cfg: dict[str, Any]) -> MemorySessionContext:
    memory_cfg = merge_memory_cfg(cfg, belief_cfg)
    sub = cfg.get("subexperiment_6") or {}
    force_6a = str(sub.get("subexp", "")).startswith("6A")
    enabled = bool(memory_cfg.get("enabled", False)) or force_6a
    mode = _normalize_mode(memory_cfg.get("mode", "memory_compatible" if enabled else "original"))
    if not enabled:
        mode = "original"
    if force_6a and mode == "original":
        mode = "memory_active"
    belief_update_mode = str(memory_cfg.get("belief_update_mode", "original_laplace"))
    if mode == "memory_active":
        belief_update_mode = str(memory_cfg.get("belief_update_mode", "adaptive_memory_laplace"))
    elif mode == "memory_compatible":
        belief_update_mode = str(memory_cfg.get("belief_update_mode", "original_laplace"))
    if force_6a:
        belief_update_mode = str(memory_cfg.get("belief_update_mode", "adaptive_memory_laplace"))
        mode = str(memory_cfg.get("mode", mode))
        mode = _normalize_mode(mode)
        if mode == "original":
            mode = "memory_active"
    return MemorySessionContext(
        enabled=enabled and mode != "original",
        mode=mode,
        belief_update_mode=belief_update_mode,
        memory_cfg=memory_cfg,
        belief_update_ablation=belief_update_ablation_mode(cfg),
    )


def initialize_belief_with_memory(
    *,
    ctx: MemorySessionContext,
    observed_history: list[str],
    index: Any,
    weights: list[float],
    belief_cfg: dict[str, Any],
    rec_cfg: dict[str, Any],
    top_k: int,
) -> tuple[BeliefState, str, dict[str, Any]]:
    base_init_belief = initialize_belief(
        observed_history=observed_history,
        index=index,
        weights=weights,
        init_mode=belief_cfg.get("init_mode", "weighted_average"),
        sigma0=float(belief_cfg.get("sigma0", 1.0)),
        jitter=float(belief_cfg.get("jitter", 1e-6)),
    )
    init_trace: dict[str, Any] = {}
    component = "belief_init"
    if not ctx.enabled:
        return base_init_belief.copy(), component, init_trace

    memory_cfg = ctx.memory_cfg
    adaptive_units_cfg = dict(memory_cfg.get("adaptive_units") or {})
    for legacy_key, nested_key in (
        ("split_dispersion_threshold", "split_dispersion_threshold"),
        ("merge_cosine_threshold", "merge_cosine_threshold"),
        ("min_support_weight", "min_support_weight"),
        ("max_granularity_level", "max_granularity_level"),
        ("max_units", "max_units"),
    ):
        if legacy_key in memory_cfg and nested_key not in adaptive_units_cfg:
            adaptive_units_cfg[nested_key] = memory_cfg[legacy_key]
    if "max_units" not in adaptive_units_cfg and ctx.mode == "memory_compatible":
        adaptive_units_cfg["max_units"] = 5
    adaptive_units_cfg.setdefault("use_as_view_only", ctx.mode == "memory_compatible")

    builder = IncrementalHistoryMemoryBuilder(
        dim=index.embeddings.shape[1],
        sigma0=float(belief_cfg.get("sigma0", 1.0)),
        jitter=float(belief_cfg.get("jitter", 1e-6)),
        adaptive_units_enabled=bool(adaptive_units_cfg.get("enabled", True)),
        adaptive_units_config=adaptive_units_cfg,
    )
    for item_id, item_weight in zip(observed_history, weights):
        if str(item_id) in index.item_id_to_index:
            builder.add_item(str(item_id), index.vector_for(str(item_id)), float(item_weight))
    intent_memory_state, equivalent_mu, equivalent_sigma = builder.finalize(original_belief=base_init_belief)
    ctx.intent_memory_state = intent_memory_state

    equivalent_belief = base_init_belief.copy()
    equivalent_belief.mu = equivalent_mu
    equivalent_belief.Sigma = equivalent_sigma
    projection_cfg = dict(memory_cfg.get("projection") or {})
    init_alpha = float(projection_cfg.get("alpha", memory_cfg.get("init_memory_belief_blend", 0.0)))
    if ctx.mode == "memory_compatible":
        init_alpha = float(projection_cfg.get("alpha", memory_cfg.get("init_memory_belief_blend", 0.0)))
    projection_source = (
        base_init_belief if ctx.mode == "memory_compatible" and init_alpha <= 0.0 else equivalent_belief
    )
    belief, memory_projection_trace = project_memory_to_belief(
        original_belief=projection_source,
        memory_state=intent_memory_state,
        alpha=init_alpha,
        sigma0=float(belief_cfg.get("sigma0", 1.0)),
        jitter=float(belief_cfg.get("jitter", 1e-6)),
    )
    rank_safe_cfg = memory_cfg.get("rank_safe_blend") or {}
    if bool(projection_cfg.get("rank_safe", False) or rank_safe_cfg.get("enabled", False)) and init_alpha > 0.0:
        belief, rank_safe_trace = rank_safe_fallback_if_needed(
            original_belief=equivalent_belief,
            projected_belief=belief,
            recommender=BeliefRecommender(index, lambda_explore=float(rec_cfg.get("lambda_explore", 0.1))),
            top_k=top_k,
            exclude=set(str(x) for x in observed_history),
            min_top10_overlap=float(rank_safe_cfg.get("min_top10_overlap", 0.8)),
        )
        memory_projection_trace.update(rank_safe_trace)
    init_trace = {
        **memory_projection_trace,
        "memory_mode": ctx.mode,
        "belief_update_mode": ctx.belief_update_mode,
        "memory_unit_count": intent_memory_state.unit_count,
        "compatible_update_ema": float(memory_cfg.get("compatible_update_ema", memory_cfg.get("update_ema", 0.0))),
    }
    ctx.memory_init_trace = init_trace
    return belief, "incremental_history_memory_init", init_trace


def _build_update_config(memory_cfg: dict[str, Any], *, trace_only: bool) -> MemoryUpdateConfig:
    keys = (
        "create_similarity_threshold",
        "merge_similarity_threshold",
        "downweight_factor",
        "update_ema",
        "support_gain",
        "min_support_weight",
        "allow_create",
        "allow_merge",
        "allow_downweight",
        "enable_competing_downweight",
        "competing_downweight_factor",
        "competing_alignment_threshold",
    )
    payload = {k: memory_cfg[k] for k in keys if k in memory_cfg}
    eta = float(memory_cfg.get("compatible_update_ema", memory_cfg.get("update_ema", 0.25)))
    payload["update_ema"] = eta
    cfg = MemoryUpdateConfig(**payload)
    if trace_only:
        cfg.allow_create = False
        cfg.allow_merge = False
        cfg.allow_downweight = False
        cfg.enable_competing_downweight = False
    return cfg


def apply_choice_belief_update(
    *,
    ctx: MemorySessionContext,
    belief: BeliefState,
    belief_pre_laplace: BeliefState,
    opt_vectors: np.ndarray,
    option_ids: list[str],
    selected_oid: str,
    sel_vec_idx: int | None,
    tau: float,
    alpha_none: float,
    ask_cfg: dict[str, Any],
    turn: int,
    recommender: BeliefRecommender,
    cur_top10: list[str],
    rec_scores: list[float],
    seen_items: set[str],
    top_k: int,
) -> tuple[BeliefState, str, dict[str, Any] | None]:
    ablation = ctx.belief_update_ablation
    if ablation == "none":
        trace = {"belief_update_ablation": "none", "skipped": True}
        return belief.copy(), "belief_update_skipped", trace
    update_sigma = ablation != "mean_only"

    if not ctx.enabled or ctx.intent_memory_state is None:
        updated = laplace_choice_update(
            belief,
            opt_vectors,
            sel_vec_idx,
            tau,
            alpha_none,
            update_sigma=update_sigma,
        )
        trace = {"belief_update_ablation": ablation} if ablation else None
        return updated, "belief_update", trace

    memory_cfg = ctx.memory_cfg
    memory_active_update = (
        ctx.mode == "memory_active"
        or ctx.belief_update_mode == "adaptive_memory_laplace"
        or bool(memory_cfg.get("active_memory_refinement", False))
    )
    memory_trace_only = ctx.mode == "memory_compatible" and not memory_active_update

    if not (memory_trace_only or memory_active_update):
        updated = laplace_choice_update(
            belief,
            opt_vectors,
            sel_vec_idx,
            tau,
            alpha_none,
            update_sigma=update_sigma,
        )
        trace = {"belief_update_ablation": ablation} if ablation else None
        return updated, "belief_update", trace

    mem_update_cfg = _build_update_config(memory_cfg, trace_only=memory_trace_only)
    sel_vec = None if sel_vec_idx is None else opt_vectors[sel_vec_idx]
    include_none = bool(ask_cfg.get("include_none", True))
    update_blend = 0.0 if memory_trace_only else float(
        memory_cfg.get(
            "update_memory_belief_blend",
            (memory_cfg.get("projection") or {}).get("alpha", memory_cfg.get("memory_belief_blend", 0.05)),
        )
    )
    intent_memory_state, updated_belief, memory_update_trace = update_intent_memory_with_choice(
        ctx.intent_memory_state,
        question_options=option_ids + (["none"] if include_none else []),
        selected_option=str(selected_oid),
        selected_vector=sel_vec,
        option_vectors=opt_vectors,
        current_belief=belief_pre_laplace,
        update_config=mem_update_cfg,
        turn_id=turn,
        tau=tau,
        alpha_none=alpha_none,
        memory_belief_blend=update_blend,
        update_sigma=update_sigma,
    )
    ctx.intent_memory_state = intent_memory_state
    memory_update_trace["memory_mode"] = ctx.mode
    memory_update_trace["compatible_update_ema"] = float(mem_update_cfg.update_ema)
    memory_update_trace["memory_update_applied_to_belief"] = bool(update_blend > 0.0 and memory_active_update)
    ctx.memory_update_traces.append(memory_update_trace)
    component = "memory_trace_update" if memory_trace_only else "adaptive_memory_update"
    return updated_belief, component, memory_update_trace
