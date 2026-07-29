from __future__ import annotations

import os
import time
from typing import Any

import numpy as np


def _top10_softmax_entropy(scores: list[float]) -> float:
    """Shannon entropy of softmax(normalize(scores[:10])). Online confidence signal (no labels)."""
    if len(scores) < 2:
        return 0.0
    x = np.asarray(scores[:10], dtype=np.float64)
    x = x - float(np.max(x))
    p = np.exp(x)
    s = float(np.sum(p)) + 1e-15
    p = p / s
    return float(-np.sum(p * np.log(p + 1e-15)))

from src.agents.direction_translator import DirectionTranslator
from src.agents.frontier_clustered_hypothesis_integration import (
    frontier_cluster_session_armed,
    run_frontier_clustered_v32_hypotheses_turn,
)
from src.agents.hypothesis_generator import HypothesisGenerator
from src.agents.option_writer import OptionWriter
from src.agents.option_neutrality import neutrality_check_options
from src.ask.anchor_selector import build_anchor_directions
from src.ask.ask_policy import should_ask
from src.ask.frontier_clustered_v32_oos_compensation import (
    build_compensation_audit,
    should_apply_oos_v32_compensation,
)
from src.ask.frontier_clustered_v32_oos_skip import (
    build_oos_skip_audit,
    should_apply_oos_contrastive_v2_skip,
)
from src.ask.frontier_clustered_v32_pregate import (
    build_budget_state,
    evaluate_frontier_clustered_v32_pre_gate,
)
from src.ask.frontier_clustered_v32_routing_repair import (
    build_cooldown_reroute_audit,
    build_near_miss_rescue_audit,
    evaluate_cooldown_reroute_strict,
    evaluate_near_miss_rescue_strict,
    new_session_state,
    record_cooldown_reroute,
    record_near_miss_rescue,
)
from src.ask.frontier_clustered_v32_rank_safe import (
    build_ask_effect_memory_entry,
    build_out_of_scope_non_v32_guard_audit,
    compute_rank_safe_guard,
    rank_safe_guard_scope_v32_only,
)
from src.ask.frontier_clustered_v32_rank_safe_mcq import (
    apply_conservative_rewrite_to_generation_block,
    is_v32_substantive_generation_block,
    resolve_conservative_rewrite_mode,
)
from src.ask.hypothesis_vectorizer import build_cluster_bound_vector_binding_audit, vectorize_hypotheses
from src.ask.cluster_bound_generation_v2 import (
    try_cb_v2_entropy_ask_gate,
    try_cb_v2_force_delivery_smoke,
)
from src.ask.midask_guard import should_apply_midask_force
from src.ask.question_selector import select_best_question
from src.ask.voi_gate import dominant_pairwise_hrv_score, empty_question_selection
from src.memory.method_views import BayesianLatentIntentMemoryView
from src.belief.initialization import initialize_belief
from src.belief.laplace_update import laplace_choice_update
from src.memory.session_memory_integration import (
    MemorySessionContext,
    apply_choice_belief_update,
    initialize_belief_with_memory,
    resolve_memory_session_context,
)
from src.diagnostics.frontier_discriminative_shadow import safe_frontier_discriminative_shadow_turn_dict
from src.recommendation.metrics import ranking_metrics, ranking_metrics_multik
from src.recommendation.recommender import BeliefRecommender
from src.recommendation.csc_readout import csc_config_from_fc32, csc_enabled, try_build_csc_overlay
from src.simulation.session_trace import SessionTraceController, TraceReplayMissError
from src.simulation.user_simulator import UserSimulator
from src.utils.run_profiling_context import append_diagnostics_row, profiling_clear_turn, profiling_set_turn
from src.telemetry.integration_hook import attach_full_rank_audit_if_enabled
from src.utils.session_guard_state import (
    configure_engineering_guards,
    session_guard_begin,
    session_guard_end,
    session_guard_snapshot,
    turn_profile_snapshot,
)


def _merge_voi_gate_softmax_diagnostics(
    voi_gate_info: dict[str, Any], stats: dict[str, Any], *, log_details: bool
) -> None:
    """Attach softmax / dominant-set diagnostics from ``dominant_pairwise_hrv_score``."""
    if "choice_prob_top1" not in stats:
        return
    voi_gate_info["choice_prob_top1"] = stats["choice_prob_top1"]
    voi_gate_info["choice_prob_top2"] = stats["choice_prob_top2"]
    voi_gate_info["choice_prob_entropy"] = stats["choice_prob_entropy"]
    if log_details:
        voi_gate_info["dominant_set_indices"] = stats.get("dominant_set_indices")
        voi_gate_info["dominant_set_probs"] = stats.get("dominant_set_probs")


def _frontier_fc32_cluster_cells_for_hypothesis(
    fc32_blk: dict[str, Any],
    hypothesis_id: str,
) -> dict[str, Any]:
    """Map simulator hypothesis_id back to frontier MCQ bundle + omap cells (sanity/logging only)."""

    cq = fc32_blk.get("choice_question") if isinstance(fc32_blk.get("choice_question"), dict) else {}
    omap = fc32_blk.get("option_to_cluster_mapping")
    omap = omap if isinstance(omap, dict) else {}
    out: dict[str, Any] = {"hypothesis_id": str(hypothesis_id), "mcq_option_letter": None, "omap_cells": None}
    for opt in cq.get("options") or []:
        if not isinstance(opt, dict):
            continue
        if str(opt.get("maps_to_hypothesis_id") or "") != str(hypothesis_id):
            continue
        letter = str(opt.get("option_id") or "").strip()
        out["mcq_option_letter"] = letter or None
        meta = omap.get(letter)
        out["omap_cells"] = meta
        cr = {}
        if isinstance(meta, dict):
            pc = meta.get("positive_cluster_id")
            cc = meta.get("contrasting_cluster_id") or meta.get("contrast_cluster_id")
            if pc is not None or cc is not None:
                try:
                    cr = {"reference_cluster_id": int(pc), "contrasting_cluster_id": int(cc)}
                except Exception:
                    cr = {}
        out["derived_contrast_ref"] = cr or None
        break
    return out


def _load_surface_writer_cls() -> type:
    import sys
    from pathlib import Path

    src_root = str(Path(__file__).resolve().parents[1])
    if src_root not in sys.path:
        sys.path.append(src_root)
    from interrec_noapi.surface_writer.writer import SurfaceWriter

    return SurfaceWriter


def _frontier_choice_options_for_selected_hypotheses(
    fc32_blk: dict[str, Any] | None,
    selected_hyps: list[dict[str, Any]],
    *,
    include_none: bool,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]] | None:
    """Return generated frontier MCQ text/options when it aligns with selected hypotheses.

    Config-gated caller only. This preserves selected hypothesis vectors and
    only changes the user-facing MCQ text, avoiding the generic OptionWriter
    template path.
    """

    if not isinstance(fc32_blk, dict):
        return None
    cq = fc32_blk.get("choice_question")
    if not isinstance(cq, dict):
        return None
    q_text = str(cq.get("final_question_text") or cq.get("question_text") or "").strip()
    opts = [o for o in (cq.get("options") or []) if isinstance(o, dict)]
    if not q_text or not opts:
        return None

    wanted = [str(h.get("hypothesis_id") or h.get("option_id") or "") for h in selected_hyps]
    wanted_set = {x for x in wanted if x}
    out: list[dict[str, Any]] = []
    for opt in opts:
        mapped = str(
            opt.get("mapped_hypothesis_id")
            or opt.get("maps_to_hypothesis_id")
            or opt.get("option_id")
            or ""
        )
        if mapped not in wanted_set:
            continue
        text = str(opt.get("text") or opt.get("option_text") or opt.get("label") or "").strip()
        if not text:
            continue
        out.append({"option_id": mapped, "option_text": text})

    # Keep option order identical to selected_hyps so selected_vector_index stays valid.
    by_id = {str(o["option_id"]): o for o in out}
    ordered = [by_id[h] for h in wanted if h in by_id]
    if len(ordered) != len(wanted):
        return None
    if include_none:
        ordered.append({"option_id": "none", "option_text": "None of the above fits well."})
    audit = {
        "used_frontier_choice_question": True,
        "question_text_chars": len(q_text),
        "option_count": len(ordered),
        "source": "frontier_clustered_v32_choice_question",
    }
    return q_text, ordered, audit


def run_session(
    session: dict[str, Any],
    theta_info: dict[str, Any],
    index: EmbeddingIndex,
    recommender: BeliefRecommender,
    translator: DirectionTranslator,
    hyp_generator: HypothesisGenerator,
    option_writer: OptionWriter,
    dataset: Any,
    cfg: dict[str, Any],
    bm25_index: Any = None,
    trace_ctrl: SessionTraceController | None = None,
) -> dict[str, Any]:
    """Run one complete episode (multiple turns) for a user session.

    Returns a list of per-turn log dicts.
    """
    ask_cfg = cfg.get("ask", {})
    sim_cfg = cfg.get("simulation", {})
    rec_cfg = cfg.get("recommendation", {})
    belief_cfg = cfg.get("belief", {})
    perf_cfg = cfg.get("performance_optimizations") or {}
    include_eigenspectrum_logging = not (
        bool(perf_cfg.get("enabled")) and bool(perf_cfg.get("disable_eigenspectrum_logging"))
    )

    K_dir = int(ask_cfg.get("K_directions", 5))
    M_hyp = int(ask_cfg.get("M_hypotheses", 5))
    n_opt = int(ask_cfg.get("n_options", 3))
    anchor_k = int(ask_cfg.get("anchor_top_k", 5))
    rep_k = int(ask_cfg.get("representative_top_k", 5))
    tau = float(ask_cfg.get("tau_choice", 0.2))
    tau_none = float(ask_cfg.get("tau_none", 0.2))
    none_bias = float(ask_cfg.get("none_bias", 0.0))
    none_threshold = float(ask_cfg.get("none_threshold", 0.25))
    alpha_none = float(ask_cfg.get("alpha_none", 1.2))
    mc_samples = int(ask_cfg.get("mc_samples", 128))
    c_ask = float(ask_cfg.get("c_ask", 0.05))
    top_k = int(rec_cfg.get("top_k", 10))
    max_turns = int(sim_cfg.get("max_turns", 5))

    observed_history = session["observed_history"]
    split_key = theta_info["theta_star_source"]
    future_items = session.get("future_test", []) or session.get("future_valid", []) or session.get("future_train", [])
    relevant = set(str(x) for x in future_items)

    weights = dataset.weights_for_items(session["user_id"], observed_history)
    memory_ctx: MemorySessionContext = resolve_memory_session_context(cfg, belief_cfg)
    t0 = time.perf_counter()
    belief, belief_init_component, memory_init_trace = initialize_belief_with_memory(
        ctx=memory_ctx,
        observed_history=observed_history,
        index=index,
        weights=weights,
        belief_cfg=belief_cfg,
        rec_cfg=rec_cfg,
        top_k=top_k,
    )
    append_diagnostics_row(
        {
            "user_id": session["user_id"],
            "episode_id": session.get("episode_id", session["user_id"]),
            "turn": 0,
            "event": "phase",
            "component": belief_init_component,
            "latency_sec": round(time.perf_counter() - t0, 6),
            "memory_mode": memory_ctx.mode if memory_ctx.enabled else "original",
            "memory_unit_count": (
                memory_ctx.intent_memory_state.unit_count if memory_ctx.intent_memory_state is not None else 0
            ),
            "compatible_update_ema": float(
                memory_ctx.memory_cfg.get("compatible_update_ema", memory_ctx.memory_cfg.get("update_ema", 0.0))
            ),
            **({k: memory_init_trace[k] for k in ("memory_update_applied_to_belief",) if k in memory_init_trace}),
        }
    )

    theta_star = theta_info["theta_star"]
    simulator = UserSimulator(
        theta_star=theta_star,
        choice_mode=sim_cfg.get("user_choice_mode", "deterministic_argmax"),
        tau=float(sim_cfg.get("tau_user_choice", 0.2)),
        tau_none=tau_none,
        none_bias=none_bias,
        none_threshold=none_threshold,
    )

    seen_items = set(str(x) for x in observed_history)
    ask_count = 0
    consecutive_no_ask_tail = 0
    turns_log: list[dict[str, Any]] = []
    fc32_v32_generation_count_session = 0
    fc32_last_v32_generation_turn = 0
    fc32_prev_turn_had_ask = False
    fc32_prev_turn_top10_snapshot: tuple[str, ...] | None = None
    fc32_prev_turn_margin12: float | None = None
    fc32_consecutive_ask_rank_unchanged = 0
    fc32_ask_effect_memory_ring: list[dict[str, Any]] = []
    fc32_last_ask_pre_top10: tuple[str, ...] | None = None
    fc32_routing_repair_state = new_session_state()
    fc32_oos_skip_count_session = 0
    fc32_cb_v2_force_delivery_session_count = 0
    fc32_cb_v2_entropy_gate_session_count = 0
    fc32_session_entropy_samples: list[float] = []
    fc32_v32_ask_count_session = 0
    fc32_compensation_used_session = False
    t_session0 = time.perf_counter()
    configure_engineering_guards(cfg)
    session_guard_begin(str(session.get("episode_id", session["user_id"])))
    voi_gate_cfg = ask_cfg.get("voi_gate") or {}
    gate_enabled = bool(voi_gate_cfg.get("enabled", False))
    gate_mode = str(voi_gate_cfg.get("mode", "dominant_pairwise_hrv"))
    gate_mass = float(voi_gate_cfg.get("mass", 0.8))
    gate_alpha = float(voi_gate_cfg.get("alpha", 0.35))
    gate_fallback_err = bool(voi_gate_cfg.get("fallback_to_full_voi_on_error", True))
    gate_log_details = bool(voi_gate_cfg.get("log_details", True))
    audit_skipped_voi = bool(voi_gate_cfg.get("audit_skipped_voi", False))
    preserve_ask_decision = bool(voi_gate_cfg.get("preserve_ask_decision", False))
    gate_online_q = float(voi_gate_cfg.get("online_quantile_q", 0.75))
    gate_online_min_hist = int(voi_gate_cfg.get("online_quantile_min_history", 2))
    session_initial_uncertainty: float | None = None
    session_unc_history: list[float] = []
    _HRV_GATE_MODES = frozenset(
        {
            "dominant_pairwise_hrv",
            "dominant_pairwise_hrv_online_quantile",
            "collapse_only",
            "conservative_dominant_hrv",
        }
    )

    base_impl_modes_template = {
        "item_encoder": cfg.get("embedding", {}).get("mode", "tfidf_svd"),
        "llm": translator.llm.implementation_mode,
        "option_neutrality": str((cfg.get("ask") or {}).get("neutrality", {}).get("mode", "rule")),
        "user_simulator": simulator.implementation_mode,
    }

    for turn in range(1, max_turns + 1):
        t_turn0 = time.perf_counter()
        fc32_routing_repair_audit = None
        fc32_oos_skip_audit: dict[str, Any] | None = None
        fc32_compensation_audit: dict[str, Any] | None = None
        profiling_set_turn(
            user_id=session["user_id"],
            episode_id=session.get("episode_id", session["user_id"]),
            turn=turn,
        )
        belief_before = belief.summary(include_eigendecomp=include_eigenspectrum_logging)

        # --- Recommend ---
        t0 = time.perf_counter()
        recs = recommender.recommend(belief, top_k=top_k, exclude=seen_items)
        append_diagnostics_row(
            {
                "event": "phase",
                "component": "recommendation_retrieval",
                "latency_sec": round(time.perf_counter() - t0, 6),
            }
        )
        rec_ids = [item_id for item_id, _ in recs]
        rec_scores = [score for _, score in recs]
        belief_snapshot_for_shadow = belief.copy()
        m5 = ranking_metrics(rec_ids, relevant, k=5)
        m10 = ranking_metrics(rec_ids, relevant, k=10)
        mk = ranking_metrics_multik(rec_ids, relevant, ks=(5, 10))
        rec_log = {
            "recommended_items": rec_ids[:10],
            "scores": rec_scores[:10],
            **m5,
            **m10,
            **mk,
        }

        cur_top10 = tuple(str(x) for x in rec_ids[:10])
        if fc32_prev_turn_had_ask and fc32_last_ask_pre_top10 is not None:
            mem_entry = build_ask_effect_memory_entry(
                pre_ask_top10=fc32_last_ask_pre_top10,
                post_ask_top10=cur_top10,
            )
            fc32_ask_effect_memory_ring.append(mem_entry)
            fc32_ask_effect_memory_ring = fc32_ask_effect_memory_ring[-2:]
        if fc32_prev_turn_had_ask and fc32_prev_turn_top10_snapshot is not None:
            if cur_top10 == fc32_prev_turn_top10_snapshot:
                fc32_consecutive_ask_rank_unchanged += 1
            else:
                fc32_consecutive_ask_rank_unchanged = 0

        # --- Uncertainty directions + anchors ---
        t0 = time.perf_counter()
        anchor_dirs = build_anchor_directions(belief, index, K=K_dir, anchor_top_k=anchor_k, exclude=seen_items)
        append_diagnostics_row(
            {
                "event": "phase",
                "component": "anchor_direction_embedding",
                "latency_sec": round(time.perf_counter() - t0, 6),
            }
        )
        anchor_dirs_log = [
            {
                "direction_id": d["direction_id"],
                "lambda": d["lambda"],
                "positive_anchors": d["positive_anchors"],
                "negative_anchors": d["negative_anchors"],
                "semantic_description": "",  # filled after translation
            }
            for d in anchor_dirs
        ]

        # --- Direction translation ---
        ep_id = str(session.get("episode_id", session["user_id"]))
        if trace_ctrl is not None and trace_ctrl.is_replay:
            tr_dirs = trace_ctrl.replay_translated(turn)
            if tr_dirs is not None:
                translated = tr_dirs
                translator.implementation_mode = "trace_replay"
                translator.last_call_audit = {
                    "component": "direction_translator",
                    "implementation_mode": "trace_replay",
                    "cache_hit": True,
                    "cache_miss": False,
                    "n_api_calls": 0,
                }
            else:
                translated = translator.translate(anchor_dirs, session_id=ep_id, turn_id=turn)
        else:
            translated = translator.translate(anchor_dirs, session_id=ep_id, turn_id=turn)
        for d_log, d_trans in zip(anchor_dirs_log, translated):
            d_log["semantic_description"] = d_trans.get("direction_name", "")

        hyp_gen_cfg = ask_cfg.get("hypothesis_generation") or {}
        rep_cfg = ask_cfg.get("representative_retrieval") or {}
        guard_cfg = ask_cfg.get("contrastive_v2_guard") or {}
        requested_hyp_mode = str(hyp_gen_cfg.get("mode", "legacy"))
        fc32_cfg_turn = dict(hyp_gen_cfg.get("frontier_clustered_v32") or {})

        frontier_gen_block_this_turn = None
        fc32_pre_gate_audit: dict[str, Any] | None = None
        fc32_budget_state_audit: dict[str, Any] | None = None

        lamb_exp = float(rec_cfg.get("lambda_explore", 0.1))
        trace_skip_llm_generation = False
        if trace_ctrl is not None and trace_ctrl.is_replay:
            trace_turn = trace_ctrl.get_turn(turn)
            if trace_turn is None:
                trace_ctrl.require_turn(turn)
            else:
                trace_skip_llm_generation = True
                trace_ctrl.mark_turn_consumed(turn)
                frontier_gen_block_this_turn = trace_ctrl.replay_frontier_block(turn)
                hypotheses = []
                oos_from_trace = trace_ctrl.replay_oos_skip_audit(turn)
                if oos_from_trace is not None:
                    fc32_oos_skip_audit = oos_from_trace
                    fc32_oos_skip_count_session += 1
                pre_from_trace = trace_turn.get("pre_gate_audit")
                if isinstance(pre_from_trace, dict) and pre_from_trace:
                    fc32_pre_gate_audit = pre_from_trace

        if not trace_skip_llm_generation and requested_hyp_mode == "frontier_clustered_v32" and frontier_cluster_session_armed(hyp_gen_cfg):
            lazy_on = bool(fc32_cfg_turn.get("lazy_generation_enabled", False))
            pre_on = bool(fc32_cfg_turn.get("pre_gate_enabled", False))
            if lazy_on and pre_on:
                margin12: float | None = None
                if len(rec_scores) >= 2:
                    margin12 = float(rec_scores[0] - rec_scores[1])
                be_raw = belief_before.get("entropy")
                belief_ent = float(be_raw) if isinstance(be_raw, (int, float)) else None
                st_raw = belief_before.get("sigma_trace")
                sigma_tr = float(st_raw) if isinstance(st_raw, (int, float)) else None
                se10 = float(_top10_softmax_entropy(rec_scores))
                gate_pass, gate_reason, feats = evaluate_frontier_clustered_v32_pre_gate(
                    fc32_cfg=fc32_cfg_turn,
                    turn_idx=turn,
                    ask_count_so_far=ask_count,
                    v32_generation_count_so_far=fc32_v32_generation_count_session,
                    last_v32_generation_turn=fc32_last_v32_generation_turn,
                    cooldown_turns=int(fc32_cfg_turn.get("ask_cooldown_turns", 0) or 0),
                    consecutive_ask_rank_unchanged=fc32_consecutive_ask_rank_unchanged,
                    belief_entropy=belief_ent,
                    sigma_trace=sigma_tr,
                    margin_top12=margin12,
                    rec_top10_ids=cur_top10,
                    softmax_entropy_top10=se10,
                )
                gen_mode_telemetry = "llm" if not fc32_cfg_turn.get("use_mock_llm", True) else "mock"
                conf_llm = str(os.environ.get("CONFIRM_FRONTIER_CLUSTERED_V32_LLM") or "").strip().lower() in (
                    "1",
                    "true",
                    "yes",
                    "on",
                )
                llm_client_mode = str(getattr(hyp_generator.llm, "mode", "mock") or "mock")
                would_call_llm = bool(
                    gate_pass and gen_mode_telemetry == "llm" and conf_llm and llm_client_mode != "mock",
                )
                fc32_pre_gate_audit = {
                    "enabled": True,
                    "lazy_generation_enabled": True,
                    "pre_gate_enabled": True,
                    "pass": bool(gate_pass),
                    "reason": gate_reason,
                    "features": feats,
                    "would_call_llm": would_call_llm,
                }
                fc32_budget_state_audit = build_budget_state(
                    fc32_cfg=fc32_cfg_turn,
                    current_turn_idx=turn,
                    ask_count_so_far=ask_count,
                    v32_generation_count_so_far=fc32_v32_generation_count_session,
                    last_v32_generation_turn=fc32_last_v32_generation_turn,
                    cooldown_turns=int(fc32_cfg_turn.get("ask_cooldown_turns", 0) or 0),
                    consecutive_ask_rank_unchanged=fc32_consecutive_ask_rank_unchanged,
                    gate_pass=gate_pass,
                    gate_reason=gate_reason,
                )
                total_cap_cfg = fc32_cfg_turn.get("max_total_asks_per_session_for_lazy_sanity")
                if total_cap_cfg is not None:
                    fc32_budget_state_audit["max_total_asks_per_session_for_lazy_sanity"] = int(total_cap_cfg)
                    fc32_budget_state_audit.setdefault("total_ask_budget_blocked", False)
                    fc32_budget_state_audit.setdefault("total_ask_budget_reason", None)

                if gate_pass:
                    hypotheses, frontier_gen_block_this_turn = run_frontier_clustered_v32_hypotheses_turn(
                        belief=belief,
                        index=index,
                        seen_items=seen_items,
                        observed_history=observed_history,
                        translated=translated,
                        hyp_generator=hyp_generator,
                        M_hyp=M_hyp,
                        hyp_gen_cfg=hyp_gen_cfg,
                        rec_lambda=lamb_exp,
                        cfg=cfg,
                        turn=turn,
                        episode_id=str(session.get("episode_id", session["user_id"])),
                    )
                    fc32_v32_generation_count_session += 1
                    fc32_last_v32_generation_turn = turn
                else:
                    reroute_ok, reroute_reasons = evaluate_cooldown_reroute_strict(
                        fc32_cfg=fc32_cfg_turn,
                        state=fc32_routing_repair_state,
                        gate_pass=bool(gate_pass),
                        gate_reason=str(gate_reason),
                        ask_count_so_far=int(ask_count),
                        belief_entropy=belief_ent,
                        margin_top12=margin12,
                    )
                    if reroute_ok:
                        hypotheses, frontier_gen_block_this_turn = run_frontier_clustered_v32_hypotheses_turn(
                            belief=belief,
                            index=index,
                            seen_items=seen_items,
                            observed_history=observed_history,
                            translated=translated,
                            hyp_generator=hyp_generator,
                            M_hyp=M_hyp,
                            hyp_gen_cfg=hyp_gen_cfg,
                            rec_lambda=lamb_exp,
                            cfg=cfg,
                            turn=turn,
                            episode_id=str(session.get("episode_id", session["user_id"])),
                        )
                        fc32_v32_generation_count_session += 1
                        fc32_last_v32_generation_turn = turn
                        record_cooldown_reroute(fc32_routing_repair_state)
                        fc32_routing_repair_audit = build_cooldown_reroute_audit(
                            turn_idx=turn,
                            reasons=reroute_reasons,
                            state=fc32_routing_repair_state,
                            gate_reason=str(gate_reason),
                            belief_entropy=belief_ent,
                            margin_top12=margin12,
                            ask_count_so_far=int(ask_count),
                        )
                        if isinstance(fc32_pre_gate_audit, dict):
                            fc32_pre_gate_audit = {
                                **fc32_pre_gate_audit,
                                "routing_repair_override": True,
                                "routing_repair_action": "cooldown_reroute",
                            }
                    elif should_apply_oos_contrastive_v2_skip(
                        fc32_cfg=fc32_cfg_turn,
                        gate_pass=bool(gate_pass),
                        reroute_ok=False,
                    ):
                        comp_ok, comp_reason = should_apply_oos_v32_compensation(
                            fc32_cfg=fc32_cfg_turn,
                            turn_idx=turn,
                            oos_skip_count_so_far=fc32_oos_skip_count_session,
                            v32_ask_count_so_far=fc32_v32_ask_count_session,
                            compensation_used=fc32_compensation_used_session,
                            belief_entropy=belief_ent,
                            margin_top12=margin12,
                        )
                        if comp_ok:
                            hypotheses, frontier_gen_block_this_turn = run_frontier_clustered_v32_hypotheses_turn(
                                belief=belief,
                                index=index,
                                seen_items=seen_items,
                                observed_history=observed_history,
                                translated=translated,
                                hyp_generator=hyp_generator,
                                M_hyp=M_hyp,
                                hyp_gen_cfg=hyp_gen_cfg,
                                rec_lambda=lamb_exp,
                                cfg=cfg,
                                turn=turn,
                                episode_id=str(session.get("episode_id", session["user_id"])),
                            )
                            fc32_v32_generation_count_session += 1
                            fc32_last_v32_generation_turn = turn
                            fc32_compensation_used_session = True
                            fc32_compensation_audit = build_compensation_audit(
                                trigger_reason=comp_reason,
                                trigger_turn=turn,
                                oos_skip_count=fc32_oos_skip_count_session,
                                v32_ask_count=fc32_v32_ask_count_session,
                                gate_reason=str(gate_reason),
                                belief_entropy=belief_ent,
                                margin_top12=margin12,
                            )
                        else:
                            fc32_oos_skip_audit = build_oos_skip_audit(gate_reason=str(gate_reason))
                            fc32_oos_skip_count_session += 1
                            hypotheses = []
                    else:
                        g2 = dict(hyp_gen_cfg)
                        g2["mode"] = "contrastive_v2"
                        hypotheses = hyp_generator.generate(
                            translated,
                            observed_history,
                            M=M_hyp,
                            generation_cfg=g2,
                        )
            else:
                hypotheses, frontier_gen_block_this_turn = run_frontier_clustered_v32_hypotheses_turn(
                    belief=belief,
                    index=index,
                    seen_items=seen_items,
                    observed_history=observed_history,
                    translated=translated,
                    hyp_generator=hyp_generator,
                    M_hyp=M_hyp,
                    hyp_gen_cfg=hyp_gen_cfg,
                    rec_lambda=lamb_exp,
                    cfg=cfg,
                    turn=turn,
                    episode_id=str(session.get("episode_id", session["user_id"])),
                )
        elif not trace_skip_llm_generation:
            hypotheses = hyp_generator.generate(
                translated,
                observed_history,
                M=M_hyp,
                generation_cfg=hyp_gen_cfg,
            )
        hyp_gen_audit_first = dict(hyp_generator.last_call_audit)

        # --- Vectorize hypotheses (contrastive_v2 first pass when configured) ---
        # item_texts for keyword fallback (used when BM25 unavailable)
        item_texts = getattr(dataset, "_item_texts_cache", None)
        hyp_timing: dict[str, float] = {}
        hyp_retrieval_diag: dict[str, Any] = {}
        trace_hv = (
            trace_ctrl.replay_hyp_vectorized(turn)
            if trace_ctrl is not None and trace_ctrl.is_replay
            else None
        )
        if trace_hv is not None:
            hyp_vectorized = trace_hv
        else:
            hyp_vectorized = vectorize_hypotheses(
                hypotheses,
                index,
                rep_k,
                bm25_index,
                item_texts=item_texts,
                timing_out=hyp_timing,
                representative_retrieval=rep_cfg,
                diagnostics_out=hyp_retrieval_diag,
            )

        # --- Optional: overlap-aware legacy fallback (contrastive_v2 only) ---
        guard_enabled_cfg = bool(guard_cfg.get("enabled", False))
        guard_mode = str(guard_cfg.get("mode", ""))
        overlap_pre = float(hyp_retrieval_diag.get("option_overlap_score", 0.0))
        mode_before = str(hyp_gen_audit_first.get("hypothesis_generation_mode_resolved") or "")

        guard_triggered = False
        guard_reason: str | None = None
        fallback_mode: str | None = None
        entropy_at_guard: float | None = None

        if (
            guard_enabled_cfg
            and requested_hyp_mode == "contrastive_v2"
            and guard_mode == "overlap_aware_legacy_fallback"
            and mode_before == "contrastive_v2"
        ):
            thr_ov = float(guard_cfg.get("overlap_threshold", 0.12))
            overlap_cond = overlap_pre >= thr_ov

            entropy_cond = True
            ent_metric = str(guard_cfg.get("entropy_metric") or "")
            ent_thr_raw = guard_cfg.get("entropy_threshold")
            if ent_thr_raw is not None and ent_metric:
                ent_thr_f = float(ent_thr_raw)
                if ent_metric == "top10_softmax_entropy":
                    entropy_at_guard = _top10_softmax_entropy(rec_scores)
                    entropy_cond = entropy_at_guard >= ent_thr_f
                elif ent_metric == "belief_entropy":
                    entropy_at_guard = float(belief_before.get("entropy", 0.0))
                    entropy_cond = entropy_at_guard >= ent_thr_f
                else:
                    entropy_cond = True

            if overlap_cond and entropy_cond:
                guard_triggered = True
                fallback_mode = str(guard_cfg.get("fallback_to", "legacy_hypothesis"))
                if ent_thr_raw is not None and ent_metric:
                    guard_reason = "high_option_overlap_high_entropy"
                else:
                    guard_reason = "high_option_overlap"

                leg_gen = dict(hyp_gen_cfg)
                leg_gen["mode"] = "legacy"
                hypotheses = hyp_generator.generate(
                    translated,
                    observed_history,
                    M=M_hyp,
                    generation_cfg=leg_gen,
                )
                leg_rep = dict(rep_cfg)
                leg_rep["mode"] = "legacy"
                hyp_retrieval_diag.clear()
                hyp_vectorized = vectorize_hypotheses(
                    hypotheses,
                    index,
                    rep_k,
                    bm25_index,
                    item_texts=item_texts,
                    timing_out=hyp_timing,
                    representative_retrieval=leg_rep,
                    diagnostics_out=hyp_retrieval_diag,
                )

        hyp_gen_resolved = str(hyp_generator.last_call_audit.get("hypothesis_generation_mode_resolved") or "")
        rep_mode_resolved = str(rep_cfg.get("mode", "legacy"))
        if guard_triggered:
            rep_mode_resolved = "legacy"

        llm_semantic_bridge: dict[str, Any] = {
            "direction_translator": dict(translator.last_call_audit),
            "hypothesis_generator": dict(hyp_generator.last_call_audit),
            "option_writer": {"skipped": True, "reason": "pending_until_question_phase"},
        }
        if requested_hyp_mode == "contrastive_v2":
            eval_guard = guard_enabled_cfg and guard_mode == "overlap_aware_legacy_fallback"
            if not eval_guard:
                g_triggered = False
                g_reason_out: str | None = "guard_config_not_active_wrong_mode" if guard_enabled_cfg else None
            else:
                g_triggered = bool(guard_triggered)
                g_reason_out = guard_reason
            glog = {
                "contrastive_v2_guard_enabled": bool(guard_enabled_cfg),
                "contrastive_v2_guard_triggered": g_triggered,
                "contrastive_v2_guard_reason": g_reason_out,
                "option_overlap_score_before_guard": overlap_pre,
                "guard_fallback_mode": fallback_mode,
                "hypothesis_generation_mode_before_guard": mode_before,
                "hypothesis_generation_mode_resolved": hyp_gen_resolved,
                "representative_retrieval_mode_resolved": rep_mode_resolved,
            }
            if entropy_at_guard is not None:
                glog["guard_entropy_value"] = entropy_at_guard
            if bool(guard_cfg.get("log_details", True)) and eval_guard:
                glog["overlap_threshold"] = float(guard_cfg.get("overlap_threshold", 0.12))
                glog["entropy_metric"] = str(guard_cfg.get("entropy_metric") or "")
                glog["entropy_threshold"] = guard_cfg.get("entropy_threshold")
            llm_semantic_bridge["contrastive_v2_overlap_guard"] = glog
        else:
            llm_semantic_bridge["contrastive_v2_overlap_guard"] = {
                "contrastive_v2_guard_enabled": False,
                "contrastive_v2_guard_triggered": False,
                "contrastive_v2_guard_reason": None,
                "option_overlap_score_before_guard": None,
                "guard_fallback_mode": None,
                "hypothesis_generation_mode_before_guard": None,
                "hypothesis_generation_mode_resolved": hyp_gen_resolved,
                "representative_retrieval_mode_resolved": rep_mode_resolved,
            }
        append_diagnostics_row(
            {
                "event": "phase",
                "component": "hypothesis_bm25",
                "latency_sec": round(hyp_timing.get("bm25_lexical_sec", 0.0), 6),
            }
        )
        append_diagnostics_row(
            {
                "event": "phase",
                "component": "hypothesis_dense_vectors",
                "latency_sec": round(hyp_timing.get("hypothesis_dense_vectors_sec", 0.0), 6),
            }
        )
        append_diagnostics_row(
            {
                "event": "phase",
                "component": "hypothesis_keyword_fallback",
                "latency_sec": round(hyp_timing.get("hypothesis_keyword_fallback_sec", 0.0), 6),
            }
        )
        log_cfg = cfg.get("logging") or {}
        hyp_log = []
        fc32_vector_binding_audit: dict[str, Any] | None = None
        for h in hyp_vectorized:
            entry: dict[str, Any] = {
                "hypothesis_id": h["hypothesis_id"],
                "text_description": h["text_description"],
                "feature_signature": h.get("feature_signature", []),
                "representative_items": h.get("representative_items", []),
                "vector_norm": h.get("vector_norm", 0.0),
            }
            if log_cfg.get("log_contrastive_hypothesis"):
                for k in (
                    "option_id",
                    "positive_intent",
                    "negative_contrast",
                    "distinguishing_features",
                    "contrast_text",
                    "short_user_facing_option_text",
                    "positive_retrieval_query",
                    "negative_retrieval_query",
                    "representative_diversity_hint",
                ):
                    if k in h:
                        entry[k] = h[k]
            if log_cfg.get("log_representative_items_scores"):
                entry["representative_items_scores"] = h.get("representative_items_scores", [])
                entry["representative_vector_source"] = h.get("representative_vector_source")
                entry["representative_diversity"] = h.get("representative_diversity")
                entry["centroid_ids"] = h.get("centroid_ids", [])
                entry["centroid_vectors_hash"] = h.get("centroid_vectors_hash", "")
            for k in (
                "vector_binding_source",
                "parsed_representative_items",
                "runtime_representative_items",
                "reps_match",
                "vector_binding_fallback",
                "fallback_reason",
            ):
                if k in h:
                    entry[k] = h[k]
            hyp_log.append(entry)

        fc32_vector_binding_audit: dict[str, Any] | None = None
        if frontier_gen_block_this_turn is not None:
            cb_v2_blk = (frontier_gen_block_this_turn.get("cluster_bound_generation_v2") or {})
            if cb_v2_blk.get("success"):
                fc32_vector_binding_audit = build_cluster_bound_vector_binding_audit(
                    frontier_gen_block_this_turn.get("parsed_hypotheses"),
                    hyp_vectorized,
                )
                frontier_gen_block_this_turn["cluster_bound_v2_vector_binding"] = fc32_vector_binding_audit

        # --- Question selection (optional cheap gate before full MC VOI) ---
        t_qsel0 = time.perf_counter()
        gate_latency_sec = 0.0
        full_voi_latency_sec = 0.0
        full_voi_called = False
        voi_gate_info: dict[str, Any] = {
            "voi_gate_enabled": gate_enabled,
            "voi_gate_mode": gate_mode if gate_enabled else None,
            "voi_gate_unc_score": None,
            "voi_gate_initial_uncertainty": None,
            "voi_gate_alpha": gate_alpha if gate_enabled else None,
            "voi_gate_mass": gate_mass if gate_enabled else None,
            "voi_gate_threshold": None,
            "voi_gate_passed": None,
            "voi_gate_skipped_full_voi": False,
            "voi_gate_reason": None,
            "dominant_set_size": None,
            "dominant_set_mass": None,
            "hypothesis_count": len(hyp_vectorized),
            "full_voi_called": False,
            "full_voi_latency_sec": 0.0,
            "gate_latency_sec": 0.0,
            "session_id": str(session.get("episode_id", session["user_id"])),
            "turn_id": turn,
        }

        fc_cfg_policy = hyp_gen_cfg.get("frontier_clustered_v32") or {}
        entropy_only_policy_enabled = (
            str(fc_cfg_policy.get("when_to_ask") or "").strip().lower() == "entropy_only"
        )
        run_full_voi = not entropy_only_policy_enabled

        if entropy_only_policy_enabled:
            selected_ids_entropy = [str(h.get("hypothesis_id")) for h in hyp_vectorized[: max(1, n_opt)]]
            sel = {
                "best_ids": selected_ids_entropy,
                "ig": 0.0,
                "option_probs": [],
                "candidate_count": len(hyp_vectorized),
                "profiling": {
                    "entropy_only_policy": True,
                    "full_voi_called": False,
                    "mc_samples": 0,
                    "hypothesis_count": len(hyp_vectorized),
                    "subset_size_n": min(n_opt, len(hyp_vectorized)),
                    "candidate_combo_evaluations": 0,
                },
            }
            voi_gate_info.update(
                {
                    "full_voi_called": False,
                    "full_voi_latency_sec": 0.0,
                    "gate_latency_sec": 0.0,
                    "voi_gate_passed": None,
                    "voi_gate_skipped_full_voi": True,
                    "voi_gate_reason": "entropy_only_policy_skips_mc_voi",
                }
            )
        elif gate_enabled and gate_mode in _HRV_GATE_MODES:
            t_gate0 = time.perf_counter()
            unc_score: float | None = None
            threshold: float | None = None
            relative_threshold: float | None = None
            online_threshold_component: float | None = None
            dom_sz: int | None = None
            dom_mass: float | None = None
            gate_reason: str | None = None
            stats: dict[str, Any] = {"ok": False}

            try:
                H_rows = [np.asarray(h["vector"], dtype=np.float64) for h in hyp_vectorized]
                if len(H_rows) == 0:
                    H_mat = np.zeros((0, belief.mu.shape[0]), dtype=np.float64)
                else:
                    H_mat = np.stack(H_rows, axis=0)

                memory = BayesianLatentIntentMemoryView.from_belief(belief)
                stats = dominant_pairwise_hrv_score(
                    H_mat, memory.covariance, memory.mean, tau=tau, mass=gate_mass
                )

                if not stats["ok"]:
                    fr = stats.get("fallback_reason") or "unknown"
                    if fr == "empty_hypothesis":
                        gate_reason = "fallback_empty_hypothesis"
                    elif fr in ("shape_mismatch_mu", "shape_mismatch_sigma", "h_bad_ndim"):
                        gate_reason = "fallback_shape_mismatch"
                    else:
                        gate_reason = f"fallback_{fr}"
                    run_full_voi = True
                    voi_gate_info.update(
                        {
                            "voi_gate_unc_score": unc_score,
                            "voi_gate_initial_uncertainty": session_initial_uncertainty,
                            "voi_gate_threshold": threshold,
                            "dominant_set_size": dom_sz,
                            "dominant_set_mass": dom_mass,
                            "voi_gate_reason": gate_reason,
                        }
                    )
                else:
                    unc_score = float(stats["score"])
                    dom_sz = int(stats["dominant_set_size"])
                    dom_mass = float(stats["dominant_set_mass"])
                    _merge_voi_gate_softmax_diagnostics(voi_gate_info, stats, log_details=gate_log_details)

                    if session_initial_uncertainty is None:
                        session_initial_uncertainty = unc_score
                        gate_reason = "first_gateable_turn_always_full_voi"
                        run_full_voi = True
                        thr_show = float(gate_alpha * session_initial_uncertainty)
                        voi_gate_info.update(
                            {
                                "voi_gate_unc_score": unc_score,
                                "voi_gate_initial_uncertainty": session_initial_uncertainty,
                                "voi_gate_threshold": thr_show,
                                "dominant_set_size": dom_sz,
                                "dominant_set_mass": dom_mass,
                                "voi_gate_reason": gate_reason,
                            }
                        )
                    else:
                        relative_threshold = float(gate_alpha * session_initial_uncertainty)
                        gate_max_skip_ds = int(voi_gate_cfg.get("max_skip_dominant_set_size", 2))
                        threshold = relative_threshold

                        if gate_mode == "dominant_pairwise_hrv_online_quantile":
                            hist_before = len(session_unc_history)
                            if hist_before < gate_online_min_hist:
                                threshold = relative_threshold
                                gate_reason_detail = "online_quantile_inactive_short_history"
                            else:
                                online_threshold_component = float(
                                    np.percentile(
                                        np.asarray(session_unc_history, dtype=np.float64),
                                        gate_online_q * 100.0,
                                    )
                                )
                                threshold = max(relative_threshold, online_threshold_component)
                                gate_reason_detail = "online_quantile_active"
                            voi_gate_info["voi_gate_online_quantile_q"] = gate_online_q
                            voi_gate_info["voi_gate_online_quantile_min_history"] = gate_online_min_hist
                            voi_gate_info["voi_gate_online_history_len_at_decision"] = hist_before
                            voi_gate_info["voi_gate_online_threshold_component"] = online_threshold_component
                            voi_gate_info["voi_gate_relative_threshold_component"] = relative_threshold
                            voi_gate_info["voi_gate_online_gate_detail"] = gate_reason_detail

                        if gate_mode in ("dominant_pairwise_hrv", "dominant_pairwise_hrv_online_quantile"):
                            assert threshold is not None
                            if unc_score > threshold:
                                gate_reason = (
                                    "unc_score_above_combined_threshold"
                                    if gate_mode == "dominant_pairwise_hrv_online_quantile"
                                    else "unc_score_above_relative_threshold"
                                )
                                run_full_voi = True
                            else:
                                gate_reason = (
                                    "unc_score_below_combined_threshold"
                                    if gate_mode == "dominant_pairwise_hrv_online_quantile"
                                    else "unc_score_below_relative_threshold"
                                )
                                run_full_voi = False
                            voi_gate_info.update(
                                {
                                    "voi_gate_unc_score": unc_score,
                                    "voi_gate_initial_uncertainty": session_initial_uncertainty,
                                    "voi_gate_threshold": threshold,
                                    "dominant_set_size": dom_sz,
                                    "dominant_set_mass": dom_mass,
                                    "voi_gate_reason": gate_reason,
                                }
                            )
                        elif gate_mode == "collapse_only":
                            threshold = relative_threshold
                            if dom_sz == 1:
                                run_full_voi = False
                                gate_reason = "dominant_set_collapse_only_skip"
                            else:
                                run_full_voi = True
                                gate_reason = "collapse_only_non_collapsed_requires_full_voi"
                            voi_gate_info.update(
                                {
                                    "voi_gate_unc_score": unc_score,
                                    "voi_gate_initial_uncertainty": session_initial_uncertainty,
                                    "voi_gate_threshold": threshold,
                                    "dominant_set_size": dom_sz,
                                    "dominant_set_mass": dom_mass,
                                    "voi_gate_reason": gate_reason,
                                }
                            )
                        elif gate_mode == "conservative_dominant_hrv":
                            threshold = relative_threshold
                            if dom_sz == 1:
                                run_full_voi = False
                                gate_reason = "conservative_dominant_set_collapse_skip"
                            elif unc_score <= relative_threshold and dom_sz is not None and dom_sz <= gate_max_skip_ds:
                                run_full_voi = False
                                gate_reason = "conservative_relative_threshold_small_dominant_skip"
                            else:
                                run_full_voi = True
                                gate_reason = "conservative_requires_full_voi"
                            voi_gate_info.update(
                                {
                                    "voi_gate_unc_score": unc_score,
                                    "voi_gate_initial_uncertainty": session_initial_uncertainty,
                                    "voi_gate_threshold": threshold,
                                    "dominant_set_size": dom_sz,
                                    "dominant_set_mass": dom_mass,
                                    "voi_gate_reason": gate_reason,
                                }
                            )

            except Exception:
                if gate_fallback_err:
                    voi_gate_info["voi_gate_reason"] = "fallback_exception"
                    run_full_voi = True
                else:
                    voi_gate_info["voi_gate_reason"] = "fallback_exception_no_full_voi"
                    run_full_voi = False

            gate_latency_sec = time.perf_counter() - t_gate0
            voi_gate_info["gate_latency_sec"] = round(gate_latency_sec, 6)
            voi_gate_info["voi_gate_initial_uncertainty"] = session_initial_uncertainty

            if gate_mode == "dominant_pairwise_hrv_online_quantile" and stats.get("ok") and unc_score is not None:
                session_unc_history.append(float(unc_score))

            if not run_full_voi:
                if preserve_ask_decision:
                    t0 = time.perf_counter()
                    sel = select_best_question(
                        belief,
                        hyp_vectorized,
                        n_opt,
                        tau,
                        none_bias,
                        none_threshold,
                        alpha_none,
                        mc_samples,
                    )
                    full_voi_latency_sec = time.perf_counter() - t0
                    full_voi_called = True
                    voi_gate_info["full_voi_called"] = True
                    voi_gate_info["full_voi_latency_sec"] = round(full_voi_latency_sec, 6)
                    voi_gate_info["voi_gate_passed"] = False
                    voi_gate_info["voi_gate_skipped_full_voi"] = False
                    voi_gate_info["voi_gate_advisory_skip"] = True
                    voi_gate_info["preserve_ask_decision"] = True
                else:
                    sel = empty_question_selection()
                    voi_gate_info["voi_gate_passed"] = False
                    voi_gate_info["voi_gate_skipped_full_voi"] = True
                    full_voi_called = False
                    if audit_skipped_voi and gate_reason in (
                        "unc_score_below_relative_threshold",
                        "unc_score_below_combined_threshold",
                        "dominant_set_collapse_only_skip",
                        "conservative_dominant_set_collapse_skip",
                        "conservative_relative_threshold_small_dominant_skip",
                    ):
                        t_aud = time.perf_counter()
                        audit_sel = select_best_question(
                            belief,
                            hyp_vectorized,
                            n_opt,
                            tau,
                            none_bias,
                            none_threshold,
                            alpha_none,
                            mc_samples,
                        )
                        aud_lat = time.perf_counter() - t_aud
                        audit_ig = float(audit_sel["ig"])
                        audit_voi_val = audit_ig - c_ask
                        voi_gate_info["voi_gate_audit"] = {
                            "audit_full_voi_ig": audit_ig,
                            "audit_full_voi_cost": c_ask,
                            "audit_full_voi_value": audit_voi_val,
                            "audit_full_voi_should_ask": bool(audit_voi_val > 0),
                            "audit_selected_question_size": len(audit_sel.get("best_ids") or []),
                            "audit_full_voi_latency_sec": round(aud_lat, 6),
                        }
            else:
                t0 = time.perf_counter()
                sel = select_best_question(
                    belief,
                    hyp_vectorized,
                    n_opt,
                    tau,
                    none_bias,
                    none_threshold,
                    alpha_none,
                    mc_samples,
                )
                full_voi_latency_sec = time.perf_counter() - t0
                full_voi_called = True
                voi_gate_info["full_voi_called"] = True
                voi_gate_info["full_voi_latency_sec"] = round(full_voi_latency_sec, 6)
                voi_gate_info["voi_gate_passed"] = True
                voi_gate_info["voi_gate_skipped_full_voi"] = False
        else:
            t0 = time.perf_counter()
            sel = select_best_question(
                belief,
                hyp_vectorized,
                n_opt,
                tau,
                none_bias,
                none_threshold,
                alpha_none,
                mc_samples,
            )
            full_voi_latency_sec = time.perf_counter() - t0
            full_voi_called = True
            voi_gate_info.update(
                {
                    "full_voi_called": True,
                    "full_voi_latency_sec": round(full_voi_latency_sec, 6),
                    "gate_latency_sec": 0.0,
                    "voi_gate_passed": None,
                    "voi_gate_skipped_full_voi": False,
                    "voi_gate_reason": "gate_disabled" if not gate_enabled else "gate_mode_no_match",
                }
            )

        ig = float(sel.get("ig", 0.0))
        do_ask, voi = should_ask(ig, c_ask)

        entropy_only_ask_gate_audit: dict[str, Any] | None = None
        if entropy_only_policy_enabled:
            belief_entropy_for_policy = (
                float(belief_before.get("entropy", 0.0))
                if belief_before.get("entropy") is not None
                else None
            )
            threshold_raw = fc_cfg_policy.get("entropy_only_min_belief_entropy")
            if threshold_raw is None:
                threshold_raw = fc_cfg_policy.get("entropy_ask_gate_min_belief_entropy")
            entropy_threshold = float(threshold_raw) if threshold_raw is not None else None
            max_sess_entropy = max(0, int(fc_cfg_policy.get("max_entropy_gate_asks_per_session", 1) or 1))
            selected_ids = list(sel.get("best_ids") or [])
            entropy_allowed = (
                belief_entropy_for_policy is not None
                and entropy_threshold is not None
                and belief_entropy_for_policy >= entropy_threshold
                and ask_count < max_sess_entropy
                and bool(selected_ids)
                and not (
                    fc32_oos_skip_audit is not None
                    and fc32_oos_skip_audit.get("oos_skip_applied")
                )
            )
            entropy_only_ask_gate_audit = {
                "enabled": True,
                "mode": "entropy_only",
                "used_original_voi_for_decision": False,
                "original_voi": float(voi),
                "original_ig": float(ig),
                "belief_entropy": belief_entropy_for_policy,
                "entropy_threshold": entropy_threshold,
                "ask_count_so_far": int(ask_count),
                "max_entropy_gate_asks_per_session": int(max_sess_entropy),
                "selected_question_ids": selected_ids,
                "ask": bool(entropy_allowed),
                "reason": (
                    "entropy_above_threshold_budget_available"
                    if entropy_allowed
                    else "entropy_below_threshold_or_budget_or_empty_selection"
                ),
                "used_future_for_runtime_decision": False,
                "used_target_for_runtime_decision": False,
            }
            do_ask = bool(entropy_allowed)
            voi = float("nan")

        midask_cfg = ask_cfg.get("midask_guard") or {}
        midask_enabled = bool(midask_cfg.get("enabled", False))
        midask_mode = str(midask_cfg.get("mode", "low_confidence_force_ask"))
        ask_min = int(midask_cfg.get("ask_count_min", 3))
        ask_max = int(midask_cfg.get("ask_count_max", 4))

        belief_entropy_val = float(belief_before.get("entropy", 0.0))
        st_raw = belief_before.get("sigma_trace")
        sigma_trace_f = float(st_raw) if st_raw is not None else None
        margin_val: float | None = None
        if len(rec_scores) >= 2:
            margin_val = float(rec_scores[0] - rec_scores[1])
        overlap_mid = float(hyp_retrieval_diag.get("option_overlap_score", 0.0))

        original_should_ask = bool(do_ask)
        midask_triggered = False
        midask_reason: str | None = None
        midask_conditions: list[str] = []
        voi_eval = float(voi)
        sel_refilled = False

        if (
            midask_enabled
            and midask_mode == "low_confidence_force_ask"
            and ask_min <= ask_count <= ask_max
        ):
            sel_eval = sel
            if not sel_eval.get("best_ids"):
                sel_eval = select_best_question(
                    belief,
                    hyp_vectorized,
                    n_opt,
                    tau,
                    none_bias,
                    none_threshold,
                    alpha_none,
                    mc_samples,
                )
                sel_refilled = True

            voi_eval = float(sel_eval.get("ig", 0.0)) - c_ask
            natural_do_ask = voi_eval > 0
            best_ids_e = list(sel_eval.get("best_ids") or [])

            apply_force, conds = should_apply_midask_force(
                voi_eval=voi_eval,
                best_ids=best_ids_e,
                natural_do_ask=natural_do_ask,
                cfg=midask_cfg,
                belief_entropy=belief_entropy_val,
                sigma_trace=sigma_trace_f,
                margin=margin_val,
                overlap=overlap_mid,
            )
            midask_conditions = list(conds)

            if apply_force:
                midask_triggered = True
                midask_reason = "midask_low_confidence_force_ask"
                sel = sel_eval
                ig = float(sel["ig"])
                voi = float(voi_eval)
                do_ask = True
            elif sel_refilled and best_ids_e:
                sel = sel_eval
                ig = float(sel["ig"])
                do_ask, voi = should_ask(ig, c_ask)

        policy_natural_should_ask = bool(do_ask)
        policy_ig_for_log = float(sel.get("ig", 0.0))
        policy_voi_for_log = float(voi)
        policy_best_ids_snap = list(sel.get("best_ids") or [])

        if fc32_oos_skip_audit is not None and fc32_oos_skip_audit.get("oos_skip_applied"):
            do_ask = False
            voi = float(voi_eval) - c_ask

        fc32_force_ask_audit: dict[str, Any] | None = None
        fc32_cb_v2_force_delivery_audit: dict[str, Any] | None = None
        fc32_cb_v2_entropy_gate_audit: dict[str, Any] | None = None
        fc_cfg_force = hyp_gen_cfg.get("frontier_clustered_v32") or {}
        n_force_cap = int(fc_cfg_force.get("force_ask_turns", 0) or 0)
        if (
            bool(fc_cfg_force.get("force_ask_for_sanity"))
            and frontier_gen_block_this_turn is not None
            and n_force_cap > 0
            and turn <= n_force_cap
        ):
            fc32_force_ask_audit = {
                "force_ask_for_sanity": True,
                "original_should_ask": policy_natural_should_ask,
                "forced_should_ask": True,
                "force_ask_reason": f"frontier_clustered_v32.force_ask_for_sanity_turn_leq_{n_force_cap}",
                "preserved_policy_IG": policy_ig_for_log,
                "preserved_policy_VOI": policy_voi_for_log,
                "preserved_policy_best_question_ids": policy_best_ids_snap,
            }
            if not sel.get("best_ids"):
                sel = select_best_question(
                    belief,
                    hyp_vectorized,
                    n_opt,
                    tau,
                    none_bias,
                    none_threshold,
                    alpha_none,
                    mc_samples,
                )
            do_ask = True

        fc_cfg_delivery = hyp_gen_cfg.get("frontier_clustered_v32") or {}
        oos_skip_applied = bool(
            fc32_oos_skip_audit is not None and fc32_oos_skip_audit.get("oos_skip_applied")
        )
        should_force_delivery, fc32_cb_v2_force_delivery_audit = try_cb_v2_force_delivery_smoke(
            fc_cfg_delivery,
            frontier_gen_block=frontier_gen_block_this_turn,
            policy_natural_should_ask=policy_natural_should_ask,
            oos_skip_applied=oos_skip_applied,
            session_forced_count=fc32_cb_v2_force_delivery_session_count,
            original_voi=float(policy_voi_for_log),
        )
        if should_force_delivery and fc32_cb_v2_force_delivery_audit is not None:
            if not sel.get("best_ids"):
                sel = select_best_question(
                    belief,
                    hyp_vectorized,
                    n_opt,
                    tau,
                    none_bias,
                    none_threshold,
                    alpha_none,
                    mc_samples,
                )
            do_ask = True
            fc32_cb_v2_force_delivery_session_count += 1
            if isinstance(frontier_gen_block_this_turn, dict):
                cb_blk = frontier_gen_block_this_turn.setdefault("cluster_bound_generation_v2", {})
                if isinstance(cb_blk, dict):
                    cb_blk.update(fc32_cb_v2_force_delivery_audit)

        fc32_cb_v2_entropy_gate_audit: dict[str, Any] | None = None
        belief_ent_for_gate = float(belief_before.get("entropy", 0.0)) if belief_before.get("entropy") is not None else None
        if belief_ent_for_gate is not None:
            fc32_session_entropy_samples.append(belief_ent_for_gate)
        should_entropy_gate, fc32_cb_v2_entropy_gate_audit = try_cb_v2_entropy_ask_gate(
            fc_cfg_delivery,
            frontier_gen_block=frontier_gen_block_this_turn,
            policy_natural_should_ask=policy_natural_should_ask,
            oos_skip_applied=oos_skip_applied,
            session_gate_count=fc32_cb_v2_entropy_gate_session_count,
            belief_entropy=belief_ent_for_gate,
            session_entropy_samples=list(fc32_session_entropy_samples),
            original_voi=float(policy_voi_for_log),
        )
        if should_entropy_gate and fc32_cb_v2_entropy_gate_audit is not None:
            if not sel.get("best_ids"):
                sel = select_best_question(
                    belief,
                    hyp_vectorized,
                    n_opt,
                    tau,
                    none_bias,
                    none_threshold,
                    alpha_none,
                    mc_samples,
                )
            do_ask = True
            fc32_cb_v2_entropy_gate_session_count += 1
            if isinstance(frontier_gen_block_this_turn, dict):
                cb_blk = frontier_gen_block_this_turn.setdefault("cluster_bound_generation_v2", {})
                if isinstance(cb_blk, dict):
                    cb_blk.update(fc32_cb_v2_entropy_gate_audit)

        # Lazy-sanity optional: cap total session asks (YAML-only; does not affect non-lazy v32 runs).
        fc_lazy_cap_cfg = hyp_gen_cfg.get("frontier_clustered_v32") or {}
        total_ask_cap_raw = fc_lazy_cap_cfg.get("max_total_asks_per_session_for_lazy_sanity")
        if (
            requested_hyp_mode == "frontier_clustered_v32"
            and frontier_cluster_session_armed(hyp_gen_cfg)
            and bool(fc_lazy_cap_cfg.get("lazy_generation_enabled", False))
            and total_ask_cap_raw is not None
        ):
            cap_n = max(0, int(total_ask_cap_raw))
            if bool(do_ask and sel.get("best_ids")) and ask_count >= cap_n:
                do_ask = False
                if fc32_budget_state_audit is not None:
                    fc32_budget_state_audit["max_total_asks_per_session_for_lazy_sanity"] = cap_n
                    fc32_budget_state_audit["total_ask_budget_blocked"] = True
                    fc32_budget_state_audit["total_ask_budget_reason"] = (
                        "max_total_asks_per_session_for_lazy_sanity_reached"
                    )
                else:
                    fc32_budget_state_audit = {
                        "lazy_total_ask_cap_only_telemetry": True,
                        "ask_count_so_far": int(ask_count),
                        "v32_generation_count_so_far": int(fc32_v32_generation_count_session),
                        "max_total_asks_per_session_for_lazy_sanity": cap_n,
                        "total_ask_budget_blocked": True,
                        "total_ask_budget_reason": "max_total_asks_per_session_for_lazy_sanity_reached",
                        "pre_gate_pass": None,
                        "budget_blocked": False,
                        "budget_block_reason": None,
                        "pre_gate_reason": None,
                        "cooldown_remaining_turns": None,
                    }

        fc32_rs_repair_cfg = dict(hyp_gen_cfg.get("frontier_clustered_v32") or {})
        if (
            requested_hyp_mode == "frontier_clustered_v32"
            and frontier_cluster_session_armed(hyp_gen_cfg)
            and fc32_routing_repair_audit is None
        ):
            pg_for_rescue = fc32_pre_gate_audit if isinstance(fc32_pre_gate_audit, dict) else {}
            rescue_ok, rescue_reasons = evaluate_near_miss_rescue_strict(
                fc32_cfg=fc32_rs_repair_cfg,
                state=fc32_routing_repair_state,
                turn_idx=turn,
                ask_count_so_far=int(ask_count),
                do_ask=bool(do_ask),
                gate_pass=pg_for_rescue.get("pass") if pg_for_rescue else None,
                gate_reason=str(pg_for_rescue.get("reason") or "pass") if pg_for_rescue else None,
                voi_eval=float(voi_eval),
                original_should_ask=bool(original_should_ask),
                belief_entropy=belief_entropy_val,
                margin_top12=margin_val,
            )
            if rescue_ok:
                if not sel.get("best_ids"):
                    sel = select_best_question(
                        belief,
                        hyp_vectorized,
                        n_opt,
                        tau,
                        none_bias,
                        none_threshold,
                        alpha_none,
                        mc_samples,
                    )
                original_decision = bool(do_ask)
                do_ask = True
                record_near_miss_rescue(fc32_routing_repair_state)
                fc32_routing_repair_audit = build_near_miss_rescue_audit(
                    turn_idx=turn,
                    reasons=rescue_reasons,
                    state=fc32_routing_repair_state,
                    original_decision=original_decision,
                    final_decision=True,
                    voi_eval=float(voi_eval),
                    belief_entropy=belief_entropy_val,
                    margin_top12=margin_val,
                    ask_count_so_far=int(ask_count),
                )

        fc32_rank_safe_guard_audit: dict[str, Any] | None = None
        if (
            requested_hyp_mode == "frontier_clustered_v32"
            and frontier_cluster_session_armed(hyp_gen_cfg)
            and bool(do_ask and sel.get("best_ids"))
        ):
            fc32_rs_cfg = dict(hyp_gen_cfg.get("frontier_clustered_v32") or {})
            if bool(fc32_rs_cfg.get("rank_safe_guard_enabled", False)):
                v32_substantive = is_v32_substantive_generation_block(frontier_gen_block_this_turn)
                scope_v32_only = rank_safe_guard_scope_v32_only(fc32_rs_cfg)

                if scope_v32_only and not v32_substantive:
                    oos_reason = "no_frontier_generation_block"
                    pg = fc32_pre_gate_audit if isinstance(fc32_pre_gate_audit, dict) else {}
                    if pg and not pg.get("pass", True):
                        oos_reason = f"pre_gate:{pg.get('reason')}"
                    fc32_rank_safe_guard_audit = build_out_of_scope_non_v32_guard_audit(
                        cfg=fc32_rs_cfg,
                        reason=oos_reason,
                        turn=turn,
                    )
                else:
                    gen_parsed = None
                    if isinstance(frontier_gen_block_this_turn, dict):
                        gen_parsed = frontier_gen_block_this_turn.get("parsed_hypotheses")
                    fc32_rank_safe_guard_audit = compute_rank_safe_guard(
                        turn_log_context={
                            "turn": turn,
                            "episode_id": session.get("episode_id", session["user_id"]),
                            "parsed_hypotheses": gen_parsed,
                            "selected_hypothesis_ids": list(sel.get("best_ids") or []),
                            "selected_hypothesis_id": (
                                str(sel["best_ids"][0]) if sel.get("best_ids") else None
                            ),
                        },
                        recommendation_scores=list(rec_scores[:10]),
                        previous_top10=list(fc32_prev_turn_top10_snapshot or ()),
                        current_top10=list(cur_top10),
                        belief_state=belief_before,
                        ask_history={
                            "ask_count_so_far": int(ask_count),
                            "consecutive_ask_no_rank_change": int(fc32_consecutive_ask_rank_unchanged),
                            "prev_margin12": fc32_prev_turn_margin12,
                            "ask_effect_memory_ring": list(fc32_ask_effect_memory_ring),
                        },
                        config=fc32_rs_cfg,
                        frontier_generation_exists=bool(v32_substantive),
                        hyp_vectorized=list(hyp_vectorized),
                        parsed_hypotheses=gen_parsed if isinstance(gen_parsed, list) else None,
                        embedding_index=index,
                    )
                    log_only = bool(fc32_rs_cfg.get("rank_safe_guard_log_only", True))
                    applied = bool(
                        not log_only
                        and (
                            fc32_rank_safe_guard_audit.get("should_block_ask")
                            or fc32_rank_safe_guard_audit.get("should_conservative_only")
                        )
                    )
                    fc32_rank_safe_guard_audit["rank_safe_guard_scope"] = "v32_generation_turn"
                    fc32_rank_safe_guard_audit["rank_safe_guard_scope_reason"] = (
                        "v32_substantive_generation_with_role_mcq"
                    )
                    fc32_rank_safe_guard_audit["v41_action_applied"] = applied
                    fc32_rank_safe_guard_audit["non_v32_original_question_preserved"] = False

                    if fc32_rank_safe_guard_audit.get("should_block_ask") and not log_only:
                        do_ask = False
                    elif fc32_rank_safe_guard_audit.get("should_conservative_only") and not log_only:
                        cons_hid = fc32_rank_safe_guard_audit.get("conservative_hypothesis_id")
                        if cons_hid and sel.get("best_ids"):
                            bid = str(cons_hid)
                            if bid in sel["best_ids"]:
                                sel = dict(sel)
                                sel["best_ids"] = [bid]
                            elif sel["best_ids"]:
                                sel = dict(sel)
                                sel["best_ids"] = [str(sel["best_ids"][0])]
                        if isinstance(frontier_gen_block_this_turn, dict):
                            rewrite_mode = resolve_conservative_rewrite_mode(fc32_rs_cfg)
                            apply_conservative_rewrite_to_generation_block(
                                frontier_gen_block_this_turn,
                                conservative_hypothesis_id=str(cons_hid) if cons_hid else None,
                                rewrite_mode=rewrite_mode,
                            )
                            fc32_rank_safe_guard_audit["conservative_rewrite_mode"] = rewrite_mode

        diag_phase: dict[str, Any] = {
            "event": "phase",
            "component": "question_selection_voi",
            "latency_sec": round(time.perf_counter() - t_qsel0, 6),
            "full_voi_called": full_voi_called,
            "full_voi_latency_sec": round(full_voi_latency_sec, 6),
            "gate_latency_sec": round(gate_latency_sec, 6),
        }
        if gate_log_details:
            diag_phase.update({k: v for k, v in voi_gate_info.items() if not k.startswith("_")})
        append_diagnostics_row(diag_phase)

        log_midask_details = bool(midask_cfg.get("log_details", True))
        midask_log: dict[str, Any] = {
            "midask_guard_enabled": midask_enabled,
            "midask_guard_triggered": midask_triggered,
            "midask_guard_reason": midask_reason,
            "midask_guard_original_should_ask": original_should_ask,
            "midask_guard_final_should_ask": policy_natural_should_ask,
            "midask_guard_ask_count_so_far": int(ask_count),
            "midask_guard_voi": float(voi),
            "midask_guard_voi_eval": float(voi_eval),
            "midask_guard_entropy": belief_entropy_val,
            "midask_guard_sigma_trace": sigma_trace_f,
            "midask_guard_margin": margin_val,
            "midask_guard_option_overlap": overlap_mid,
            "midask_guard_sel_refilled": sel_refilled,
            "midask_guard_consecutive_no_ask_tail_before": int(consecutive_no_ask_tail),
        }
        if log_midask_details:
            midask_log["midask_guard_conditions_met"] = midask_conditions

        if trace_ctrl is not None and trace_ctrl.is_replay:
            pol = trace_ctrl.replay_policy(turn)
            if pol is None:
                trace_ctrl.require_turn(turn)
            else:
                do_ask, sel = pol
                if "best_ids" not in sel and sel.get("best_question_ids"):
                    sel["best_ids"] = list(sel.get("best_question_ids") or [])
                sel.setdefault("candidate_count", len(sel.get("best_ids") or []))
                sel.setdefault("profiling", {})
                oos_from_trace = trace_ctrl.replay_oos_skip_audit(turn)
                if oos_from_trace is not None and fc32_oos_skip_audit is None:
                    fc32_oos_skip_audit = oos_from_trace
                    fc32_oos_skip_count_session += 1
                if oos_from_trace is not None and oos_from_trace.get("oos_skip_applied"):
                    do_ask = False

        q_sel_log = {
            "candidate_question_count": sel["candidate_count"],
            "best_question_ids": sel["best_ids"],
            "IG": policy_ig_for_log,
            "VOI": policy_voi_for_log,
            "natural_policy_should_ask": policy_natural_should_ask,
            "ask": bool(do_ask),
            "c_ask": c_ask,
            "voi_gate": voi_gate_info,
            "full_voi_profiling": sel.get("profiling"),
            "midask_guard": midask_log,
        }
        if entropy_only_ask_gate_audit is not None:
            q_sel_log["entropy_only_ask_gate"] = entropy_only_ask_gate_audit

        # --- Question + user choice ---
        belief_update_executed = False
        fc32_closed_loop_audit = {}
        csc_readout_audit: dict[str, Any] | None = None
        question_log: dict[str, Any] = {}
        user_choice_log: dict[str, Any] = {}
        belief_after_log: dict[str, Any] = {}
        surface_writer_audit: dict[str, Any] = {"skipped": True, "reason": "not_ask_turn"}

        if do_ask and sel["best_ids"]:
            ask_count += 1
            if frontier_gen_block_this_turn is not None:
                fc32_v32_ask_count_session += 1
            selected_hyps = (
                trace_ctrl.replay_selected_hypotheses(turn, hyp_vectorized, sel["best_ids"])
                if trace_ctrl is not None and trace_ctrl.is_replay
                else [h for h in hyp_vectorized if h["hypothesis_id"] in sel["best_ids"]]
            )
            if trace_ctrl is not None and trace_ctrl.is_replay and not selected_hyps:
                trace_ctrl.record_missing_field(turn, "hyp_vectorized", "missing_for_ask_turn")
                raise TraceReplayMissError(
                    f"trace missing vectorized hypotheses session={ep_id} turn={turn}"
                )
            rep_items_map = {h["hypothesis_id"]: h.get("representative_items", []) for h in selected_hyps}
            frontier_choice_reuse: tuple[str, list[dict[str, Any]], dict[str, Any]] | None = None
            if (
                frontier_gen_block_this_turn is not None
                and bool(fc32_cfg_turn.get("use_frontier_choice_question_for_ask", False))
                and trace_ctrl is None
            ):
                frontier_choice_reuse = _frontier_choice_options_for_selected_hypotheses(
                    frontier_gen_block_this_turn,
                    selected_hyps,
                    include_none=bool(cfg.get("ask", {}).get("include_none", True)),
                )
            if trace_ctrl is not None and trace_ctrl.is_replay:
                include_none_opt = bool(cfg.get("ask", {}).get("include_none", True))
                options = trace_ctrl.replay_option_writer_output(
                    turn, selected_hyps, include_none=include_none_opt
                )
                option_writer.implementation_mode = "trace_replay"
                option_writer.last_call_audit = {
                    "component": "option_writer",
                    "implementation_mode": "trace_replay",
                    "cache_hit": True,
                    "cache_miss": False,
                    "n_api_calls": 0,
                    "parse_success": True,
                }
                neutrality_log = {
                    "mode": "trace_replay",
                    "skipped": True,
                    "balanced_final": True,
                    "retries_used": 0,
                }
            elif frontier_choice_reuse is not None:
                q_text, options, frontier_choice_audit = frontier_choice_reuse
                option_writer.implementation_mode = "frontier_choice_question_reuse"
                option_writer.last_call_audit = {
                    "component": "option_writer",
                    "implementation_mode": "frontier_choice_question_reuse",
                    "parse_success": True,
                    "cache_hit": False,
                    "cache_miss": False,
                    "n_api_calls": 0,
                    "fallback_reason": None,
                    **frontier_choice_audit,
                }
                llm_semantic_bridge["option_writer"] = dict(option_writer.last_call_audit)
                neutrality_log = {
                    "mode": "frontier_choice_question_reuse",
                    "skipped": True,
                    "balanced_final": True,
                    "retries_used": 0,
                }
            else:
                options_raw = option_writer.write(
                    selected_hyps,
                    rep_items_map,
                    session_id=ep_id,
                    turn_id=turn,
                )
                llm_semantic_bridge["option_writer"] = dict(option_writer.last_call_audit)
                neut_cfg = ask_cfg.get("neutrality") or {}
                t0 = time.perf_counter()
                options, neutrality_log = neutrality_check_options(
                    translator.llm,
                    selected_hyps,
                    options_raw,
                    mode=str(neut_cfg.get("mode", "rule")),
                    max_retries=int(neut_cfg.get("max_retries", 3)),
                    include_none=cfg.get("ask", {}).get("include_none", True),
                )
                append_diagnostics_row(
                    {
                        "event": "phase",
                        "component": "neutrality_total",
                        "latency_sec": round(time.perf_counter() - t0, 6),
                        "neutrality_retries_used": neutrality_log.get("retries_used"),
                        "neutrality_balanced_final": neutrality_log.get("balanced_final"),
                    }
                )
            if trace_ctrl is not None and trace_ctrl.is_replay:
                llm_semantic_bridge["option_writer"] = dict(option_writer.last_call_audit)
                trace_turn = trace_ctrl.get_turn(turn) or {}
                q_text = str(trace_turn.get("question_text") or "")
            elif frontier_choice_reuse is None:
                q_text = ""
            option_ids = [o["option_id"] for o in options if o["option_id"] != "none"]
            option_texts = [o["option_text"] for o in options]
            opt_vectors = np.stack([h["vector"] for h in selected_hyps])

            if trace_ctrl is not None and trace_ctrl.is_replay:
                choice_result = trace_ctrl.replay_user_choice(turn)
                if choice_result is None:
                    trace_ctrl.require_turn(turn)
                    choice_result = simulator.choose(opt_vectors, option_ids, include_none=True)
            else:
                choice_result = simulator.choose(opt_vectors, option_ids, include_none=True)
            selected_oid = choice_result["selected_option_id"]
            sel_vec_idx = choice_result["selected_vector_index"]

            t0 = time.perf_counter()
            belief_pre_laplace = belief.copy()
            belief, belief_update_component, memory_update_trace = apply_choice_belief_update(
                ctx=memory_ctx,
                belief=belief,
                belief_pre_laplace=belief_pre_laplace,
                opt_vectors=opt_vectors,
                option_ids=option_ids,
                selected_oid=str(selected_oid),
                sel_vec_idx=sel_vec_idx,
                tau=tau,
                alpha_none=alpha_none,
                ask_cfg=ask_cfg,
                turn=turn,
                recommender=recommender,
                cur_top10=[str(x) for x in cur_top10],
                rec_scores=list(rec_scores),
                seen_items=seen_items,
                top_k=top_k,
            )
            append_diagnostics_row(
                {
                    "event": "phase",
                    "component": belief_update_component,
                    "latency_sec": round(time.perf_counter() - t0, 6),
                    "memory_mode": memory_ctx.mode if memory_ctx.enabled else "original",
                    "memory_operation": (memory_update_trace or {}).get("operation"),
                    "compatible_update_ema": (memory_update_trace or {}).get("compatible_update_ema"),
                    "memory_update_applied_to_belief": (memory_update_trace or {}).get(
                        "memory_update_applied_to_belief"
                    ),
                }
            )

            if (
                frontier_gen_block_this_turn is not None
                and csc_enabled(fc32_cfg_turn)
            ):
                theta_items = [str(x) for x in (theta_info.get("theta_star_items") or [])[:1]]
                audit_target_id = theta_items[0] if theta_items else None
                overlay, csc_readout_audit = try_build_csc_overlay(
                    index=index,
                    belief_before=belief_pre_laplace,
                    belief_after=belief,
                    lambda_explore=float(rec_cfg.get("lambda_explore", 0.1)),
                    seen=seen_items,
                    fc32_gen=frontier_gen_block_this_turn,
                    selected_option_id=str(selected_oid),
                    hyp_vectorized=selected_hyps,
                    csc_cfg=csc_config_from_fc32(fc32_cfg_turn),
                    audit_target_item_id=audit_target_id,
                    logged_hypotheses=hyp_log,
                )
                if overlay is not None:
                    recommender.set_csc_overlay(overlay)
                elif csc_readout_audit and csc_readout_audit.get("csc_fallback_reason"):
                    recommender.clear_csc_overlay()

            belief_update_executed = True

            surface_writer_cfg = dict(cfg.get("surface_writer") or {})
            if trace_ctrl is None and surface_writer_cfg.get("mode"):
                SurfaceWriter = _load_surface_writer_cls()
                sw = SurfaceWriter(surface_writer_cfg)
                sem_opts: list[dict[str, Any]] = []
                for h, opt in zip(selected_hyps, [o for o in options if o.get("option_id") != "none"]):
                    sem_opts.append(
                        {
                            **h,
                            "option_id": str(opt.get("option_id") or h.get("hypothesis_id") or ""),
                            "option_text": opt.get("option_text"),
                        }
                    )
                sw_ctx = {
                    "dataset": str(cfg.get("dataset", {}).get("name") or ""),
                    "episode_id": ep_id,
                    "turn": turn,
                    "run_id": surface_writer_cfg.get("cache_run_id") or "",
                    "implementation_mode": str(hyp_generator.implementation_mode or ""),
                    "option_ids": option_ids,
                    "cluster_phrase": str(
                        (frontier_gen_block_this_turn or {}).get("choice_question", {}).get("cluster_phrase")
                        if isinstance(frontier_gen_block_this_turn, dict)
                        else ""
                    ),
                    "memory_summary": "",
                    "ranking_frontier_summary": "",
                }
                try:
                    from interrec_noapi.surface_writer.hybrid_cache_keys import compute_option_payload_hash

                    sw_ctx["option_payload_hash"] = compute_option_payload_hash(
                        sw_ctx["dataset"], ep_id, turn, selected_hyps
                    )
                except Exception:
                    pass
                sw_result = sw.write_question(sem_opts, sw_ctx, option_id_order=option_ids)
                if sw_result.mode == "local_small_model" and sw_result.local_model_loaded:
                    q_text = sw_result.question
                    id_to_text = {oid: txt for oid, txt in zip(option_ids, sw_result.options)}
                    option_texts = [
                        id_to_text.get(str(o.get("option_id")), str(o.get("option_text") or ""))
                        if o.get("option_id") != "none"
                        else str(o.get("option_text") or "")
                        for o in options
                    ]
                surface_writer_audit = {
                    "mode": sw_result.mode,
                    "local_model_path": surface_writer_cfg.get("local_model_path"),
                    "local_model_loaded": bool(sw_result.local_model_loaded),
                    "fallback_used": sw_result.mode != "local_small_model",
                    "fallback_reason": sw_result.fallback_reason,
                    "latency_sec": sw_result.latency_sec,
                    "api_calls": sw_result.api_calls,
                    **dict(sw.last_audit),
                }
            elif trace_ctrl is not None:
                surface_writer_audit = {"skipped": True, "reason": "trace_replay"}

            question_log = {
                "question_text": q_text
                or "Which of the following best describes your current interest?",
                "options": option_texts,
                "neutrality": neutrality_log,
            }
            user_choice_log = {
                "theta_star_source": split_key,
                "theta_star_items": theta_info["theta_star_items"][:5],
                **choice_result,
            }
            belief_after_log = belief.summary(include_eigendecomp=include_eigenspectrum_logging)

            if frontier_gen_block_this_turn is not None:
                hid_sel = ""
                if str(selected_oid) != "none":
                    hid_sel = str(selected_oid)
                cmap_audit = (
                    _frontier_fc32_cluster_cells_for_hypothesis(frontier_gen_block_this_turn, hid_sel)
                    if hid_sel
                    else {}
                )
                cq = frontier_gen_block_this_turn.get("choice_question")
                cq = cq if isinstance(cq, dict) else {}
                fc32_closed_loop_audit = {
                    "do_ask": True,
                    "frontier_clustered_v32_generation_attached_turn": True,
                    "planned_cluster_contrasts": frontier_gen_block_this_turn.get(
                        "planned_cluster_contrasts",
                    ),
                    "parsed_hypotheses_n": len(
                        frontier_gen_block_this_turn.get("parsed_hypotheses") or [],
                    ),
                    "choice_question": cq,
                    "option_to_cluster_mapping": frontier_gen_block_this_turn.get(
                        "option_to_cluster_mapping",
                    ),
                    "session_presented_question_text": question_log.get("question_text"),
                    "session_presented_option_ids": list(option_ids),
                    "session_presented_option_texts": list(option_texts),
                    "schema_validation_result": frontier_gen_block_this_turn.get(
                        "schema_validation_result",
                    ),
                    "fallback_used": frontier_gen_block_this_turn.get("fallback_used"),
                    "fallback_reason": frontier_gen_block_this_turn.get("fallback_reason"),
                    "simulated_user_choice": dict(choice_result),
                    "selected_option_id": str(selected_oid),
                    "selected_hypothesis_id": hid_sel or None,
                    "selected_cluster_ref_audit": cmap_audit,
                    "belief_update_executed": True,
                    "belief_entropy_delta_vs_before_turn": (
                        float(belief_before.get("entropy") or 0.0)
                        - float(belief_after_log.get("entropy") or 0.0)
                    ),
                }
        else:
            belief_after_log = belief_before.copy()
            llm_semantic_bridge["option_writer"] = {
                "skipped": True,
                "reason": "no_ask_selected_or_selection_empty",
            }

        impl_modes_turn = dict(base_impl_modes_template)
        impl_modes_turn["direction_translator"] = translator.implementation_mode
        impl_modes_turn["hypothesis_generator"] = hyp_generator.implementation_mode
        impl_modes_turn["option_writer"] = option_writer.implementation_mode
        impl_modes_turn["surface_writer"] = surface_writer_audit.get("mode") if isinstance(surface_writer_audit, dict) else None
        llm_semantic_bridge["surface_writer"] = surface_writer_audit

        # preference_error: ||mu_t - theta*||
        pref_error = float(np.linalg.norm(belief.mu - theta_star))
        final_recs = recommender.recommend(belief, top_k=top_k, exclude=seen_items)
        final_rec_ids = [item_id for item_id, _ in final_recs]
        final_rec_scores = [score for _, score in final_recs]
        final_m5 = ranking_metrics(final_rec_ids, relevant, k=5)
        final_m10 = ranking_metrics(final_rec_ids, relevant, k=10)
        final_mk = ranking_metrics_multik(final_rec_ids, relevant, ks=(5, 10))
        final_rec_log = {
            "metric_source": "post_update_final_topk",
            "recommended_items": final_rec_ids[:10],
            "scores": final_rec_scores[:10],
            **final_m5,
            **final_m10,
            **final_mk,
        }

        turn_log: dict[str, Any] = {
            "user_id": session["user_id"],
            "episode_id": session.get("episode_id", session["user_id"]),
            "turn": turn,
            "data_split": {
                "observed_history_count": len(observed_history),
                "theta_star_source": split_key,
                "future_train_count": len(session.get("future_train", [])),
                "future_valid_count": len(session.get("future_valid", [])),
                "future_test_count": len(session.get("future_test", [])),
            },
            "implementation_modes": impl_modes_turn,
            "llm_semantic_bridge": llm_semantic_bridge,
            "belief_before": belief_before,
            "recommendation": rec_log,
            "final_recommendation_corrected": final_rec_log,
            "uncertainty_directions": anchor_dirs_log,
            "hypotheses": hyp_log,
            "question_selection": q_sel_log,
            "question": question_log,
            "user_choice": user_choice_log,
            "belief_after": belief_after_log,
            "metrics": {
                "entropy_delta": belief_before["entropy"] - belief_after_log.get("entropy", belief_before["entropy"]),
                "preference_error": pref_error,
                # Session-level metrics must describe the post-update ranking
                # exposed as final_recommendation_corrected.
                **final_mk,
                "ask_count_so_far": ask_count,
            },
            "belief_update_executed": belief_update_executed,
            "do_ask": bool(do_ask and sel.get("best_ids")),
        }
        turn_log.update(midask_log)
        gbr = llm_semantic_bridge.get("contrastive_v2_overlap_guard") or {}
        turn_log["contrastive_v2_guard_enabled"] = bool(gbr.get("contrastive_v2_guard_enabled", False))
        turn_log["contrastive_v2_guard_triggered"] = bool(gbr.get("contrastive_v2_guard_triggered", False))
        turn_log["contrastive_v2_guard_reason"] = gbr.get("contrastive_v2_guard_reason")
        turn_log["option_overlap_score_before_guard"] = gbr.get("option_overlap_score_before_guard")
        turn_log["guard_fallback_mode"] = gbr.get("guard_fallback_mode")
        turn_log["hypothesis_generation_mode_before_guard"] = gbr.get("hypothesis_generation_mode_before_guard")
        turn_log["hypothesis_generation_mode_resolved"] = gbr.get("hypothesis_generation_mode_resolved")
        turn_log["representative_retrieval_mode_resolved"] = gbr.get("representative_retrieval_mode_resolved")
        if log_cfg.get("log_option_overlap_score"):
            turn_log["option_overlap_score"] = float(hyp_retrieval_diag.get("option_overlap_score", 0.0))
        turn_log["question_selection"]["voi_gate"]["turn_wall_time_sec"] = round(
            time.perf_counter() - t_turn0, 6
        )
        if frontier_gen_block_this_turn is not None:
            turn_log["frontier_clustered_v32_generation"] = frontier_gen_block_this_turn
        if fc32_vector_binding_audit is not None:
            turn_log["cluster_bound_v2_vector_binding"] = fc32_vector_binding_audit
        if fc32_force_ask_audit is not None:
            turn_log["frontier_clustered_v32_force_ask_sanity"] = fc32_force_ask_audit
        if fc32_cb_v2_force_delivery_audit is not None:
            turn_log["cluster_bound_v2_force_delivery"] = fc32_cb_v2_force_delivery_audit
        if fc32_cb_v2_entropy_gate_audit is not None:
            ent_audit = dict(fc32_cb_v2_entropy_gate_audit)
            uc = user_choice_log or {}
            ent_audit["user_choice_recorded"] = bool(
                uc.get("selected_option_id")
                or uc.get("choice")
                or uc.get("selected_hypothesis_id")
            )
            ent_audit["belief_update_executed"] = bool(belief_update_executed)
            bb = belief_before or {}
            ba = belief_after_log or {}
            mn_b = float(bb.get("mu_norm") or 0)
            mn_a = float(ba.get("mu_norm") or 0)
            ent_b = float(bb.get("entropy") or 0)
            ent_a = float(ba.get("entropy") or 0)
            ent_audit["mu_shift_norm"] = round(mn_a - mn_b, 6)
            ent_audit["entropy_delta"] = round(ent_a - ent_b, 6)
            if belief_update_executed and fc32_prev_turn_top10_snapshot is not None:
                ent_audit["rerank_after_update"] = bool(cur_top10 != fc32_prev_turn_top10_snapshot)
            else:
                ent_audit["rerank_after_update"] = False
            turn_log["cluster_bound_v2_entropy_ask_gate"] = ent_audit
        if entropy_only_ask_gate_audit is not None:
            turn_log["entropy_only_ask_gate"] = entropy_only_ask_gate_audit
        if fc32_closed_loop_audit:
            turn_log["frontier_clustered_v32_closed_loop"] = fc32_closed_loop_audit
        if fc32_pre_gate_audit is not None:
            turn_log["frontier_clustered_v32_pre_gate"] = fc32_pre_gate_audit
        if fc32_budget_state_audit is not None:
            turn_log["frontier_clustered_v32_budget_state"] = fc32_budget_state_audit
        if fc32_routing_repair_audit is not None:
            turn_log["frontier_clustered_v32_routing_repair"] = fc32_routing_repair_audit
        if fc32_oos_skip_audit is not None:
            turn_log["frontier_clustered_v32_oos_skip"] = fc32_oos_skip_audit
        if fc32_compensation_audit is not None:
            turn_log["frontier_clustered_v32_oos_compensation"] = fc32_compensation_audit
        if fc32_rank_safe_guard_audit is not None:
            turn_log["frontier_clustered_v32_rank_safe_guard"] = fc32_rank_safe_guard_audit
        if trace_ctrl is not None and trace_ctrl.is_replay:
            cons = trace_ctrl.consumption_summary()
            turn_log["trace_replay"] = {
                "trace_loaded": cons["trace_loaded"],
                "consumed_trace_turn": trace_ctrl.turn_was_consumed(turn),
                "trace_consumed_turns_session": len(trace_ctrl._consumed_turn_ids),
            }
        if csc_readout_audit is not None:
            turn_log["csc_readout_audit"] = csc_readout_audit
        full_rank_audit = attach_full_rank_audit_if_enabled(
            cfg=cfg,
            recommender=recommender,
            belief=belief,
            seen_items=seen_items,
            theta_star_items=theta_info.get("theta_star_items"),
            future_test_items=[str(x) for x in future_items],
        )
        if full_rank_audit is not None:
            if csc_readout_audit is not None:
                full_rank_audit["csc_readout_active"] = bool(csc_readout_audit.get("csc_applied"))
                full_rank_audit["csc_overlay_on_recommender"] = recommender.csc_overlay_active
            turn_log["full_rank_effect_audit"] = full_rank_audit
        if bool(log_cfg.get("frontier_discriminative_shadow")):
            lamb_shadow = float(rec_cfg.get("lambda_explore", 0.1))
            frontier_shadow_k = int(log_cfg.get("frontier_discriminative_frontier_k", 50))
            turn_log["frontier_discriminative_shadow"] = safe_frontier_discriminative_shadow_turn_dict(
                belief_snapshot_for_shadow,
                index,
                seen_items=seen_items,
                hyp_vectorized=hyp_vectorized,
                selected_hypothesis_ids=list(sel.get("best_ids") or []),
                recommendation_scores_topk=rec_scores,
                lambda_explore=lamb_shadow,
                frontier_k=frontier_shadow_k,
            )
        if do_ask and sel.get("best_ids"):
            consecutive_no_ask_tail = 0
        else:
            consecutive_no_ask_tail += 1

        fc32_prev_turn_top10_snapshot = cur_top10
        fc32_prev_turn_margin12 = (
            float(rec_scores[0] - rec_scores[1]) if len(rec_scores) >= 2 else None
        )
        fc32_prev_turn_had_ask = bool(do_ask and sel.get("best_ids"))
        if fc32_prev_turn_had_ask:
            fc32_last_ask_pre_top10 = cur_top10

        if trace_ctrl is not None and trace_ctrl.is_record:
            trace_ctrl.record_turn(
                turn_log,
                hyp_vectorized=hyp_vectorized,
                translated=translated,
            )

        turns_log.append(turn_log)

        append_diagnostics_row(
            {
                "event": "turn_total",
                "component": "per_turn_wall",
                "latency_sec": round(time.perf_counter() - t_turn0, 6),
                "ask_this_turn": bool(do_ask and sel["best_ids"]),
                "ask_count_so_far": ask_count,
                **turn_profile_snapshot(),
            }
        )

    snap = session_guard_snapshot()
    guard_diag = {f"guard_{k}": v for k, v in snap.items() if k != "started_mono"}

    append_diagnostics_row(
        {
            "user_id": session["user_id"],
            "episode_id": session.get("episode_id", session["user_id"]),
            "turn": 0,
            "event": "session_total",
            "component": "per_session_wall",
            "latency_sec": round(time.perf_counter() - t_session0, 6),
            "turns_in_session": len(turns_log),
            "total_ask_count": ask_count,
            **guard_diag,
        }
    )
    session_guard_end()
    profiling_clear_turn()

    out: dict[str, Any] = {
        "user_id": session["user_id"],
        "episode_id": session.get("episode_id", session["user_id"]),
        "turns": turns_log,
        "total_ask_count": ask_count,
        "final_metrics": turns_log[-1]["metrics"] if turns_log else {},
    }
    if trace_ctrl is not None:
        cons = trace_ctrl.consumption_summary()
        out["trace_replay"] = {
            "mode": trace_ctrl.mode,
            "trace_loaded": cons["trace_loaded"],
            "trace_consumed_turns": cons["trace_consumed_turns"],
            "trace_missing": cons["trace_missing"],
            "trace_missing_fields": cons["trace_missing_fields"],
            **cons,
        }
    return out
