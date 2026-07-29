"""Session-level frontier_clustered_v32 hypothesis integration (pool → cluster → contrast → mock/LLM gated).

Default OFF — only runs when ``ask.hypothesis_generation`` requests this mode *and*
``frontier_clustered_v32.{enabled|simulator_integration_enabled|integrated}`` is true."""

from __future__ import annotations

import os
import time
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.ask.cluster_bound_generation_v2 import (
    ClusterBoundV2Result,
    cluster_bound_generation_v2_enabled,
    cluster_bound_v2_to_hypotheses,
    resolve_domain,
    validate_cluster_bound_v2_schema,
    validate_cluster_bound_v2_self_check,
    run_cluster_bound_generation_v2,
)
from src.ask.frontier_clustered_v32 import (
    apply_cluster_semantic_enrichment_mock,
    build_frontier_cluster_prompt_payload,
    build_frontier_pool,
    cluster_frontier_pool,
    compact_frontier_prompt_payload,
    generate_choice_question_mock,
    generate_frontier_cluster_hypotheses_mock_or_llm,
    select_frontier_cluster_contrasts,
)
from src.ask.frontier_clustered_v32_rank_safe_mcq import assign_rank_safety_to_hypotheses
from src.ask.frontier_clustered_v32_schema import (
    apply_compact_option_field_shims,
    infer_option_mappings_if_missing,
    planned_contrast_pairs_from_diag,
    validate_generation_bundle,
)
from src.ask.integrated_representative_backend import IntegratedRepresentativeSemanticBackend
from src.embedding.index import EmbeddingIndex

from src.ask.movielens_candidate_recall import (
    build_movielens_candidate_pool,
    movielens_candidate_recall_enabled,
)
from src.ask.movielens_speed_optimization import movielens_speed_opt_cfg, movielens_speed_opt_enabled
from src.agents.hypothesis_generator import HypothesisGenerator
from src.agents.prompt_templates import _item_label


def frontier_cluster_session_armed(hyp_gen_cfg: dict[str, Any]) -> bool:
    if str(hyp_gen_cfg.get("mode", "")).strip() != "frontier_clustered_v32":
        return False
    fc = hyp_gen_cfg.get("frontier_clustered_v32") or {}
    return bool(fc.get("enabled") or fc.get("simulator_integration_enabled") or fc.get("integrated"))


def _semantic_backend_cfg(cfg: dict[str, Any], fc32_cfg: dict[str, Any]) -> dict[str, Any]:
    root = dict(cfg.get("semantic_backend") or {})
    nested = dict(fc32_cfg.get("semantic_backend") or {})
    out = {**root, **nested}
    typ = str(out.get("type") or "").strip()
    if not typ and str(out.get("backend") or "").strip():
        out["type"] = str(out.get("backend"))
    return out


def _representative_backend_enabled(cfg: dict[str, Any], fc32_cfg: dict[str, Any]) -> bool:
    sb = _semantic_backend_cfg(cfg, fc32_cfg)
    if bool(sb.get("preserve_reference_route", False)):
        return False
    impl = str(sb.get("implementation_mode") or "").strip()
    if impl == "frontier_clustered_v32_cluster_bound_v2_mock":
        return False
    if impl and impl != "frontier_clustered_v32_integrated_representative_item":
        return False
    return bool(sb.get("enabled")) and str(sb.get("type") or "") == "integrated_representative_item"


def _build_representative_backend(cfg: dict[str, Any], fc32_cfg: dict[str, Any]) -> IntegratedRepresentativeSemanticBackend:
    sb = _semantic_backend_cfg(cfg, fc32_cfg)
    return IntegratedRepresentativeSemanticBackend(
        mode=str(sb.get("variant") or sb.get("mode") or "representative_topk"),
        top_k=int(sb.get("top_k", fc32_cfg.get("max_representatives_per_cluster", 5))),
        alpha=float(sb.get("alpha", 0.7)),
        beta=float(sb.get("beta", 0.3)),
        gamma=float(sb.get("gamma", 0.05)),
    )


def _history_centroid(history: list[str], index: EmbeddingIndex) -> np.ndarray | None:
    if not history:
        return None
    vecs = []
    for iid in history[-24:]:
        s = str(iid)
        if s in index.item_id_to_index:
            vecs.append(index.embeddings[int(index.item_id_to_index[s])].astype(float))
    if not vecs:
        return None
    return np.mean(np.stack(vecs, axis=0), axis=0)


def _stable_episode_hash(episode_id: str) -> int:
    digest = hashlib.sha256(str(episode_id).encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _pool_summary(pool: list[Any], top10: list[str]) -> dict[str, Any]:
    return {
        "pool_size": len(pool),
        "top10_preview_ids": [str(x) for x in top10[:10]],
        "top_ranks": [getattr(p, "rank", None) for p in pool[:8]],
    }


def _cluster_summaries_log(clusters: list[Any], fc32_cfg: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    max_rep = 5
    sem_max = 220
    if fc32_cfg:
        max_rep = max(1, int(fc32_cfg.get("max_representatives_per_cluster", 5)))
        sem_max = int(fc32_cfg.get("semantic_summary_max_chars", sem_max))
    out: list[dict[str, Any]] = []
    for c in clusters:
        cen = list(getattr(c, "mean_embedding_direction", None) or ())
        out.append(
            {
                "cluster_id": int(getattr(c, "cluster_id", -1)),
                "size": int(getattr(c, "size", 0)),
                "mean_rank": getattr(c, "mean_rank", None),
                "mean_belief_score": getattr(c, "mean_belief_score", None),
                "mean_uncertainty": getattr(c, "mean_uncertainty", None),
                "overlap_top10_frac": getattr(c, "overlap_top10_frac", None),
                "history_similarity": getattr(c, "history_similarity", None),
                "representative_names": list(getattr(c, "representative_names", ()) or ())[:max_rep],
                "semantic_summary": str(getattr(c, "semantic_summary", "") or "")[:sem_max],
                "mean_embedding_direction_preview": [round(float(x), 5) for x in cen[:8]],
            },
        )
    return out


def _attach_option_contrast_refs(
    options: list[dict[str, Any]], omap: dict[str, Any]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for o in options:
        dd = dict(o)
        oid = str(dd.get("option_id") or "")
        mp = omap.get(oid) if isinstance(omap, dict) else None
        if isinstance(mp, dict) and dd.get("contrast_ref") is None:
            try:
                a = mp.get("positive_cluster_id")
                b = mp.get("contrasting_cluster_id") or mp.get("contrast_cluster_id")
                if a is not None and b is not None:
                    dd["contrast_ref"] = {
                        "reference_cluster_id": int(a),
                        "contrasting_cluster_id": int(b),
                    }
            except Exception:
                pass
        if "mapped_hypothesis_id" not in dd and dd.get("maps_to_hypothesis_id"):
            dd["mapped_hypothesis_id"] = str(dd["maps_to_hypothesis_id"])
        out.append(dd)
    return out


def _hypotheses_session_shape(raw: list[dict[str, Any]], M: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for h in raw:
        if len(out) >= max(1, M):
            break
        hh = dict(h)
        hid = str(hh.get("hypothesis_id") or hh.get("option_id") or "").strip()
        if not hid:
            continue
        text = str(
            hh.get("text_description")
            or hh.get("short_user_facing_option_text")
            or hh.get("intent_text")
            or hh.get("positive_intent")
            or "",
        ).strip()
        fs = hh.get("feature_signature")
        if not isinstance(fs, list) or len(fs) == 0:
            feats = list(hh.get("distinguishing_features") or ())[:6]
            fs = [str(x) for x in feats if str(x)] or [f"sig_{hid}"]
        row = {
            **hh,
            "hypothesis_id": hid,
            "option_id": str(hh.get("option_id") or hid),
            "text_description": text or hid,
            "positive_intent": str(hh.get("positive_intent") or hh.get("intent_text") or text),
            "feature_signature": fs,
        }
        if not str(row.get("negative_contrast") or "").strip():
            row["negative_contrast"] = str(hh.get("negative_contrast") or "") or "n/a"
        out.append(row)
    return out


def _fallback_contrastive_v2(
    hyp_generator: HypothesisGenerator,
    translated: list[dict[str, Any]],
    observed: list[str],
    M: int,
    base_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    g2 = dict(base_cfg)
    g2["mode"] = "contrastive_v2"
    return hyp_generator.generate(translated, observed, M=M, generation_cfg=g2)


def _fallback_legacy(
    hyp_generator: HypothesisGenerator,
    translated: list[dict[str, Any]],
    observed: list[str],
    M: int,
    base_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    g2 = dict(base_cfg)
    g2["mode"] = "legacy"
    return hyp_generator.generate(translated, observed, M=M, generation_cfg=g2)


def run_frontier_clustered_v32_hypotheses_turn(
    *,
    belief: BeliefState,
    index: EmbeddingIndex,
    seen_items: set[str],
    observed_history: list[str],
    translated: list[dict[str, Any]],
    hyp_generator: HypothesisGenerator,
    M_hyp: int,
    hyp_gen_cfg: dict[str, Any],
    rec_lambda: float,
    cfg: dict[str, Any],
    turn: int,
    episode_id: str,
    session_speed_stats: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Runs v32 hypothesis path with schema validation + fallback ladder. Never raises."""

    fc32_cfg = dict(hyp_gen_cfg.get("frontier_clustered_v32") or {})
    use_mock_llm = bool(fc32_cfg.get("use_mock_llm", True))
    gen_mode = "mock" if use_mock_llm else "llm"

    log_cfg = cfg.get("logging") or {}
    verbose = bool(log_cfg.get("frontier_clustered_v32_debug"))

    t0 = time.perf_counter()
    block: dict[str, Any] = {
        "enabled": True,
        "mode": gen_mode,
        "frontier_pool_summary": {},
        "cluster_summaries": [],
        "planned_cluster_contrasts": [],
        "raw_generation_output": "",
        "parsed_hypotheses": [],
        "choice_question": {},
        "option_to_cluster_mapping": {},
        "schema_validation_result": {},
        "cluster_bound_generation_v2": {},
        "fallback_used": False,
        "fallback_reason": None,
        "latency_ms": 0.0,
        "token_usage": None,
        "api_call_count": 0,
        "hypothesis_generator_audit": {},
    }

    def _finish(
        hyps: list[dict[str, Any]],
        *,
        fb: bool,
        fb_reason: str | None,
        audit: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        block["fallback_used"] = bool(fb)
        block["fallback_reason"] = fb_reason
        block["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        hyp_generator.last_call_audit = dict(audit)
        hyp_generator.implementation_mode = str(audit.get("implementation_mode", "unknown"))
        block["hypothesis_generator_audit"] = dict(audit)
        if not verbose and isinstance(block.get("raw_generation_output"), str):
            ro = block["raw_generation_output"]
            if len(ro) > 4000:
                block["raw_generation_output"] = ro[:4000] + "\n…[truncated]"
        return hyps, block

    base_audit: dict[str, Any] = {
        "component": "hypothesis_generator",
        "hypothesis_generation_mode_requested": "frontier_clustered_v32",
        "hypothesis_generation_mode_resolved": "frontier_clustered_v32_mock",
        "implementation_mode": "frontier_clustered_v32_mock",
        "fallback_reason": None,
        "cache_hit": False,
    }

    try:
        seed = int(cfg.get("seed", 42))
        rng = np.random.default_rng(seed + (_stable_episode_hash(str(episode_id)) % 10_000) + turn * 17)

        if gen_mode == "llm":
            conf = str(os.environ.get("CONFIRM_FRONTIER_CLUSTERED_V32_LLM") or "").strip().lower()
            if conf not in ("1", "true", "yes", "on"):
                hyps = _fallback_contrastive_v2(
                    hyp_generator, translated, observed_history, M_hyp, hyp_gen_cfg
                )
                return _finish(
                    hyps,
                    fb=True,
                    fb_reason="frontier_clustered_v32_llm_confirm_missing_fell_back_contrastive_v2",
                    audit={
                        **base_audit,
                        "implementation_mode": hyp_generator.implementation_mode,
                        "hypothesis_generation_mode_resolved": str(
                            hyp_generator.last_call_audit.get("hypothesis_generation_mode_resolved")
                            or "contrastive_v2",
                        ),
                        "fallback_reason": "llm_confirm_missing",
                    },
                )
            if getattr(hyp_generator.llm, "mode", "mock") == "mock":
                hyps = _fallback_contrastive_v2(
                    hyp_generator, translated, observed_history, M_hyp, hyp_gen_cfg
                )
                return _finish(
                    hyps,
                    fb=True,
                    fb_reason="frontier_clustered_v32_llm_requires_non_mock_llm_client",
                    audit={
                        **base_audit,
                        "implementation_mode": hyp_generator.implementation_mode,
                        "hypothesis_generation_mode_resolved": str(
                            hyp_generator.last_call_audit.get("hypothesis_generation_mode_resolved")
                            or "contrastive_v2",
                        ),
                        "fallback_reason": "llm_client_still_mock",
                    },
                )

        embeddings = index.embeddings.astype(float)
        dataset_name = str((cfg.get("dataset") or {}).get("name", ""))
        mcr_cfg = dict(cfg.get("movielens_candidate_recall") or {})
        allowed_ids: set[str] | None = None
        mcr_audit: dict[str, Any] = {"enabled": False}
        if movielens_candidate_recall_enabled(mcr_cfg, dataset_name):
            proc = Path(str((cfg.get("dataset") or {}).get("processed_dir", "")))
            speed_opt = movielens_speed_opt_cfg(cfg)
            allowed_ids, mcr_audit = build_movielens_candidate_pool(
                cfg=mcr_cfg,
                processed_dir=proc,
                embeddings=embeddings,
                item_ids=list(index.item_ids),
                item_id_to_index=dict(index.item_id_to_index),
                observed_history=list(observed_history),
                seen_items=set(seen_items),
                session_id=str(episode_id),
                session_pool_cache=bool(
                    movielens_speed_opt_enabled(cfg)
                    and speed_opt.get("itemknn_pool_session_cache", False)
                ),
                session_speed_stats=session_speed_stats,
            )
            mcr_audit["enabled"] = True
        block["movielens_candidate_recall"] = mcr_audit

        pool, top10 = build_frontier_pool(
            belief,
            index,
            seen_items,
            embeddings=embeddings,
            lamb=rec_lambda,
            cfg=fc32_cfg,
            allowed_item_ids=allowed_ids,
        )
        if len(pool) < 3 and allowed_ids is not None and bool(mcr_cfg.get("fallback_to_semantic", True)):
            pool, top10 = build_frontier_pool(
                belief,
                index,
                seen_items,
                embeddings=embeddings,
                lamb=rec_lambda,
                cfg=fc32_cfg,
                allowed_item_ids=None,
            )
            block["movielens_candidate_recall"]["fallback_to_semantic_used"] = True
        block["frontier_pool_summary"] = _pool_summary(pool, top10)
        if len(pool) < 3:
            if cluster_bound_generation_v2_enabled(fc32_cfg):
                hyps = _fallback_legacy(hyp_generator, translated, observed_history, M_hyp, hyp_gen_cfg)
                fb_reason = "frontier_pool_too_small_fell_back_legacy_cb_v2"
            else:
                hyps = _fallback_contrastive_v2(
                    hyp_generator, translated, observed_history, M_hyp, hyp_gen_cfg
                )
                fb_reason = "frontier_pool_too_small_fell_back_contrastive_v2"
            return _finish(
                hyps,
                fb=True,
                fb_reason=fb_reason,
                audit={
                    **base_audit,
                    **dict(hyp_generator.last_call_audit),
                    "hypothesis_generation_mode_resolved": str(
                        hyp_generator.last_call_audit.get("hypothesis_generation_mode_resolved")
                        or "contrastive_v2",
                    ),
                },
            )

        item_lookup = hyp_generator.item_lookup or {}
        hc = _history_centroid(observed_history, index)
        clusters, _ = cluster_frontier_pool(
            pool,
            index,
            embeddings,
            item_lookup,
            top10_ids=top10,
            history_centroid=hc,
            cfg=fc32_cfg,
            rng=rng,
        )
        block["cluster_summaries"] = _cluster_summaries_log(clusters, fc32_cfg)

        if fc32_cfg.get("apply_cluster_semantic_enrichment_mock"):
            clusters_payload = []
            for cl in clusters:
                clusters_payload.append(
                    {
                        "cluster_id": cl.cluster_id,
                        "size": cl.size,
                        "mean_rank": cl.mean_rank,
                        "mean_belief_score": cl.mean_belief_score,
                        "mean_uncertainty": cl.mean_uncertainty,
                        "overlap_top10_frac": cl.overlap_top10_frac,
                        "history_similarity": cl.history_similarity,
                        "representative_names": cl.representative_names,
                        "semantic_summary": cl.semantic_summary,
                    },
                )
            apply_cluster_semantic_enrichment_mock(clusters_payload, item_lookup, rng, cfg=fc32_cfg)
            # Re-hydrate summaries for prompt + choice (minimal merge)
            id2 = {int(x["cluster_id"]): x for x in clusters_payload if "cluster_id" in x}
            for c in clusters:
                row = id2.get(int(c.cluster_id))
                if row and row.get("cluster_semantic_summary"):
                    c.semantic_summary = str(row.get("cluster_semantic_summary") or c.semantic_summary)

        contrasts = select_frontier_cluster_contrasts(clusters, cfg=fc32_cfg)
        contrasts_ser = [
            {
                "contrast_id": x.contrast_id,
                "reference_cluster_id": x.safe_or_reference_cluster_id,
                "contrasting_cluster_id": x.contrasting_cluster_id,
                "rationale": x.rationale,
            }
            for x in contrasts
        ]
        block["planned_cluster_contrasts"] = contrasts_ser

        history_summary = ", ".join(_item_label(str(x), item_lookup) for x in observed_history[-8:])
        anchor_names = [
            _squish_short(str(d.get("direction_name") or d.get("semantic_description") or ""))
            for d in translated[:6]
        ]

        payload = build_frontier_cluster_prompt_payload(
            clusters,
            contrasts,
            history_summary,
            [a for a in anchor_names if a],
            cfg=fc32_cfg,
        )
        payload = compact_frontier_prompt_payload(payload, fc32_cfg)
        planned_pairs = planned_contrast_pairs_from_diag(contrasts_ser)

        m_eff = max(3, min(8, M_hyp))
        if fc32_cfg.get("prompt_compact_mode"):
            cap_m = int(fc32_cfg.get("max_hypotheses", m_eff))
            m_eff = max(3, min(cap_m, m_eff))
        mx_tok = fc32_cfg.get("frontier_llm_max_output_tokens")
        mx_tok_i = int(mx_tok) if mx_tok is not None else None

        cb_v2_on = cluster_bound_generation_v2_enabled(fc32_cfg)
        v2_used = False
        choice_question_bundle: dict[str, Any] | None = None
        omap_prebuilt: dict[str, Any] | None = None
        representative_backend_on = _representative_backend_enabled(cfg, fc32_cfg)
        representative_backend = _build_representative_backend(cfg, fc32_cfg) if representative_backend_on else None

        if cb_v2_on:
            top_names = [
                _item_label(str(x), item_lookup)
                for x in (payload.get("current_top_style_summary") or {}).get("representative_names", [])
                if str(x).strip()
            ][:6]
            if not top_names:
                top_names = [a for a in anchor_names if a][:6]
            if representative_backend is not None:
                local_out = representative_backend.generate_cluster_bound_options(
                    {
                        "clusters": clusters,
                        "contrasts": contrasts,
                        "contrasts_ser": contrasts_ser,
                        "history_summary": history_summary,
                        "anchor_names": [a for a in anchor_names if a],
                        "item_lookup": item_lookup,
                        "index": index,
                        "cfg": cfg,
                        "fc32_cfg": fc32_cfg,
                        "current_top_names": top_names,
                    }
                )
                parsed_local = dict(local_out.get("parsed") or {})
                schema_val_local = validate_cluster_bound_v2_schema(parsed_local)
                sc_val_local = validate_cluster_bound_v2_self_check(
                    parsed_local,
                    domain=resolve_domain(cfg, fc32_cfg),
                )
                if schema_val_local.get("schema_hard_pass") and sc_val_local.get("self_check_pass"):
                    hyps_local, cq_local, _opts_local, omap_local = cluster_bound_v2_to_hypotheses(
                        parsed_local,
                        contrasts=contrasts_ser,
                    )
                    v2_res = ClusterBoundV2Result(
                        success=True,
                        raw_text=str(local_out.get("raw_text") or ""),
                        parsed=parsed_local,
                        hypotheses=hyps_local,
                        choice_question=cq_local,
                        option_to_cluster_mapping=omap_local,
                        schema_validation=schema_val_local,
                        self_check_validation=sc_val_local,
                        telemetry={
                            "cluster_bound_generation_v2_enabled": True,
                            "semantic_backend": dict(local_out.get("audit") or {}),
                            "backend_type": "integrated_representative_item",
                            "backend_variant": representative_backend.mode,
                            "api_call_count": 0,
                            "trace_replay_used": False,
                            "schema_pass": True,
                            "self_check_pass": True,
                        },
                        api_call_count=0,
                        token_usage=None,
                    )
                else:
                    v2_res = ClusterBoundV2Result(
                        success=False,
                        raw_text=str(local_out.get("raw_text") or ""),
                        parsed=parsed_local,
                        fallback_reason=(
                            sc_val_local.get("fallback_reason")
                            or schema_val_local.get("fallback_reason")
                            or "integrated_representative_backend_validation_failed"
                        ),
                        schema_validation=schema_val_local,
                        self_check_validation=sc_val_local,
                        telemetry={
                            "cluster_bound_generation_v2_enabled": True,
                            "semantic_backend": dict(local_out.get("audit") or {}),
                            "backend_type": "integrated_representative_item",
                            "backend_variant": representative_backend.mode,
                            "api_call_count": 0,
                            "schema_pass": False,
                            "self_check_pass": False,
                        },
                        api_call_count=0,
                    )
            else:
                v2_res = run_cluster_bound_generation_v2(
                    clusters=clusters,
                    contrasts=contrasts,
                    contrasts_ser=contrasts_ser,
                    history_summary=history_summary,
                    anchor_names=[a for a in anchor_names if a],
                    item_lookup=item_lookup,
                    llm_client=hyp_generator.llm,
                    gen_mode=gen_mode,
                    cfg=cfg,
                    fc32_cfg=fc32_cfg,
                    current_top_names=top_names,
                    max_output_tokens=mx_tok_i,
                    session_id=str(episode_id) if episode_id else None,
                    turn_id=int(turn) if turn is not None else None,
                )
            block["cluster_bound_generation_v2"] = dict(v2_res.telemetry)
            block["cluster_bound_generation_v2"]["fallback_reason"] = v2_res.fallback_reason
            block["cluster_bound_generation_v2"]["success"] = v2_res.success
            if v2_res.api_call_count:
                block["api_call_count"] = int(v2_res.api_call_count)
            if v2_res.token_usage:
                block["token_usage"] = v2_res.token_usage
            if v2_res.success:
                v2_used = True
                raw_text = v2_res.raw_text
                hyps_raw = list(v2_res.hypotheses)
                choice_question_bundle = dict(v2_res.choice_question)
                omap_prebuilt = dict(v2_res.option_to_cluster_mapping)
                block["schema_validation_result"] = {
                    "parse_success": True,
                    "schema_hard_pass": True,
                    "cluster_bound_v2_schema": v2_res.schema_validation,
                    "cluster_bound_v2_self_check": v2_res.self_check_validation,
                }
            else:
                block["cluster_bound_generation_v2"]["legacy_v32_fallback"] = True

        if not v2_used:
            if representative_backend is not None:
                local_frontier = representative_backend.generate_frontier_cluster_hypotheses(
                    {
                        "clusters": clusters,
                        "contrasts": contrasts,
                        "contrasts_ser": contrasts_ser,
                        "history_summary": history_summary,
                        "anchor_names": [a for a in anchor_names if a],
                        "item_lookup": item_lookup,
                        "index": index,
                        "cfg": cfg,
                        "fc32_cfg": fc32_cfg,
                        "M": m_eff,
                    }
                )
                raw_text = str(local_frontier.get("raw_text") or "")
                hyps_raw = list(local_frontier.get("hypotheses") or [])
                block["semantic_backend"] = dict(local_frontier.get("audit") or {})
                block["mode"] = "integrated_representative_item"
                block["api_call_count"] = 0
            else:
                raw_text, hyps_raw = generate_frontier_cluster_hypotheses_mock_or_llm(
                    payload,
                    hyp_generator.llm,
                    gen_mode,
                    M=m_eff,
                    max_output_tokens=mx_tok_i,
                    prompt_compact_mode=bool(fc32_cfg.get("prompt_compact_mode", False)),
                )
            if cb_v2_on and block.get("cluster_bound_generation_v2"):
                block["fallback_used"] = True
                v2_fb = (block.get("cluster_bound_generation_v2") or {}).get("fallback_reason")
                block["fallback_reason"] = f"cluster_bound_v2_failed_legacy_v32;{v2_fb or 'unknown'}"

        block["raw_generation_output"] = str(raw_text)
        block["parsed_hypotheses"] = list(hyps_raw)
        for _h in block["parsed_hypotheses"]:
            if isinstance(_h, dict) and _h.get("rank_safety_role"):
                _h["llm_declared_rank_safety_role"] = str(_h.get("rank_safety_role"))

        if bool(fc32_cfg.get("rank_safe_mcq_roles_enabled", False)):
            assign_rank_safety_to_hypotheses(
                block["parsed_hypotheses"],
                top10_ids=[str(x) for x in top10],
                index=index,
                cfg=fc32_cfg,
            )

        if gen_mode == "llm" and not v2_used:
            lau = getattr(hyp_generator.llm, "last_structured_audit", {}) or {}
            block["api_call_count"] = int(lau.get("n_api_calls") or 0)
            tu = getattr(hyp_generator.llm, "_last_token_usage", None)
            block["token_usage"] = dict(tu) if isinstance(tu, dict) else tu

        clusters_choice = []
        for c in clusters:
            cc = dict(asdict(c))
            cen = cc.get("mean_embedding_direction")
            if isinstance(cen, list) and len(cen) > 16:
                cc["mean_embedding_direction"] = [float(x) for x in cen[:16]] + ["…"]
            clusters_choice.append(cc)

        if v2_used and choice_question_bundle is not None:
            cq = choice_question_bundle
            q_final = str(cq.get("final_question_text") or cq.get("question_text") or "")
            opts3 = list(cq.get("options") or [])
            omap = dict(omap_prebuilt or {})
            cq_extra = dict(cq.get("extra") or {})
            shim_warns: list[str] = []
        else:
            _raw_q_unused, q_final, opts, omap, cq_extra = generate_choice_question_mock(
                hyps_raw,
                cfg=fc32_cfg,
                clusters_payload=clusters_choice,
            )
            opts2 = _attach_option_contrast_refs(list(opts), dict(omap))
            if len(opts2) == len(hyps_raw):
                opts3, shim_warns = infer_option_mappings_if_missing(hyps_raw, opts2)
            else:
                opts3, shim_warns = list(opts2), ["hypothesis_option_len_mismatch_skip_positional_shim"]

            cq = {
                "final_question_text": q_final,
                "question_text": q_final,
                "options": opts3,
                "extra": cq_extra,
            }
            choice_question_bundle = cq
        opt_field_shims: list[str] = []
        if fc32_cfg.get("prompt_compact_mode"):
            choice_question_bundle, opt_field_shims = apply_compact_option_field_shims(
                hyps_raw,
                cq,
                dict(omap),
            )
        block["choice_question"] = {
            "final_question_text": q_final[:4000],
            "options": choice_question_bundle.get("options", opts3),
            "generation_extra_keys": sorted(str(k) for k in (cq_extra or {}).keys())[:40],
        }
        block["option_to_cluster_mapping"] = dict(omap)

        sess_hyps = _hypotheses_session_shape(hyps_raw, M_hyp)

        # --- schema validate (local shim + contrast refs attached) ---
        rolefix_mode = str(os.environ.get("V41_PROMPT_ROLE_FIX") or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if v2_used:
            val = dict(block.get("schema_validation_result") or {})
            val.setdefault("schema_hard_pass", True)
            val.setdefault("parse_success", True)
            val["cluster_bound_v2_validated"] = True
        else:
            val = validate_generation_bundle(
                block["parsed_hypotheses"],
                choice_question_bundle,
                planned_pairs,
                hypothesis_count_band=(3, 3) if rolefix_mode else None,
                rolefix_mode=rolefix_mode,
            )
        for w in shim_warns + opt_field_shims:
            val.setdefault("schema_soft_warnings", []).append(w)

        block["schema_validation_result"] = val

        if representative_backend_on:
            resolved_lbl = "frontier_clustered_v32_integrated_representative_item"
        elif v2_used:
            resolved_lbl = (
                "frontier_clustered_v32_cluster_bound_v2_llm"
                if gen_mode == "llm"
                else "frontier_clustered_v32_cluster_bound_v2_mock"
            )
        else:
            resolved_lbl = "frontier_clustered_v32_llm" if gen_mode == "llm" else "frontier_clustered_v32_mock"
        audit_ok: dict[str, Any] = {
            **base_audit,
            "parse_success": bool(val.get("parse_success")),
            "hypothesis_generation_mode_resolved": resolved_lbl,
            "implementation_mode": resolved_lbl,
        }

        if not val.get("schema_hard_pass", False):
            # cluster-bound v2 lane: no contrastive_v2 fallback (legacy v32 already attempted upstream)
            if cb_v2_on:
                hy_leg = _fallback_legacy(hyp_generator, translated, observed_history, M_hyp, hyp_gen_cfg)
                if hy_leg:
                    merged_audit = dict(hyp_generator.last_call_audit)
                    merged_audit["frontier_clustered_v32_schema_failure"] = val.get("fallback_reason")
                    return _finish(
                        hy_leg,
                        fb=True,
                        fb_reason="cluster_bound_v2_schema_hard_fail_fallback_legacy;"
                        + str(val.get("fallback_reason") or ""),
                        audit=merged_audit,
                    )
                fb = hyp_generator._rule_based(translated, observed_history, M_hyp)  # type: ignore[attr-defined]
                return _finish(
                    fb,
                    fb=True,
                    fb_reason="cluster_bound_v2_schema_hard_fail_fallback_rule_based;"
                    + str(val.get("fallback_reason") or ""),
                    audit={
                        **base_audit,
                        "implementation_mode": "rule_based",
                        "hypothesis_generation_mode_resolved": "rule_based",
                        "fallback_reason": "schema_fail_cb_v2_rule_based",
                    },
                )

            # fallback ladder: contrastive_v2 → legacy → rule_based
            hy_cv = _fallback_contrastive_v2(
                hyp_generator, translated, observed_history, M_hyp, hyp_gen_cfg
            )
            if hy_cv and isinstance(hy_cv, list):
                merged_audit = dict(hyp_generator.last_call_audit)
                merged_audit["frontier_clustered_v32_schema_failure"] = val.get("fallback_reason")
                return _finish(
                    hy_cv,
                    fb=True,
                    fb_reason=(
                        "schema_hard_fail_fallback_contrastive_v2;"
                        + str(val.get("fallback_reason") or "")
                    ),
                    audit=merged_audit,
                )

            hy_leg = _fallback_legacy(hyp_generator, translated, observed_history, M_hyp, hyp_gen_cfg)
            if hy_leg:
                merged_audit = dict(hyp_generator.last_call_audit)
                merged_audit["frontier_clustered_v32_schema_failure"] = val.get("fallback_reason")
                return _finish(
                    hy_leg,
                    fb=True,
                    fb_reason="schema_hard_fail_fallback_legacy;" + str(val.get("fallback_reason") or ""),
                    audit=merged_audit,
                )

            fb = hyp_generator._rule_based(translated, observed_history, M_hyp)  # type: ignore[attr-defined]
            return _finish(
                fb,
                fb=True,
                fb_reason="schema_hard_fail_fallback_rule_based_safe;" + str(val.get("fallback_reason") or ""),
                audit={
                    **base_audit,
                    "implementation_mode": "rule_based",
                    "hypothesis_generation_mode_resolved": "rule_based",
                    "fallback_reason": "schema_fail_all_llm_fallbacks_exhausted",
                },
            )

        return _finish(sess_hyps[:M_hyp], fb=False, fb_reason=None, audit=audit_ok)

    except Exception as ex:  # noqa: BLE001 — session safety contract
        cb_v2_on_ex = cluster_bound_generation_v2_enabled(fc32_cfg)
        try:
            if cb_v2_on_ex:
                hyps = _fallback_legacy(hyp_generator, translated, observed_history, M_hyp, hyp_gen_cfg)
            else:
                hyps = _fallback_contrastive_v2(
                    hyp_generator, translated, observed_history, M_hyp, hyp_gen_cfg
                )
        except Exception:
            hyps = hyp_generator._rule_based(translated, observed_history, M_hyp)  # type: ignore[attr-defined]
            return _finish(
                hyps,
                fb=True,
                fb_reason=f"frontier_clustered_v32_exception_rule_based:{type(ex).__name__}:{ex!s}",
                audit={
                    **base_audit,
                    "implementation_mode": "rule_based",
                    "hypothesis_generation_mode_resolved": "rule_based",
                    "parse_success": False,
                },
            )
        fb_lbl = (
            "frontier_clustered_v32_exception_fell_back_legacy"
            if cb_v2_on_ex
            else "frontier_clustered_v32_exception_fell_back_contrastive_v2"
        )
        return _finish(
            hyps,
            fb=True,
            fb_reason=f"{fb_lbl}:{type(ex).__name__}:{ex!s}",
            audit=dict(hyp_generator.last_call_audit),
        )


def _squish_short(x: str) -> str:
    return " ".join(str(x or "").strip().split())[:240]
