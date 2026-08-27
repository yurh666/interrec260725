from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.agents.direction_translator import DirectionTranslator
from src.agents.hypothesis_generator import HypothesisGenerator
from src.agents.llm_client import LLMCacheReplayMissError, LLMClient
from src.simulation.session_trace import SessionTraceController, TraceReplayMissError
from src.agents.option_writer import OptionWriter
from src.data.dataset import InterRecDataset
from src.embedding.bm25_index import BM25Index
from src.embedding.index import EmbeddingIndex
from src.logging.run_logger import RunLogger
from src.recommendation.recommender import BeliefRecommender
from src.simulation.session_simulator import run_session
from src.simulation.theta_builder import build_theta_star


def _sessions_for_simulation(dataset: InterRecDataset, sim_cfg: dict[str, Any]) -> list[Any]:
    """Select session subset: episode_ids, user slice, max_users, or all."""

    eids = sim_cfg.get("episode_ids")
    if eids:
        order = [str(x) for x in eids]
        by_id = {
            str(s.get("episode_id") or s.get("user_id")): s for s in dataset.sessions
        }
        missing = [e for e in order if e not in by_id]
        if missing:
            raise ValueError(f"simulation.episode_ids not found in dataset: {missing[:5]}")
        return [by_id[e] for e in order]

    us = sim_cfg.get("user_slice_start")
    ue = sim_cfg.get("user_slice_end")
    if us is not None or ue is not None:
        if us is None or ue is None:
            raise ValueError("simulation.user_slice_start and user_slice_end must be set together")
        i0 = int(us) - 1
        i1 = int(ue)
        if i0 < 0 or i1 < i0:
            raise ValueError(f"Invalid user slice: start={us}, end={ue}")
        return dataset.sessions[i0:i1]
    mu = sim_cfg.get("max_users")
    if mu is not None:
        return dataset.sessions[: int(mu)]
    return list(dataset.sessions)


def build_components(
    cfg: dict[str, Any],
    dataset: InterRecDataset | None = None,
) -> dict[str, Any]:
    """Instantiate all runtime components from config.

    dataset is passed so that DirectionTranslator and HypothesisGenerator can use
    real item metadata (artist names, tags) in LLM prompts instead of raw numeric IDs.
    """
    llm_cfg = cfg.get("llm", {})
    base_url = llm_cfg.get("base_url")
    if isinstance(base_url, str) and not base_url.strip():
        base_url = None
    cache_cfg = llm_cfg.get("cache") or {}

    llm = LLMClient(
        mode=llm_cfg.get("mode", "mock"),
        provider=llm_cfg.get("provider", "none"),
        base_url=base_url,
        model=llm_cfg.get("model"),
        timeout=float(llm_cfg.get("timeout", 60)),
        log_prompts=llm_cfg.get("log_prompts", True),
        log_responses=llm_cfg.get("log_responses", True),
        max_retry=int(llm_cfg.get("max_retry", 3)),
        cache_enabled=bool(cache_cfg.get("enabled", True)),
        cache_dir=cache_cfg.get("dir"),
        max_structured_parse_retries=int(
            cache_cfg.get("max_parse_retries", llm_cfg.get("max_parse_retries", llm_cfg.get("max_retry", 3)))
        ),
        cache_memory_max_entries=int(cache_cfg.get("memory_max_entries", 0)),
        api_key=str(llm_cfg["api_key"]).strip() if llm_cfg.get("api_key") else None,
        cache_mode=str(cache_cfg.get("mode", "normal")),
        cache_fail_fast_on_miss=bool(cache_cfg.get("fail_fast_on_miss", False)),
        use_stable_cache_keys=bool(cache_cfg.get("use_stable_keys", False)),
        replay_lookup_aliases=bool(cache_cfg.get("replay_lookup_aliases", False)),
        run_seed=int(cfg.get("seed", 42)),
    )
    # Build item_lookup once if dataset is available — used to send real names to LLM
    item_lookup = dataset.item_lookup() if dataset is not None else None
    return {
        "llm": llm,
        "translator": DirectionTranslator(llm, item_lookup=item_lookup),
        "hyp_generator": HypothesisGenerator(llm, item_lookup=item_lookup),
        "option_writer": OptionWriter(llm, include_none=cfg.get("ask", {}).get("include_none", True)),
    }


def run_interrec_experiment(
    cfg: dict[str, Any],
    run_dir: Path,
    dataset: InterRecDataset,
    index: EmbeddingIndex,
    logger: RunLogger,
    bm25_index: BM25Index | None = None,
) -> dict[str, Any]:
    from src.utils.session_guard_state import configure_engineering_guards

    configure_engineering_guards(cfg)
    if bool((cfg.get("network_guard") or {}).get("enabled", False)):
        from src.utils.network_guard import install_network_guard

        install_network_guard(enabled=True)
    sim_cfg = cfg.get("simulation", {})
    split = str(sim_cfg.get("split", "test"))
    top_k = int(cfg.get("recommendation", {}).get("top_k", 10))
    lambda_explore = float(cfg.get("recommendation", {}).get("lambda_explore", 0.1))

    components = build_components(cfg, dataset=dataset)
    recommender = BeliefRecommender(index, lambda_explore=lambda_explore)

    sessions = _sessions_for_simulation(dataset, sim_cfg)

    fc32_cfg = (cfg.get("ask") or {}).get("hypothesis_generation", {}).get("frontier_clustered_v32") or {}
    if fc32_cfg.get("cluster_bound_v2_force_delivery_smoke"):
        from src.ask.cluster_bound_generation_v2 import reset_cb_v2_force_delivery_run_counter

        reset_cb_v2_force_delivery_run_counter()
    if fc32_cfg.get("enable_cluster_bound_v2_entropy_ask_gate"):
        from src.ask.cluster_bound_generation_v2 import reset_cb_v2_entropy_gate_run_counter

        reset_cb_v2_entropy_gate_run_counter()

    cache_cfg = (cfg.get("llm") or {}).get("cache") or {}
    all_results: list[dict[str, Any]] = []
    try:
        for session in tqdm(sessions, desc="Running InterRec episodes"):
            trace_ctrl = SessionTraceController.from_cfg(
                cfg,
                run_dir=run_dir,
                session=session,
                run_id=logger.run_id,
            )
            try:
                if str(sim_cfg.get("protocol", "")).lower() == "choicerec":
                    import numpy as np

                    dim = index.embeddings.shape[1]
                    theta_info = {
                        "theta_star": np.zeros(dim, dtype=float),
                        "theta_star_source": "choicerec_dynamic",
                        "theta_star_items": [],
                    }
                else:
                    theta_info = build_theta_star(session, index, split=split, dataset=dataset)
                episode = run_session(
                    session=session,
                    theta_info=theta_info,
                    index=index,
                    recommender=recommender,
                    translator=components["translator"],
                    hyp_generator=components["hyp_generator"],
                    option_writer=components["option_writer"],
                    dataset=dataset,
                    cfg=cfg,
                    bm25_index=bm25_index,
                    trace_ctrl=trace_ctrl,
                )
                all_results.append(episode)
                for turn_log in episode["turns"]:
                    turn_log["run_id"] = logger.run_id
                    turn_log["dataset"] = cfg.get("dataset", {}).get("name", "unknown")
                    turn_log["method"] = "interrec"
                    turn_log["seed"] = cfg.get("seed", 42)
                    logger.log_turn(turn_log)
                if trace_ctrl is not None and trace_ctrl.is_record:
                    try:
                        trace_ctrl.finalize()
                    except Exception as trace_exc:
                        trace_ctrl.write_finalize_error_report(trace_exc)
                        logger.log_error(
                            str(session.get("episode_id", session["user_id"])),
                            f"trace_finalize_failed: {trace_exc}",
                        )
            except (LLMCacheReplayMissError, TraceReplayMissError) as exc:
                uid = str(session.get("episode_id", session["user_id"]))
                logger.log_error(uid, traceback.format_exc())
                failed = {
                    "user_id": session["user_id"],
                    "episode_id": session.get("episode_id", session["user_id"]),
                    "turns": [],
                    "session_failed": True,
                    "failure_reason": str(exc),
                }
                all_results.append(failed)
                if trace_ctrl is not None and trace_ctrl.is_record:
                    try:
                        trace_ctrl.finalize()
                    except Exception as trace_exc:
                        trace_ctrl.write_finalize_error_report(trace_exc)
                if cache_cfg.get("fail_fast_on_miss") and str(cache_cfg.get("mode")) in (
                    "replay_strict",
                    "cache_only",
                    "trace_replay",
                ):
                    raise
            except Exception:
                logger.log_error(
                    str(session.get("episode_id", session["user_id"])),
                    traceback.format_exc(),
                )
    finally:
        aggregate = _aggregate_metrics(all_results, top_k)
        logger.write_metrics(aggregate, all_results)
    return aggregate


def run_bm25_baseline(
    cfg: dict[str, Any],
    dataset: InterRecDataset,
    index: EmbeddingIndex,
    logger: RunLogger,
    bm25_index: BM25Index,
) -> dict[str, Any]:
    """Static BM25 retrieval baseline: no belief update, no interaction."""
    from src.recommendation.metrics import ranking_metrics

    sim_cfg = cfg.get("simulation", {})
    split = str(sim_cfg.get("split", "test"))
    split_key = {"train": "future_train", "valid": "future_valid", "test": "future_test"}.get(split, "future_test")
    top_k = int(cfg.get("recommendation", {}).get("top_k", 10))

    sessions = _sessions_for_simulation(dataset, sim_cfg)
    all_hr, all_ndcg, all_mrr = [], [], []

    for session in tqdm(sessions, desc="BM25 baseline"):
        observed = session["observed_history"]
        query = " ".join(str(x) for x in observed[-20:])
        relevant = set(str(x) for x in session.get(split_key, []))
        seen = set(str(x) for x in observed)
        results = bm25_index.search(query, top_k=top_k, exclude=seen)
        rec_ids = [r[0] for r in results]
        m = ranking_metrics(rec_ids, relevant, k=top_k)
        all_hr.append(m[f"HitRate@{top_k}"])
        all_ndcg.append(m[f"NDCG@{top_k}"])
        all_mrr.append(m[f"MRR@{top_k}"])

    aggregate = {
        "method": "bm25_baseline",
        "dataset": cfg.get("dataset", {}).get("name", "unknown"),
        "seed": cfg.get("seed", 42),
        f"HitRate@{top_k}": float(sum(all_hr) / max(len(all_hr), 1)),
        f"NDCG@{top_k}": float(sum(all_ndcg) / max(len(all_ndcg), 1)),
        f"MRR@{top_k}": float(sum(all_mrr) / max(len(all_mrr), 1)),
        "n_users": len(all_hr),
    }
    logger.write_metrics(aggregate, [])
    return aggregate


def _aggregate_metrics(results: list[dict[str, Any]], top_k: int = 10) -> dict[str, Any]:
    hr_list, ndcg_list, mrr_list, ask_list, pref_err_list = [], [], [], [], []
    for ep in results:
        turns = ep.get("turns", [])
        if not turns:
            continue
        last_m = turns[-1].get("metrics", {})
        hr_list.append(last_m.get(f"HitRate@{top_k}", 0.0))
        ndcg_list.append(last_m.get(f"NDCG@{top_k}", 0.0))
        mrr_list.append(last_m.get(f"MRR@{top_k}", 0.0))
        pref_err_list.append(last_m.get("preference_error", 0.0))
        ask_list.append(ep.get("total_ask_count", 0))

    def safe_mean(lst: list[float]) -> float:
        return float(sum(lst) / max(len(lst), 1))

    return {
        "method": "interrec",
        f"HitRate@{top_k}": safe_mean(hr_list),
        f"NDCG@{top_k}": safe_mean(ndcg_list),
        f"MRR@{top_k}": safe_mean(mrr_list),
        "avg_preference_error": safe_mean(pref_err_list),
        "avg_ask_count": safe_mean(ask_list),
        "n_users": len(results),
    }
