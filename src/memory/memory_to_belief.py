from __future__ import annotations

from typing import Any

import numpy as np

from src.belief.belief_state import BeliefState
from src.memory.adaptive_intent_units import adaptive_memory_to_belief
from src.memory.intent_memory_state import IntentMemoryState
from src.utils.math import l2_normalize


def project_memory_to_belief(
    *,
    original_belief: BeliefState,
    memory_state: IntentMemoryState | None,
    alpha: float = 0.0,
    sigma0: float = 1.0,
    jitter: float = 1e-6,
) -> tuple[BeliefState, dict[str, Any]]:
    """Blend adaptive memory projection into a belief only when alpha > 0."""

    blend = min(max(float(alpha), 0.0), 1.0)
    audit: dict[str, Any] = {
        "alpha": blend,
        "projection_enabled": bool(memory_state is not None and blend > 0.0),
        "memory_units_count": memory_state.unit_count if memory_state is not None else 0,
        "granularity_levels": (
            sorted({u.granularity_level for u in memory_state.units})
            if memory_state is not None
            else []
        ),
        "rank_safe_triggered": False,
    }
    if memory_state is None or blend <= 0.0:
        audit["memory_update_applied_to_belief"] = False
        return original_belief.copy(), audit

    memory_belief = adaptive_memory_to_belief(memory_state, sigma0=sigma0, jitter=jitter)
    mu = (1.0 - blend) * original_belief.mu + blend * memory_belief.mu
    Sigma = (1.0 - blend) * original_belief.Sigma + blend * memory_belief.Sigma
    projected = BeliefState(
        mu=l2_normalize(mu.reshape(1, -1))[0],
        Sigma=Sigma,
        jitter=original_belief.jitter,
    )
    projected.ensure_psd(force=True)
    audit["memory_update_applied_to_belief"] = True
    audit["memory_mu_cosine_to_original"] = float(np.dot(memory_belief.mu, original_belief.mu))
    audit["projected_mu_shift_norm"] = float(np.linalg.norm(projected.mu - original_belief.mu))
    audit["projected_sigma_trace_delta"] = float(np.trace(projected.Sigma) - np.trace(original_belief.Sigma))
    return projected, audit


def rank_safe_fallback_if_needed(
    *,
    original_belief: BeliefState,
    projected_belief: BeliefState,
    recommender,
    top_k: int,
    exclude: set[str],
    min_top10_overlap: float = 0.8,
) -> tuple[BeliefState, dict[str, Any]]:
    original_top = [str(i) for i, _ in recommender.recommend(original_belief, top_k=top_k, exclude=exclude)[:10]]
    projected_top = [str(i) for i, _ in recommender.recommend(projected_belief, top_k=top_k, exclude=exclude)[:10]]
    overlap = len(set(original_top) & set(projected_top)) / max(len(original_top), 1)
    triggered = overlap < float(min_top10_overlap)
    return (
        original_belief.copy() if triggered else projected_belief,
        {
            "rank_safe_triggered": bool(triggered),
            "top10_overlap": float(overlap),
            "pre_memory_top10": original_top,
            "post_memory_top10": projected_top,
            "used_future_for_runtime_decision": False,
            "used_target_for_runtime_decision": False,
        },
    )
