from __future__ import annotations

from typing import Any

import numpy as np

from src.belief.belief_state import BeliefState
from src.belief.laplace_update import laplace_choice_update
from src.memory.adaptive_intent_units import adaptive_memory_to_belief
from src.memory.intent_memory_state import IntentMemoryState, IntentMemoryUnit
from src.memory.memory_update_policy import MemoryUpdateConfig, decide_memory_operation
from src.utils.math import cosine, l2_normalize


def _next_unit_id(memory_state: IntentMemoryState) -> str:
    existing = {u.unit_id for u in memory_state.units}
    n = 0
    while True:
        uid = f"unit_{n}"
        if uid not in existing:
            return uid
        n += 1


def update_intent_memory_with_choice(
    memory_state: IntentMemoryState,
    question_options: list[str],
    selected_option: str | None,
    selected_vector: np.ndarray | None,
    option_vectors: np.ndarray,
    current_belief: BeliefState,
    update_config: MemoryUpdateConfig | dict[str, Any] | None = None,
    *,
    turn_id: int = 0,
    tau: float = 0.2,
    alpha_none: float = 1.2,
    memory_belief_blend: float = 0.65,
) -> tuple[IntentMemoryState, BeliefState, dict[str, Any]]:
    """Incrementally refine intent memory units from a user choice, then sync belief."""

    cfg = update_config if isinstance(update_config, MemoryUpdateConfig) else MemoryUpdateConfig(
        **(update_config or {})
    )
    state = memory_state.copy()
    state.turn_id = turn_id

    option_ids = list(question_options)
    selected_index: int | None = None
    if selected_option is not None and selected_option != "none":
        for i, oid in enumerate(option_ids):
            if str(oid) == str(selected_option):
                selected_index = i
                break
    elif selected_option == "none" or selected_option is None:
        selected_index = None

    decision = decide_memory_operation(
        memory_state=state,
        option_vectors=option_vectors,
        selected_index=selected_index,
        selected_vector=selected_vector,
        config=cfg,
    )
    op = decision["operation"]
    target_indices = decision.get("target_unit_indices", [])

    unit_support_before = {u.unit_id: u.support_weight for u in state.units}
    uncertainty_before = {u.unit_id: u.uncertainty for u in state.units}

    delta_mu = np.zeros(state.dim, dtype=float)
    if op == "update" and target_indices and selected_vector is not None:
        idx = int(target_indices[0])
        unit = state.units[idx]
        old_centroid = unit.centroid.copy()
        new_centroid = (1.0 - cfg.update_ema) * unit.centroid + cfg.update_ema * selected_vector
        unit.centroid = l2_normalize(new_centroid.reshape(1, -1))[0]
        unit.support_weight += cfg.support_gain * float(decision.get("confidence", 0.5))
        unit.uncertainty *= 0.95
        delta_mu = unit.centroid - old_centroid

    elif op == "create" and selected_vector is not None:
        dim = state.dim
        uid = _next_unit_id(state)
        state.units.append(
            IntentMemoryUnit(
                unit_id=uid,
                centroid=l2_normalize(selected_vector.reshape(1, -1))[0],
                covariance=np.eye(dim, dtype=float) * current_belief.jitter,
                support_items=[],
                support_weight=cfg.support_gain,
                granularity_level=max((u.granularity_level for u in state.units), default=0) + 1,
                uncertainty=float(np.trace(current_belief.Sigma) / max(dim, 1)),
            )
        )
        delta_mu = selected_vector.copy()

    elif op == "downweight":
        for unit in state.units:
            unit.support_weight *= cfg.downweight_factor
            unit.uncertainty *= 1.05

    elif op == "noop":
        pass

    elif op == "merge":
        pass

    competing_downweighted_ids: list[str] = []
    if (
        cfg.enable_competing_downweight
        and selected_index is not None
        and option_vectors.size
        and state.units
    ):
        target_set = set(int(i) for i in target_indices)
        centroids = np.stack([u.centroid for u in state.units], axis=0)
        align = centroids @ option_vectors.T
        factor = min(max(float(cfg.competing_downweight_factor), 0.0), 1.0)
        for idx, unit in enumerate(state.units):
            if idx in target_set:
                continue
            other_scores = [
                float(align[idx, j])
                for j in range(align.shape[1])
                if j != selected_index
            ]
            if other_scores and max(other_scores) >= cfg.competing_alignment_threshold:
                unit.support_weight *= factor
                unit.uncertainty *= 1.02
                competing_downweighted_ids.append(unit.unit_id)

    # Merge pass: combine units that became too similar.
    merged_units: list[IntentMemoryUnit] = []
    used = [False] * len(state.units)
    for i, ui in enumerate(state.units):
        if used[i]:
            continue
        merged = ui
        used[i] = True
        for j in range(i + 1, len(state.units)):
            if used[j]:
                continue
            uj = state.units[j]
            if cfg.allow_merge and cosine(merged.centroid, uj.centroid) >= cfg.merge_similarity_threshold:
                w_i = max(merged.support_weight, 1e-8)
                w_j = max(uj.support_weight, 1e-8)
                merged.centroid = l2_normalize(
                    ((w_i * merged.centroid + w_j * uj.centroid) / (w_i + w_j)).reshape(1, -1)
                )[0]
                merged.support_items = list(dict.fromkeys(merged.support_items + uj.support_items))
                merged.support_weight = w_i + w_j
                merged.uncertainty = 0.5 * (merged.uncertainty + uj.uncertainty)
                used[j] = True
                op = "merge"
        if merged.support_weight >= cfg.min_support_weight:
            merged_units.append(merged)
    state.units = merged_units or state.units

    # Laplace update on aggregated belief for compatibility with downstream ranking.
    laplace_belief = laplace_choice_update(
        current_belief,
        option_vectors,
        selected_index,
        tau,
        alpha_none,
    )
    blend = min(max(float(memory_belief_blend), 0.0), 1.0)
    if blend <= 0.0:
        # Compatibility mode must be an exact Laplace pass-through; normalizing
        # the posterior mean or aggregating memory here is enough to perturb
        # downstream ranking ties.
        updated_belief = laplace_belief
    else:
        memory_belief = adaptive_memory_to_belief(
            state,
            sigma0=float(np.sqrt(max(float(np.trace(current_belief.Sigma) / max(state.dim, 1)), 1e-6))),
            jitter=current_belief.jitter,
        )
        # Blend memory-refined belief with Laplace posterior. Lower values keep
        # the original InterRec ask/VOI geometry better aligned while still
        # logging and updating explicit intent memory units.
        mu = blend * memory_belief.mu + (1.0 - blend) * laplace_belief.mu
        Sigma = blend * memory_belief.Sigma + (1.0 - blend) * laplace_belief.Sigma
        updated_belief = BeliefState(mu=l2_normalize(mu.reshape(1, -1))[0], Sigma=Sigma, jitter=current_belief.jitter)
        updated_belief.ensure_psd(force=True)

    unit_support_after = {u.unit_id: u.support_weight for u in state.units}
    uncertainty_after = {u.unit_id: u.uncertainty for u in state.units}

    trace = {
        "turn_id": turn_id,
        "selected_option_id": selected_option,
        "operation": op,
        "target_unit_ids": decision.get("target_unit_ids", []),
        "delta_mu_norm": float(np.linalg.norm(delta_mu)),
        "delta_sigma_trace": float(np.trace(updated_belief.Sigma) - np.trace(current_belief.Sigma)),
        "memory_belief_blend": blend,
        "competing_downweighted_ids": competing_downweighted_ids,
        "competing_downweight_count": len(competing_downweighted_ids),
        "unit_support_before": unit_support_before,
        "unit_support_after": unit_support_after,
        "uncertainty_before": uncertainty_before,
        "uncertainty_after": uncertainty_after,
        "memory_unit_count": state.unit_count,
    }
    return state, updated_belief, trace
