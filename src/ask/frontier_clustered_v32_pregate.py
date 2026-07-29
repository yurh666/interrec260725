"""Cheap pre-gate + budget bookkeeping for frontier_clustered_v32 (default OFF).

Uses only session-so-far diagnostics (belief summaries, reco scores/lists, ask counters).

No latent ``future_*`` / oracle targets are read here.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def top10_softmax_entropy(scores: list[float]) -> float:
    """Shannon entropy of softmax(top10 scores) — online-only signal."""

    if len(scores) < 2:
        return 0.0
    x = np.asarray(scores[:10], dtype=np.float64)
    x = x - float(np.max(x))
    p = np.exp(x)
    s = float(np.sum(p)) + 1e-15
    p = p / s
    return float(-np.sum(p * np.log(p + 1e-15)))


def evaluate_frontier_clustered_v32_pre_gate(
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
    """Return ``(passes, reason, feature_dict)`` — ``passes`` means caller may invoke v32 path."""

    features: dict[str, Any] = {
        "turn": int(turn_idx),
        "ask_count_so_far": int(ask_count_so_far),
        "v32_generation_count_so_far": int(v32_generation_count_so_far),
        "last_v32_generation_turn": int(last_v32_generation_turn),
        "cooldown_turns_config": int(cooldown_turns),
        "consecutive_ask_rank_unchanged": int(consecutive_ask_rank_unchanged),
        "belief_entropy": belief_entropy,
        "sigma_trace": sigma_trace,
        "margin_top1_minus_top2": margin_top12,
        "top10_ids_head": list(rec_top10_ids),
        "top10_softmax_entropy": softmax_entropy_top10,
    }

    max_asks = fc32_cfg.get("max_v32_asks_per_session")
    if max_asks is not None and ask_count_so_far >= int(max_asks):
        return False, "budget_max_v32_asks_reached", features

    max_gens = fc32_cfg.get("max_v32_generations_per_session")
    if max_gens is not None and v32_generation_count_so_far >= int(max_gens):
        return False, "budget_max_v32_generations_reached", features

    if last_v32_generation_turn > 0 and cooldown_turns > 0:
        delta = turn_idx - last_v32_generation_turn
        if delta <= int(cooldown_turns):
            return False, "cooldown_active", features

    stop_streak = fc32_cfg.get("disable_generation_after_consecutive_no_rank_change")
    if stop_streak is not None and consecutive_ask_rank_unchanged >= int(stop_streak):
        return False, "stagnation_consecutive_ask_without_top10_change", features

    mthr = fc32_cfg.get("pre_gate_max_top10_margin")
    ethr = fc32_cfg.get("pre_gate_min_uncertainty")
    if mthr is not None and ethr is not None and margin_top12 is not None and belief_entropy is not None:
        if float(margin_top12) > float(mthr) and float(belief_entropy) < float(ethr):
            return False, "high_margin_and_low_entropy_skip", features

    return True, "pass", features


def build_budget_state(
    *,
    fc32_cfg: dict[str, Any],
    current_turn_idx: int,
    ask_count_so_far: int,
    v32_generation_count_so_far: int,
    last_v32_generation_turn: int,
    cooldown_turns: int,
    consecutive_ask_rank_unchanged: int,
    gate_pass: bool,
    gate_reason: str,
) -> dict[str, Any]:
    cool_rem = 0
    if last_v32_generation_turn > 0 and cooldown_turns > 0:
        delta = max(0, int(current_turn_idx) - int(last_v32_generation_turn))
        cool_rem = max(0, int(cooldown_turns) + 1 - delta)

    budget_blocked = (not gate_pass) and gate_reason.startswith("budget_")
    return {
        "ask_count_so_far": int(ask_count_so_far),
        "v32_generation_count_so_far": int(v32_generation_count_so_far),
        "max_v32_asks_per_session": fc32_cfg.get("max_v32_asks_per_session"),
        "max_v32_generations_per_session": fc32_cfg.get("max_v32_generations_per_session"),
        "cooldown_turns_config": int(cooldown_turns),
        "cooldown_remaining_turns": int(cool_rem),
        "consecutive_ask_no_rank_change": int(consecutive_ask_rank_unchanged),
        "budget_blocked": bool(budget_blocked),
        "budget_block_reason": gate_reason if budget_blocked else None,
        "pre_gate_pass": bool(gate_pass),
        "pre_gate_reason": gate_reason,
    }
