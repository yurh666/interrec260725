"""Runtime helpers for ChoiceRec subexperiments (fixed ask schedule, update ablation)."""
from __future__ import annotations

import hashlib
from typing import Any


def choicerec_subexp_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    block = cfg.get("subexperiment_choicerec")
    return dict(block) if isinstance(block, dict) else {}


def evenly_spaced_ask_turns(n_asks: int, *, max_turns: int = 15) -> set[int]:
    """Place ``n_asks`` ask turns evenly across ``1..max_turns`` (non-consecutive when n<max)."""
    n = int(n_asks)
    if n <= 0:
        return set()
    if n >= max_turns:
        return set(range(1, max_turns + 1))
    out: set[int] = set()
    for i in range(n):
        t = int(round((i + 1) * max_turns / (n + 1)))
        t = max(1, min(max_turns, t))
        out.add(t)
    return out


def early_consecutive_ask_turns(n_asks: int, *, max_turns: int = 15) -> set[int]:
    """Ask on the first ``n_asks`` turns (1..n)."""
    n = int(n_asks)
    if n <= 0:
        return set()
    return set(range(1, min(n, max_turns) + 1))


def matched_ask_count_for_session(
    episode_id: str,
    *,
    n_high: int,
    n_low: int,
    high_count: int,
    low_count: int,
    seed: int = 42,
) -> int:
    """Deterministically assign high/low ask counts across sessions."""
    key = f"{seed}:{episode_id}".encode()
    bucket = int(hashlib.md5(key).hexdigest(), 16) % max(n_high + n_low, 1)
    return int(high_count if bucket < n_high else low_count)


def fixed_ask_turn_set(
    cfg: dict[str, Any],
    *,
    max_turns: int = 15,
    episode_id: str | None = None,
) -> set[int] | None:
    sub = choicerec_subexp_cfg(cfg)
    strategy = str(sub.get("fixed_ask_strategy") or "evenly_spaced").strip().lower()
    schedule_fn = evenly_spaced_ask_turns
    if strategy in {"early", "early_consecutive", "prefix", "front_loaded"}:
        schedule_fn = early_consecutive_ask_turns

    if sub.get("fixed_ask_matched_full_avg"):
        if not episode_id:
            return set()
        n = matched_ask_count_for_session(
            str(episode_id),
            n_high=int(sub.get("fixed_ask_matched_n_high", 0) or 0),
            n_low=int(sub.get("fixed_ask_matched_n_low", 0) or 0),
            high_count=int(sub.get("fixed_ask_matched_high_count", 5) or 5),
            low_count=int(sub.get("fixed_ask_matched_low_count", 4) or 4),
            seed=int(sub.get("fixed_ask_match_seed", 42) or 42),
        )
        return schedule_fn(n, max_turns=max_turns)

    if "fixed_ask_count" not in sub:
        return None
    n = int(sub.get("fixed_ask_count", 0) or 0)
    if strategy == "evenly_spaced" and bool(sub.get("fixed_ask_evenly_spaced", True)):
        return evenly_spaced_ask_turns(n, max_turns=max_turns)
    return early_consecutive_ask_turns(n, max_turns=max_turns)


def belief_update_ablation_mode(cfg: dict[str, Any]) -> str | None:
    mode = str(choicerec_subexp_cfg(cfg).get("belief_update_mode") or "").strip().lower()
    if mode in {"mean_only", "none", "skip", "no_update"}:
        return "none" if mode in {"none", "skip", "no_update"} else "mean_only"
    return None


def apply_fixed_ask_schedule(
    *,
    cfg: dict[str, Any],
    turn: int,
    max_turns: int,
    natural_do_ask: bool,
    episode_id: str | None = None,
) -> tuple[bool, dict[str, Any]]:
    schedule = fixed_ask_turn_set(cfg, max_turns=max_turns, episode_id=episode_id)
    if schedule is None:
        return natural_do_ask, {}
    forced = turn in schedule
    sub = choicerec_subexp_cfg(cfg)
    per_session_n = None
    if sub.get("fixed_ask_matched_full_avg") and episode_id:
        per_session_n = matched_ask_count_for_session(
            str(episode_id),
            n_high=int(sub.get("fixed_ask_matched_n_high", 0) or 0),
            n_low=int(sub.get("fixed_ask_matched_n_low", 0) or 0),
            high_count=int(sub.get("fixed_ask_matched_high_count", 5) or 5),
            low_count=int(sub.get("fixed_ask_matched_low_count", 4) or 4),
            seed=int(sub.get("fixed_ask_match_seed", 42) or 42),
        )
    audit = {
        "enabled": True,
        "fixed_ask_count": int(sub.get("fixed_ask_count", 0) or 0),
        "fixed_ask_matched_full_avg": bool(sub.get("fixed_ask_matched_full_avg")),
        "fixed_ask_per_session_count": per_session_n,
        "fixed_ask_strategy": str(sub.get("fixed_ask_strategy") or "evenly_spaced"),
        "scheduled_turns": sorted(schedule),
        "turn": int(turn),
        "natural_should_ask": bool(natural_do_ask),
        "forced_should_ask": bool(forced),
        "disable_voi_selection": True,
    }
    return forced, audit
