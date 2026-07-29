"""Optional per-session engineering guards (latency caps). Default: disabled via cfg."""

from __future__ import annotations

import time
from typing import Any

from src.utils.run_profiling_context import profiling_get_ctx

_GUARD_CFG: dict[str, Any] = {}

# Session-level aggregates (cleared in session_guard_end).
_SESSION: dict[str, Any] = {}

# Turn-level profile (reset each profiling_set_turn).
_TURN_PROF: dict[str, int] = {
    "llm_http_calls_turn": 0,
    "structured_cache_hits_turn": 0,
    "structured_cache_misses_turn": 0,
    "neutrality_judge_turn": 0,
    "neutrality_rewrite_turn": 0,
}


def configure_engineering_guards(cfg: dict[str, Any]) -> None:
    """Load once per process / experiment from merged config."""
    global _GUARD_CFG
    block = cfg.get("engineering_guards")
    _GUARD_CFG = dict(block) if isinstance(block, dict) else {}


def session_guard_begin(episode_id: str) -> None:
    eid = str(episode_id)
    global _SESSION
    _SESSION = {
        "episode_id": eid,
        "llm_http_total": 0,
        "timeout_total": 0,
        "http_fail_total": 0,
        "neutrality_judge_total": 0,
        "neutrality_rewrite_total": 0,
        "neutrality_short_circuit_rule": 0,
        "started_mono": time.monotonic(),
    }


def session_guard_end() -> None:
    global _SESSION
    _SESSION = {}


def _guard_enabled() -> bool:
    return bool(_GUARD_CFG.get("enabled"))


def _cap_llm() -> int | None:
    v = _GUARD_CFG.get("max_llm_http_calls_per_session")
    return int(v) if v is not None and str(v).strip() != "" else None


def _cap_neutrality_rounds() -> int | None:
    v = _GUARD_CFG.get("max_neutrality_rounds_per_session")
    return int(v) if v is not None and str(v).strip() != "" else None


def _soft_wall_sec() -> float | None:
    v = _GUARD_CFG.get("max_session_wall_time_soft_limit_sec")
    return float(v) if v is not None and str(v).strip() != "" else None


def session_wall_so_far_sec() -> float:
    t0 = float(_SESSION.get("started_mono") or time.monotonic())
    return max(0.0, time.monotonic() - t0)


def guard_note_llm_http(*, success: bool, timeout: bool = False) -> None:
    if not _SESSION:
        return
    _SESSION["llm_http_total"] = int(_SESSION.get("llm_http_total", 0)) + 1
    if timeout:
        _SESSION["timeout_total"] = int(_SESSION.get("timeout_total", 0)) + 1
    if not success:
        _SESSION["http_fail_total"] = int(_SESSION.get("http_fail_total", 0)) + 1
    _TURN_PROF["llm_http_calls_turn"] = int(_TURN_PROF.get("llm_http_calls_turn", 0)) + 1


def guard_note_structured_cache(cache_hit: bool) -> None:
    if cache_hit:
        _TURN_PROF["structured_cache_hits_turn"] = int(_TURN_PROF.get("structured_cache_hits_turn", 0)) + 1
    else:
        _TURN_PROF["structured_cache_misses_turn"] = int(_TURN_PROF.get("structured_cache_misses_turn", 0)) + 1


def guard_note_neutrality_judge() -> None:
    _SESSION["neutrality_judge_total"] = int(_SESSION.get("neutrality_judge_total", 0)) + 1
    _TURN_PROF["neutrality_judge_turn"] = int(_TURN_PROF.get("neutrality_judge_turn", 0)) + 1


def guard_note_neutrality_rewrite() -> None:
    _SESSION["neutrality_rewrite_total"] = int(_SESSION.get("neutrality_rewrite_total", 0)) + 1
    _TURN_PROF["neutrality_rewrite_turn"] = int(_TURN_PROF.get("neutrality_rewrite_turn", 0)) + 1


def guard_note_neutrality_short_circuit() -> None:
    _SESSION["neutrality_short_circuit_rule"] = int(_SESSION.get("neutrality_short_circuit_rule", 0)) + 1


def reset_turn_profile_counters() -> None:
    global _TURN_PROF
    _TURN_PROF = {k: 0 for k in _TURN_PROF}


def turn_profile_snapshot() -> dict[str, int]:
    return dict(_TURN_PROF)


def session_guard_snapshot() -> dict[str, Any]:
    return dict(_SESSION) if _SESSION else {}


def force_neutrality_llm_short_circuit() -> bool:
    """If True, skip expensive LLM neutrality (caller falls back to rule heuristic)."""
    if not _guard_enabled():
        return False
    cap = _cap_llm()
    if cap is not None and int(_SESSION.get("llm_http_total", 0)) >= cap:
        return True
    nr_cap = _cap_neutrality_rounds()
    if nr_cap is not None:
        rounds = int(_SESSION.get("neutrality_judge_total", 0)) + int(_SESSION.get("neutrality_rewrite_total", 0))
        if rounds >= nr_cap:
            return True
    sw = _soft_wall_sec()
    if sw is not None and session_wall_so_far_sec() >= sw:
        return True
    return False


def disable_neutrality_rewrite_only() -> bool:
    """When True, neutrality judge may run but rewrite rounds should be skipped (budget)."""
    if not _guard_enabled():
        return False
    if not bool(_GUARD_CFG.get("disable_neutrality_rewrite_when_over_budget", True)):
        return False
    cap = _cap_llm()
    if cap is not None and int(_SESSION.get("llm_http_total", 0)) >= max(1, cap - 2):
        return True
    sw = _soft_wall_sec()
    if sw is not None and session_wall_so_far_sec() >= sw * 0.85:
        return True
    return False


def current_episode_id() -> str:
    return str(profiling_get_ctx().get("episode_id") or "")
