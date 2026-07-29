"""Optional profiling context for latency / diagnostics JSONL (no algorithm impact)."""

from __future__ import annotations

import contextvars
import json
import os
import threading
import time
from typing import Any

_DIAG_IO_LOCK = threading.Lock()

_ctx: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "interrec_profile_ctx", default=None
)
_llm_component: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "interrec_llm_component", default=None
)


def profiling_set_turn(
    *,
    user_id: Any,
    episode_id: Any,
    turn: int,
) -> None:
    try:
        from src.utils.session_guard_state import reset_turn_profile_counters

        reset_turn_profile_counters()
    except Exception:
        pass
    _ctx.set(
        {
            "user_id": user_id,
            "episode_id": episode_id,
            "turn": int(turn),
        }
    )


def profiling_clear_turn() -> None:
    _ctx.set(None)


def profiling_get_ctx() -> dict[str, Any]:
    return dict(_ctx.get() or {})


def set_active_llm_component(name: str | None) -> None:
    _llm_component.set(name)


def get_active_llm_component() -> str | None:
    return _llm_component.get()


def append_diagnostics_row(rec: dict[str, Any]) -> None:
    """Append one JSON line if INTERREC_DIAGNOSTICS_JSONL is set."""
    path = os.environ.get("INTERREC_DIAGNOSTICS_JSONL")
    if not path:
        return
    row = {"ts": time.time(), **profiling_get_ctx(), **rec}
    try:
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with _DIAG_IO_LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
    except OSError:
        pass
