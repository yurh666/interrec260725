"""Accumulates neutrality disk-cache / dedup / noop-retry statistics (profiling-only)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

_lock = threading.Lock()
_stats: dict[str, int | float] = {
    "judge_calls": 0,
    "judge_cache_hits": 0,
    "judge_cache_misses": 0,
    "rewrite_calls": 0,
    "rewrite_cache_hits": 0,
    "rewrite_cache_misses": 0,
    "retries_skipped_noop_rewrite": 0,
    "loops_exhausted": 0,
}


def reset() -> None:
    with _lock:
        for k in list(_stats):
            _stats[k] = 0


def _bump(key: str, inc: int = 1) -> None:
    with _lock:
        _stats[key] = int(_stats.get(key, 0)) + inc


def record_judge(*, cache_hit: bool) -> None:
    _bump("judge_calls")
    _bump("judge_cache_hits" if cache_hit else "judge_cache_misses")


def record_rewrite(*, cache_hit: bool) -> None:
    _bump("rewrite_calls")
    _bump("rewrite_cache_hits" if cache_hit else "rewrite_cache_misses")


def record_noop_rewrite_skip() -> None:
    _bump("retries_skipped_noop_rewrite")


def record_loop_exhausted() -> None:
    _bump("loops_exhausted")


def snapshot() -> dict[str, Any]:
    with _lock:
        return dict(_stats)


def write_json(path: Path) -> None:
    path.write_text(json.dumps(snapshot(), indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
