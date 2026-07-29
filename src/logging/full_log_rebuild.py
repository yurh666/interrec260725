"""Rebuild per-episode structures + aggregates from full_log.jsonl (for resume / repair)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_episodes_from_full_log(run_dir: str | Path) -> list[dict[str, Any]]:
    """Parse full_log.jsonl into episode dicts compatible with experiment_runner._aggregate_metrics."""

    run_dir = Path(run_dir)
    path = run_dir / "full_log.jsonl"
    if not path.is_file():
        return []

    by_ep: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(o, dict):
                continue
            if o.get("type") in ("embedding_build_audit", "warning", "error"):
                continue
            if "turn" not in o or "metrics" not in o:
                continue
            uid = str(o.get("user_id", ""))
            eid = str(o.get("episode_id", ""))
            by_ep[(uid, eid)].append(o)

    episodes: list[dict[str, Any]] = []
    for turns in by_ep.values():
        turns_sorted = sorted(turns, key=lambda t: int(t.get("turn", 0)))
        if not turns_sorted:
            continue
        last_m = turns_sorted[-1].get("metrics") or {}
        ask_ct = last_m.get("ask_count_so_far", 0)
        episodes.append({"turns": turns_sorted, "total_ask_count": int(ask_ct)})

    return episodes
