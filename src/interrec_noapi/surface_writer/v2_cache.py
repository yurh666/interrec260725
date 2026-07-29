"""Surface Writer v2 offline cache lookup."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _hash_list(items: Sequence[str]) -> str:
    return hashlib.sha256("|".join(sorted(items)).encode()).hexdigest()[:16]


def make_cache_key(
    dataset: str,
    episode_id: str,
    turn: int | str,
    semantic_options: Sequence[Mapping[str, Any]],
    semantic_backend_mode: str = "",
) -> str:
    option_ids = [str(o.get("option_id") or o.get("hypothesis_id") or "") for o in semantic_options]
    rep_ids = []
    for o in semantic_options:
        rep_ids.extend(str(x) for x in (o.get("representative_item_ids") or o.get("representative_items") or []))
    parts = [
        str(dataset),
        str(episode_id),
        str(turn),
        semantic_backend_mode,
        _hash_list(option_ids),
        _hash_list(rep_ids),
        "|".join(option_ids),
    ]
    return hashlib.sha256("::".join(parts).encode()).hexdigest()


def lookup_cache(cache_dir: Path, cache_key: str) -> dict[str, Any] | None:
    for path in cache_dir.glob("*_surface_cache.jsonl"):
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("cache_key") == cache_key and row.get("validated"):
                return row.get("value")
    return None
