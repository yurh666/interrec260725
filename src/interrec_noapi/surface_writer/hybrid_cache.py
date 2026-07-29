"""Hybrid surface cache lookup for strict runtime mode."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping, Sequence

from interrec_noapi.surface_writer.hybrid_cache_keys import (
    build_event_cache_key_from_runtime_context,
    build_unique_cache_key_from_runtime,
    debug_key_bundle,
)


class CacheMissError(RuntimeError):
    pass


def _row_to_hit(row: tuple, lookup_key: str, lookup_method: str) -> dict[str, Any]:
    question_text, options_json, source, validated = row
    if not question_text or not options_json:
        return {}
    opts = json.loads(options_json)
    if not opts:
        return {}
    return {
        "question_text": question_text,
        "options": opts,
        "source": source,
        "validated": bool(validated),
        "lookup_key": lookup_key,
        "lookup_method": lookup_method,
        "dedup_key": lookup_key,
    }


def lookup_hybrid_cache(
    cache_path: Path | str,
    dataset: str,
    semantic_options: Sequence[Mapping[str, Any]],
    implementation_mode: str = "",
    *,
    context: Mapping[str, Any] | None = None,
    event_cache_key: str | None = None,
    unique_cache_key: str | None = None,
    lookup_priority: Sequence[str] | None = None,
    debug: bool = False,
) -> dict[str, Any] | None:
    path = Path(cache_path)
    if not path.exists():
        return None

    ctx = dict(context or {})
    ctx.setdefault("dataset", dataset)
    ctx.setdefault("implementation_mode", implementation_mode)

    priority = list(lookup_priority or ("event_cache_key", "canonical_unique_key"))
    event_key = event_cache_key or build_event_cache_key_from_runtime_context(ctx, semantic_options)
    unique_key = unique_cache_key or build_unique_cache_key_from_runtime(
        str(dataset or ""),
        semantic_options,
    )
    debug_info = debug_key_bundle(ctx, semantic_options) if debug else None

    conn = sqlite3.connect(str(path))
    try:
        for step in priority:
            if step == "event_cache_key":
                row = conn.execute(
                    "SELECT question_text, options_json, source, validated FROM event_cache WHERE event_cache_key = ?",
                    (event_key,),
                ).fetchone()
                if row:
                    hit = _row_to_hit(row, event_key, "event_cache_key")
                    if hit:
                        if debug_info:
                            hit["debug"] = debug_info
                        return hit
            elif step in {"canonical_unique_key", "unique_cache_key", "dedup_key"}:
                row = conn.execute(
                    "SELECT question_text, options_json, source, validated FROM unique_cache WHERE unique_cache_key = ?",
                    (unique_key,),
                ).fetchone()
                if row:
                    hit = _row_to_hit(row, unique_key, "canonical_unique_key")
                    if hit:
                        if debug_info:
                            hit["debug"] = debug_info
                        return hit
    finally:
        conn.close()

    if debug:
        return {"miss": True, "debug": debug_info, "event_cache_key": event_key, "unique_cache_key": unique_key}
    return None


def lookup_hybrid_cache_strict(
    cache_path: Path | str,
    dataset: str,
    semantic_options: Sequence[Mapping[str, Any]],
    implementation_mode: str = "",
    *,
    context: Mapping[str, Any] | None = None,
    event_cache_key: str | None = None,
    unique_cache_key: str | None = None,
    lookup_priority: Sequence[str] | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    hit = lookup_hybrid_cache(
        cache_path,
        dataset,
        semantic_options,
        implementation_mode,
        context=context,
        event_cache_key=event_cache_key,
        unique_cache_key=unique_cache_key,
        lookup_priority=lookup_priority,
        debug=debug,
    )
    if not hit or hit.get("miss"):
        dbg = hit.get("debug") if isinstance(hit, dict) else None
        msg = f"hybrid cache miss for dataset={dataset}"
        if dbg:
            msg += f" event_key={dbg.get('event_cache_key')} unique_key={dbg.get('unique_cache_key')}"
        raise CacheMissError(msg)
    return hit
