"""Unified hybrid surface cache key generation for build and runtime lookup."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

# Canonical empty mode used in unique/dedup keys extracted from existing logs (metadata was blank).
UNIQUE_KEY_IMPLEMENTATION_MODE = ""


def normalize_implementation_mode(mode: str | None) -> str:
    return str(mode or "").strip()


def normalize_option_ids(option_ids: Sequence[str] | None) -> list[str]:
    return [str(x) for x in (option_ids or []) if str(x)]


def short_phrase(opt: Mapping[str, Any]) -> str:
    for k in ("cluster_phrase", "feature_signature", "text_description"):
        v = opt.get(k)
        if v is None:
            continue
        if isinstance(v, list):
            return ",".join(str(x) for x in v)[:120]
        s = str(v).strip()
        if s:
            return s[:120]
    return ""


def normalize_representative_ids(options: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    out: list[list[str]] = []
    for opt in options:
        rep = opt.get("representative_item_ids") or opt.get("representative_items") or []
        if rep and not isinstance(rep[0], str):
            rep = [str(x) for x in rep]
        else:
            rep = [str(x) for x in rep]
        out.append(rep)
    return out


def stable_json_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def compute_option_payload_hash(
    dataset: str,
    episode_id: str,
    turn: int | str,
    hypotheses: Sequence[Mapping[str, Any]],
) -> str:
    """Match scripts/ask_event_utils.payload_hash."""
    payload = {
        "dataset": dataset,
        "episode_id": episode_id,
        "turn": int(turn) if str(turn).isdigit() else turn,
        "options": [
            {
                "option_id": h.get("hypothesis_id") or h.get("option_id"),
                "representative_item_ids": h.get("representative_items")
                or h.get("parsed_representative_items")
                or h.get("representative_item_ids"),
                "feature_signature": h.get("feature_signature"),
            }
            for h in hypotheses
        ],
    }
    return stable_json_hash(payload)


def build_event_cache_key(
    dataset: str,
    episode_id: str,
    turn: int | str,
    *,
    run_id: str = "",
    implementation_mode: str = "",
    option_ids: Sequence[str] | None = None,
    option_payload_hash: str = "",
    ask_index: int | None = None,
    seed: int | None = None,
) -> str:
    """Event-level exact key; matches scripts/build_gpu_writer_input_from_ask_events.cache_key."""
    _ = ask_index, seed  # reserved for future ask-occurrence disambiguation
    parts = [
        str(dataset or ""),
        str(run_id or ""),
        str(episode_id or ""),
        str(turn or ""),
        normalize_implementation_mode(implementation_mode),
        "|".join(normalize_option_ids(option_ids)),
        str(option_payload_hash or ""),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def build_event_cache_key_from_ask_event(event: Mapping[str, Any]) -> str:
    return build_event_cache_key(
        str(event.get("dataset") or ""),
        str(event.get("episode_id") or ""),
        event.get("turn") or 0,
        run_id=str(event.get("run_id") or ""),
        implementation_mode=str(event.get("implementation_mode") or event.get("semantic_backend_mode") or ""),
        option_ids=event.get("option_ids") or [],
        option_payload_hash=str(event.get("option_payload_hash") or ""),
    )


def build_event_cache_key_from_runtime_context(
    context: Mapping[str, Any],
    semantic_options: Sequence[Mapping[str, Any]],
) -> str:
    option_ids = context.get("option_ids") or [
        str(o.get("option_id") or o.get("hypothesis_id") or "") for o in semantic_options
    ]
    payload_hash = context.get("option_payload_hash")
    if not payload_hash:
        payload_hash = compute_option_payload_hash(
            str(context.get("dataset") or ""),
            str(context.get("episode_id") or context.get("session_id") or ""),
            context.get("turn") or 0,
            semantic_options,
        )
    impl = context.get("implementation_mode") or context.get("semantic_backend_mode") or ""
    return build_event_cache_key(
        str(context.get("dataset") or ""),
        str(context.get("episode_id") or context.get("session_id") or ""),
        context.get("turn") or 0,
        run_id=str(context.get("run_id") or context.get("cache_run_id") or ""),
        implementation_mode=str(impl),
        option_ids=normalize_option_ids(option_ids),
        option_payload_hash=str(payload_hash or ""),
    )


def build_unique_cache_key_from_parts(
    dataset: str,
    options: Sequence[Mapping[str, Any]],
    *,
    implementation_mode: str = UNIQUE_KEY_IMPLEMENTATION_MODE,
    rep_ids_by_index: Sequence[Sequence[str]] | None = None,
) -> str:
    parts: list[str] = [str(dataset or ""), normalize_implementation_mode(implementation_mode)]
    rep_lists = list(rep_ids_by_index) if rep_ids_by_index is not None else None
    for i, opt in enumerate(options):
        parts.append(str(opt.get("option_id") or opt.get("hypothesis_id") or ""))
        parts.append(short_phrase(opt))
        names = [str(x) for x in (opt.get("representative_item_names") or [])]
        parts.append("|".join(names))
        rep_ids = opt.get("representative_item_ids") or opt.get("representative_items")
        if rep_ids is None and rep_lists is not None and i < len(rep_lists):
            rep_ids = rep_lists[i]
        parts.append("|".join(str(x) for x in (rep_ids or [])))
    return hashlib.sha256("||".join(parts).encode()).hexdigest()


def build_unique_cache_key_from_writer_row(
    row: Mapping[str, Any],
    rep_ids_by_event: Mapping[str, list[list[str]]] | None = None,
) -> str:
    inp = row.get("input") or {}
    meta = row.get("metadata") or {}
    opts = inp.get("options") or []
    eid = str(meta.get("event_id") or "")
    rep_from_events = (rep_ids_by_event or {}).get(eid)
    impl = str(meta.get("semantic_backend_mode") or meta.get("implementation_mode") or UNIQUE_KEY_IMPLEMENTATION_MODE)
    return build_unique_cache_key_from_parts(
        str(row.get("dataset") or ""),
        opts,
        implementation_mode=impl,
        rep_ids_by_index=rep_from_events,
    )


def build_unique_cache_key_from_runtime(
    dataset: str,
    options: Sequence[Mapping[str, Any]],
    implementation_mode: str = "",
    semantic_backend_mode: str | None = None,
) -> str:
    """Dedup fallback key. Existing cache unique keys use empty metadata mode."""
    mode = semantic_backend_mode if semantic_backend_mode is not None else UNIQUE_KEY_IMPLEMENTATION_MODE
    if not mode:
        mode = UNIQUE_KEY_IMPLEMENTATION_MODE
    return build_unique_cache_key_from_parts(dataset, options, implementation_mode=mode)


def debug_key_bundle(
    context: Mapping[str, Any],
    semantic_options: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    event_key = build_event_cache_key_from_runtime_context(context, semantic_options)
    unique_key = build_unique_cache_key_from_runtime(
        str(context.get("dataset") or ""),
        semantic_options,
        semantic_backend_mode=UNIQUE_KEY_IMPLEMENTATION_MODE,
    )
    return {
        "event_cache_key": event_key,
        "unique_cache_key": unique_key,
        "dataset": context.get("dataset"),
        "episode_id": context.get("episode_id"),
        "turn": context.get("turn"),
        "run_id": context.get("run_id") or context.get("cache_run_id"),
        "implementation_mode": context.get("implementation_mode"),
        "option_ids": context.get("option_ids"),
        "option_payload_hash": context.get("option_payload_hash"),
    }
