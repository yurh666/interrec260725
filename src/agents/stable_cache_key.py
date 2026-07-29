"""Canonical cache payloads for cluster-bound v2 integrated reproducibility protocol."""

from __future__ import annotations

import hashlib
import json
from typing import Any

CACHE_KEY_VERSION = "cb_v2_stable_v1"
METHOD_VERSION = "interrec_cb_v2_integrated_v1"
PROMPT_TEMPLATE_VERSION = "cluster_bound_v2_system_v2_anchors_user_v1"


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def payload_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def prompt_hash(messages: list[dict[str, str]]) -> str:
    slim = [{"role": m.get("role"), "content": str(m.get("content") or "")[:2000]} for m in messages]
    return payload_hash({"prompt_template_version": PROMPT_TEMPLATE_VERSION, "messages": slim})


def cache_context_base(
    *,
    session_id: str | None = None,
    turn_id: int | None = None,
    domain: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    _ = seed  # telemetry only; not included in cb_v2_stable_v1 disk keys
    return {
        "cache_key_version": CACHE_KEY_VERSION,
        "session_id": str(session_id) if session_id else None,
        "turn_id": int(turn_id) if turn_id is not None else None,
        "domain": str(domain) if domain else None,
    }


def normalize_text_for_cache(text: Any) -> str:
    import re

    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def normalize_hypothesis_for_cache(h: dict[str, Any]) -> dict[str, Any]:
    hid = str(h.get("hypothesis_id") or h.get("option_id") or "")
    bound = h.get("bound_cluster_id")
    if bound is None:
        bound = h.get("parsed_bound_cluster_id")
    if bound in (None, "", 0, "0"):
        bound_s = ""
    else:
        bound_s = str(bound)
    return {
        **h,
        "hypothesis_id": hid,
        "role": str(h.get("role") or "").strip(),
        "bound_cluster_id": bound_s,
    }


def legacy_direction_translator_list_payload(stable_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for d in stable_payload.get("directions") or []:
        if not isinstance(d, dict):
            continue
        rows.append(
            {
                "direction_id": int(d["direction_id"]),
                "positive_anchors": list(d.get("positive_anchors") or []),
                "negative_anchors": list(d.get("negative_anchors") or []),
                "lambda": round(float(d.get("lambda", 0.0)), 8),
            }
        )
    return rows


def replay_cache_lookup_variants(namespace: str, payload: Any) -> list[tuple[str, Any, str]]:
    """Return (tag, payload, miss_reason_hint) variants for replay_strict alias lookup."""

    from src.agents.llm_cache import payload_hex_key

    variants: list[tuple[str, Any, str]] = []
    seen: set[str] = set()

    def add(p: Any, tag: str, hint: str = "") -> None:
        if p is None:
            return
        hk = payload_hex_key(namespace, p)
        if hk in seen:
            return
        seen.add(hk)
        variants.append((tag, p, hint))

    add(payload, "primary", "")
    if not isinstance(payload, dict):
        return variants

    base = dict(payload)
    tid = base.get("turn_id")
    sid = base.get("session_id")
    comp = str(base.get("component") or namespace)

    for alt_tid in range(1, 16):
        if alt_tid != tid:
            trial = dict(base)
            trial["turn_id"] = alt_tid
            add(trial, f"turn_id_alias_{alt_tid}", "turn_id_mismatch")

    no_turn = {k: v for k, v in base.items() if k != "turn_id"}
    add(no_turn, "no_turn_id", "turn_id_mismatch")

    no_sess = {k: v for k, v in base.items() if k != "session_id"}
    add(no_sess, "no_session_id", "session_id_mismatch")

    structural = {
        k: v
        for k, v in base.items()
        if k
        not in (
            "session_id",
            "turn_id",
            "seed",
        )
    }
    add(structural, "structural_only", "session_or_turn_mismatch")

    if comp == "direction_translator" and base.get("directions"):
        add(legacy_direction_translator_list_payload(base), "legacy_list_payload", "cache_key_version_mismatch")

    return variants


def stable_direction_translator_payload(
    directions: list[dict[str, Any]],
    *,
    session_id: str | None = None,
    turn_id: int | None = None,
) -> dict[str, Any]:
    rows = []
    for d in sorted(directions, key=lambda x: int(x.get("direction_id") or 0)):
        rows.append(
            {
                "direction_id": int(d["direction_id"]),
                "positive_anchors": sorted(str(x) for x in d.get("positive_anchors", [])),
                "negative_anchors": sorted(str(x) for x in d.get("negative_anchors", [])),
                "lambda": round(float(d.get("lambda", 0.0)), 6),
            }
        )
    return {
        **cache_context_base(session_id=session_id, turn_id=turn_id),
        "component": "direction_translator",
        "directions": rows,
    }


def stable_cluster_bound_generation_payload(
    input_payload: dict[str, Any],
    *,
    session_id: str | None = None,
    turn_id: int | None = None,
) -> dict[str, Any]:
    candidates = []
    for c in sorted(
        input_payload.get("candidate_options") or [],
        key=lambda x: str(x.get("option_id") or ""),
    ):
        if not isinstance(c, dict):
            continue
        candidates.append(
            {
                "option_id": str(c.get("option_id") or ""),
                "role": str(c.get("role") or ""),
                "bound_cluster_id": str(c.get("bound_cluster_id") or ""),
                "representative_item_ids": sorted(str(x) for x in (c.get("representative_item_ids") or [])),
                "cluster_keywords": sorted(str(x) for x in (c.get("cluster_keywords") or [])),
            }
        )
    return {
        **cache_context_base(
            session_id=session_id,
            turn_id=turn_id,
            domain=str(input_payload.get("domain") or ""),
        ),
        "component": "cluster_bound_generation_v2",
        "candidate_options": candidates,
    }


def stable_hypothesis_generator_payload(
    directions: list[dict[str, Any]],
    observed_items: list[str],
    M: int,
    *,
    schema: str = "legacy",
    session_id: str | None = None,
    turn_id: int | None = None,
) -> dict[str, Any]:
    dirs = []
    for d in sorted(directions, key=lambda x: int(x.get("direction_id") or 0)):
        dirs.append(
            {
                "direction_id": int(d["direction_id"]),
                "lambda": round(float(d.get("lambda", 0.0)), 6),
                "positive_anchors": sorted(str(x) for x in d.get("positive_anchors", [])),
                "negative_anchors": sorted(str(x) for x in d.get("negative_anchors", [])),
                "is_meaningful": bool(d.get("is_meaningful", True)),
            }
        )
    return {
        **cache_context_base(session_id=session_id, turn_id=turn_id),
        "component": "hypothesis_generator",
        "schema": str(schema),
        "directions": dirs,
        "history_tail_ids": sorted(str(x) for x in observed_items[-20:]),
        "M": int(M),
    }


def stable_option_writer_payload(
    hypotheses: list[dict[str, Any]],
    representative_items: dict[str, list[str]],
    *,
    session_id: str | None = None,
    turn_id: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    rows = []
    for h in sorted(hypotheses, key=lambda x: str(x.get("hypothesis_id") or x.get("option_id") or "")):
        hid = str(h.get("hypothesis_id") or h.get("option_id") or "")
        rep_ids = sorted(
            str(x)
            for x in (
                h.get("representative_items")
                or h.get("representative_item_ids")
                or representative_items.get(hid)
                or []
            )[:12]
        )
        role = str(h.get("role") or "").strip()
        if not role:
            rs = str(h.get("rank_safety_role") or "")
            if rs == "current_top_preserve":
                role = "preserve"
            elif rs == "boundary_challenge":
                role = "boundary"
            elif rs:
                role = rs
        rows.append(
            {
                "option_id": hid,
                "role": role,
                "bound_cluster_id": str(h.get("bound_cluster_id") or h.get("parsed_bound_cluster_id") or ""),
                "representative_item_ids": rep_ids,
            }
        )
    return {
        **cache_context_base(session_id=session_id, turn_id=turn_id),
        "component": "option_writer",
        "options": rows,
    }
