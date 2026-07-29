from __future__ import annotations

import os
import re
from typing import Any

from src.agents.llm_client import LLMClient, _is_timeout_exc
from src.agents.prompt_templates import neutrality_judge_prompt, neutrality_rewrite_prompt
from src.utils.session_guard_state import (
    disable_neutrality_rewrite_only,
    force_neutrality_llm_short_circuit,
    guard_note_neutrality_judge,
    guard_note_neutrality_rewrite,
    guard_note_neutrality_short_circuit,
)
from src.utils.neutrality_cache_stats import (
    record_judge,
    record_loop_exhausted,
    record_noop_rewrite_skip,
    record_rewrite,
)


def _strip_none(options: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(o) for o in options if o.get("option_id") != "none"]


def _append_none(options_core: list[dict[str, Any]], include_none: bool) -> list[dict[str, Any]]:
    if not include_none:
        return list(options_core)
    return list(options_core) + [
        {"option_id": "none", "option_text": "None of the above fits well."},
    ]


def _sync_ids_hypotheses(
    core: list[dict[str, Any]],
    hypotheses: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    hid_order = [h["hypothesis_id"] for h in hypotheses]
    by_id = {o["option_id"]: o["option_text"] for o in core}
    missing = any(h not in by_id for h in hid_order)
    if missing:
        return [], False
    synced = [{"option_id": h, "option_text": str(by_id[h])} for h in hid_order]
    return synced, True


def _normalize_text(s: str) -> str:
    """Whitespace-insensitive canonical form for signatures (no semantic change to model)."""
    t = str(s or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


def _options_norm_signature(opts: list[dict[str, Any]]) -> tuple[tuple[str, str], ...]:
    rows = sorted(
        ((str(o["option_id"]), _normalize_text(str(o.get("option_text", "")))) for o in opts),
        key=lambda x: x[0],
    )
    return tuple(rows)


def _judge_cache_payload(working: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"option_id": str(o["option_id"]), "option_text": _normalize_text(str(o.get("option_text", "")))}
        for o in sorted(working, key=lambda x: str(x["option_id"]))
    ]


def _rewrite_cache_payload(
    hypotheses: list[dict[str, Any]],
    working: list[dict[str, Any]],
    favored_option_id: str | None,
    critique: str,
) -> dict[str, Any]:
    hs = [
        {
            "hypothesis_id": str(h["hypothesis_id"]),
            "text_description": _normalize_text(str(h.get("text_description", ""))),
        }
        for h in sorted(hypotheses, key=lambda x: str(x["hypothesis_id"]))
    ]
    opts00 = [
        {"option_id": str(o["option_id"]), "option_text": _normalize_text(str(o.get("option_text", "")))}
        for o in sorted(working, key=lambda x: str(x["option_id"]))
    ]
    fav = None if favored_option_id in (None, "", "null") else str(favored_option_id)
    return {
        "hypotheses": hs,
        "options": opts00,
        "favored_option_id": fav,
        "critique": _normalize_text(str(critique or "")),
    }


def _neutrality_disk_cache_enabled(llm: LLMClient) -> bool:
    if llm.mode == "mock":
        return False
    return os.environ.get("INTERREC_DISABLE_NEUTRALITY_CACHE", "").strip() not in ("1", "true", "yes")


def neutrality_check_options(
    llm: LLMClient,
    hypotheses: list[dict[str, Any]],
    options: list[dict[str, Any]],
    *,
    mode: str = "rule",
    max_retries: int = 3,
    include_none: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Judge / rewrite MCQ option text for neutrality (method §4.2)."""

    log: dict[str, Any] = {
        "mode": mode,
        "balanced_final": False,
        "retries_used": 0,
        "judgments": [],
        "reverted_to_original": False,
        "llm_structured_audits": [],
        "retry_traces": [],
    }

    if mode == "off":
        log["skipped"] = "off"
        log["balanced_final"] = True
        return options, log

    if mode == "rule":
        from src.agents.option_balancer import OptionBalancer

        b = OptionBalancer()
        core0 = _strip_none(options)
        pad = [{"option_id": "none", "option_text": "none"}] if include_none else []
        log["balanced_final"] = b.check(core0 + pad)
        log["skipped"] = "rule_heuristic_only"
        return options, log

    core0 = _strip_none(options)

    if llm.mode == "mock":
        from src.agents.option_balancer import OptionBalancer

        pad = [{"option_id": "none", "option_text": "none"}] if include_none else []
        b = OptionBalancer()
        log["balanced_final"] = b.check(core0 + pad)
        log["skipped"] = "mock_llm_uses_rule"
        return options, log

    working, ok = _sync_ids_hypotheses(core0, hypotheses)
    if not ok:
        log["balanced_final"] = False
        log["reverted_to_original"] = True
        log["recover"] = "id_mismatch"
        return options, log

    if mode == "llm" and force_neutrality_llm_short_circuit():
        from src.agents.option_balancer import OptionBalancer

        pad = [{"option_id": "none", "option_text": "none"}] if include_none else []
        b = OptionBalancer()
        log["balanced_final"] = b.check(core0 + pad)
        log["skipped"] = "engineering_guard_llm_quota"
        guard_note_neutrality_short_circuit()
        return options, log

    max_retries = max(1, int(max_retries))
    last_balanced = False
    use_cache = _neutrality_disk_cache_enabled(llm)

    for attempt in range(max_retries):
        jp = _judge_cache_payload(working)
        msgs = neutrality_judge_prompt(working)
        if attempt > 0:
            msgs[-1]["content"] += (
                f"\n(Round {attempt + 1}/{max_retries}. JSON only.)\n"
            )

        try:
            if use_cache:
                parsed = llm.structured_json(
                    msgs,
                    cache_namespace="neutrality_judge",
                    cache_payload=jp,
                    profile_component="neutrality_judge",
                )
            else:
                parsed = llm.structured_json(msgs, profile_component="neutrality_judge")
        except Exception as exc:
            if _is_timeout_exc(exc):
                log["recover"] = "neutrality_judge_timeout"
                log["reverted_to_original"] = True
                log["timeout_exception"] = type(exc).__qualname__
                return options, log
            raise
        guard_note_neutrality_judge()
        audit = dict(llm.last_structured_audit)
        record_judge(cache_hit=bool(audit.get("cache_hit")))
        log["llm_structured_audits"].append({"phase": "neutrality_judge", **audit})
        if not isinstance(parsed, dict):
            parsed = {}
        log["judgments"].append(parsed)

        balanced = bool(parsed.get("balanced", False))
        last_balanced = balanced
        log["retries_used"] = attempt
        log["retry_traces"].append(
            {
                "attempt": attempt,
                "balanced": balanced,
                "cache_hit_judge": bool(audit.get("cache_hit")),
                "fallback_reason": audit.get("fallback_reason"),
            }
        )

        if balanced:
            log["balanced_final"] = True
            return _append_none(working, include_none), log

        favored = parsed.get("favored_option_id")
        critique = str(parsed.get("critique", "") or "").strip()

        pre_sig = _options_norm_signature(working)

        if disable_neutrality_rewrite_only():
            log["rewrite_skipped_budget"] = True
            log["balanced_final"] = False
            log["best_effort"] = True
            return _append_none(working, include_none), log

        rp = _rewrite_cache_payload(
            hypotheses,
            working,
            str(favored) if favored else None,
            critique,
        )
        rmsgs = neutrality_rewrite_prompt(
            hypotheses,
            working,
            favored_option_id=str(favored) if favored else None,
            critique=critique,
        )
        try:
            if use_cache:
                revised = llm.structured_json(
                    rmsgs,
                    cache_namespace="neutrality_rewrite",
                    cache_payload=rp,
                    profile_component="neutrality_rewrite",
                )
            else:
                revised = llm.structured_json(rmsgs, profile_component="neutrality_rewrite")
        except Exception as exc:
            if _is_timeout_exc(exc):
                log["recover"] = "neutrality_rewrite_timeout"
                log["reverted_to_original"] = True
                log["timeout_exception"] = type(exc).__qualname__
                return options, log
            raise
        guard_note_neutrality_rewrite()
        raudit = dict(llm.last_structured_audit)
        record_rewrite(cache_hit=bool(raudit.get("cache_hit")))
        log["llm_structured_audits"].append({"phase": "neutrality_rewrite", **raudit})
        log["retry_traces"][-1]["cache_hit_rewrite"] = bool(raudit.get("cache_hit"))

        if not isinstance(revised, list):
            log["balanced_final"] = False
            log["reverted_to_original"] = True
            log["recover"] = "rewrite_not_list"
            return options, log

        by_txt: dict[str, str] = {}
        for o in revised:
            oid = str(o.get("option_id", ""))
            txt = str(o.get("option_text", ""))
            by_txt[oid] = txt

        working = [
            {"option_id": h["hypothesis_id"], "option_text": by_txt.get(h["hypothesis_id"], "")}
            for h in hypotheses
        ]

        missing_empty = any(not w["option_text"].strip() for w in working)
        if missing_empty:
            log["balanced_final"] = False
            log["reverted_to_original"] = True
            log["recover"] = "rewrite_incomplete"
            return options, log

        post_sig = _options_norm_signature(working)
        if post_sig == pre_sig:
            log["retry_stopped"] = "noop_rewrite_identical_normalized"
            log["balanced_final"] = False
            log["best_effort"] = True
            record_noop_rewrite_skip()
            return _append_none(working, include_none), log

    log["balanced_final"] = bool(last_balanced)
    log["best_effort"] = True
    record_loop_exhausted()
    return _append_none(working, include_none), log
