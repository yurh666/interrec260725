from __future__ import annotations

from typing import Any

from src.agents.llm_client import LLMClient
from src.agents.prompt_templates import direction_translator_prompt
from src.agents.stable_cache_key import stable_direction_translator_payload


def _stable_directions_for_llm(directions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic anchor ordering for prompts + cache payloads (same multiset of items)."""

    out: list[dict[str, Any]] = []
    for d in directions:
        dd = dict(d)
        dd["positive_anchors"] = sorted(str(x) for x in d.get("positive_anchors", []))
        dd["negative_anchors"] = sorted(str(x) for x in d.get("negative_anchors", []))
        out.append(dd)
    return out


def _direction_cache_payload(directions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for d in directions:
        out.append(
            {
                "direction_id": int(d["direction_id"]),
                "positive_anchors": [str(x) for x in d.get("positive_anchors", [])],
                "negative_anchors": [str(x) for x in d.get("negative_anchors", [])],
                "lambda": round(float(d.get("lambda", 0.0)), 8),
            }
        )
    return out


def _merged_directions_valid(merged: list[dict[str, Any]]) -> bool:
    required = {"direction_name", "positive_side", "negative_side", "is_meaningful"}
    for row in merged:
        if not required.issubset(row.keys()):
            return False
        if str(row.get("direction_name") or "").strip() == "":
            return False
        if str(row.get("positive_side") or "").strip() == "":
            return False
        if str(row.get("negative_side") or "").strip() == "":
            return False
    return True


class DirectionTranslator:
    """Converts anchor-item pairs (positive/negative) into semantic direction descriptions.

    ``last_call_audit`` holds per-call implementation_mode, parse_success, cache info, etc.
    """

    def __init__(
        self,
        llm: LLMClient,
        item_lookup: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.llm = llm
        self.item_lookup = item_lookup
        self.implementation_mode = "mock"
        self.last_call_audit: dict[str, Any] = {}
        self._session_translation_cache: dict[str, list[dict[str, Any]]] = {}

    def translate(
        self,
        directions: list[dict[str, Any]],
        *,
        session_id: str | None = None,
        turn_id: int | None = None,
        session_cache_enabled: bool = False,
        session_cache_key: str | None = None,
        session_speed_stats: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if session_cache_enabled and session_cache_key and session_cache_key in self._session_translation_cache:
            cached = self._session_translation_cache[session_cache_key]
            self.implementation_mode = "session_cache_hit"
            self.last_call_audit = {
                "component": "direction_translator",
                "implementation_mode": "session_cache_hit",
                "parse_success": True,
                "cache_hit": True,
                "cache_miss": False,
                "session_cache_key": session_cache_key,
                "n_api_calls": 0,
            }
            if session_speed_stats is not None:
                session_speed_stats["direction_translator_cache_hits"] = int(
                    session_speed_stats.get("direction_translator_cache_hits", 0)
                ) + 1
            return [dict(x) for x in cached]

        if self.llm.mode == "mock":
            self.implementation_mode = "mock"
            self.last_call_audit = {
                "component": "direction_translator",
                "implementation_mode": "mock",
                "parse_success": True,
                "cache_hit": False,
                "cache_miss": False,
                "cache_key_sha256": None,
                "retry_count": 0,
                "raw_response_saved": False,
                "raw_storage_path": None,
                "fallback_reason": None,
            }
            return self._rule_based(directions)

        dirs_s = _stable_directions_for_llm(directions)
        if getattr(self.llm, "use_stable_cache_keys", False):
            payload = stable_direction_translator_payload(
                dirs_s, session_id=session_id, turn_id=turn_id
            )
        else:
            payload = _direction_cache_payload(dirs_s)
        msgs = direction_translator_prompt(dirs_s, item_lookup=self.item_lookup)
        parsed = self.llm.structured_json(
            msgs,
            cache_namespace="direction_translator",
            cache_payload=payload,
        )
        la = dict(self.llm.last_structured_audit)
        la["component"] = "direction_translator"

        if isinstance(parsed, list) and len(parsed) == len(directions):
            merged = self._merge_with_input(directions, parsed)
            if _merged_directions_valid(merged):
                self.implementation_mode = "real_llm_ok"
                self.last_call_audit = {
                    **la,
                    "implementation_mode": "real_llm_ok",
                    "parse_success": True,
                    "fallback_reason": None,
                }
                if session_cache_enabled and session_cache_key:
                    self._session_translation_cache[session_cache_key] = [dict(x) for x in merged]
                    if session_speed_stats is not None:
                        session_speed_stats["direction_translator_cache_misses"] = int(
                            session_speed_stats.get("direction_translator_cache_misses", 0)
                        ) + 1
                return merged

        fb = self._rule_based(directions)
        why = (
            "llm_returned_invalid_or_incomplete_structure"
            if isinstance(parsed, list)
            else "llm_parse_not_a_list_or_length_mismatch"
        )
        if not isinstance(parsed, list):
            why += f" (parsed_type={type(parsed).__name__})"
        self.implementation_mode = "real_llm_fallback_rule_based"
        self.last_call_audit = {
            **la,
            "implementation_mode": "real_llm_fallback_rule_based",
            "parse_success": bool(la.get("parse_success")),
            "fallback_reason": why,
        }
        if session_cache_enabled and session_cache_key:
            self._session_translation_cache[session_cache_key] = [dict(x) for x in fb]
            if session_speed_stats is not None:
                session_speed_stats["direction_translator_cache_misses"] = int(
                    session_speed_stats.get("direction_translator_cache_misses", 0)
                ) + 1
        return fb

    def rule_based_translate(
        self,
        directions: list[dict[str, Any]],
        *,
        reason: str = "speed_opt_skipped",
    ) -> list[dict[str, Any]]:
        """Non-LLM translation for MovieLens speed-opt no-ask turns."""
        out = self._rule_based(directions)
        self.implementation_mode = "speed_opt_skipped"
        self.last_call_audit = {
            "component": "direction_translator",
            "implementation_mode": "speed_opt_skipped",
            "parse_success": True,
            "cache_hit": False,
            "cache_miss": False,
            "n_api_calls": 0,
            "skip_reason": reason,
        }
        return out

    def _rule_based(self, directions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from src.agents.prompt_templates import _item_label

        out = []
        for d in directions:
            pos = d.get("positive_anchors", [])[:2]
            neg = d.get("negative_anchors", [])[:2]
            pos_desc = ", ".join(_item_label(x, self.item_lookup) for x in pos)
            neg_desc = ", ".join(_item_label(x, self.item_lookup) for x in neg)
            out.append(
                {
                    "direction_id": d["direction_id"],
                    "direction_name": f"direction_{d['direction_id']}",
                    "positive_side": f"items like: {pos_desc}",
                    "negative_side": f"items like: {neg_desc}",
                    "is_meaningful": True,
                    "lambda": d.get("lambda", 0.0),
                    "positive_anchors": d.get("positive_anchors", []),
                    "negative_anchors": d.get("negative_anchors", []),
                }
            )
        return out

    def _merge_with_input(
        self, directions: list[dict[str, Any]], parsed: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        by_id = {int(p["direction_id"]): p for p in parsed if "direction_id" in p}
        out = []
        for d in directions:
            did = int(d["direction_id"])
            merged = dict(d)
            if did in by_id:
                merged.update(by_id[did])
            else:
                merged.setdefault("direction_name", f"direction_{did}")
                merged.setdefault("positive_side", "")
                merged.setdefault("negative_side", "")
                merged.setdefault("is_meaningful", True)
            out.append(merged)
        return out
