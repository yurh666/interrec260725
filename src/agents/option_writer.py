from __future__ import annotations

from typing import Any

from src.agents.llm_client import LLMClient
from src.agents.prompt_templates import option_writer_prompt
from src.agents.stable_cache_key import stable_option_writer_payload


def _option_writer_cache_payload(
    hypotheses: list[dict[str, Any]],
    representative_items: dict[str, list[str]],
) -> list[dict[str, Any]]:
    rows = []
    for h in hypotheses:
        hid = str(h["hypothesis_id"])
        rep = sorted(str(x) for x in representative_items.get(hid, [])[:8])
        rows.append(
            {
                "hypothesis_id": hid,
                "text_description": str(h.get("text_description", "")),
                "representative_items": rep,
            }
        )
    return rows


def _options_aligned(hypotheses: list[dict[str, Any]], parsed: Any) -> bool:
    if not isinstance(parsed, list) or len(parsed) != len(hypotheses):
        return False
    for h, o in zip(hypotheses, parsed):
        if not isinstance(o, dict):
            return False
        if str(o.get("option_id", "")) != str(h["hypothesis_id"]):
            return False
        if not str(o.get("option_text", "")).strip():
            return False
    return True


class OptionWriter:
    """Converts selected hypotheses into user-facing multiple-choice option text."""

    def __init__(self, llm: LLMClient, include_none: bool = True) -> None:
        self.llm = llm
        self.include_none = include_none
        self.implementation_mode = "template"
        self.last_call_audit: dict[str, Any] = {}

    def write(
        self,
        hypotheses: list[dict[str, Any]],
        representative_items: dict[str, list[str]],
        *,
        session_id: str | None = None,
        turn_id: int | None = None,
    ) -> list[dict[str, Any]]:
        if self.llm.mode == "mock":
            self.implementation_mode = "mock"
            self.last_call_audit = {
                "component": "option_writer",
                "implementation_mode": "mock_template",
                "parse_success": True,
                "cache_hit": False,
                "cache_miss": False,
                "cache_key_sha256": None,
                "retry_count": 0,
                "raw_response_saved": False,
                "raw_storage_path": None,
                "fallback_reason": None,
            }
            opts = self._template_options(hypotheses)
            return self._append_none(opts)

        rep_for_prompt = {
            str(h["hypothesis_id"]): sorted(
                str(x) for x in representative_items.get(str(h["hypothesis_id"]), [])[:8]
            )
            for h in hypotheses
        }
        if getattr(self.llm, "use_stable_cache_keys", False):
            payload = stable_option_writer_payload(
                hypotheses,
                rep_for_prompt,
                session_id=session_id,
                turn_id=turn_id,
            )
        else:
            payload = _option_writer_cache_payload(hypotheses, rep_for_prompt)
        msgs = option_writer_prompt(hypotheses, rep_for_prompt)
        parsed = self.llm.structured_json(
            msgs,
            cache_namespace="option_writer",
            cache_payload=payload,
        )
        la = dict(self.llm.last_structured_audit)
        la["component"] = "option_writer"

        if _options_aligned(hypotheses, parsed):
            opts = parsed
            self.implementation_mode = "real_llm_ok"
            self.last_call_audit = {
                **la,
                "implementation_mode": "real_llm_ok",
                "parse_success": True,
                "fallback_reason": None,
            }
        else:
            opts = self._template_options(hypotheses)
            self.implementation_mode = "real_llm_fallback_rule_based"
            self.last_call_audit = {
                **la,
                "implementation_mode": "real_llm_fallback_rule_based",
                "parse_success": bool(la.get("parse_success")),
                "fallback_reason": "parsed_options_misaligned_or_invalid",
            }

        return self._append_none(opts)

    def _append_none(self, options: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.include_none:
            return list(options) + [
                {"option_id": "none", "option_text": "None of the above fits well."},
            ]
        return list(options)

    def _template_options(self, hypotheses: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {"option_id": h["hypothesis_id"], "option_text": h["text_description"]}
            for h in hypotheses
        ]
