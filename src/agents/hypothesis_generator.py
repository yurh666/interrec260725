from __future__ import annotations

import re
from typing import Any

from src.agents.llm_client import LLMClient
from src.agents.stable_cache_key import stable_hypothesis_generator_payload
from src.agents.prompt_templates import (
    hypothesis_generator_contrastive_v2_prompt,
    hypothesis_generator_prompt,
)


def _squish(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip())


def _as_str_list(x: Any, *, max_n: int = 12) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str) and x.strip():
        return [x.strip()]
    if isinstance(x, list):
        out: list[str] = []
        for el in x[:max_n]:
            s = _squish(el)
            if s:
                out.append(s)
        return out
    return []


def _hypothesis_cache_payload(
    directions: list[dict[str, Any]],
    observed_items: list[str],
    M: int,
    *,
    schema: str = "legacy",
) -> dict[str, Any]:
    dirs = []
    for d in directions:
        dirs.append(
            {
                "direction_id": int(d["direction_id"]),
                "lambda": round(float(d.get("lambda", 0.0)), 8),
                "direction_name": _squish(str(d.get("direction_name", ""))),
                "positive_side": _squish(str(d.get("positive_side", ""))),
                "negative_side": _squish(str(d.get("negative_side", ""))),
                "is_meaningful": bool(d.get("is_meaningful", True)),
            }
        )
    return {
        "schema": str(schema),
        "directions": dirs,
        "history_tail_ids": [str(x) for x in observed_items[-20:]],
        "M": int(M),
    }


def _hypothesis_list_ok(rows: Any, M: int) -> bool:
    if not isinstance(rows, list) or len(rows) < M:
        return False
    for h in rows[:M]:
        if not isinstance(h, dict):
            return False
        if not str(h.get("hypothesis_id", "")):
            return False
        if not str(h.get("text_description", "")).strip():
            return False
        fs = h.get("feature_signature")
        if not isinstance(fs, list) or len(fs) == 0:
            return False
    return True


def _max_pairwise_jaccard(sig_lists: list[list[str]]) -> float:
    sets = [set(s) for s in sig_lists if s]
    if len(sets) < 2:
        return 0.0
    mx = 0.0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i], sets[j]
            inter = len(a & b)
            union = len(a | b) or 1
            mx = max(mx, inter / union)
    return mx


def _normalize_contrastive_hypotheses(parsed: Any, M: int) -> list[dict[str, Any]]:
    if not isinstance(parsed, list):
        return []
    out: list[dict[str, Any]] = []
    for h in parsed:
        if len(out) >= M:
            break
        if not isinstance(h, dict):
            continue
        oid = str(h.get("option_id") or h.get("hypothesis_id") or "").strip()
        if not oid:
            continue
        pos = _squish(h.get("positive_intent", ""))
        short_t = _squish(h.get("short_user_facing_option_text", ""))
        dist = _as_str_list(h.get("distinguishing_features"))
        fs = _as_str_list(h.get("feature_signature"))
        if not fs and dist:
            fs = dist[:6]
        if not fs and pos:
            fs = [w for w in pos.split() if len(w) > 2][:5]
        row: dict[str, Any] = {
            "hypothesis_id": oid,
            "option_id": oid,
            "text_description": short_t or pos,
            "positive_intent": pos or short_t,
            "negative_contrast": _squish(h.get("negative_contrast", "")),
            "distinguishing_features": dist,
            "feature_signature": fs,
            "positive_retrieval_query": _squish(h.get("positive_retrieval_query", "")),
            "negative_retrieval_query": _squish(h.get("negative_retrieval_query", "")),
            "representative_diversity_hint": _squish(h.get("representative_diversity_hint", "")),
            "contrast_text": _squish(h.get("contrast_text", "")),
            "short_user_facing_option_text": short_t or pos,
            "rationale": _squish(h.get("rationale", "")),
        }
        if not row["text_description"]:
            row["text_description"] = pos or oid
        out.append(row)
    return out


def _contrastive_v2_hypothesis_list_ok(
    rows: list[dict[str, Any]], M: int, gen_cfg: dict[str, Any]
) -> bool:
    if len(rows) < M:
        return False
    use = rows[:M]
    for h in use:
        if not str(h.get("positive_intent", "")).strip():
            return False
        if gen_cfg.get("require_negative_contrast", True) and len(str(h.get("negative_contrast", "")).strip()) < 4:
            return False
        if gen_cfg.get("require_distinguishing_features", True) and len(h.get("distinguishing_features") or []) < 2:
            return False
        fs = h.get("feature_signature")
        if not isinstance(fs, list) or len(fs) == 0:
            return False
    if gen_cfg.get("enforce_feature_signature_diversity", True):
        mx_j = float(gen_cfg.get("feature_signature_max_jaccard", 0.85))
        if _max_pairwise_jaccard([list(h.get("feature_signature") or []) for h in use]) > mx_j:
            return False
    return True


class HypothesisGenerator:
    """Generates M intent hypotheses given translated uncertainty directions."""

    _STOP = {
        "items",
        "like:",
        "like",
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "in",
        "with",
        "user",
        "may",
        "prefer",
        "vs",
        "versus",
        "side",
        "positive",
        "negative",
    }

    def __init__(
        self,
        llm: LLMClient,
        item_lookup: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.llm = llm
        self.item_lookup = item_lookup
        self.implementation_mode = "mock"
        self.last_call_audit: dict[str, Any] = {}

    def generate(
        self,
        directions: list[dict[str, Any]],
        observed_items: list[str],
        M: int = 5,
        *,
        generation_cfg: dict[str, Any] | None = None,
        session_id: str | None = None,
        turn_id: int | None = None,
    ) -> list[dict[str, Any]]:
        generation_cfg = dict(generation_cfg or {})
        requested_mode = str(generation_cfg.get("mode", "legacy"))
        if requested_mode == "frontier_clustered_v32":
            # Session integration owns the frontier pool/cluster path; HypothesisGenerator can only safely
            # execute surrogate legacy/contrastive_v2 payloads here when called directly without belief/index.
            generation_cfg["mode"] = "legacy"
        mode = str(generation_cfg.get("mode", "legacy"))

        base_audit_template: dict[str, Any] = {
            "component": "hypothesis_generator",
            "hypothesis_generation_mode_requested": requested_mode,
            "hypothesis_generation_mode_resolved": None,
            "schema_fallback_steps": 0,
            "contrastive_v2_attempted": False,
            "contrastive_v2_accepted": False,
            "parse_success": True,
            "cache_hit": False,
            "cache_miss": False,
            "cache_key_sha256": None,
            "retry_count": 0,
            "raw_response_saved": False,
            "raw_storage_path": None,
            "fallback_reason": None,
        }

        if self.llm.mode == "mock":
            self.implementation_mode = "mock"
            self.last_call_audit = {
                **base_audit_template,
                "implementation_mode": "mock",
                "hypothesis_generation_mode_resolved": "mock_contrastive_v2"
                if mode == "contrastive_v2"
                else "mock_rule_based",
            }
            if mode == "contrastive_v2":
                return self._rule_based_contrastive_v2(directions, observed_items, M)
            return self._rule_based(directions, observed_items, M)

        from src.agents.prompt_templates import _item_label

        summary = ", ".join(_item_label(x, self.item_lookup) for x in observed_items[-10:])
        meaningful_dirs = [d for d in directions if d.get("is_meaningful", True)]
        if not meaningful_dirs:
            meaningful_dirs = list(directions)

        use_stable = getattr(self.llm, "use_stable_cache_keys", False)

        if mode != "contrastive_v2":
            if use_stable:
                payload = stable_hypothesis_generator_payload(
                    meaningful_dirs,
                    observed_items,
                    M,
                    schema="legacy",
                    session_id=session_id,
                    turn_id=turn_id,
                )
            else:
                payload = _hypothesis_cache_payload(meaningful_dirs, observed_items, M, schema="legacy")
            msgs = hypothesis_generator_prompt(meaningful_dirs, summary, M)
            parsed = self.llm.structured_json(
                msgs,
                cache_namespace="hypothesis_generator",
                cache_payload=payload,
            )
            la = dict(self.llm.last_structured_audit)
            la["component"] = "hypothesis_generator"
            la["hypothesis_generation_mode_requested"] = "legacy"
            if _hypothesis_list_ok(parsed, M):
                self.implementation_mode = "real_llm_ok"
                self.last_call_audit = {
                    **base_audit_template,
                    **la,
                    "implementation_mode": "real_llm_ok",
                    "hypothesis_generation_mode_resolved": "legacy",
                    "parse_success": True,
                    "fallback_reason": None,
                }
                return parsed[:M]
            fb = self._rule_based(directions, observed_items, M)
            self.implementation_mode = "real_llm_fallback_rule_based"
            self.last_call_audit = {
                **base_audit_template,
                **la,
                "implementation_mode": "real_llm_fallback_rule_based",
                "hypothesis_generation_mode_resolved": "rule_based",
                "schema_fallback_steps": 1,
                "parse_success": bool(la.get("parse_success")),
                "fallback_reason": "hypothesis_json_invalid_or_empty_after_merge",
            }
            return fb

        # --- contrastive_v2 path ---
        if use_stable:
            payload_cv2 = stable_hypothesis_generator_payload(
                meaningful_dirs,
                observed_items,
                M,
                schema="contrastive_v2",
                session_id=session_id,
                turn_id=turn_id,
            )
        else:
            payload_cv2 = _hypothesis_cache_payload(
                meaningful_dirs, observed_items, M, schema="contrastive_v2"
            )
        msgs_cv2 = hypothesis_generator_contrastive_v2_prompt(meaningful_dirs, summary, M)
        parsed_cv2 = self.llm.structured_json(
            msgs_cv2,
            cache_namespace="hypothesis_generator",
            cache_payload=payload_cv2,
        )
        la1 = dict(self.llm.last_structured_audit)
        la1["component"] = "hypothesis_generator"

        rows_cv2 = _normalize_contrastive_hypotheses(parsed_cv2, M)
        ok_cv2 = _contrastive_v2_hypothesis_list_ok(rows_cv2, M, generation_cfg)
        if ok_cv2:
            self.implementation_mode = "real_llm_ok"
            self.last_call_audit = {
                **base_audit_template,
                **la1,
                "implementation_mode": "real_llm_ok",
                "hypothesis_generation_mode_resolved": "contrastive_v2",
                "contrastive_v2_attempted": True,
                "contrastive_v2_accepted": True,
                "parse_success": True,
                "fallback_reason": None,
                "schema_fallback_steps": 0,
            }
            return rows_cv2[:M]

        # fallback: legacy LLM prompt
        if use_stable:
            payload_leg = stable_hypothesis_generator_payload(
                meaningful_dirs,
                observed_items,
                M,
                schema="legacy_after_contrastive_fail",
                session_id=session_id,
                turn_id=turn_id,
            )
        else:
            payload_leg = _hypothesis_cache_payload(
                meaningful_dirs, observed_items, M, schema="legacy_after_contrastive_fail"
            )
        msgs_leg = hypothesis_generator_prompt(meaningful_dirs, summary, M)
        parsed_leg = self.llm.structured_json(
            msgs_leg,
            cache_namespace="hypothesis_generator",
            cache_payload=payload_leg,
        )
        la2 = dict(self.llm.last_structured_audit)
        la2["component"] = "hypothesis_generator"

        if _hypothesis_list_ok(parsed_leg, M):
            self.implementation_mode = "real_llm_ok"
            self.last_call_audit = {
                **base_audit_template,
                **la2,
                "implementation_mode": "real_llm_ok",
                "hypothesis_generation_mode_resolved": "legacy_after_contrastive_invalid",
                "contrastive_v2_attempted": True,
                "contrastive_v2_accepted": False,
                "parse_success": True,
                "fallback_reason": "contrastive_v2_invalid_fell_back_to_legacy_llm",
                "schema_fallback_steps": 1,
            }
            return parsed_leg[:M]

        fb = self._rule_based(directions, observed_items, M)
        self.implementation_mode = "real_llm_fallback_rule_based"
        self.last_call_audit = {
            **base_audit_template,
            **la2,
            "implementation_mode": "real_llm_fallback_rule_based",
            "hypothesis_generation_mode_resolved": "rule_based_after_contrastive_and_legacy_invalid",
            "contrastive_v2_attempted": True,
            "contrastive_v2_accepted": False,
            "parse_success": bool(la2.get("parse_success")),
            "fallback_reason": "contrastive_v2_and_legacy_json_invalid",
            "schema_fallback_steps": 2,
        }
        return fb

    def _rule_based(
        self,
        directions: list[dict[str, Any]],
        observed_items: list[str],
        M: int,
    ) -> list[dict[str, Any]]:
        meaningful = [d for d in directions if d.get("is_meaningful", True)]
        hypotheses: list[dict[str, Any]] = []
        for i, d in enumerate(meaningful[:M]):
            hid = f"h{i + 1}"
            pos_side = d.get("positive_side", f"direction_{d['direction_id']} positive")
            neg_side = d.get("negative_side", f"direction_{d['direction_id']} negative")
            hypotheses.append(
                {
                    "hypothesis_id": hid,
                    "text_description": f"User may prefer: {pos_side}",
                    "feature_signature": self._extract_keywords(pos_side),
                    "rationale": f"Mock rule-based from direction {d['direction_id']}",
                    "direction_id": d["direction_id"],
                }
            )
            if len(hypotheses) < M:
                hypotheses.append(
                    {
                        "hypothesis_id": f"h{i + 1}b",
                        "text_description": f"User may prefer: {neg_side}",
                        "feature_signature": self._extract_keywords(neg_side),
                        "rationale": f"Mock rule-based from direction {d['direction_id']} negative",
                        "direction_id": d["direction_id"],
                    }
                )
            if len(hypotheses) >= M:
                break
        _ = observed_items
        return hypotheses[:M]

    def _rule_based_contrastive_v2(
        self,
        directions: list[dict[str, Any]],
        observed_items: list[str],
        M: int,
    ) -> list[dict[str, Any]]:
        meaningful = [d for d in directions if d.get("is_meaningful", True)]
        hypotheses: list[dict[str, Any]] = []
        for i, d in enumerate(meaningful[:M]):
            if len(hypotheses) >= M:
                break
            hid = f"opt{i + 1}"
            pos_side = str(d.get("positive_side", f"direction_{d['direction_id']} positive"))
            neg_side = str(d.get("negative_side", f"direction_{d['direction_id']} negative"))
            fk = self._extract_keywords(pos_side)
            nk = self._extract_keywords(neg_side)
            fs = list(dict.fromkeys(fk + nk))[:6] or fk
            hypotheses.append(
                {
                    "hypothesis_id": hid,
                    "option_id": hid,
                    "text_description": f"Interest leaning toward: {_squish(pos_side)[:90]}",
                    "positive_intent": pos_side,
                    "negative_contrast": neg_side,
                    "distinguishing_features": fk[:4] if len(fk) >= 2 else fk + nk[:2],
                    "feature_signature": fs,
                    "positive_retrieval_query": " ".join(fk[:6]) if fk else _squish(pos_side)[:120],
                    "negative_retrieval_query": " ".join(nk[:6]) if nk else _squish(neg_side)[:120],
                    "representative_diversity_hint": "Spread representatives across distinct artists.",
                    "contrast_text": f"{_squish(pos_side)[:80]} vs {_squish(neg_side)[:80]}",
                    "short_user_facing_option_text": f"More interested in preferences similar to: {_squish(pos_side)[:80]}",
                    "rationale": f"mock rule-based contrastive_v2 dir {d['direction_id']}",
                    "direction_id": d["direction_id"],
                }
            )
        _ = observed_items
        return hypotheses[:M]

    def _extract_keywords(self, text: str, max_k: int = 4) -> list[str]:
        import re

        tokens = re.sub(r"[(),:;]", " ", text).split()
        keywords = [t for t in tokens if t.lower() not in self._STOP and len(t) >= 3]
        return keywords[:max_k] if keywords else tokens[:max_k]
