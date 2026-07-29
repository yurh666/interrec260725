"""Runtime validation + normalization helpers for frontier_clustered_v32 generation.

Keeps validators side-effect free: never raises into the session simulator — returns dicts."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

INTERNAL_PAT = re.compile(
    r"\bcluster\b|\bfrontier\b|\brank\b|\bboundary\b|top\s*10(?!\w)",
    re.IGNORECASE,
)

_MOJIB = ("\ufffd", "Ã©", "â€™", "â€œ")

LEGACY_KEYS = ("hypothesis_generation_output", "choice_question_output")


def planned_contrast_pairs_from_diag(contrasts_payload: list[dict[str, Any]]) -> set[tuple[int, int]]:
    s: set[tuple[int, int]] = set()
    for x in contrasts_payload or []:
        try:
            a = int(x.get("reference_cluster_id"))
            b = int(x.get("contrasting_cluster_id"))
        except Exception:
            continue
        s.add((a, b))
        s.add((b, a))
    return s


def normalize_llm_parsed_root(root: dict[str, Any] | None) -> dict[str, Any] | None:
    """Map dry-run symmetric keys onto canonical hypotheses + choice_question."""
    if not isinstance(root, dict):
        return None
    hyp_key = LEGACY_KEYS[0] if LEGACY_KEYS[0] in root else "hypotheses"
    ch_key = LEGACY_KEYS[1] if LEGACY_KEYS[1] in root else "choice_question"

    hypo = root.get(hyp_key)
    ch = root.get(ch_key)
    # Handle nested choice dict if present as choice_question_output only
    if isinstance(hypo, str):
        try:
            hypo = json.loads(str(hypo))
        except Exception:
            hypo = None
    if isinstance(ch, str):
        try:
            ch = json.loads(str(ch))
        except Exception:
            ch = None

    cq: dict[str, Any] = {}
    if isinstance(ch, dict):
        cq["question_text"] = ch.get("question_text") or ch.get("final_question_text")
        cq["options"] = ch.get("options") or ch.get("choices") or []

    hyps_norm: list[dict[str, Any]] = []
    if isinstance(hypo, list):
        hyps_norm = [_ensure_hypothesis_defaults(h) for h in hypo if isinstance(h, dict)]

    return {"hypotheses": hyps_norm, "choice_question": cq}


def _ensure_hypothesis_defaults(h: dict[str, Any]) -> dict[str, Any]:
    out = dict(h)
    if "contrast_ref" not in out and out.get("positive_cluster_id") is not None:
        try:
            ccid = out.get("contrast_cluster_id")
            if ccid is None:
                ccid = out.get("contrasting_cluster_id")
            out["contrast_ref"] = {
                "reference_cluster_id": int(out.get("positive_cluster_id")),
                "contrasting_cluster_id": int(ccid),
            }
        except Exception:
            pass
    return out


def apply_compact_option_field_shims(
    hypotheses: list[dict[str, Any]],
    choice_question: dict[str, Any],
    option_to_cluster_mapping: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Copy missing ``mapped_hypothesis_id`` / ``contrast_ref`` onto MCQ options (compact lane only)."""

    warns: list[str] = []
    cq = dict(choice_question or {})
    opts_in = cq.get("options") or []
    if not isinstance(opts_in, list):
        return choice_question, warns

    hyp_by_id: dict[str, dict[str, Any]] = {}
    for h in hypotheses:
        if not isinstance(h, dict):
            continue
        hid = str(h.get("hypothesis_id") or h.get("option_id") or "")
        if hid:
            hyp_by_id[hid] = _ensure_hypothesis_defaults(h)

    omap = option_to_cluster_mapping if isinstance(option_to_cluster_mapping, dict) else {}
    out_opts: list[dict[str, Any]] = []
    for o in opts_in:
        if not isinstance(o, dict):
            continue
        oo = dict(o)
        oid = str(oo.get("option_id") or "")
        if not oo.get("mapped_hypothesis_id"):
            m1 = oo.get("maps_to_hypothesis_id")
            if m1 is not None and str(m1).strip():
                oo["mapped_hypothesis_id"] = str(m1)
        mh = str(oo.get("mapped_hypothesis_id") or "")
        if mh and mh in hyp_by_id and not isinstance(oo.get("contrast_ref"), dict):
            hh = hyp_by_id[mh]
            cr = hh.get("contrast_ref")
            if isinstance(cr, dict):
                oo["contrast_ref"] = dict(cr)
            else:
                warns.append(f"shim_no_contrast_ref_for_hyp:{mh}")
        if not isinstance(oo.get("contrast_ref"), dict) and oid and oid in omap:
            cell = omap[oid]
            if isinstance(cell, dict):
                pc = cell.get("positive_cluster_id")
                cc = cell.get("contrast_cluster_id") if cell.get("contrast_cluster_id") is not None else cell.get(
                    "contrasting_cluster_id",
                )
                try:
                    if pc is not None and cc is not None:
                        oo["contrast_ref"] = {
                            "reference_cluster_id": int(pc),
                            "contrasting_cluster_id": int(cc),
                        }
                except Exception:
                    warns.append(f"shim_omap_contrast_parse_fail:{oid}")
        out_opts.append(oo)

    cq["options"] = out_opts
    return cq, warns


def _squish_lower_bundle(parts: list[str]) -> str:
    return " ".join(" ".join((p or "").split()).lower() for p in parts if p)


def _trigrams(text: str) -> Counter[str]:
    t = "".join(ch if ch.isalnum() else " " for ch in _squish_lower_bundle([text]))
    tok = [x for x in t.split() if x]
    if len(tok) < 3:
        return Counter((" ".join(tok),))
    ctr: Counter[str] = Counter()
    for i in range(len(tok) - 2):
        ctr[" ".join(tok[i : i + 3])] += 1
    return ctr


def _jac(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    num = sum(min(a[k], b[k]) for k in keys)
    den = sum(max(a[k], b[k]) for k in keys) or 1
    return num / den


def validate_generation_bundle(
    hypotheses: list[dict[str, Any]],
    choice_question: dict[str, Any],
    planned_pairs: set[tuple[int, int]],
    *,
    hypothesis_count_band: tuple[int, int] | None = None,
    require_mapped_hypothesis_id: bool = False,
    require_option_contrast_ref: bool = False,
    rolefix_mode: bool = False,
) -> dict[str, Any]:
    """Return validation_result dict consumed by telemetry + aggregates."""

    soft: list[str] = []
    hard: list[str] = []

    band_lo, band_hi = hypothesis_count_band or ((3, 3) if rolefix_mode else (3, 8))
    n_h = len(hypotheses)
    if n_h < band_lo or n_h > band_hi:
        hard.append(f"hypothesis_count_out_of_band:{n_h}")

    qtext = str((choice_question or {}).get("question_text") or "")
    opts_raw = (choice_question or {}).get("options") or []

    substantive: list[dict[str, Any]] = []
    unsure = False
    if isinstance(opts_raw, list):
        for o in opts_raw:
            if not isinstance(o, dict):
                continue
            blob = (
                str(o.get("text", "")) + str(o.get("label", "")) + str(o.get("option_id", ""))
            ).lower()
            if "not sure" in blob or "no strong preference" in blob:
                unsure = True
                continue
            substantive.append(dict(o))

    n_sub = len(substantive)

    if n_sub < 3:
        hard.append(f"too_few_substantive_options:{n_sub}")
    if n_sub > 5:
        hard.append(f"too_many_substantive_options:{n_sub}")

    if not qtext.strip():
        hard.append("empty_question_text")

    bundle = qtext + " " + " ".join(str(o.get("text") or "") for o in substantive)
    jargon = INTERNAL_PAT.findall(bundle)
    if jargon:
        hard.append(f"internal_jargon_USER_VISIBLE:{','.join(sorted(set(jargon)))[:240]}")

    if any(m in bundle for m in _MOJIB):
        hard.append("possible_mojibake_bundle")

    def _normalize_cr(cr: Any) -> tuple[int, int] | None:
        if not isinstance(cr, dict):
            return None
        try:
            return int(cr["reference_cluster_id"]), int(cr["contrasting_cluster_id"])
        except Exception:
            return None

    hypo_ids: set[str] = set()
    for i, h in enumerate(hypotheses):
        if not isinstance(h, dict):
            hard.append(f"hypothesis_{i}_not_dict")
            continue
        hh = _ensure_hypothesis_defaults(h)
        hid = str(hh.get("hypothesis_id") or hh.get("option_id") or "")
        if hid:
            hypo_ids.add(hid)
        if not str(hh.get("intent_text") or hh.get("positive_intent") or "").strip():
            hard.append(f"hypothesis_{i}_empty_primary_text")
        cr = _normalize_cr(hh.get("contrast_ref"))
        if cr is None:
            hard.append(f"hypothesis_{i}_bad_contrast_ref")
        elif planned_pairs and cr not in planned_pairs:
            msg = f"hypothesis_{i}_contrast_out_of_plan:{cr}"
            if rolefix_mode:
                soft.append(msg)
            else:
                hard.append(msg)

    if rolefix_mode:
        roles = [str(h.get("rank_safety_role") or "") for h in hypotheses if isinstance(h, dict)]
        rc = Counter(r for r in roles if r)
        if rc.get("current_top_preserve", 0) != 1:
            hard.append(f"role_count_current_top_preserve={rc.get('current_top_preserve', 0)}")
        if rc.get("boundary_challenge", 0) < 1:
            hard.append("role_missing_boundary_challenge")
        if rc.get("explore_uncertain", 0) < 1:
            hard.append("role_missing_explore_uncertain")

    if substantive and not unsure:
        soft.append("optional_unsure_absent_observed_but_not_HARD")

    pairwise_max = 0.0
    if len(substantive) >= 2:
        grams = [_trigrams(str(o.get("text") or "")) for o in substantive]
        for i in range(len(grams)):
            for j in range(i + 1, len(grams)):
                pairwise_max = max(pairwise_max, _jac(grams[i], grams[j]))
        if pairwise_max >= 0.82:
            soft.append(f"high_option_overlap_trigram:{pairwise_max:.3f}")

    mapped_issues = False
    for k, opt in enumerate(substantive):
        mid = opt.get("mapped_hypothesis_id")
        if mid is None or str(mid).strip() == "":
            if require_mapped_hypothesis_id:
                hard.append(f"option_{k}_missing_mapped_hypothesis_id")
            mapped_issues = True
        elif str(mid) not in hypo_ids:
            hard.append(f"option_{k}_unknown_mapped_hypothesis:{mid}")

        crt = _normalize_cr(opt.get("contrast_ref"))
        if crt is None:
            if require_option_contrast_ref:
                hard.append(f"option_{k}_missing_contrast_ref_object")
            else:
                pass
        elif planned_pairs and crt not in planned_pairs:
            msg = f"option_{k}_contrast_out_of_plan:{crt}"
            if rolefix_mode:
                soft.append(msg)
            else:
                hard.append(msg)

    if mapped_issues:
        soft.append("mapped_hypothesis_id_contract_gap_maybe_derived_upstream")

    hard_pass = len(hard) == 0
    return {
        "parse_success": hard_pass,
        "schema_hard_pass": hard_pass,
        "schema_soft_warnings": soft,
        "fallback_required": not hard_pass,
        "fallback_reason": None if hard_pass else "; ".join(sorted(set(hard))[:12]),
        "hard_errors_detail": hard,
        "metrics": {
            "hypothesis_count": n_h,
            "substantive_option_count": n_sub,
            "pairwise_trigram_jaccard_max": round(pairwise_max, 6),
            "planned_pairs_undirected_unique": _planned_pairs_undirected_unique(planned_pairs),
        },
    }


def _planned_pairs_undirected_unique(pairs: set[tuple[int, int]]) -> int:
    if not pairs:
        return 0
    return len({tuple(sorted((int(a), int(b)))) for a, b in pairs})


def infer_option_mappings_if_missing(
    hypotheses: list[dict[str, Any]],
    options: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Best-effort: attach mapped_hypothesis_id + contrast_ref to options by positional order."""

    warns: list[str] = []
    if len(options) != len(hypotheses):
        warns.append(f"positional_bind_len_mismatch_hyp={len(hypotheses)}_opt={len(options)}")
        return options, warns
    filled: list[dict[str, Any]] = []
    for opt, hyp in zip(options, hypotheses):
        o = dict(opt)
        if not o.get("mapped_hypothesis_id"):
            hid = str(hyp.get("hypothesis_id") or hyp.get("option_id") or "")
            if hid:
                o["mapped_hypothesis_id"] = hid
        if hyp.get("contrast_ref"):
            if not o.get("contrast_ref"):
                o["contrast_ref"] = hyp["contrast_ref"]
        filled.append(o)
    return filled, warns
