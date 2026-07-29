"""Shared utilities for hybrid surface cache generation and lookup."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping, Sequence

HINT_WORDS = ["correct", "best", "recommended", "right answer", "obviously", "clearly better", "target item"]
INTERNAL = ["theta_star", "option_vector", "belief state", "semantic backend", "embedding", "mu_norm", "entropy="]

QUESTIONS = {
    "lastfm": "Which of the following preference directions is closest to what you would like to explore in music?",
    "movielens": "Which of the following preference directions is closest to what you would like to explore in movies?",
    "book": "Which of the following preference directions is closest to what you would like to explore in books?",
}
DEFAULT_QUESTION = "Which of the following preference directions is closest to what you would like to explore?"


def short_phrase(opt: Mapping[str, Any]) -> str:
    for k in ("cluster_phrase", "feature_signature", "text_description"):
        v = str(opt.get(k) or "").strip()
        if v:
            return v[:120]
    return ""


def examples_str(names: Sequence[str], limit: int = 3) -> str:
    xs = [str(x).strip() for x in names if str(x).strip()][:limit]
    if len(xs) >= 3:
        return f"{xs[0]}, {xs[1]}, and {xs[2]}"
    if len(xs) == 2:
        return f"{xs[0]} and {xs[1]}"
    if len(xs) == 1:
        return xs[0]
    return ""


def render_lightweight_option(opt: Mapping[str, Any]) -> str:
    phrase = short_phrase(opt)
    names = opt.get("representative_item_names") or opt.get("representative_items") or []
    ex = examples_str(list(names))
    if phrase and ex:
        text = f"A preference for {phrase}, for example {ex}."
    elif phrase:
        text = f"A preference for {phrase}."
    elif ex:
        text = f"A preference direction represented by examples such as {ex}."
    else:
        text = f"A preference direction ({opt.get('option_id', 'option')})."
    return text[:240]


def render_lightweight_question(dataset: str) -> str:
    return QUESTIONS.get(str(dataset), DEFAULT_QUESTION)


def dedup_key_from_options(
    dataset: str,
    implementation_mode: str,
    options: Sequence[Mapping[str, Any]],
) -> str:
    parts = [str(dataset or ""), str(implementation_mode or "")]
    for o in options:
        parts.append(str(o.get("option_id") or o.get("hypothesis_id") or ""))
        parts.append(short_phrase(o))
        names = [str(x) for x in (o.get("representative_item_names") or [])]
        parts.append("|".join(names))
        rep_ids = o.get("representative_item_ids") or o.get("representative_items") or []
        if rep_ids and not isinstance(rep_ids[0], str):
            rep_ids = [str(x) for x in rep_ids]
        parts.append("|".join(str(x) for x in rep_ids))
    return hashlib.sha256("||".join(parts).encode()).hexdigest()


def dedup_key_from_writer_row(row: Mapping[str, Any]) -> str:
    inp = row.get("input") or {}
    meta = row.get("metadata") or {}
    impl = str(meta.get("semantic_backend_mode") or meta.get("implementation_mode") or "")
    return dedup_key_from_options(str(row.get("dataset") or ""), impl, inp.get("options") or [])


def validate_surface_record(rec: Mapping[str, Any], order: Sequence[str] | None = None) -> tuple[bool, list[str]]:
    issues: list[str] = []
    q = str(rec.get("question_text") or "").strip()
    if not q:
        issues.append("empty_question")
    opts = rec.get("options") or []
    texts = []
    ids = []
    for o in opts:
        oid = str(o.get("option_id") or "")
        txt = str(o.get("option_text") or "").strip()
        ids.append(oid)
        texts.append(txt)
        if not txt:
            issues.append(f"empty_option:{oid}")
    if order and ids[: len(order)] != list(order):
        issues.append("option_id_order_mismatch")
    blob = (q + " " + " ".join(texts)).lower()
    for w in HINT_WORDS + INTERNAL:
        if w.lower() in blob:
            issues.append(f"bad_wording:{w}")
    if len(set(texts)) < len(texts):
        issues.append("duplicate_options")
    return len(issues) == 0, issues
