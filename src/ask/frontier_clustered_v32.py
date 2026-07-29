"""Frontier-clustered What-to-Ask (experimental v3.2).

Pure offline helpers: frontier pool → clusters → contrasts → **mock**
hypothesis + MCQ artefacts. Never reads latent ``future_*`` targets.
"""

from __future__ import annotations

import json
import math
import re
import os
from collections import Counter
from dataclasses import dataclass, field
from textwrap import dedent
from typing import Any

import numpy as np

from src.belief.belief_state import BeliefState
from src.embedding.index import EmbeddingIndex
from src.recommendation.scorer import belief_scores

try:
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover
    KMeans = None


_JARGON_PATTERNS_UI = (
    r"\bfrontier\b",
    r"\bcluster\b",
    r"\bclusters\b",
    r"\bboundary\b",
    r"\bboundaries\b",
    r"\btop\s*10\b",
    r"\btopten\b",
    r"\branking\b",
    r"\brank\b",
    r"\branks\b",
    r"\bposterior\b",
    r"\bembedding\b",
)


def _squish(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip())


def sanitize_user_question_copy(text: str) -> str:
    """Rewrite obvious internal wording in MCQ preamble (mock-only sanitiser)."""

    t = _squish(str(text or ""))
    if not t:
        return t
    swaps = (
        ("frontier", "unfamiliar picks"),
        ("cluster", "grouping"),
        ("clusters", "groupings"),
        ("boundary", "split"),
        ("boundaries", "splits"),
        ("rank", "ordering"),
        ("ranking", "ordering"),
    )
    low = t.lower()
    for a, b in swaps:
        if a in low:
            t = re.sub(rf"\b{re.escape(a)}\b", b, t, flags=re.I)
            low = t.lower()
    return _squish(t)


_MOJIBAKE_MARKERS = ("\ufffd", "�")


def sanitize_item_name_for_prompt(raw: str) -> tuple[str | None, str]:
    """Strip / repair Mojibake for user-facing prompts; drops unusable cues.

    Raw hypothesis JSON may still keep the untouched string; callers track dropped counts.
    """

    s = _squish(str(raw or ""))
    if not s:
        return None, "blank"

    if any(m in s for m in _MOJIBAKE_MARKERS):
        return None, "dropped_bad_encoding"

    suspicious_utf8_mix = ("Ð" in s or "Ã" in s)
    if suspicious_utf8_mix:
        repaired: str | None = None
        for enc in ("latin1", "cp1252"):
            try:
                t = _squish(s.encode(enc).decode("utf-8"))
                if len(t) < 2:
                    continue
                if "Ð" in t or "Ã" in t:
                    continue
                repaired = t
                break
            except (UnicodeDecodeError, UnicodeEncodeError):
                continue
        if repaired:
            s = repaired
            tag = "repaired_utf8_via_latin1"
        else:
            return None, "dropped_mojibake"
    else:
        tag = "ok"

    if "Ð" in s or "Ã" in s:
        return None, "dropped_mojibake"

    s_out = sanitize_user_question_copy(_squish(str(s)))[:74]
    if not _squish(s_out):
        return None, "blank_after_strip"
    return s_out, tag


def _style_output_banned(low: str) -> bool:
    if re.search(
        r"\b(accent|boundary|boundaries|frontier|cluster|clusters|rank|ranks|ranking|"
        r"embedding|posterior)\b|\btop\s*10\b|\btopten\b",
        low,
        flags=re.I,
    ):
        return True
    return "exploratory unfamiliar picks" in low


def make_cluster_style_phrase(cluster: dict[str, Any]) -> str:
    """Concise preference-direction labels for MCQ wording (no internal template lemmas).

    Builds from semantic_summary cues + familiarity / energy summaries. Every phrase avoids
    ``accent/frontier/cluster/...`` scaffolding that collided across options in legacy mocks.
    """

    try:
        cid = int(cluster.get("cluster_id"))
    except Exception:
        cid = 0

    def _nf(key: str, default: float = 0.0) -> float:
        try:
            v = float(cluster.get(key) or default)
            return v if math.isfinite(v) else default
        except (TypeError, ValueError):
            return default

    o10 = _nf("overlap_top10_frac")
    hi = _nf("history_similarity")
    ur = _nf("mean_uncertainty")
    mr = _nf("mean_rank")
    ss = _squish(str(cluster.get("semantic_summary") or "")).lower()

    safe_pool = [
        "familiar guitar-driven indie direction",
        "more melodic alternative-rock direction",
        "rougher garage-rock leaning",
        "softer singer-songwriter leaning",
        "darker post-punk leaning",
        "brighter power-pop leaning",
        "more exploratory niche direction",
        "safer recent-session neighbor",
        "mellow synth-leaning indie direction",
        "raw guitar-forward rock lean",
        "polished radio-friendly alt lean",
        "indie-leaning mellow direction",
        "bright uptempo alt-pop lean",
    ]

    def _pick_fallback() -> str:
        return safe_pool[cid % len(safe_pool)]

    genre_hit: str | None = None
    genre_rules: tuple[tuple[tuple[str, ...], str], ...] = (
        (("singer songwriter", "singer-songwriter"), "softer singer-songwriter leaning"),
        (("folk", "acoustic"), "soft acoustic folk lean"),
        (("britpop",), "bright Britpop-leaning direction"),
        (("post punk", "post-punk", "postpunk"), "darker post-punk leaning"),
        (("garage",), "rougher garage-rock leaning"),
        (("shoegaze", "dream pop", "dream-pop"), "dreamy shoegaze-leaning direction"),
        (("hip hop", "hip-hop", "rap"), "groove-forward hip-hop lean"),
        (("edm", "techno", "house"), "electronic dance-energy lean"),
        (("country",), "roots-country melodic lean"),
        (("metal",), "heavy guitar-driven rock lean"),
        (("indie rock", "indie-rock"), "guitar-driven indie-rock lean"),
        (("alternative rock", "alt rock", "alt-rock"), "melodic alternative-rock direction"),
        (("alternative", "alt."), "alternative rock-leaning direction"),
        (("punk",), "raw punk-leaning direction"),
    )
    for keys, lbl in genre_rules:
        if any(k in ss for k in keys):
            genre_hit = lbl
            break

    out: str
    if genre_hit:
        out = _squish(genre_hit)
    elif hi >= 0.5:
        out = _squish(f"adjacent to recent listening — {_pick_fallback()}")
    elif o10 >= 0.38:
        out = _squish(f"familiar radio-leaning — {_pick_fallback()}")
    elif ur >= 0.5 or (0 < o10 <= 0.18):
        out = _squish(f"more exploratory niche — {_pick_fallback()}")
    elif hi <= 0.26 and hi == hi:
        out = _squish(f"further from recent-session tie — {_pick_fallback()}")
    elif mr < 55 and mr == mr:
        out = _squish(f"tighter discovery band — {_pick_fallback()}")
    elif mr >= 115 and mr == mr:
        out = _squish(f"deeper-catalog stretch — {_pick_fallback()}")
    else:
        out = _pick_fallback()

    out = sanitize_user_question_copy(_squish(out[:200]))
    low = out.lower()
    if _style_output_banned(low) or len(_squish(out)) < 6:
        out = _pick_fallback()
    out = sanitize_user_question_copy(_squish(out[:160]))
    if _style_output_banned(out.lower()):
        out = safe_pool[(cid + 9) % len(safe_pool)]
    return _squish(out[:150])


def _clip_to_max_words(txt: str, max_words: int) -> str:
    toks = txt.split()
    if len(toks) <= max_words:
        return _squish(txt)
    clipped = _squish(" ".join(toks[:max_words]))
    return clipped + (" …" if clipped else clipped)


_SHORT_OPTION_BANNED = (
    "style read",
    "explicit steer-away",
    "representative cues",
    "same preferred mood leaning",
    "triple shown",
    "tilt discoveries",
    "tilt discovery",
)


def build_mcq_option_text_short(h: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Compact human-readable option (35–45 word budget, short template)."""

    pos_style_raw = _squish(str(h.get("positive_style_phrase") or "warm familiar-session hues"))
    neg_style_raw = _squish(str(h.get("contrast_style_phrase") or "more adventurous unfamiliar hues"))

    moj_counts = {"sanitized_drop": 0, "repair": 0}
    reps_pos = list(h.get("representative_items") or [])[:4]
    reps_neg = list(h.get("contrast_representative_items") or [])[:4]

    def _fmt_three(names: list[Any]) -> str:
        chunks: list[str] = []
        for nm in names:
            disp, lab = sanitize_item_name_for_prompt(str(nm))
            if lab.startswith("repaired"):
                moj_counts["repair"] += 1
            if lab.startswith("dropped") or lab == "blank_after_strip" or lab == "blank":
                moj_counts["sanitized_drop"] += 1
                continue
            if disp:
                chunks.append(str(disp))
            if len(chunks) >= 3:
                break
        if not chunks:
            return "moods implied by illustrative artists"
        if len(chunks) == 1:
            return f"artists like {chunks[0]}"
        if len(chunks) == 2:
            return f"artists like {chunks[0]} and {chunks[1]}"
        return f"artists such as {chunks[0]}, {chunks[1]}, and {chunks[2]}"

    pos_art = _fmt_three(reps_pos)
    neg_art = _fmt_three(reps_neg)

    pos_style = _clip_to_max_words(pos_style_raw, max_words=12)
    neg_style = _clip_to_max_words(neg_style_raw, max_words=12)

    def _a_article(adj_phrase: str) -> str:
        trimmed = adj_phrase.lstrip().lower()
        if not trimmed:
            return "a"
        return "an" if trimmed[:1] in "aeiou" else "a"

    body = sanitize_user_question_copy(
        _squish(
            f"Prefer a more {pos_style} direction, e.g. {pos_art}, "
            f"rather than {_a_article(neg_style)} {neg_style} direction like {neg_art}.",
        ),
    )
    body_lc = body.lower()

    bw = len(body.split())
    if bw > 45:
        body = sanitize_user_question_copy(_clip_to_max_words(body, 44))
        body_lc = body.lower()
        bw = len(body.split())

    for ban in _SHORT_OPTION_BANNED:
        if ban in body_lc:
            body = sanitize_user_question_copy(
                body.replace(ban.title(), "").replace(ban, "").strip(),
            )
            body_lc = body.lower()

    meta = {
        "word_est": bw,
        "mojibake_events": int(moj_counts["sanitized_drop"] + moj_counts["repair"]),
        "dropped_artists_ui": int(moj_counts["sanitized_drop"]),
        "repaired_names": int(moj_counts["repair"]),
    }
    return body, meta


def build_mcq_option_text_v4(h: dict[str, Any], *, variant: int) -> tuple[str, dict[str, Any]]:
    """Preference-shaped MCQ option with rotated openings (v4 semantic labels)."""

    pos_style_raw = _squish(str(h.get("positive_style_phrase") or "melodic alternative-rock direction"))
    neg_style_raw = _squish(str(h.get("contrast_style_phrase") or "softer singer-songwriter leaning"))

    moj_counts = {"sanitized_drop": 0, "repair": 0}
    reps_pos = list(h.get("representative_items") or [])[:4]
    reps_neg = list(h.get("contrast_representative_items") or [])[:4]

    def _fmt_three(names: list[Any]) -> str:
        chunks: list[str] = []
        for nm in names:
            disp, lab = sanitize_item_name_for_prompt(str(nm))
            if lab.startswith("repaired"):
                moj_counts["repair"] += 1
            if lab.startswith("dropped") or lab == "blank_after_strip" or lab == "blank":
                moj_counts["sanitized_drop"] += 1
                continue
            if disp:
                chunks.append(str(disp))
            if len(chunks) >= 3:
                break
        if not chunks:
            return "illustrative artists for each side of the split"
        if len(chunks) == 1:
            return f"artists like {chunks[0]}"
        if len(chunks) == 2:
            return f"{chunks[0]} and {chunks[1]}"
        return f"{chunks[0]}, {chunks[1]}, and {chunks[2]}"

    pos_art = _fmt_three(reps_pos)
    neg_art = _fmt_three(reps_neg)

    pos_core = _clip_to_max_words(pos_style_raw, max_words=14)
    neg_core = _clip_to_max_words(neg_style_raw, max_words=14)

    v = int(variant) % 4
    if v == 0:
        stem = f"I want {pos_core}, like {pos_art}, more than {neg_core}, like {neg_art}."
    elif v == 1:
        stem = (
            f"I would rather explore {pos_core}, like {pos_art}, "
            f"than stay with {neg_core}, like {neg_art}."
        )
    elif v == 2:
        stem = (
            f"Keep recommendations closer to {pos_core}, like {pos_art}, "
            f"instead of {neg_core}, like {neg_art}."
        )
    else:
        stem = (
            f"I prefer leaning toward {pos_core}, with artists like {pos_art}, "
            f"over {neg_core}, like {neg_art}."
        )

    body = sanitize_user_question_copy(_squish(stem))
    body_lc = body.lower()

    bw = len(body.split())
    if bw > 45:
        body = sanitize_user_question_copy(_clip_to_max_words(body, 45))
        body_lc = body.lower()
        bw = len(body.split())

    for ban in _SHORT_OPTION_BANNED:
        if ban in body_lc:
            body = sanitize_user_question_copy(
                body.replace(ban.title(), "").replace(ban, "").strip(),
            )
            body_lc = body.lower()

    if _style_output_banned(body_lc):
        body = sanitize_user_question_copy(
            _squish(
                f"Lean toward {pos_core}, illustrated by {pos_art}, "
                f"rather than {neg_core}, illustrated by {neg_art}.",
            ),
        )
        bw = len(body.split())
        body_lc = body.lower()

    meta = {
        "word_est": bw,
        "mojibake_events": int(moj_counts["sanitized_drop"] + moj_counts["repair"]),
        "dropped_artists_ui": int(moj_counts["sanitized_drop"]),
        "repaired_names": int(moj_counts["repair"]),
    }
    return body, meta


def _pick_hypotheses_style_disjoint(
    hypotheses: list[dict[str, Any]],
    *,
    target_max: int,
    rep_thresholds: tuple[float, ...],
    pos_conflict: float = 0.45,
    paired_pos_floor: float = 0.35,
    paired_neg_floor: float = 0.45,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pool = _uniq_pair_pool_ordered(hypotheses)
    if not pool:
        return [], {
            "semantic_filter_dropped_count": 0,
            "semantic_filter_drop_rep": 0,
            "semantic_filter_drop_style": 0,
            "semantic_filter_relaxed": False,
        }

    def _styles(h: dict[str, Any]) -> tuple[str, str]:
        return (
            _squish(str(h.get("positive_style_phrase") or "")),
            _squish(str(h.get("contrast_style_phrase") or "")),
        )

    def _style_clash(h: dict[str, Any], chosen: list[dict[str, Any]]) -> bool:
        ph, nh = _styles(h)
        for c in chosen:
            pc, nc = _styles(c)
            jp = _token_jaccard(ph, pc)
            jn = _token_jaccard(nh, nc)
            if jp > pos_conflict:
                return True
            if jp > paired_pos_floor and jn > paired_neg_floor:
                return True
        return False

    def _attempt(use_style_filter: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        drop_rep = drop_style = 0
        for th in rep_thresholds:
            chosen: list[dict[str, Any]] = []
            for h in pool:
                if len(chosen) >= target_max:
                    break
                if len(chosen) >= 1 and _max_positive_rep_overlap(h, chosen) > th:
                    drop_rep += 1
                    continue
                if use_style_filter and chosen and _style_clash(h, chosen):
                    drop_style += 1
                    continue
                chosen.append(h)
            if len(chosen) >= 2:
                meta = {
                    "semantic_filter_dropped_count": int(drop_rep + drop_style),
                    "semantic_filter_drop_rep": int(drop_rep),
                    "semantic_filter_drop_style": int(drop_style),
                    "semantic_filter_relaxed": bool(not use_style_filter),
                    "style_pick_rep_threshold_used": float(th),
                }
                return chosen, meta
        chosen = pool[: max(1, min(target_max, len(pool)))]
        meta = {
            "semantic_filter_dropped_count": int(drop_rep + drop_style),
            "semantic_filter_drop_rep": int(drop_rep),
            "semantic_filter_drop_style": int(drop_style),
            "semantic_filter_relaxed": bool(not use_style_filter),
            "style_pick_rep_threshold_used": float("nan"),
        }
        return chosen, meta

    chosen1, meta1 = _attempt(True)
    score1 = len(chosen1) * 1000 - int(meta1.get("semantic_filter_dropped_count", 0))

    chosen2, meta2 = _attempt(False)
    score2 = len(chosen2) * 1000 - int(meta2.get("semantic_filter_dropped_count", 0))

    if len(chosen1) >= 2 and (len(chosen2) < 2 or score1 >= score2):
        return chosen1, meta1
    return chosen2, meta2


def _hypothesis_positive_rep_set(hypothesis: dict[str, Any]) -> set[str]:
    return {_squish(str(x))[:120] for x in list(hypothesis.get("representative_items") or [])[:3] if _squish(str(x))}


def _max_positive_rep_overlap(candidate: dict[str, Any], chosen: list[dict[str, Any]]) -> float:
    s_new = _hypothesis_positive_rep_set(candidate)
    if not s_new:
        return 0.0
    worst = 0.0
    for ch in chosen:
        s_old = _hypothesis_positive_rep_set(ch)
        if not s_old:
            continue
        worst = max(worst, len(s_new & s_old) / max(1, len(s_new | s_old)))
    return worst


def _uniq_pair_pool_ordered(hypotheses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pair_seen: set[tuple[int, int]] = set()
    pool: list[dict[str, Any]] = []
    for h in hypotheses:
        try:
            pid = int(h["positive_cluster_id"])
            cid = int(h["contrast_cluster_id"])
        except Exception:
            continue
        key = (pid, cid)
        if key in pair_seen:
            continue
        pair_seen.add(key)
        pool.append(h)

    used_pos_cover: set[int] = set()
    prioritized: list[dict[str, Any]] = []
    for hh in pool:
        pid = int(hh["positive_cluster_id"])
        if pid not in used_pos_cover:
            prioritized.append(hh)
            used_pos_cover.add(pid)
    for hh in pool:
        if hh not in prioritized:
            prioritized.append(hh)
    return prioritized


def _pick_hypotheses_disjoint_positive_reps(
    hypotheses: list[dict[str, Any]],
    *,
    target_max: int,
    target_min: int,
    thresholds: tuple[float, ...] = (0.32, 0.44, 0.55),
) -> list[dict[str, Any]]:
    pool = _uniq_pair_pool_ordered(hypotheses)
    if not pool:
        return []

    for th in thresholds:
        chosen: list[dict[str, Any]] = []
        for h in pool:
            if len(chosen) >= target_max:
                break
            if len(chosen) >= 1 and _max_positive_rep_overlap(h, chosen) > th:
                continue
            chosen.append(h)
        if len(chosen) >= target_min:
            return chosen

    return pool[: max(target_min, min(target_max, len(pool)))]


def _token_jaccard(a: str, b: str) -> float:
    ta = {w for w in re.split(r"\W+", a.lower()) if len(w) > 2}
    tb = {w for w in re.split(r"\W+", b.lower()) if len(w) > 2}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _hypothesis_negative_rep_set(hypothesis: dict[str, Any]) -> set[str]:
    return {
        _squish(str(x))[:120] for x in list(hypothesis.get("contrast_representative_items") or [])[:3] if _squish(str(x))
    }


def _max_contrast_rep_overlap(candidate: dict[str, Any], chosen: list[dict[str, Any]]) -> float:
    s_new = _hypothesis_negative_rep_set(candidate)
    if not s_new:
        return 0.0
    worst = 0.0
    for ch in chosen:
        s_old = _hypothesis_negative_rep_set(ch)
        if not s_old:
            continue
        worst = max(worst, len(s_new & s_old) / max(1, len(s_new | s_old)))
    return worst


def _cluster_centroid_unit(clusters_map: dict[int, dict[str, Any]], cluster_id: int) -> np.ndarray | None:
    row = clusters_map.get(int(cluster_id)) or {}
    v = row.get("mean_embedding_direction")
    if not isinstance(v, (list, tuple)) or len(v) < 4:
        return None
    a = np.array([float(x) for x in v], dtype=float)
    n = float(np.linalg.norm(a))
    if not math.isfinite(n) or n < 1e-9:
        return None
    return a / n


def style_phrase_support_score_mock(cluster_semantic_summary: str, phrase: str) -> float:
    """Offline rule overlap between phrase lexical cues vs cluster semantic summary."""

    ss = _squish(str(cluster_semantic_summary or "")).lower()
    ps = _squish(str(phrase or "")).lower()
    if not ps:
        return 0.35
    phrase_toks = {w for w in re.split(r"\W+", ps) if len(w) > 3}
    stop = {
        "adjacent",
        "recent",
        "listening",
        "more",
        "less",
        "exploratory",
        "niche",
        "familiar",
        "direction",
        "leaning",
        "session",
        "neighbor",
        "radio",
        "catalog",
        "stretch",
        "tighter",
        "deeper",
    }
    phrase_toks = {t for t in phrase_toks if t not in stop}
    if not phrase_toks:
        return 0.72
    summary_toks = {w for w in re.split(r"\W+", ss) if len(w) > 2}
    frac = len(phrase_toks & summary_toks) / max(1, len(phrase_toks))
    lexical_hits = 0
    if "garage" in ps and any(k in ss for k in ("garage", "punk", "raw", "grunge", "noise")):
        lexical_hits += 1
    if "punk" in ps and "punk" in ss:
        lexical_hits += 1
    if "songwriter" in ps or "acoustic" in ps:
        if any(k in ss for k in ("acoustic", "folk", "singer", "soft")):
            lexical_hits += 1
        else:
            lexical_hits -= 1
    if "melodic" in ps and any(k in ss for k in ("pop", "melodic", "alt")):
        lexical_hits += 1
    if "alternative" in ps:
        if "rock" in ss or "indie" in ss or "alt" in ss:
            lexical_hits += 1
    base = 0.42 + 0.48 * min(1.0, frac + 0.15 * float(lexical_hits))
    return float(max(0.0, min(1.0, base)))


_CONSERVATIVE_STYLE_FALLBACK_PHRASE = "broader indie-leaning direction"


def _cluster_semantic_summary_genre_lexical_thin(summary: str) -> bool:
    """True when semantic_summary looks like representative-name blob (no audible genre/lemmas).

    Offline rule-only: avoids mass-replacement of style phrases whenever cluster text has no lexical hooks.
    """

    ss = _squish(str(summary or "")).lower()
    if not ss:
        return True
    if "lexical_cues:" in ss or ("axes:" in ss and len(ss) >= 36):
        return False
    hints = (
        "indie",
        "rock",
        "punk",
        "folk",
        "pop",
        "alt",
        "acoustic",
        "metal",
        "jazz",
        "garage",
        "melodic",
        "songwriter",
        "post-punk",
        "alternative",
        "electron",
        "dance",
        "hip hop",
        "rap",
        "country",
        "soul",
        "blues",
    )
    return not any(tok in ss for tok in hints)


def _hypothesis_has_conservative_style_fallback_phrasing(h: dict[str, Any]) -> bool:
    ph, nh = _v5_styles(h)
    core = _squish(_CONSERVATIVE_STYLE_FALLBACK_PHRASE).lower()
    if not ph.lower().startswith(core) or not nh.lower().startswith(core):
        return False
    return "positive cue" in ph.lower() and "contrast cue" in nh.lower()


def apply_hypothesis_style_phrase_support_mock(
    hypotheses: list[dict[str, Any]],
    *,
    clusters_by_id: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    low_conf = insufficient = fallback = 0
    adjusted: list[dict[str, Any]] = []
    for h in hypotheses:
        try:
            pid = int(h.get("positive_cluster_id"))
            nid = int(h.get("contrast_cluster_id"))
        except Exception:
            adjusted.append(dict(h))
            continue
        ca = clusters_by_id.get(pid) or {}
        cb = clusters_by_id.get(nid) or {}
        pos_phr = _squish(str(h.get("positive_style_phrase") or ""))
        neg_phr = _squish(str(h.get("contrast_style_phrase") or ""))
        ss_pos = _squish(
            str(ca.get("cluster_semantic_summary") or ca.get("semantic_summary") or ""),
        )
        ss_neg = _squish(
            str(cb.get("cluster_semantic_summary") or cb.get("semantic_summary") or ""),
        )
        sp = style_phrase_support_score_mock(ss_pos, pos_phr)
        sn = style_phrase_support_score_mock(ss_neg, neg_phr)
        pos_thin = _cluster_semantic_summary_genre_lexical_thin(ss_pos)
        neg_thin = _cluster_semantic_summary_genre_lexical_thin(ss_neg)
        pos_replace_th = 0.40 if pos_thin else 0.46
        neg_replace_th = 0.40 if neg_thin else 0.46
        hh = dict(h)
        if sp < pos_replace_th:
            insufficient += 1
            hh["positive_style_phrase"] = _CONSERVATIVE_STYLE_FALLBACK_PHRASE + " — positive cue"
            low_conf += 1
            fallback += 1
        if sn < neg_replace_th:
            insufficient += 1
            hh["contrast_style_phrase"] = _CONSERVATIVE_STYLE_FALLBACK_PHRASE + " — contrast cue"
            low_conf += 1
            fallback += 1
        hh["style_phrase_support_score_positive"] = float(sp)
        hh["style_phrase_support_score_contrast"] = float(sn)
        adjusted.append(hh)

    avg_pos_sp = (
        float(np.mean([float(x.get("style_phrase_support_score_positive") or 0.0) for x in adjusted]))
        if adjusted
        else float("nan")
    )
    avg_neg_sp = (
        float(np.mean([float(x.get("style_phrase_support_score_contrast") or 0.0) for x in adjusted]))
        if adjusted
        else float("nan")
    )

    meta = {
        "style_phrase_low_confidence_count": int(low_conf),
        "metadata_insufficient_count": int(insufficient),
        "conservative_phrase_fallback_count": int(fallback),
        "hypothesis_avg_style_support_positive": avg_pos_sp,
        "hypothesis_avg_style_support_contrast": avg_neg_sp,
        "hypothesis_avg_style_support_global": (
            float((avg_pos_sp + avg_neg_sp) / 2.0) if math.isfinite(avg_pos_sp) and math.isfinite(avg_neg_sp) else float("nan")
        ),
    }
    return adjusted, meta


def _v5_styles(h: dict[str, Any]) -> tuple[str, str]:
    return (
        _squish(str(h.get("positive_style_phrase") or "")),
        _squish(str(h.get("contrast_style_phrase") or "")),
    )


def _v5_hypothesis_same_strand_pair(a: dict[str, Any], b: dict[str, Any]) -> bool:
    try:
        return (
            int(a["positive_cluster_id"]),
            int(a["contrast_cluster_id"]),
        ) == (
            int(b["positive_cluster_id"]),
            int(b["contrast_cluster_id"]),
        )
    except Exception:
        return False


def _v5_style_clash(h: dict[str, Any], chosen: list[dict[str, Any]]) -> bool:
    ph, nh = _v5_styles(h)
    paired_pos_floor = 0.36
    paired_neg_floor = 0.44
    pos_conflict = 0.44
    h_fallback = _hypothesis_has_conservative_style_fallback_phrasing(h)

    def _paired_clash_regular(pc: str, nc: str) -> bool:
        jp = _token_jaccard(ph, pc)
        jn = _token_jaccard(nh, nc)
        if jp > pos_conflict:
            return True
        return jp > paired_pos_floor and jn > paired_neg_floor

    for c in chosen:
        pc, nc = _v5_styles(c)
        c_fallback = _hypothesis_has_conservative_style_fallback_phrasing(c)
        # Mass conservative fallback repeats the same scaffold; discriminate by strand + contrast artists instead.
        if h_fallback and c_fallback:
            if _v5_hypothesis_same_strand_pair(h, c):
                return True
            continue
        if h_fallback ^ c_fallback:
            jp = _token_jaccard(ph, pc)
            jn = _token_jaccard(nh, nc)
            if jp > 0.55 and jn > 0.55:
                return True
            continue
        if _paired_clash_regular(pc, nc):
            return True
    return False


def _cluster_contrast_diversity_score(
    chosen: list[dict[str, Any]],
    clusters_map: dict[int, dict[str, Any]],
) -> float:
    if len(chosen) < 2:
        return 1.0
    vecs: list[np.ndarray] = []
    for h in chosen:
        try:
            cid = int(h["contrast_cluster_id"])
        except Exception:
            continue
        v = _cluster_centroid_unit(clusters_map, cid)
        if v is not None:
            vecs.append(v)
    if len(vecs) < 2:
        return 0.5
    acc: list[float] = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            acc.append(1.0 - float(np.clip(float(np.dot(vecs[i], vecs[j])), -1.0, 1.0)))
    return float(np.mean(acc)) if acc else 0.5


def _contrast_cluster_reuse_count(chosen: list[dict[str, Any]]) -> int:
    ctr: Counter[int] = Counter()
    for h in chosen:
        try:
            ctr[int(h["contrast_cluster_id"])] += 1
        except Exception:
            continue
    extra = sum(max(0, c - 1) for c in ctr.values())
    return int(extra)


def _positive_style_pairwise_max_jaccard(chosen: list[dict[str, Any]]) -> float:
    phs = [_squish(str(h.get("positive_style_phrase") or "")) for h in chosen if h]
    phs = [p for p in phs if p]
    mxp = 0.0
    for ii in range(len(phs)):
        for jj in range(ii + 1, len(phs)):
            mxp = max(mxp, _token_jaccard(phs[ii], phs[jj]))
    return float(mxp)


def _contrast_style_pairwise_max_jaccard(chosen: list[dict[str, Any]]) -> float:
    phs = [_squish(str(h.get("contrast_style_phrase") or "")) for h in chosen if h]
    phs = [p for p in phs if p]
    mxp = 0.0
    for ii in range(len(phs)):
        for jj in range(ii + 1, len(phs)):
            mxp = max(mxp, _token_jaccard(phs[ii], phs[jj]))
    return float(mxp)


def _pick_hypotheses_v5_cluster_contrast(
    hypotheses: list[dict[str, Any]],
    *,
    clusters_map: dict[int, dict[str, Any]],
    target_max: int,
    reuse_contrast_phrase_jaccard_tol: float = 0.45,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pool = _uniq_pair_pool_ordered(hypotheses)
    if not pool:
        zeros = {
            "contrast_filter_dropped_count": 0,
            "cluster_contrast_diversity_score": 0.0,
            "contrast_cluster_reuse_count": 0,
            "few_options_after_contrast_filter": True,
            "contrast_cluster_reuse_penalty": 1.0,
            "contrast_style_phrase_overlap_penalty": 1.0,
            "contrast_artist_overlap_penalty": 1.0,
            "positive_style_pairwise_max_jaccard": 0.0,
            "contrast_style_pairwise_max_jaccard": 0.0,
            "pairwise_contrast_artist_overlap_max": 0.0,
        }
        return [], zeros

    def _spread(h: dict[str, Any]) -> float:
        try:
            pid = int(h["positive_cluster_id"])
            nid = int(h["contrast_cluster_id"])
        except Exception:
            return -1.0
        vp = _cluster_centroid_unit(clusters_map, pid)
        vn = _cluster_centroid_unit(clusters_map, nid)
        if vp is None or vn is None:
            return -0.25
        return float(1.0 - float(np.clip(float(np.dot(vp, vn)), -1.0, 1.0)))

    spreads = {_squish(str(h.get("hypothesis_id"))): _spread(h) for h in pool}
    pool_sorted = sorted(
        pool,
        key=lambda hh: (
            -(spreads.get(_squish(str(hh.get("hypothesis_id"))), -999.0)),
            str(hh.get("hypothesis_id")),
        ),
    )

    drops = 0

    def try_pack(*, uniq_contrast: bool, neg_cap: float) -> tuple[list[dict[str, Any]], int]:
        out: list[dict[str, Any]] = []
        dup_drops_local = 0
        ctr_phr_map: dict[int, str] = {}
        for h in pool_sorted:
            if len(out) >= target_max:
                break
            try:
                ccid = int(h["contrast_cluster_id"])
            except Exception:
                dup_drops_local += 1
                continue
            cphrase = _squish(str(h.get("contrast_style_phrase") or ""))
            if uniq_contrast:
                if ccid in ctr_phr_map:
                    dup_drops_local += 1
                    continue
            elif ccid in ctr_phr_map:
                prev = ctr_phr_map[ccid]
                if prev and _token_jaccard(cphrase, prev) >= reuse_contrast_phrase_jaccard_tol - 1e-12:
                    dup_drops_local += 1
                    continue
            if out and _max_contrast_rep_overlap(h, out) > neg_cap:
                dup_drops_local += 1
                continue
            if out and _v5_style_clash(h, out):
                dup_drops_local += 1
                continue
            out.append(dict(h))
            if ccid not in ctr_phr_map:
                ctr_phr_map[ccid] = cphrase or ""
        return out, dup_drops_local

    chosen, d1 = try_pack(uniq_contrast=True, neg_cap=0.5)
    drops += d1
    if len(chosen) < 2:
        chosen, d2 = try_pack(uniq_contrast=True, neg_cap=0.67)
        drops += d2
    if len(chosen) < 2:
        chosen, d3 = try_pack(uniq_contrast=False, neg_cap=0.67)
        drops += d3

    div = _cluster_contrast_diversity_score(chosen, clusters_map)
    reuse_extra = _contrast_cluster_reuse_count(chosen)
    cjac = _contrast_style_pairwise_max_jaccard(chosen)
    pairwise_neg_ov_mx = 0.0
    for ii in range(len(chosen)):
        for jj in range(ii + 1, len(chosen)):
            pairwise_neg_ov_mx = max(pairwise_neg_ov_mx, _max_contrast_rep_overlap(chosen[ii], [chosen[jj]]))

    pos_j = _positive_style_pairwise_max_jaccard(chosen)

    meta = {
        "contrast_filter_dropped_count": int(max(0, drops)),
        "cluster_contrast_diversity_score": float(div),
        "contrast_cluster_reuse_count": int(reuse_extra),
        "few_options_after_contrast_filter": bool(len(chosen) < 3),
        "contrast_style_pairwise_max_jaccard": float(cjac),
        "pairwise_contrast_artist_overlap_max": float(pairwise_neg_ov_mx),
        "positive_style_pairwise_max_jaccard": float(pos_j),
        "contrast_cluster_reuse_penalty": float(min(1.0, reuse_extra / max(1, len(chosen)))),
        "contrast_style_phrase_overlap_penalty": float(min(1.0, cjac)),
        "contrast_artist_overlap_penalty": float(min(1.0, pairwise_neg_ov_mx)),
        "reuse_contrast_phrase_jaccard_tolerance": float(reuse_contrast_phrase_jaccard_tol),
    }
    return chosen, meta


def repair_sentence_mojibake_visible(s: str) -> tuple[str, int]:
    """Repair full option/preamble sentences; avoids per-item truncation from sanitize_item_name."""

    t = _squish(str(s or ""))
    if not t:
        return t, 0
    suspicious = ("\ufffd" in t or "�" in t or "Ð" in t or "Ã¤" in t or "Ã©" in t or "Ã­" in t or "Ã³" in t)
    if not suspicious:
        return sanitize_user_question_copy(t), 0
    if "\ufffd" in t or "�" in t:
        t2 = sanitize_user_question_copy(t.replace("\ufffd", "").replace("�", "").strip())
        return t2, 1
    if "Ð" in t or "Ã" in t:
        for enc in ("latin1", "cp1252"):
            try:
                cand = sanitize_user_question_copy(_squish(str(t).encode(enc).decode("utf-8")))
                if cand and ("Ð" not in cand) and ("Ã¡" not in cand):
                    return cand, 1
            except Exception:
                continue
    return sanitize_user_question_copy(t), 1


def scrub_visible_mcq_bundle_user_text(bundle: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Rebuild final_question_text after sanitising substantive option paragraphs."""

    opts = bundle.get("options") or []
    touched = 0
    preamble, pcount = repair_sentence_mojibake_visible(str(bundle.get("question") or ""))
    touched += pcount
    bundle["question"] = preamble
    new_opts: list[dict[str, Any]] = []
    for o in opts:
        oo = dict(o)
        txt = str(oo.get("text") or "")
        if str(oo.get("maps_to_hypothesis_id") or "") == "__none__":
            new_opts.append(oo)
            continue
        fixed, k = repair_sentence_mojibake_visible(txt)
        touched += k
        oo["text"] = fixed
        new_opts.append(oo)
    bundle["options"] = new_opts
    bundle["final_question_text"] = preamble + "\n\n" + "\n".join(
        [f"{oo['option_id']}) {_squish(str(oo.get('text') or ''))}" for oo in new_opts],
    )
    return bundle, int(touched)


def item_uncertainty_sigma_rows(emb_rows: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    diag = np.einsum("ij,jk,ik->i", emb_rows.astype(float), Sigma.astype(float), emb_rows.astype(float))
    return np.sqrt(np.maximum(diag, 0.0))


@dataclass
class FrontierPoolItem:
    item_id: str
    rank: int
    belief_score: float
    uncertainty: float


@dataclass
class FrontierCluster:
    cluster_id: int
    member_ids: list[str]
    size: int
    mean_rank: float
    mean_belief_score: float
    mean_uncertainty: float
    overlap_top10_frac: float
    history_similarity: float
    representative_names: list[str]
    semantic_summary: str
    mean_embedding_direction: list[float] = field(default_factory=list)


@dataclass
class ClusterContrast:
    contrast_id: str
    safe_or_reference_cluster_id: int
    contrasting_cluster_id: int
    rationale: str


def build_frontier_pool(
    belief: BeliefState,
    index: EmbeddingIndex,
    seen_items: set[str],
    *,
    embeddings: np.ndarray,
    lamb: float,
    cfg: dict[str, Any],
    allowed_item_ids: set[str] | None = None,
) -> tuple[list[FrontierPoolItem], list[str]]:
    """FrontierPool = TopK ∪ boundary ranks ∪ uncertainty slice (all unseen)."""

    vc = cfg or {}
    top_k = int(vc.get("frontier_top_k", 100))
    unc_k = int(vc.get("frontier_uncertain_k", 50))
    b0 = int(vc.get("boundary_rank_start", 10))
    b1 = int(vc.get("boundary_rank_end", 100))
    perf_cfg = vc.get("_performance_optimizations") or {}
    perf_on = bool(perf_cfg.get("enabled"))
    use_topk_argpartition = perf_on and bool(perf_cfg.get("topk_argpartition"))
    use_vectorized_uncertainty = perf_on and bool(perf_cfg.get("vectorized_uncertainty", True))

    unseen_mask = np.array([str(iid) not in seen_items for iid in index.item_ids], dtype=bool)
    if allowed_item_ids is not None:
        unseen_mask &= np.array([str(iid) in allowed_item_ids for iid in index.item_ids], dtype=bool)
    sco = belief_scores(belief, embeddings, lambda_explore=lamb)
    flat = np.flatnonzero(unseen_mask)
    if flat.size == 0:
        return [], []

    if use_topk_argpartition:
        rank_cap = min(len(flat), max(top_k, b1, 10))
        if rank_cap <= 0:
            return [], []
        part = np.argpartition(-sco[flat], rank_cap - 1)[:rank_cap]
        rank_order = flat[part[np.argsort(-sco[flat][part])]]
        order = rank_order
    else:
        order = flat[np.argsort(-sco[flat])]

    ranks: dict[str, int] = {}
    for r, fid in enumerate(order, start=1):
        ranks[str(index.item_ids[int(fid)])] = r

    emb_order = embeddings[order].astype(float)
    if use_vectorized_uncertainty:
        unc_all = item_uncertainty_sigma_rows(emb_order, belief.Sigma)
    else:
        unc_all = item_uncertainty_sigma_rows(emb_order, belief.Sigma)
    unc_flat: dict[int, float] = {
        int(order[ii]): float(unc_all[ii]) for ii in range(len(order))
    }

    picks: list[int] = []
    kk = min(top_k, len(order))
    picks.extend(order[:kk].tolist())
    rb = min(max(b1, b0), len(order))
    for r_target in range(max(1, b0), rb + 1):
        picks.append(int(order[r_target - 1]))

    head = min(len(order), max(200, unc_k * 4))
    unc_ord = np.argsort(-unc_all[:head])
    for ji in range(min(unc_k, head)):
        picks.append(int(order[unc_ord[ji]]))

    uniq: list[int] = []
    saw: set[int] = set()
    for ij in picks:
        if ij not in saw:
            saw.add(int(ij))
            uniq.append(int(ij))

    rows = [
        FrontierPoolItem(
            item_id=str(index.item_ids[ij]),
            rank=int(ranks[str(index.item_ids[ij])]),
            belief_score=float(sco[int(ij)]),
            uncertainty=float(unc_flat.get(ij, float("nan"))),
        )
        for ij in uniq
    ]
    rows.sort(key=lambda z: z.rank)
    top10_ids = [str(index.item_ids[int(order[i])]) for i in range(min(10, len(order)))]
    return rows, top10_ids


def cluster_frontier_pool(
    pool: list[FrontierPoolItem],
    index: EmbeddingIndex,
    embeddings: np.ndarray,
    item_lookup: dict[str, dict[str, Any]],
    *,
    top10_ids: list[str],
    history_centroid: np.ndarray | None,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[list[FrontierCluster], dict[str, Any]]:
    vc = cfg or {}
    cluster_count = int(vc.get("cluster_count", 5))
    min_sz = max(3, int(vc.get("min_cluster_size", 5)))
    max_rep = int(vc.get("max_representatives_per_cluster", 5))

    rows_keep = [p for p in pool if p.item_id in index.item_id_to_index]
    flat_idx = [int(index.item_id_to_index[p.item_id]) for p in rows_keep]
    if len(rows_keep) < min_sz:
        return [], {"failure": "pool_too_small", "pool_size": len(rows_keep)}

    ids = [p.item_id for p in rows_keep]
    rank_by = {p.item_id: p.rank for p in rows_keep}
    score_by = {p.item_id: p.belief_score for p in rows_keep}
    unc_by = {p.item_id: p.uncertainty for p in rows_keep}

    E = embeddings[np.array(flat_idx, dtype=np.int64)].astype(float)
    nr = np.linalg.norm(E, axis=1, keepdims=True).clip(min=1e-12)
    En = E / nr

    kk = max(2, min(cluster_count, max(2, len(En) // max(3, min_sz // 3))))
    kk = min(kk, len(En))

    aux: dict[str, Any] = {"k_requested": cluster_count, "k_effective": kk}

    labels: np.ndarray
    if KMeans is None or kk < 2 or len(En) < kk:
        aux["method"] = "single_cluster_fallback"
        labels = np.zeros(len(En), dtype=int)
    else:
        aux["method"] = "kmeans_sklearn"
        km = KMeans(n_clusters=kk, n_init=5, random_state=int(rng.integers(0, 2**31 - 1)), max_iter=40)
        labels = np.asarray(km.fit_predict(En), dtype=int)

    top10_set = set(top10_ids)
    hc_n = None
    if history_centroid is not None:
        hc = history_centroid.astype(float).ravel()
        hc_n = hc / max(float(np.linalg.norm(hc)), 1e-12)

    clusters: list[FrontierCluster] = []

    for cid in sorted(set(int(x) for x in labels)):
        m = labels == cid
        mids = [ids[i] for i in range(len(ids)) if m[i]]
        rs = np.array([rank_by[j] for j in mids], dtype=float)
        bs = np.array([score_by[j] for j in mids], dtype=float)
        unc_ar = np.array([unc_by.get(j, math.nan) for j in mids], dtype=float)
        overlap = float(np.mean([j in top10_set for j in mids])) if mids else 0.0

        hsims: list[float] = []
        if hc_n is not None:
            for j in mids:
                jj = index.item_id_to_index.get(str(j))
                if jj is None:
                    continue
                v = embeddings[int(jj)].astype(float)
                vn = v / max(np.linalg.norm(v), 1e-12)
                hsims.append(float(np.dot(vn, hc_n)))
        hs = float(np.mean(hsims)) if hsims else float("nan")

        reps = mids[:max_rep]
        names: list[str] = []
        for rid in reps:
            row = item_lookup.get(str(rid), {})
            nm = row.get("name") or row.get("title") or row.get("item_name") or str(rid)[:48]
            names.append(str(nm)[:120])
        summary = _squish(" · ".join(names[:3]))
        cen = np.mean(En[m], axis=0)
        cen_vec = cen / max(float(np.linalg.norm(cen)), 1e-12)
        cen_list = [float(x) for x in cen_vec.tolist()]
        clusters.append(
            FrontierCluster(
                cluster_id=int(cid),
                member_ids=mids,
                size=len(mids),
                mean_rank=float(np.nanmean(rs)) if rs.size else float("nan"),
                mean_belief_score=float(np.nanmean(bs)) if bs.size else float("nan"),
                mean_uncertainty=float(np.nanmean(unc_ar[np.isfinite(unc_ar)])) if np.any(np.isfinite(unc_ar)) else float("nan"),
                overlap_top10_frac=float(overlap),
                history_similarity=float(hs) if math.isfinite(hs) else float("nan"),
                representative_names=names,
                semantic_summary=summary,
                mean_embedding_direction=cen_list,
            ),
        )
    return clusters, aux


def select_frontier_cluster_contrasts(
    clusters: list[FrontierCluster],
    *,
    cfg: dict[str, Any],
) -> list[ClusterContrast]:
    vc = cfg or {}
    max_contrast = max(1, int(vc.get("max_contrast_pairs", 4)))
    compact = bool(vc.get("prompt_compact_mode"))
    cap_final = max(max_contrast, 3) if compact else max_contrast
    if len(clusters) < 2:
        return []

    top_safe = sorted(
        clusters,
        key=lambda c: (
            -(c.overlap_top10_frac * 2.5 + float(c.history_similarity if math.isfinite(c.history_similarity) else -1.5))
            + 0.01 * float(c.mean_rank),
        ),
    )[0]

    boundary_sorted = sorted(
        clusters,
        key=lambda c: (
            -(c.mean_uncertainty if math.isfinite(c.mean_uncertainty) else -1.0),
            -abs(float(c.mean_rank) - 25.0),
        ),
    )
    high_unc = boundary_sorted[0]
    bc = high_unc if high_unc.cluster_id != top_safe.cluster_id else boundary_sorted[-1]

    def _uo_key(a_id: Any, b_id: Any) -> tuple[int, int]:
        aa, bb = int(a_id), int(b_id)
        return (aa, bb) if aa <= bb else (bb, aa)

    have_keys: set[tuple[int, int]] = set()

    out: list[ClusterContrast] = []
    out.append(
        ClusterContrast(
            contrast_id=f"{top_safe.cluster_id}_vs_{bc.cluster_id}",
            safe_or_reference_cluster_id=top_safe.cluster_id,
            contrasting_cluster_id=bc.cluster_id,
            rationale="Compare **history-aligned frontier mass** against a **higher-uncertainty** cluster elsewhere in the frontier.",
        ),
    )
    have_keys.add(_uo_key(top_safe.cluster_id, bc.cluster_id))

    extras = sorted(
        clusters,
        key=lambda c: -float(c.overlap_top10_frac),
    )
    for ci, c in enumerate(extras[: min(8, len(extras))]):
        buddy = clusters[(clusters.index(c) + 1) % len(clusters)]
        if c.cluster_id == buddy.cluster_id:
            continue
        if buddy.cluster_id == c.cluster_id:
            continue
        kk_extra = _uo_key(c.cluster_id, buddy.cluster_id)
        if kk_extra in have_keys:
            continue
        have_keys.add(kk_extra)
        out.append(
            ClusterContrast(
                contrast_id=f"{c.cluster_id}_{buddy.cluster_id}",
                safe_or_reference_cluster_id=c.cluster_id,
                contrasting_cluster_id=buddy.cluster_id,
                rationale=f"Alternate contrast #{ci}: distinct ranked clusters `{c.cluster_id}` vs `{buddy.cluster_id}` by overlap profile.",
            ),
        )
        if len(out) >= cap_final:
            break

    if compact:
        ids_sorted = sorted({int(c.cluster_id) for c in clusters})
        for i in range(len(ids_sorted)):
            if len(out) >= cap_final:
                break
            for j in range(i + 1, len(ids_sorted)):
                if len(out) >= cap_final:
                    break
                a_i, b_i = ids_sorted[i], ids_sorted[j]
                kk = _uo_key(a_i, b_i)
                if kk in have_keys:
                    continue
                have_keys.add(kk)
                out.append(
                    ClusterContrast(
                        contrast_id=f"{a_i}_vs_{b_i}_compact_fill",
                        safe_or_reference_cluster_id=a_i,
                        contrasting_cluster_id=b_i,
                        rationale=(
                            "`prompt_compact_mode` pairwise filler — ensures sufficient planned contrasts "
                            "for mocked v3.2 hypotheses while staying within pooled cluster identities."
                        ),
                    ),
                )
    return out[:cap_final]


def build_cluster_semantic_summary(
    cluster: FrontierCluster,
    item_lookup: dict[str, dict[str, Any]],
    *,
    history_context: dict[str, Any],
) -> dict[str, Any]:
    """Compose an offline-readable cluster synopsis from CSV item rows + frontier stats — no invented genres."""

    reps = [_squish(str(x))[:120] for x in cluster.representative_names[:12]]

    mids = list(cluster.member_ids)[:52]
    pooled: list[str] = []
    enriched_fields = 0
    evidence_items: list[str] = []

    genre_stop = {
        "unknown",
        "misc",
        "other",
        "various",
    }
    kw_stop = {
        "title",
        "artist",
        "track",
        "music",
        "song",
        "album",
        "audio",
        "category",
        "tags",
        "description",
        "the",
        "and",
        "with",
        "from",
    }

    for mid in mids:
        row = item_lookup.get(str(mid)) or {}
        evidence_items.append(str(mid))
        g = _squish(str(row.get("genre") or row.get("category") or "")).lower()
        tg = _squish(str(row.get("tags") or "")).lower()
        desc = _squish(str(row.get("description") or "")).lower()
        artist = _squish(str(row.get("artist_name") or ""))
        title = _squish(str(row.get("title") or row.get("track_name") or ""))
        if g and len(g) > 1 and g not in genre_stop:
            enriched_fields += 1
            pooled.append(g)
        if tg and len(tg) > 1:
            enriched_fields += 1
            pooled.append(re.sub(r"[,|;]", " ", tg))
        if len(desc) > 8:
            enriched_fields += 1
            pooled.append(desc[:440])
        elif artist:
            pooled.append(re.sub(r"[,|;]", " ", artist)[:160])
            if title:
                pooled.append(re.sub(r"[,|;]", " ", title)[:140])

    blob = _squish(" ".join(pooled)).lower()
    reps_blob = (_squish(" ".join(reps))).lower()

    vibe_tokens = (
        ("mellow", "mellow_energy"),
        ("sparse", "sparse_arrangement_texture"),
        ("acoustic", "acoustic-forward"),
        ("upbeat", "upbeat_motion"),
        ("ambient", "ambient_pad"),
        ("aggressive", "aggressive_drive"),
        ("dramatic", "dramatic_swell"),
        ("raw", "raw_edge"),
        ("electronic", "electronic_timbre"),
        ("percussive", "percussive_forward"),
        ("dark", "darker_palette"),
        ("bright", "brighter_palette"),
        ("minimal", "minimal_arrangement"),
    )
    energy_or_texture_hint = ""
    for needle, lbl in vibe_tokens:
        if needle in blob or needle in reps_blob:
            energy_or_texture_hint = lbl
            break

    toks = [w for w in re.split(r"\W+", blob) if len(w) > 3 and w not in kw_stop]
    semantic_keywords = [w for w, _ in Counter(toks).most_common(14)]

    ah_raw = [_squish(str(x))[:112] for x in (history_context.get("anchor_names") or []) if _squish(str(x))]
    ah = ah_raw[:32]
    anchors_touch_cluster = False
    base_rep_low = reps_blob.replace(" · ", " ")
    for a in ah:
        al = a.lower()
        if len(al) <= 3:
            continue
        sparts = al.split()
        anchor_prefix_ok = False
        if sparts:
            pref = sparts[0][: min(8, len(sparts[0]))]
            if len(pref) > 4 and pref in base_rep_low:
                anchor_prefix_ok = True
        if al in base_rep_low or anchor_prefix_ok:
            anchors_touch_cluster = True
            continue
        tokens_a = [t for t in re.split(r"\W+", al) if len(t) > 4]
        if tokens_a:
            anchors_touch_cluster = anchors_touch_cluster or any(t in blob for t in tokens_a[:6])

    style_axes: list[str] = []
    ot = cluster.overlap_top10_frac
    if isinstance(ot, (int, float)) and math.isfinite(float(ot)):
        fot = float(ot)
        if fot >= 0.35:
            style_axes.append("high_surface_alignment_with_candidate_top_band")
        elif fot <= 0.12:
            style_axes.append("low_surface_alignment_with_candidate_top_band")

    hf = cluster.history_similarity
    if isinstance(hf, (int, float)) and math.isfinite(float(hf)):
        hfv = float(hf)
        if hfv >= 0.05:
            style_axes.append("tighter_recent_history_similarity_mass")
        elif hfv <= -0.02:
            style_axes.append("offset_from_recent_history_centroid")

    if isinstance(cluster.mean_uncertainty, (int, float)) and math.isfinite(float(cluster.mean_uncertainty)):
        mu_c = float(cluster.mean_uncertainty)
        g_mu = history_context.get("global_mean_uncertainty")
        if isinstance(g_mu, (int, float)) and math.isfinite(float(g_mu)) and abs(mu_c - float(g_mu)) >= 0.02:
            if mu_c > float(g_mu):
                style_axes.append("elevated_uncertainty_within_frontier_band")
            else:
                style_axes.append("tighter_uncertainty_within_frontier_band")

    try:
        mr = float(cluster.mean_rank)
    except (TypeError, ValueError):
        mr = float("nan")
    if math.isfinite(mr):
        if mr <= 22.0:
            style_axes.append("near_top_of_exploration_stack")
        elif mr >= 55.0:
            style_axes.append("deeper_stack_more_speculative_candidates")

    ot_f = float(ot) if isinstance(ot, (int, float)) and math.isfinite(float(ot)) else 0.0
    hf_usable = (
        float(hf)
        if isinstance(hf, (int, float)) and math.isfinite(float(hf))
        else 0.0
    )
    if ot_f >= 0.42 and hf_usable >= 0.03:
        familiarity_level = "frequently_near_current_recommendations"
    elif anchors_touch_cluster:
        familiarity_level = "touches_recent_anchor_artists_lexically"
    elif ot_f <= 0.15:
        familiarity_level = "sparse_in_immediate_top_band"
    else:
        familiarity_level = "mixed_frontier_profile"

    confidence = (
        0.2
        + 0.016 * float(min(enriched_fields, 20))
        + 0.055 * float(min(len(semantic_keywords), 12))
        + (0.11 if energy_or_texture_hint else 0.0)
        + (0.08 if len(style_axes) >= 2 else 0.05)
        + (0.06 if anchors_touch_cluster else 0.0)
    )

    fallback_reason = ""
    if enriched_fields < 2 and len(semantic_keywords) < 3:
        fallback_reason = "metadata_sparse_item_rows"
        confidence = min(float(confidence), 0.46)

    metadata_enrichment_used = bool(enriched_fields >= 2 or len(semantic_keywords) >= 4)

    confidence = float(max(0.08, min(1.0, confidence)))

    kw_snip = ", ".join(semantic_keywords[:10]) if semantic_keywords else "(no_stable_lexical_overlap)"
    axes_snip = "; ".join(style_axes[:7]) if style_axes else "frontier_statistics_only_without_item_text_cues"

    chunks = [
        f"representatives: {_squish(' · '.join(reps[:3]))[:220]}",
        f"lexical_cues: {kw_snip}",
        f"axes: {axes_snip}",
        f"familiarity: {familiarity_level}",
    ]
    if energy_or_texture_hint:
        chunks.insert(2, f"texture_hint: {energy_or_texture_hint}")
    cluster_semantic_summary = _squish("; ".join(chunks))

    return {
        "cluster_semantic_summary": cluster_semantic_summary,
        "semantic_keywords": semantic_keywords,
        "style_axes": style_axes,
        "familiarity_level": familiarity_level,
        "energy_or_texture_hint": energy_or_texture_hint or "",
        "confidence": confidence,
        "evidence_items": evidence_items[:32],
        "metadata_enrichment_used": metadata_enrichment_used,
        "fallback_reason": fallback_reason,
        "anchors_touch_cluster": anchors_touch_cluster,
    }


def apply_cluster_semantic_enrichment_mock(
    clusters_payload: list[dict[str, Any]],
    frontier_clusters: list[FrontierCluster],
    item_lookup: dict[str, dict[str, Any]],
    *,
    anchor_names: list[str],
    history_summary: str,
) -> dict[str, Any]:
    """Mutates clusters_payload dict rows with enrichment fields."""

    fc_by_id = {int(fc.cluster_id): fc for fc in frontier_clusters}
    ucs = [
        float(fc.mean_uncertainty)
        for fc in frontier_clusters
        if isinstance(fc.mean_uncertainty, (int, float)) and math.isfinite(float(fc.mean_uncertainty))
    ]
    global_mu = float(np.mean(ucs)) if ucs else float("nan")
    enriched = thin = 0

    hc: dict[str, Any] = {
        "anchor_names": anchor_names,
        "history_summary": history_summary,
        "global_mean_uncertainty": global_mu,
    }
    for row in clusters_payload:
        try:
            cid = int(row.get("cluster_id"))
        except Exception:
            continue
        fc = fc_by_id.get(cid)
        if fc is None:
            continue
        summ = build_cluster_semantic_summary(fc, item_lookup, history_context=hc)
        row["cluster_semantic_summary"] = summ["cluster_semantic_summary"]
        row["cluster_semantic_keywords"] = summ["semantic_keywords"]
        row["cluster_semantic_style_axes"] = summ["style_axes"]
        row["cluster_familiarity_level"] = summ["familiarity_level"]
        row["cluster_energy_or_texture_hint"] = summ["energy_or_texture_hint"]
        row["cluster_semantic_confidence"] = float(summ["confidence"])
        row["cluster_semantic_evidence_items"] = summ["evidence_items"]
        row["metadata_enrichment_used"] = bool(summ["metadata_enrichment_used"])
        row["semantic_summary_fallback_reason"] = str(summ.get("fallback_reason") or "")
        if summ.get("metadata_enrichment_used"):
            enriched += 1
        if summ.get("fallback_reason"):
            thin += 1

    return {
        "clusters_semantic_fields_written": len(clusters_payload),
        "clusters_metadata_enrichment_used_count": enriched,
        "clusters_semantic_thin_fallback_count": thin,
    }


def compact_frontier_prompt_payload(payload: dict[str, Any], fc32_cfg: dict[str, Any]) -> dict[str, Any]:
    """Truncate cluster / contrast payloads for cheaper prompts (still schema-safe for mock)."""

    if not fc32_cfg.get("prompt_compact_mode"):
        return payload
    out = json.loads(json.dumps(payload, ensure_ascii=False, default=str))
    max_rep = max(1, int(fc32_cfg.get("max_representatives_per_cluster", 2)))
    ss_max = int(fc32_cfg.get("semantic_summary_max_chars", 120))
    mx_pairs = max(1, int(fc32_cfg.get("max_contrast_pairs", fc32_cfg.get("max_contrast_pairs_in_payload", 3))))
    cl_out: list[dict[str, Any]] = []
    for c in out.get("clusters") or []:
        if not isinstance(c, dict):
            continue
        cc = dict(c)
        reps = list(cc.get("representative_names") or [])[:max_rep]
        cc["representative_names"] = [_squish(str(x))[:64] for x in reps]
        ss = _squish(str(cc.get("semantic_summary") or ""))
        cc["semantic_summary"] = ss[:ss_max]
        if fc32_cfg.get("prompt_compact_strip_embedding_direction", True):
            cc.pop("mean_embedding_direction", None)
        cl_out.append(cc)
    out["clusters"] = cl_out
    ct0 = [x for x in (out.get("contrasts") or []) if isinstance(x, dict)][:mx_pairs]
    out["contrasts"] = ct0
    anch = list(out.get("anchor_item_names") or [])[:6]
    out["anchor_item_names"] = anch
    hist = _squish(str(out.get("history_summary") or ""))
    out["history_summary"] = hist[: int(fc32_cfg.get("history_summary_max_chars", 200))]
    return out


def build_frontier_cluster_prompt_payload(
    clusters: list[FrontierCluster],
    contrasts: list[ClusterContrast],
    history_summary: str,
    anchor_names: list[str],
    *,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    _ = cfg
    return {
        "schema": "frontier_clustered_v32_v0",
        "history_summary": history_summary,
        "anchor_item_names": anchor_names,
        "clusters": [
            {
                "cluster_id": c.cluster_id,
                "size": c.size,
                "mean_rank": c.mean_rank,
                "mean_belief_score": c.mean_belief_score,
                "mean_uncertainty": c.mean_uncertainty,
                "overlap_top10_frac": c.overlap_top10_frac,
                "history_similarity": c.history_similarity,
                "representative_names": c.representative_names,
                "semantic_summary": c.semantic_summary,
                # Mock-only centroid for semantic distance gates (offline; not shown to users).
                "mean_embedding_direction": list(c.mean_embedding_direction or []),
            }
            for c in clusters
        ],
        "contrasts": [
            {
                "contrast_id": x.contrast_id,
                "reference_cluster_id": x.safe_or_reference_cluster_id,
                "contrasting_cluster_id": x.contrasting_cluster_id,
                "rationale": x.rationale,
            }
            for x in contrasts
        ],
    }


def _representative_three(cl: dict[str, Any]) -> list[str]:
    reps = list(cl.get("representative_names") or [])
    if len(reps) >= 3:
        return [str(x)[:80] for x in reps[:3]]
    fallback = _squish(str(cl.get("semantic_summary") or ""))
    chunks = [c.strip() for c in fallback.split("·")] if fallback else []
    out = ([str(x)[:80] for x in reps] + chunks)[:3]
    return out or ["(unknown cue)", "(unknown cue)", "(unknown cue)"]


def _names_anchor_phrase(names: list[str]) -> str:
    clean = [_squish(str(n))[:64] for n in names[:3]]
    clean = [c for c in clean if c]
    return _squish(" · ".join(clean))[:180]


def _pair_distinguishing_features(ca: dict[str, Any], cb: dict[str, Any]) -> list[str]:
    """2–4 human-readable contrasts derived from pooled cluster stats for mock artefacts."""
    feats: list[str] = []
    try:
        oa = float(ca.get("overlap_top10_frac") or 0)
        ob = float(cb.get("overlap_top10_frac") or 0)
    except (TypeError, ValueError):
        oa = ob = 0.0
    if abs(oa - ob) >= 0.05:
        if oa > ob:
            feats.append("Weights cues that resemble artists you bump into frequently in mixes like these.")
        else:
            feats.append("Weights cues that resemble artists you seldom see surfaced in mixes like these.")

    ha = ca.get("history_similarity")
    hb = cb.get("history_similarity")
    if isinstance(ha, (int, float)) and isinstance(hb, (int, float)):
        hf = float(ha)
        hg = float(hb)
        if math.isfinite(hf) and math.isfinite(hg) and abs(hf - hg) >= 0.03:
            if hf > hg:
                feats.append("Sticks closer to the immediate stylistic neighborhood of your recent sessions.")
            else:
                feats.append("Steps slightly away from the tightest match to your very recent sessions.")

    ua = ca.get("mean_uncertainty")
    ub = cb.get("mean_uncertainty")
    if isinstance(ua, (int, float)) and isinstance(ub, (int, float)):
        x = float(ua)
        y = float(ub)
        if math.isfinite(x) and math.isfinite(y) and abs(x - y) >= 0.02:
            if x < y:
                feats.append("Favors a steadier band of suggestions than the alternate trio.")
            else:
                feats.append("Allows a looser, more open-ended band than the alternate trio.")

    try:
        ra = float(ca.get("mean_rank") or 0)
        rb = float(cb.get("mean_rank") or 0)
    except (TypeError, ValueError):
        ra = rb = 0.0
    if math.isfinite(ra) and math.isfinite(rb) and abs(ra - rb) >= 8.0:
        if ra < rb:
            feats.append("Emphasizes a tighter band among unfamiliar suggestions versus the alternate trio.")
        else:
            feats.append("Emphasizes a looser band among unfamiliar suggestions versus the alternate trio.")

    if len(feats) < 2:
        pa = _names_anchor_phrase(_representative_three(ca))
        pb = _names_anchor_phrase(_representative_three(cb))
        feats.append(f"Keeps tonal weight on {pa} versus {pb}.")
    if len(feats) < 2:
        feats.append(
            "This row is deliberately paired so each option contrasts two different stylistic strands, not synonyms.",
        )
    return feats[:4]


def _differs_sentence(this_pos: str, this_neg: str, others: list[tuple[str, str]]) -> str:
    if not others:
        return (
            _squish(
                "This pairing is scoped to two specific artist triples; sibling options highlight different anchors.",
            )[:380]
        )
    tails = [_squish(f"{p} versus {q}")[:120] for p, q in others[:2]]
    joiner = "; ".join(tails)
    return _squish(
        f"This option contrasts {_squish(this_pos)[:140]} against {_squish(this_neg)[:140]}. "
        f"Peers instead separate {joiner}.",
    )[:420]


def _ranking_boundary_rationale_plain(ca: dict[str, Any], cb: dict[str, Any], pos_names: str, neg_names: str) -> str:
    return _squish(
        f"Choosing this tilts which unseen creators get emphasized toward artists like {pos_names[:120]}, "
        f"versus steering away from the mood implied by artists like {neg_names[:120]}. "
        "The split is anchored on measurable divergence in overlap with your current mixes and similarity to "
        "recent-session taste cues.",
    )[:480]


def trim_frontier_payload_for_llm(payload: dict[str, Any]) -> dict[str, Any]:
    """Shrink JSON sent to LM (drop centroid vectors — not shown to users)."""

    slim: dict[str, Any] = {
        "schema": payload.get("schema"),
        "history_summary": payload.get("history_summary"),
        "anchor_item_names": list(payload.get("anchor_item_names") or []),
        "clusters": [],
        "contrasts": list(payload.get("contrasts") or []),
    }
    clusters_slim: list[dict[str, Any]] = []
    for c in payload.get("clusters") or []:
        if not isinstance(c, dict):
            continue
        cc = dict(c)
        cc.pop("mean_embedding_direction", None)
        clusters_slim.append(cc)
    slim["clusters"] = clusters_slim

    # v4.1 role-fix: give LLM an explicit "what to preserve" anchor (no internal jargon in user copy).
    if clusters_slim:
        try:
            top_cl = max(
                clusters_slim,
                key=lambda x: float(x.get("overlap_top10_frac") or 0),
            )
            slim["current_top_style_summary"] = {
                "cluster_id": top_cl.get("cluster_id"),
                "representative_names": list(top_cl.get("representative_names") or [])[:4],
                "semantic_summary": _squish(str(top_cl.get("semantic_summary") or ""))[:240],
            }
        except (TypeError, ValueError):
            pass
        try:
            unc_cl = max(
                clusters_slim,
                key=lambda x: float(x.get("mean_uncertainty") or 0),
            )
            slim["uncertain_style_summary"] = {
                "cluster_id": unc_cl.get("cluster_id"),
                "representative_names": list(unc_cl.get("representative_names") or [])[:4],
                "semantic_summary": _squish(str(unc_cl.get("semantic_summary") or ""))[:240],
            }
        except (TypeError, ValueError):
            pass
    return slim


def _normalize_llm_hypothesis_object(h_raw: dict[str, Any]) -> dict[str, Any]:
    """Coerce LM keys into shapes expected by choice + schema validators."""

    h = dict(h_raw)
    hid = str(h.get("hypothesis_id") or h.get("option_id") or "").strip()
    if hid:
        h["hypothesis_id"] = hid
        h["option_id"] = str(h.get("option_id") or hid)

    try:
        pa = int(h.get("positive_cluster_id") or h.get("positive_cluster_ref") or -1)
    except Exception:
        pa = -1
    try:
        pb = int(h.get("contrast_cluster_id") or h.get("contrasting_cluster_id") or h.get("contrast_cluster_ref") or -1)
    except Exception:
        pb = -1
    if pa >= 0:
        h["positive_cluster_id"] = pa
    if pb >= 0:
        h["contrast_cluster_id"] = pb
    cr = h.get("contrast_ref")
    if not isinstance(cr, dict) and pa >= 0 and pb >= 0:
        h["contrast_ref"] = {"reference_cluster_id": pa, "contrasting_cluster_id": pb}

    it = _squish(str(h.get("intent_text") or h.get("positive_intent") or ""))
    if it:
        h.setdefault("intent_text", it)
        h.setdefault("positive_intent", it)
    txt = _squish(str(h.get("text_description") or h.get("short_user_facing_option_text") or it))
    if txt:
        h.setdefault("text_description", txt[:360])

    if not isinstance(h.get("distinguishing_features"), list):
        h["distinguishing_features"] = []
    fs = list(h.get("feature_signature") or [])
    if not fs and h.get("distinguishing_features"):
        fs = [str(x) for x in (h["distinguishing_features"] or [])][:6]
    if fs:
        h["feature_signature"] = fs

    rep = list(h.get("representative_items") or h.get("representative_examples") or [])
    cre = list(h.get("contrast_representative_items") or [])
    if rep:
        h["representative_items"] = [_squish(str(x))[:80] for x in rep[:3]]
    if cre:
        h["contrast_representative_items"] = [_squish(str(x))[:80] for x in cre[:3]]

    ng = _squish(str(h.get("negative_contrast") or ""))
    if ng:
        h["negative_contrast"] = ng
    elif it:
        h["negative_contrast"] = (
            _squish("Prefer directions different from the counter-moods implied by the contrasting exemplars.")[:320]
        )

    suf = _squish(str(h.get("short_user_facing_option_text") or txt or it))
    if suf:
        h["short_user_facing_option_text"] = suf[:720]

    # v4.1 role-aware / prompt-role-fix fields (runtime-safe metadata)
    if h.get("rank_safety_role") is not None:
        h["rank_safety_role"] = _squish(str(h.get("rank_safety_role") or ""))[:48]
    if h.get("role_rationale") is not None:
        h["role_rationale"] = _squish(str(h.get("role_rationale") or ""))[:360]

    for ev_key in (
        "current_top_evidence_items",
        "boundary_evidence_items",
        "uncertain_evidence_items",
    ):
        ev = h.get(ev_key)
        if isinstance(ev, list):
            h[ev_key] = [_squish(str(x))[:80] for x in ev[:6] if _squish(str(x))]

    for k_missing, template in (
        ("why_this_differs_from_other_options", ""),
        ("ranking_boundary_rationale", ""),
        ("why_this_question_may_affect_ranking_boundary", ""),
        ("positive_style_phrase", ""),
        ("contrast_style_phrase", ""),
    ):
        if not _squish(str(h.get(k_missing) or "")):
            if k_missing == "positive_style_phrase" and it:
                h[k_missing] = it[:180]
            elif k_missing == "contrast_style_phrase" and ng:
                h[k_missing] = ng[:180]
            elif it:
                h[k_missing] = it[:380]
            else:
                h[k_missing] = template
    return h


def generate_frontier_cluster_hypotheses_via_llm_live(
    payload: dict[str, Any],
    llm_client: Any,
    *,
    M: int,
    max_tokens: int | None = None,
    prompt_compact_mode: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    """Call configured OpenAI-compatible client (DeepSeek) for clustered-contrast hypotheses JSON."""

    if llm_client is None:
        raise ValueError("llm_client is required for frontier_clustered_v32 LLM path")
    if getattr(llm_client, "mode", "mock") == "mock":
        raise RuntimeError(
            "frontier_clustered_v32 LLM branch requires configs with llm.mode != mock "
            "(and API keys loaded).",
        )

    trimmed = trim_frontier_payload_for_llm(payload)
    payload_json = json.dumps(trimmed, ensure_ascii=False, indent=2)
    if len(payload_json) > 28_000:
        payload_json = payload_json[:28_000] + "\n…[payload_truncated]"

    contrasts = trimmed.get("contrasts") or []
    cid_lines = []
    for c in contrasts[:12]:
        if not isinstance(c, dict):
            continue
        try:
            cid_lines.append(
                f"pair: reference_cluster_id={int(c.get('reference_cluster_id'))} "
                f"contrasting_cluster_id={int(c.get('contrasting_cluster_id'))}",
            )
        except Exception:
            continue
    cid_blob = "; ".join(cid_lines) if cid_lines else "(follow contrasts[] in payload JSON)"

    role_fix = str(os.environ.get("V41_PROMPT_ROLE_FIX") or "").strip().lower() in ("1", "true", "yes", "on")

    m_goal = max(3, min(8, int(M)))
    if role_fix:
        m_goal = 3

    schema_rules = ""
    if role_fix:
        schema_rules = dedent(
            """\
INTERREC_FRONTIER_CLUSTERED_V32_SCHEMA_V41_ROLEFIX (strict JSON):

Return a single JSON object with key "hypotheses" ONLY.
Produce EXACTLY 3 hypothesis objects, with EXACTLY these roles (one each):
  1) current_top_preserve
  2) boundary_challenge
  3) explore_uncertain

Each hypothesis MUST contain:
  - hypothesis_id AND option_id (same string, e.g. "hA","hB","hC" or "h1","h2","h3")
  - rank_safety_role: "current_top_preserve" | "boundary_challenge" | "explore_uncertain"
  - role_rationale: one sentence explaining why this role fits (no internal jargon)
  - positive_cluster_id, contrast_cluster_id (integers copied from contrasts[] only)
  - contrast_ref object: {{"reference_cluster_id": <int>, "contrasting_cluster_id": <int>}} matching that pair ordering
  - intent_text, positive_intent (user-safe, <= ~45 tokens)
  - negative_contrast (explicit counter-moods, NO empty)
  - distinguishing_features: array of 2–4 SHORT strings (music words only)
  - representative_items AND contrast_representative_items arrays (3 SHORT artist/track names EACH)
  - current_top_evidence_items: 2–4 strings (artist/track cues) grounding the preserve role (required for current_top_preserve; optional for others)
  - boundary_evidence_items: 2–4 strings grounding the boundary role (required for boundary_challenge; optional for others)
  - uncertain_evidence_items: 2–4 strings grounding the explore role (required for explore_uncertain; optional for others)
  - positive_style_phrase, contrast_style_phrase
  - short_user_facing_option_text (<=720 chars after trim)
  - text_description (canonical short copy)
  - feature_signature array (4–6 lowercase tokens)

Hard constraints:
  - Exactly ONE hypothesis must be rank_safety_role=current_top_preserve and must include non-empty current_top_evidence_items.
  - Exactly ONE must be boundary_challenge and must include non-empty boundary_evidence_items.
  - Exactly ONE must be explore_uncertain and must include non-empty uncertain_evidence_items.
  - Do NOT output duplicate roles.

USER-VISIBLE HARD BAN inside ALL user-facing strings: do NOT write the substring "cluster",
"clusters", "frontier", "ranking", "rank", "boundary", "top 10", "posterior", "embedding".
Use conversational music words instead.

Planned contrast pairs MUST be grounded strictly to contrasts in the payload JSON:
{cid_blob}
""",
        ).format(cid_blob=cid_blob)
    else:
        schema_rules = dedent(
            """\
INTERREC_FRONTIER_CLUSTERED_V32_SCHEMA (strict JSON):

Return a single JSON object with key "hypotheses" ONLY (model key optional).
Produce EXACTLY {m_goal} hypothesis objects covering DISTINCT planned contrast pairs.

Each hypothesis MUST contain:
  - hypothesis_id AND option_id (same string, e.g. "h1"..)
  - positive_cluster_id, contrast_cluster_id (integers copied from TWO clusters in contrasts[] only)
  - contrast_ref object: {{"reference_cluster_id": <int>, "contrasting_cluster_id": <int>}} matching THAT pair ordering
  - intent_text, positive_intent (user-safe, <= ~45 tokens)
  - negative_contrast (explicit counter-moods, NO empty)
  - distinguishing_features: array of 2-4 SHORT strings WITHOUT internal system vocabulary
  - representative_items AND contrast_representative_items arrays (3 SHORT artist/track names EACH)
  - positive_style_phrase, contrast_style_phrase
  - short_user_facing_option_text summarizing stance (<=720 chars after trim)
  - text_description (canonical short copy)
  - feature_signature array (4–6 lowercase tokens)

USER-VISIBLE HARD BAN inside ALL user-facing strings: do NOT write the substring "cluster",
"clusters", "frontier", "ranking", "rank", "boundary", "top 10", "posterior", "embedding".
Use conversational music words instead.

planned contrast pairs MUST be grounded strictly to contrasts in the payload JSON:
{cid_blob}
""",
        ).format(m_goal=m_goal, cid_blob=cid_blob)

    system_blob = dedent(
        """\
You serialize ranking-aware preference hypotheses for personalized music browsing.
Respond with JSON only (no markdown fences). Respect contrast pair grounding.
Tone: helpful, conversational, culturally neutral.""",
    )

    extra_user_instructions = ""
    if role_fix:
        extra_user_instructions = dedent(
            """\

ROLEFIX NOTES:
- Make the three roles clearly different for a human.
- Use current_top_style_summary in the payload for Option A (preserve): cite 2–4 artist/track cues from representative_names.
- Use contrasts[] for Option B (boundary): pick a pair where overlap with current favorites is middling, not zero.
- Use uncertain_style_summary for Option C (explore): cite uncertain cluster representative_names.
- current_top_preserve must feel like "keep leaning into what already seems to work" (non-empty current_top_evidence_items).
- boundary_challenge must feel like "a nearby pivot that could reorder some candidates" (non-empty boundary_evidence_items).
- explore_uncertain must feel like "a more exploratory direction" (non-empty uncertain_evidence_items).
- Do not use internal terms; keep user-facing wording crisp and concrete.
""",
        )

    msgs = [
        {"role": "system", "content": system_blob + "\n\n" + schema_rules},
        {
            "role": "user",
            "content": (
                "Below is CLUSTERING CONTEXT JSON (clusters + contrasts + summaries). "
                f"Produce {m_goal} hypotheses per rules.\n"
                + extra_user_instructions
                + "\n"
                + payload_json
            ),
        },
    ]

    chat_kw: dict[str, Any] = {"temperature": 0.15}
    if max_tokens is not None:
        chat_kw["max_tokens"] = int(max_tokens)

    parsed = llm_client.structured_json(
        msgs,
        profile_component="frontier_clustered_v32_llm_hypotheses",
        **chat_kw,
    )

    hypotheses_in: Any = []
    if isinstance(parsed, list):
        hypotheses_in = parsed
    elif isinstance(parsed, dict):
        hypotheses_in = (
            parsed.get("hypotheses")
            or parsed.get("hypothesis_generation_output")
            or []
        )
    elif parsed is None:
        hypotheses_in = []

    cleaned: list[dict[str, Any]] = []
    for h_raw in hypotheses_in:
        if not isinstance(h_raw, dict):
            continue
        hh = _normalize_llm_hypothesis_object(h_raw)
        if str(hh.get("hypothesis_id") or "").strip():
            cleaned.append(hh)

    cleaned = cleaned[:m_goal]

    root_out = {"model": getattr(llm_client, "model", "unknown"), "hypotheses": cleaned}
    if isinstance(parsed, dict) and "mock_warnings" in parsed:
        root_out["mock_warnings"] = parsed.get("mock_warnings")

    raw_text = json.dumps(root_out, ensure_ascii=False, indent=2)
    return raw_text, cleaned


def generate_frontier_cluster_hypotheses_mock_or_llm(
    payload: dict[str, Any],
    llm_client: Any,
    mode: str,
    *,
    M: int = 4,
    max_output_tokens: int | None = None,
    prompt_compact_mode: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    """``mock``: deterministic templates. ``llm``: OpenAI-compat HTTP (requires non-mock client)."""

    if str(mode or "").strip().lower() == "llm":
        return generate_frontier_cluster_hypotheses_via_llm_live(
            payload,
            llm_client,
            M=M,
            max_tokens=max_output_tokens,
            prompt_compact_mode=prompt_compact_mode,
        )

    del llm_client

    clusters_list = payload.get("clusters", [])
    clusters = {int(c["cluster_id"]): c for c in clusters_list}
    contrasts = payload.get("contrasts", [])
    mock_warnings: list[str] = []
    if len(clusters) < 2:
        mock_warnings.append("cannot_build_contrast_pair")
        raw_obj_short: dict[str, Any] = {
            "model": "mock_rule_v32",
            "mock_warnings": mock_warnings,
            "hypotheses": [],
        }
        return json.dumps(raw_obj_short, ensure_ascii=False, indent=2), []
    if len(clusters_list) < 3:
        mock_warnings.append("few_clusters_for_cover")
    dedup_pairs: set[tuple[int, int]] = set()

    planned: list[tuple[int, int]] = []
    for ct in contrasts:
        a = int(ct.get("reference_cluster_id", ct.get("safe_or_reference_cluster_id", -1)))
        b = int(ct.get("contrasting_cluster_id", -1))
        if (a, b) in dedup_pairs or (b, a) in dedup_pairs:
            continue
        dedup_pairs.add((a, b))
        cc_obj = clusters.get(a)
        cb_obj = clusters.get(b)
        if not cc_obj or not cb_obj:
            continue
        planned.append((a, b))

    raw_obj: dict[str, Any] = {"model": "mock_rule_v32", "mock_warnings": mock_warnings, "hypotheses": []}
    hyps: list[dict[str, Any]] = []

    take = planned[: max(1, min(M, len(planned)))]
    if not take:
        mock_warnings.append("no_valid_contrast_after_dedupe")
        raw_obj["hypotheses"] = []
        return json.dumps(raw_obj, ensure_ascii=False, indent=2), []

    for i, tri in enumerate(take):
        a, b = tri
        ca = clusters.get(a) or {}
        cb = clusters.get(b) or {}
        rep_pos = _representative_three(ca)
        rep_neg = _representative_three(cb)
        pos_names = _names_anchor_phrase(rep_pos)
        neg_names = _names_anchor_phrase(rep_neg)
        intent_text = _squish(
            f"Tilt discoveries toward moods suggested by artists like {pos_names}, "
            f"rather than moods suggested by artists like {neg_names}.",
        )
        negative_contrast = _squish(
            f"Explicit steer-away from tonal cues resembling {neg_names} (triple shown as shorthand, not exhaustive).",
        )
        feats = _pair_distinguishing_features(ca, cb)
        hid = f"fc{i + 1}"
        row_seed = {"pos_trim": pos_names, "neg_trim": neg_names, "intent": intent_text}

        row = {
            "option_id": hid,
            "hypothesis_id": hid,
            "positive_cluster_id": a,
            "contrast_cluster_id": b,
            "intent_text": intent_text,
            "positive_intent": intent_text,
            "negative_contrast": negative_contrast,
            "distinguishing_features": feats[:4],
            "representative_items": rep_pos[:3],
            "contrast_representative_items": rep_neg[:3],
            "positive_style_phrase": make_cluster_style_phrase(ca),
            "contrast_style_phrase": make_cluster_style_phrase(cb),
            "why_this_differs_from_other_options": "",  # filled after siblings exist
            "ranking_boundary_rationale": _ranking_boundary_rationale_plain(ca, cb, pos_names, neg_names),
            "why_this_question_may_affect_ranking_boundary": _ranking_boundary_rationale_plain(
                ca, cb, pos_names, neg_names,
            ),
            "short_user_facing_option_text": "",
            "text_description": intent_text[:360],
            "feature_signature": [f"c{a}", f"c{b}", "contrast_pair"],
            "_scratch": row_seed,
        }
        hyps.append(row)

    pairs_trim: list[tuple[str, str]] = [(str(h["_scratch"]["pos_trim"]), str(h["_scratch"]["neg_trim"])) for h in hyps]
    for hi, row in enumerate(hyps):
        others_pairs = [p for j, p in enumerate(pairs_trim) if j != hi]
        pos_t, neg_t = pairs_trim[hi]
        row["why_this_differs_from_other_options"] = _differs_sentence(pos_t, neg_t, others_pairs)
        ca_d = clusters[int(row["positive_cluster_id"])]
        cb_d = clusters[int(row["contrast_cluster_id"])]
        lead_feat = _pair_distinguishing_features(ca_d, cb_d)[0]
        pos_line = _squish(str(row.get("positive_intent") or row.get("intent_text") or ""))
        neg_line = _squish(str(row.get("negative_contrast") or ""))
        row["short_user_facing_option_text"] = _squish(
            f"{lead_feat} Preferred lean: {pos_line} Counter-steer / lighten: {neg_line}",
        )[:720]
        del row["_scratch"]

    serial = [{k: v for k, v in h.items() if not str(k).startswith("_")} for h in hyps]
    raw_obj["hypotheses"] = serial
    return json.dumps(raw_obj, ensure_ascii=False, indent=2), hyps


def _hypothesis_lookup(hypotheses: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    keys: dict[str, dict[str, Any]] = {}
    for h in hypotheses:
        hid = str(h.get("hypothesis_id") or h.get("option_id") or "")
        if hid:
            keys[hid] = h
    return keys


def _pick_hypotheses_for_question(
    hypotheses: list[dict[str, Any]],
    target_n: int,
) -> list[dict[str, Any]]:
    """Keep unique `(positive_cluster_id, contrast_cluster_id)` pairs — prioritises diverse positives."""

    if not hypotheses or target_n <= 0:
        return []
    pool = _uniq_pair_pool_ordered(hypotheses)
    return pool[: min(len(pool), target_n)]


def build_mcq_option_text(h: dict[str, Any], *, same_positive_strand_siblings: bool) -> str:
    """Style read + paired intent / negative contrast once + reps; sibling rows lead with differing ease-off artists."""

    feats = list(h.get("distinguishing_features") or ())
    seg_style_parts = [_squish(str(feats[ix])).strip().rstrip(".") for ix in range(min(2, len(feats))) if ix < len(feats)]
    seg_style = "; ".join(p for p in seg_style_parts if p)

    pos_line = sanitize_user_question_copy(_squish(str(h.get("positive_intent") or h.get("intent_text") or "")))
    neg_line = sanitize_user_question_copy(_squish(str(h.get("negative_contrast") or "")))

    rep_pos = " · ".join(_squish(str(x)) for x in (h.get("representative_items") or ())[:3])
    rep_neg = " · ".join(_squish(str(x)) for x in (h.get("contrast_representative_items") or ())[:3])

    chunks: list[str] = []

    # Same positive / different counters: surface the discriminator first so truncation cannot hide it.
    if same_positive_strand_siblings:
        sibling = sanitize_user_question_copy(
            _squish(
                f"Shares the same preferred mood leaning as another option here; differs by easing off tonal cues tied to "
                f"{rep_neg} (pick this if those moods are what you want to dial down)."
            ),
        )
        chunks.append(sibling)

    if seg_style:
        chunks.append(sanitize_user_question_copy(_squish(f"Style read: {seg_style}.")))

    pos_clean = pos_line.rstrip(".").strip()
    neg_clean = neg_line.rstrip(".").strip()
    chunks.append(sanitize_user_question_copy(_squish(f"{pos_clean}. {neg_clean}.")))

    if rep_pos or rep_neg:
        chunks.append(
            sanitize_user_question_copy(
                _squish(f"Representative cues — moods to lean toward: {rep_pos or '—'}; moods to ease off: {rep_neg or '—'}."),
            ),
        )

    merged = sanitize_user_question_copy(_squish(" ".join(c for c in chunks if c)))
    limit = 820
    if len(merged) > limit:
        merged = sanitize_user_question_copy(_squish(merged[:limit].rsplit(" ", 1)[0] + " …"))
    return merged


def generate_choice_question_mock(
    hypotheses: list[dict[str, Any]],
    *,
    cfg: dict[str, Any],
    clusters_payload: list[dict[str, Any]] | None = None,
) -> tuple[str, str, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    vc = cfg or {}
    render_mode = _squish(str(vc.get("mock_mcq_render", "legacy"))) or "legacy"
    clusters_map: dict[int, dict[str, Any]] = {}
    for c in clusters_payload or []:
        try:
            clusters_map[int(c.get("cluster_id"))] = dict(c)
        except Exception:
            continue

    if render_mode == "v4_semantic_labels" or render_mode in ("v5_cluster_contrast", "v51_metadata_summary"):
        configured = max(2, min(5, int(vc.get("choice_options_per_question", 4))))
        floor_hyp = 2
    else:
        configured = max(3, min(5, int(vc.get("choice_options_per_question", 4))))
        floor_hyp = 3
    avail = len(hypotheses)
    target_n = max(floor_hyp, min(configured, max(floor_hyp, avail)))

    disjoint = bool(vc.get("mock_disjoint_rep_pick", False))
    ro_th = float(vc.get("mock_rep_overlap_pick_threshold", 0.32))

    extra_warnings: list[str] = []
    semantic_style_pick: dict[str, Any] = {}

    chosen: list[dict[str, Any]] = []

    if render_mode in ("v5_cluster_contrast", "v51_metadata_summary"):
        cap_sub = min(4, target_n)
        base_h = [dict(x) for x in hypotheses]
        adjusted_hyps, sup_meta_global = apply_hypothesis_style_phrase_support_mock(
            base_h,
            clusters_by_id=clusters_map,
        )
        chosen, v5_meta = _pick_hypotheses_v5_cluster_contrast(
            adjusted_hyps,
            clusters_map=clusters_map,
            target_max=cap_sub,
            reuse_contrast_phrase_jaccard_tol=float(vc.get("mock_v5_contrast_reuse_style_jaccard_tol", 0.45)),
        )

        mxp_pos = _positive_style_pairwise_max_jaccard(chosen)
        mxc_neg = float(v5_meta.get("contrast_style_pairwise_max_jaccard") or 0.0)
        combined_style_ov = float(max(mxp_pos, mxc_neg))
        few_sem = len(chosen) < 3
        few_contrast = bool(v5_meta.get("few_options_after_contrast_filter"))

        semantic_style_pick = {
            **sup_meta_global,
            **v5_meta,
            "style_phrase_support_score": float(
                sup_meta_global.get("hypothesis_avg_style_support_global") or float("nan"),
            ),
            "pairwise_option_semantic_distance": float(v5_meta.get("cluster_contrast_diversity_score") or 0.0),
            "semantic_filter_dropped_count": int(v5_meta.get("contrast_filter_dropped_count", 0)),
            "style_phrase_overlap_rate": combined_style_ov,
            "style_phrase_diversity_score": float(max(0.0, min(1.0, 1.0 - combined_style_ov))),
            "few_options_after_semantic_filter": bool(few_sem),
            "few_options_after_contrast_filter": bool(few_contrast),
            "hypothesis_dual_style_overlap_max_probe": combined_style_ov,
        }
        if few_sem:
            extra_warnings.append("few_options_after_semantic_filter")
        if few_contrast:
            extra_warnings.append("few_options_after_contrast_filter")

        if len(chosen) < 1:
            chosen = _pick_hypotheses_for_question(hypotheses, max(2, min(4, avail)))
            extra_warnings.append("v5_cluster_contrast_degraded_fallback")
    elif render_mode == "v4_semantic_labels":
        cap_sub = min(4, target_n)
        chosen, sem_meta = _pick_hypotheses_style_disjoint(
            hypotheses,
            target_max=cap_sub,
            rep_thresholds=(
                ro_th,
                min(0.55, ro_th + 0.12),
                min(0.62, ro_th + 0.24),
            ),
        )
        phs = [_squish(str(h.get("positive_style_phrase") or "")) for h in chosen]
        mxp = 0.0
        for ii in range(len(phs)):
            for jj in range(ii + 1, len(phs)):
                mxp = max(mxp, _token_jaccard(phs[ii], phs[jj]))
        few_after = len(chosen) < 3
        if few_after:
            extra_warnings.append("few_options_after_semantic_filter")
        semantic_style_pick = {
            "style_phrase_overlap_rate": float(mxp),
            "style_phrase_diversity_score": float(max(0.0, min(1.0, 1.0 - mxp))),
            "semantic_filter_dropped_count": int(sem_meta.get("semantic_filter_dropped_count", 0)),
            "few_options_after_semantic_filter": bool(few_after),
        }

    elif disjoint:
        cap_sub = min(4, target_n)
        chosen = _pick_hypotheses_disjoint_positive_reps(
            hypotheses,
            target_max=cap_sub,
            target_min=3,
            thresholds=(
                ro_th,
                min(0.55, ro_th + 0.12),
                min(0.62, ro_th + 0.24),
            ),
        )
        if len(chosen) > cap_sub:
            chosen = chosen[:cap_sub]
    else:
        chosen = _pick_hypotheses_for_question(hypotheses, target_n)

    preamble_raw = (
        "New suggestions sometimes feel unfamiliar—which stylistic leaning should personalization favor next?\n\n"
        "Artist names are shorthand for mood—they are illustrative cues rather than trivia."
    )
    preamble = sanitize_user_question_copy(preamble_raw)

    pos_counts: dict[int, int] = {}
    for hh in chosen:
        try:
            pki = int(hh["positive_cluster_id"])
        except Exception:
            continue
        pos_counts[pki] = pos_counts.get(pki, 0) + 1

    letters = [chr(ord("A") + k) for k in range(12)]
    opts: list[dict[str, Any]] = []
    omap: dict[str, Any] = {}
    omap_reps: dict[str, Any] = {}
    omap_feats: dict[str, Any] = {}

    moj_accum = 0
    word_counts: list[int] = []
    char_lens: list[int] = []

    for i, h in enumerate(chosen):
        opt_id = letters[i]
        pc = h.get("positive_cluster_id")
        cc = h.get("contrast_cluster_id")
        try:
            pc_i = int(pc)
        except Exception:
            pc_i = -1
        sibling_same_positive = pc_i >= 0 and pos_counts.get(pc_i, 0) > 1
        if render_mode in ("v4_semantic_labels", "v5_cluster_contrast", "v51_metadata_summary"):
            text, hm = build_mcq_option_text_v4(h, variant=i)
            text = sanitize_user_question_copy(_squish(text))
            moj_accum += int(hm.get("mojibake_events") or 0)
            word_counts.append(int(hm.get("word_est") or len(text.split())))
            char_lens.append(len(text))
        elif render_mode == "short_v3":
            text, hm = build_mcq_option_text_short(h)
            text = sanitize_user_question_copy(_squish(text))
            moj_accum += int(hm.get("mojibake_events") or 0)
            word_counts.append(int(hm.get("word_est") or len(text.split())))
            char_lens.append(len(text))
        else:
            text = sanitize_user_question_copy(
                build_mcq_option_text(h, same_positive_strand_siblings=sibling_same_positive),
            )
            char_lens.append(len(text))
            word_counts.append(len(text.split()))
        max_ow = vc.get("max_option_words")
        if max_ow is not None:
            text = sanitize_user_question_copy(_clip_to_max_words(text, int(max_ow)))
        hid = str(h.get("hypothesis_id", ""))
        opt_row: dict[str, Any] = {"option_id": opt_id, "text": text, "maps_to_hypothesis_id": hid}
        for rk in (
            "rank_safety_role",
            "expected_rank_effect",
            "covered_current_top_items",
            "risk_note",
        ):
            if h.get(rk) is not None:
                opt_row[rk] = h.get(rk)
        opts.append(opt_row)
        omap[opt_id] = {"positive_cluster_id": pc, "contrast_cluster_id": cc}
        omap_reps[opt_id] = {
            "representative_items": list(h.get("representative_items") or ()),
            "contrast_representative_items": list(h.get("contrast_representative_items") or ()),
        }
        omap_feats[opt_id] = list(h.get("distinguishing_features") or ())

    unsure_long = sanitize_user_question_copy(
        _squish(str(vc.get("mock_choice_unsure_text_long", "") or "")),
    ) or sanitize_user_question_copy(
        "Not sure — either leaning could work; skip narrowing for now.",
    )
    unsure_short = sanitize_user_question_copy(
        _squish(str(vc.get("mock_choice_unsure_text_short", "") or "")),
    ) or "Not sure — skip narrowing for now."
    unsure_text = (
        unsure_short
        if render_mode in ("short_v3", "v4_semantic_labels", "v5_cluster_contrast", "v51_metadata_summary")
        else unsure_long
    )

    if bool(vc.get("mock_choice_append_unsure", True)) and len(opts) <= 5 and len(opts) < len(letters):
        next_lab = letters[len(opts)]
        uns = {
            "option_id": next_lab,
            "text": unsure_text,
            "maps_to_hypothesis_id": "__none__",
        }
        opts.append(uns)
        omap[next_lab] = {"positive_cluster_id": None, "contrast_cluster_id": None}
        omap_reps[next_lab] = {"representative_items": [], "contrast_representative_items": []}
        omap_feats[next_lab] = []

    bundle = {
        "question": preamble,
        "final_question_text": preamble + "\n\n" + "\n".join([f"{o['option_id']}) {o['text']}" for o in opts]),
        "options": opts,
        "option_to_cluster_mapping": omap,
        "option_to_representatives": omap_reps,
        "option_to_distinguishing_features": omap_feats,
    }
    bundle, user_scrub_touches = scrub_visible_mcq_bundle_user_text(bundle)
    q_final = str(bundle.get("final_question_text") or "")
    opts = bundle.get("options") or opts

    def _dirty_substantive(tx: str) -> int:
        ss = str(tx or "")
        if "Ð" in ss or "Ã" in ss or "\ufffd" in ss or "�" in ss:
            return 1
        return 0

    bad_moj = sum(
        _dirty_substantive(str(o.get("text") or ""))
        for o in opts
        if str(o.get("maps_to_hypothesis_id") or "") not in {"", "__none__"}
    )
    raw = json.dumps(bundle, ensure_ascii=False, indent=2)
    avg_chars = float(sum(char_lens) / max(1, len(char_lens)))
    extra: dict[str, Any] = {
        "chosen_hypothesis_ids": [str(x.get("hypothesis_id")) for x in chosen],
        "choice_preamble": preamble,
        "mock_mcq_render": render_mode,
        "mock_disjoint_rep_pick": disjoint,
        "human_option_audit": {
            "dropped_or_sanitized_item_count": int(moj_accum),
            "option_word_counts": word_counts,
            "substantive_option_char_mean": avg_chars,
            "user_visible_mojibake_scrub_touch_count": int(user_scrub_touches),
            "mojibake_in_user_text_count": int(bad_moj),
        },
    }
    if semantic_style_pick:
        extra["semantic_style_pick"] = semantic_style_pick
    if extra_warnings:
        extra["extra_warnings"] = extra_warnings
    return raw, q_final, opts, omap, extra


def internal_jargon_hits_visible(texts: list[str]) -> int:
    joined = "\n".join(texts).lower()
    hits = 0
    for pat in _JARGON_PATTERNS_UI:
        hits += len(re.findall(pat, joined, flags=re.I))
    return hits


def audit_heuristic_scores(
    hypotheses: list[dict[str, Any]],
    opts: list[dict[str, Any]],
    contrasts: list[ClusterContrast],
    clusters: list[FrontierCluster],
    *,
    preamble: str | None = None,
    min_substantive_mcq: int = 3,
) -> dict[str, Any]:
    texts = [str(o.get("text", "")) for o in opts if str(o.get("maps_to_hypothesis_id", "")) != "__none__"]
    hypo_by_id = _hypothesis_lookup(hypotheses)
    mh: list[dict[str, Any]] = []
    for o in opts:
        hid = str(o.get("maps_to_hypothesis_id") or "")
        if hid == "__none__" or not hid:
            continue
        if hid in hypo_by_id:
            mh.append(hypo_by_id[hid])

    mxjac = 0.0
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            mxjac = max(mxjac, _token_jaccard(texts[i], texts[j]))
    option_semantic_distinctness = float(max(0.0, min(1.0, 1.0 - mxjac)))
    legacy_distinct_alias = option_semantic_distinctness

    reps_sets = [{str(x) for x in (h.get("representative_items") or [])} for h in mh]
    ov_rep = 0.0
    representative_overlap_rate = 0.0
    if reps_sets:
        ratios: list[float] = []
        for i in range(len(reps_sets)):
            for j in range(i + 1, len(reps_sets)):
                a, bbb = reps_sets[i], reps_sets[j]
                if not a or not bbb:
                    continue
                ratios.append(len(a & bbb) / max(1, len(a | bbb)))
        ov_rep = float(max(ratios) if ratios else 0.0)
        representative_overlap_rate = ov_rep

    cov_pos = len(
        {
            int(h["positive_cluster_id"])
            for h in mh
            if isinstance(h.get("positive_cluster_id"), (int, float))
        },
    )
    pair_keys = {(int(h["positive_cluster_id"]), int(h["contrast_cluster_id"])) for h in mh if isinstance(h.get("positive_cluster_id"), (int, float)) and isinstance(h.get("contrast_cluster_id"), (int, float))}
    denom_clusters = len({c.cluster_id for c in clusters}) or 1
    cluster_binding_coverage = float(
        sum(
            1
            for h in mh
            if isinstance(h.get("positive_cluster_id"), (int, float))
            and isinstance(h.get("contrast_cluster_id"), (int, float))
        )
        / max(1, len(mh)),
    )
    cluster_coverage_score = cov_pos / denom_clusters

    id2c = {c.cluster_id: c for c in clusters}
    gaps: list[float] = []
    for cx in contrasts:
        aa = id2c.get(cx.safe_or_reference_cluster_id)
        bb = id2c.get(cx.contrasting_cluster_id)
        if not aa or not bb:
            continue
        gaps.append(abs(aa.mean_belief_score - bb.mean_belief_score))
        gaps.append(abs(aa.overlap_top10_frac - bb.overlap_top10_frac))
    fz_mean = float(np.mean(gaps)) if gaps else float("nan")

    leng = [float(len(str(o.get('text')))) for o in opts]
    avg_len = float(np.mean(leng)) if leng else 0.0
    std_len = float(np.std(leng)) if len(leng) > 1 else 0.0
    ratio = float(std_len / max(avg_len, 1.0))
    option_length_balance = float(max(0.0, min(1.0, 1.0 - min(1.0, ratio * 6.0))))

    visible_all = ([] if preamble is None else [preamble]) + [str(o.get("text")) for o in opts]
    jhits = internal_jargon_hits_visible([_squish(str(x).lower()) for x in visible_all])
    internal_jargon_leakage = float(min(1.0, jhits / 4.0))

    contrastiveness_parts: list[float] = []
    for h in mh:
        feats = len(list(h.get("distinguishing_features") or []))
        neg = len(_squish(str(h.get("negative_contrast") or "")))
        contrastiveness_parts.append(0.33 * feats + min(240.0, float(neg)) / 720.0)
    contrastiveness_score = (
        float(np.mean(contrastiveness_parts)) / 1.67 if contrastiveness_parts else float("nan")
    )
    readable = avg_len >= 80.0 and len(texts) >= 2 and all(len(_squish(str(t))) > 48 for t in texts)
    question_answerability_score = (
        float(1.0 if readable and jhits == 0 else (0.6 if readable else 0.2))
        * (1.0 - internal_jargon_leakage * 0.5)
        * float(min(1.0, 0.3 + len([o for o in opts if str(o.get('maps_to_hypothesis_id'))!='__none__']) * 0.2))
    )

    uniq_pos_pairs = cov_pos >= 3 or len(pair_keys) >= 3

    warnings: list[str] = []
    if len(clusters) < 2:
        warnings.append("too_few_clusters")
    if len(hypotheses) < 2:
        warnings.append("too_few_hypotheses")
    n_mcq_substantive_options = sum(1 for o in opts if str(o.get("maps_to_hypothesis_id", "")) not in {"", "__none__"})
    floor_sub = max(2, min(4, int(min_substantive_mcq)))
    if n_mcq_substantive_options < floor_sub:
        warnings.append("too_few_mcq_hypotheses")
    if ov_rep > 0.38:
        warnings.append("representative_overlap_rate_high")
        warnings.append("high_cross_option_rep_overlap")
    elif ov_rep > 0.33:
        warnings.append("representative_overlap_rate_medium")
    if option_semantic_distinctness < 0.52:
        warnings.append("options_too_similar")
        warnings.append("similar_option_wording")
    elif option_semantic_distinctness < 0.62:
        warnings.append("similar_option_wording")

    cluster_binding_weak = cluster_binding_coverage < 0.9 or cov_pos < 2
    if cluster_binding_weak:
        warnings.append("hypothesis_not_bound_to_cluster_weak")

    if jhits > 0:
        warnings.append("internal_jargon_leakage_detected")

    if not uniq_pos_pairs and len(pair_keys) < len(mh):
        warnings.append("low_contrast_pair_diversity")

    if question_answerability_score < 0.45:
        warnings.append("choice_not_answerable_heuristic")

    if option_length_balance < 0.48:
        warnings.append("mcq_length_imbalance")

    if fz_mean < float("inf") and (math.isnan(fz_mean) or fz_mean < 0.002):
        warnings.append("boundary_signal_flat")

    feats_ok = mh and all(len(list(h.get("distinguishing_features") or [])) >= 2 for h in mh)
    if not feats_ok:
        warnings.append("thin_distinguishing_features")

    scores_out: dict[str, Any] = {
        "are_options_distinct_auto_score": legacy_distinct_alias,
        "option_semantic_distinctness": option_semantic_distinctness,
        "option_overlap_score": ov_rep,
        "representative_overlap_rate": representative_overlap_rate,
        "cluster_coverage_score": float(cluster_coverage_score),
        "cluster_binding_coverage": cluster_binding_coverage,
        "frontier_contrast_score": fz_mean,
        "question_answerability_score": question_answerability_score,
        "internal_jargon_leak_score": internal_jargon_leakage,
        "internal_jargon_hits": jhits,
        "option_length_balance": option_length_balance,
        "contrastiveness_score": contrastiveness_score,
        "uniq_positive_clusters_in_mcq": int(cov_pos),
        "distinct_contrast_pairs_in_mcq": int(len(pair_keys)),
        "warnings": warnings,
    }
    return scores_out


def auto_case_quality(scores: dict[str, Any], hypotheses: list[dict[str, Any]]) -> str:
    warns_raw = scores.get("warnings") or []
    warns = set(str(x) for x in warns_raw)
    if "too_few_clusters" in warns or len(hypotheses) < 2:
        return "bad_case"

    n_sub = float(scores.get("uniq_positive_clusters_in_mcq", 0) or 0)
    n_pairs = float(scores.get("distinct_contrast_pairs_in_mcq", 0) or 0)
    sem = float(scores.get("option_semantic_distinctness", 0) or 0)
    rep_raw = scores.get("representative_overlap_rate")
    if rep_raw is None:
        rep_raw = scores.get("option_overlap_score", 1.0)
    rep_ov = float(rep_raw)
    jleak = float(scores.get("internal_jargon_leak_score", 0) or 0)
    answerable = float(scores.get("question_answerability_score", 0) or 0)

    bad_triggers = {
        "options_too_similar",
        "internal_jargon_leakage_detected",
        "choice_not_answerable_heuristic",
        "too_few_mcq_hypotheses",
        "representative_overlap_rate_high",
    }
    if bad_triggers.intersection(warns):
        return "bad_case"
    if rep_ov > 0.34 or sem < 0.48 or len(hypotheses) < 3:
        return "bad_case"
    uc_raw = scores.get("uniq_positive_clusters_in_mcq")
    if isinstance(uc_raw, (int, float)) and int(uc_raw) == 1 and n_pairs <= 2:
        return "bad_case"

    good_need = (
        n_sub >= 3
        or n_pairs >= 3
    )
    good_sem = sem >= 0.58
    good_rep = rep_ov <= 0.30
    good_jargon = jleak < 0.12
    good_feel = answerable >= 0.55

    if good_need and good_sem and good_rep and good_jargon and good_feel and "thin_distinguishing_features" not in warns:
        return "good_case"

    if sem >= 0.48 and rep_ov <= 0.36 and answerable >= 0.42 and jleak < 0.25:
        return "uncertain_case"
    return "bad_case"
