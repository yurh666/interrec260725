"""Question granularity controls for cluster-bound generation (config-only hook).

Only affects prompt planning / cluster-bound input enrichment — not ranking,
belief update, or candidate retrieval. Uses runtime-visible cluster/item signals only.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from typing import Any

from src.ask.frontier_clustered_v32 import FrontierCluster

VALID_MODES = frozenset({"original", "coarse", "adaptive"})
DEFAULT_COARSE_BASIS = ("genre", "era", "theme", "mood", "style")

ERA_BUCKETS = (
    (1900, 1959, "classic_pre1960"),
    (1960, 1979, "era_1960s_1970s"),
    (1980, 1999, "era_1980s_1990s"),
    (2000, 2019, "era_2000s_2010s"),
    (2020, 2100, "era_modern"),
)

GENRE_THEME_MAP = {
    "action": "theme:action",
    "adventure": "theme:adventure",
    "animation": "theme:family",
    "children": "theme:family",
    "comedy": "theme:comedy",
    "crime": "theme:crime",
    "documentary": "theme:realism",
    "drama": "theme:drama",
    "fantasy": "theme:fantasy",
    "film-noir": "mood:dark",
    "horror": "mood:dark",
    "musical": "style:light",
    "mystery": "theme:mystery",
    "romance": "theme:romance",
    "sci-fi": "theme:sci-fi",
    "thriller": "theme:thriller",
    "war": "theme:serious",
    "western": "theme:adventure",
}


def question_granularity_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    ask = cfg.get("ask") or {}
    qg = dict(ask.get("question_granularity") or {})
    if not qg:
        sc = ask.get("strict_cascade_when_to_ask") or {}
        if sc.get("question_granularity_adaptive"):
            qg = {"enabled": True, "mode": "adaptive"}
    mode = str(qg.get("mode") or "original").strip().lower()
    if mode not in VALID_MODES:
        mode = "original"
    basis = qg.get("coarse_basis") or list(DEFAULT_COARSE_BASIS)
    if not isinstance(basis, list):
        basis = list(DEFAULT_COARSE_BASIS)
    policy = str(qg.get("coarse_selfcheck_policy") or "default").strip().lower()
    return {
        "enabled": bool(qg.get("enabled", False)),
        "mode": mode,
        "coarse_basis": [str(x).strip().lower() for x in basis if str(x).strip()],
        "coarse_selfcheck_policy": policy,
        "require_representative_anchors": bool(qg.get("require_representative_anchors", True)),
        "require_cluster_binding": bool(qg.get("require_cluster_binding", True)),
        "min_anchor_items_per_option": int(qg.get("min_anchor_items_per_option", 2)),
        "coarse_prompt_version": str(qg.get("coarse_prompt_version") or "v2"),
        "keep_choice_format": bool(qg.get("keep_choice_format", True)),
        "keep_option_balance_check": bool(qg.get("keep_option_balance_check", True)),
        "log_granularity": bool(qg.get("log_granularity", True)),
        "adaptive_dispersion_threshold": float(qg.get("adaptive_dispersion_threshold", 0.35)),
        "adaptive_min_clusters": int(qg.get("adaptive_min_clusters", 3)),
    }


def _parse_genres(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip().lower() for x in raw if str(x).strip()]
    text = str(raw)
    parts = re.split(r"[|,;/]", text)
    return [p.strip().lower() for p in parts if p.strip()]


def _era_bucket(year: int | None) -> str | None:
    if year is None or year < 1900:
        return None
    for lo, hi, label in ERA_BUCKETS:
        if lo <= year <= hi:
            return label
    return None


def _item_year(row: dict[str, Any]) -> int | None:
    for key in ("year", "release_year", "movie_year"):
        val = row.get(key)
        if val is None:
            continue
        try:
            return int(val)
        except (TypeError, ValueError):
            continue
    return None


def extract_item_facets(item_row: dict[str, Any], basis: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {b: [] for b in basis}
    genres = _parse_genres(item_row.get("genre") or item_row.get("genres"))
    if "genre" in basis:
        out["genre"] = genres[:6]
    if "era" in basis:
        yr = _item_year(item_row)
        bucket = _era_bucket(yr)
        if bucket:
            out["era"].append(bucket)
        if yr and yr < 1980:
            out["era"].append("classic")
        elif yr and yr >= 2000:
            out["era"].append("modern")
    if "theme" in basis:
        themes = []
        for g in genres:
            if g in GENRE_THEME_MAP:
                themes.append(GENRE_THEME_MAP[g])
        out["theme"] = list(dict.fromkeys(themes))[:6]
    if "mood" in basis:
        moods = []
        for g in genres:
            if g in ("horror", "film-noir", "thriller"):
                moods.append("dark_intense")
            elif g in ("comedy", "musical", "animation"):
                moods.append("light_entertaining")
            elif g in ("drama", "war"):
                moods.append("serious_emotional")
        out["mood"] = list(dict.fromkeys(moods))[:4]
    if "style" in basis:
        styles = []
        for g in genres:
            if g in ("documentary", "drama", "war"):
                styles.append("serious_narrative")
            elif g in ("action", "adventure", "sci-fi"):
                styles.append("spectacle_driven")
            elif g in ("comedy", "romance"):
                styles.append("character_light")
        out["style"] = list(dict.fromkeys(styles))[:4]
    return out


def aggregate_cluster_coarse_facets(
    cluster: FrontierCluster,
    item_lookup: dict[str, dict[str, Any]],
    basis: list[str],
) -> dict[str, Any]:
    genre_ctr: Counter[str] = Counter()
    facet_lists: dict[str, list[str]] = {b: [] for b in basis}
    years: list[int] = []
    for iid in list(cluster.member_ids)[:24]:
        row = item_lookup.get(str(iid), {})
        facets = extract_item_facets(row, basis)
        for b in basis:
            facet_lists[b].extend(facets.get(b) or [])
        for g in facets.get("genre") or []:
            genre_ctr[g] += 1
        yr = _item_year(row)
        if yr:
            years.append(yr)
    dominant_genres = [g for g, _ in genre_ctr.most_common(4)]
    era_label = None
    if years:
        med = int(statistics.median(years))
        era_label = _era_bucket(med)
    return {
        "dominant_genres": dominant_genres,
        "facet_tags": {b: list(dict.fromkeys(facet_lists.get(b) or []))[:6] for b in basis},
        "era_median": int(statistics.median(years)) if years else None,
        "era_bucket": era_label,
        "genre_entropy": _entropy(list(genre_ctr.values())),
    }


def _entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log2(p)
    return float(ent)


def compute_cluster_dispersion(clusters: list[FrontierCluster]) -> dict[str, Any]:
    if not clusters:
        return {"dispersion_score": 0.0, "n_clusters": 0, "overlap_std": 0.0}
    overlaps = [float(c.overlap_top10_frac) for c in clusters if math.isfinite(c.overlap_top10_frac)]
    overlap_std = float(statistics.pstdev(overlaps)) if len(overlaps) > 1 else 0.0
    mean_unc = [float(c.mean_uncertainty) for c in clusters if math.isfinite(c.mean_uncertainty)]
    unc_std = float(statistics.pstdev(mean_unc)) if len(mean_unc) > 1 else 0.0
    low_overlap_frac = sum(1 for o in overlaps if o < 0.05) / max(len(overlaps), 1)
    dispersion = min(1.0, 0.5 * overlap_std + 0.3 * unc_std + 0.2 * low_overlap_frac)
    return {
        "dispersion_score": round(dispersion, 4),
        "n_clusters": len(clusters),
        "overlap_std": round(overlap_std, 4),
        "low_overlap_cluster_frac": round(low_overlap_frac, 4),
    }


def resolve_effective_granularity_mode(
    qg: dict[str, Any],
    clusters: list[FrontierCluster],
    item_lookup: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    if not qg.get("enabled"):
        return "original", {"reason": "granularity_disabled"}
    mode = str(qg.get("mode") or "original")
    if mode in ("original", "coarse"):
        return mode, {"reason": f"fixed_mode_{mode}"}
    disp = compute_cluster_dispersion(clusters)
    genre_entropies = []
    for cl in clusters:
        agg = aggregate_cluster_coarse_facets(cl, item_lookup, qg.get("coarse_basis") or list(DEFAULT_COARSE_BASIS))
        genre_entropies.append(float(agg.get("genre_entropy") or 0.0))
    mean_ge = statistics.mean(genre_entropies) if genre_entropies else 0.0
    threshold = float(qg.get("adaptive_dispersion_threshold", 0.35))
    min_clusters = int(qg.get("adaptive_min_clusters", 3))
    use_coarse = (
        disp["dispersion_score"] >= threshold
        or disp["n_clusters"] >= min_clusters
        and disp.get("low_overlap_cluster_frac", 0) >= 0.4
        or mean_ge >= 1.8
    )
    effective = "coarse" if use_coarse else "original"
    audit = {
        "reason": "adaptive_dispersion" if use_coarse else "adaptive_concentrated",
        "dispersion": disp,
        "mean_cluster_genre_entropy": round(mean_ge, 4),
        "threshold": threshold,
    }
    return effective, audit


def coarse_prompt_suffix(domain: str, basis: list[str], *, version: str = "v2") -> str:
    if version == "v2" and domain == "movielens_movie":
        return coarse_prompt_suffix_v2_movielens(basis)
    basis_txt = ", ".join(basis) if basis else ", ".join(DEFAULT_COARSE_BASIS)
    dom_hint = {
        "movielens_movie": "movie genres, eras, themes, moods, and viewing styles",
        "lastfm_music": "music genres, eras, moods, and listening styles",
    }.get(domain, "domain facets")
    return (
        "\n\nQUESTION GRANULARITY = COARSE:\n"
        f"- Frame the question and ALL options at COARSE granularity using {dom_hint}.\n"
        f"- Emphasize these facet dimensions: {basis_txt}.\n"
        "- Each option must contrast broad taste directions while staying grounded in bound cluster items.\n"
        "- Still bind each option to its bound_cluster_id and representative_item_names.\n"
    )


def coarse_prompt_suffix_v2_movielens(basis: list[str]) -> str:
    basis_txt = ", ".join(basis) if basis else ", ".join(DEFAULT_COARSE_BASIS)
    return (
        "\n\nQUESTION GRANULARITY = COARSE V2 (facet + representative anchors):\n"
        f"- Facet dimensions: {basis_txt}.\n"
        "- Each option MUST combine: (1) a coarse facet label in cluster_phrase, AND "
        "(2) 2-3 exact movie titles copied from representative_item_names in option_text.\n"
        "- option_text format example: "
        "\"偏向轻松喜剧/家庭向电影，例如 Movie A (1995), Movie B (2001), Movie C (1988)\".\n"
        "- Do NOT emit facet-only options (e.g. just \"Comedy\"). Do NOT omit representative titles.\n"
        "- Copy representative_item_names verbatim into option_text and positive_item_anchors (2-4 names).\n"
        "- preserve/boundary/explore roles unchanged; explore must bind to the high-uncertainty cluster provided.\n"
        "- question_text: coarse taste-direction question (not item-pair micro-preference).\n"
        "- distinction_from_other_options must explain facet contrast vs other options.\n"
    )


def enrich_cluster_bound_input(
    input_payload: dict[str, Any],
    *,
    effective_mode: str,
    qg: dict[str, Any],
    clusters: list[FrontierCluster],
    item_lookup: dict[str, dict[str, Any]],
    adaptive_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = dict(input_payload)
    out["question_granularity"] = {
        "enabled": bool(qg.get("enabled")),
        "configured_mode": qg.get("mode"),
        "effective_mode": effective_mode,
        "coarse_basis": list(qg.get("coarse_basis") or []),
        "coarse_selfcheck_policy": qg.get("coarse_selfcheck_policy"),
        "coarse_prompt_version": qg.get("coarse_prompt_version"),
        "adaptive_audit": adaptive_audit,
    }
    if effective_mode != "coarse":
        return out
    cluster_by_id = {int(c.cluster_id): c for c in clusters}
    enriched_candidates = []
    for cand in out.get("candidate_options") or []:
        if not isinstance(cand, dict):
            continue
        row = dict(cand)
        try:
            cid = int(str(row.get("bound_cluster_id")).replace("cluster_", ""))
        except Exception:
            cid = -1
        cl = cluster_by_id.get(cid)
        if cl is not None:
            agg = aggregate_cluster_coarse_facets(
                cl,
                item_lookup,
                qg.get("coarse_basis") or list(DEFAULT_COARSE_BASIS),
            )
            row["coarse_facets"] = agg.get("facet_tags") or {}
            row["dominant_genres"] = agg.get("dominant_genres") or []
            row["era_bucket"] = agg.get("era_bucket")
            row["coarse_direction_hint"] = _coarse_direction_hint(agg)
        enriched_candidates.append(row)
    out["candidate_options"] = enriched_candidates
    out["question_context"] = dict(out.get("question_context") or {})
    out["question_context"]["granularity_instruction"] = (
        "Use coarse-grained facet contrasts across options; avoid fine item-level wording."
    )
    return out


def _coarse_direction_hint(agg: dict[str, Any]) -> str:
    genres = agg.get("dominant_genres") or []
    era = agg.get("era_bucket") or ""
    themes = (agg.get("facet_tags") or {}).get("theme") or []
    parts = []
    if genres:
        parts.append("/".join(genres[:3]))
    if era:
        parts.append(str(era))
    if themes:
        parts.append(themes[0].replace("theme:", ""))
    return " · ".join(parts) if parts else "mixed catalog cluster"


def granularity_telemetry(
    qg: dict[str, Any],
    effective_mode: str,
    adaptive_audit: dict[str, Any] | None,
    input_payload: dict[str, Any],
) -> dict[str, Any]:
    if not qg.get("log_granularity"):
        return {}
    coarse_facets = []
    for cand in input_payload.get("candidate_options") or []:
        if isinstance(cand, dict) and cand.get("coarse_facets"):
            coarse_facets.append(
                {
                    "option_id": cand.get("option_id"),
                    "coarse_facets": cand.get("coarse_facets"),
                    "dominant_genres": cand.get("dominant_genres"),
                }
            )
    return {
        "question_granularity_enabled": bool(qg.get("enabled")),
        "question_granularity_configured_mode": qg.get("mode"),
        "question_granularity_effective_mode": effective_mode,
        "question_granularity_coarse_basis": list(qg.get("coarse_basis") or []),
        "question_granularity_adaptive_audit": adaptive_audit,
        "question_granularity_coarse_facets_per_option": coarse_facets,
    }
