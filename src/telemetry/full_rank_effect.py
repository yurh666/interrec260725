"""Full-catalog rank effect telemetry for ask-turn audit (not guard input)."""

from __future__ import annotations

from typing import Any

from src.recommendation.recommender import BeliefRecommender


def _target_items_from_context(
    *,
    theta_star_items: list[str] | None,
    future_test_items: list[str] | None,
) -> list[str]:
    if theta_star_items:
        return [str(x) for x in theta_star_items]
    if future_test_items:
        return [str(x) for x in future_test_items]
    return []


def _rank_map(ranked: list[tuple[str, float]]) -> dict[str, int]:
    return {item_id: idx + 1 for idx, (item_id, _score) in enumerate(ranked)}


def _score_map(ranked: list[tuple[str, float]]) -> dict[str, float]:
    return {item_id: float(score) for item_id, score in ranked}


def best_target_rank(targets: list[str], ranks: dict[str, int]) -> int | None:
    vals = [ranks[t] for t in targets if t in ranks]
    return min(vals) if vals else None


def best_target_score(targets: list[str], scores: dict[str, float]) -> float | None:
    vals = [scores[t] for t in targets if t in scores]
    return float(max(vals)) if vals else None


def compute_full_rank_snapshot(
    recommender: BeliefRecommender,
    belief: Any,
    *,
    exclude: set[str],
    target_items: list[str],
) -> dict[str, Any]:
    ranked = recommender.rank_all(belief, exclude=exclude)
    ranks = _rank_map(ranked)
    scores = _score_map(ranked)
    targets = [str(x) for x in target_items]
    best_rank = best_target_rank(targets, ranks)
    per_target = {
        t: {"rank": ranks.get(t), "score": scores.get(t)} for t in targets
    }
    return {
        "catalog_size_ranked": len(ranked),
        "best_target_rank": best_rank,
        "best_target_score": best_target_score(targets, scores),
        "target_ranks": per_target,
        "hit_at_10": bool(best_rank is not None and best_rank <= 10),
        "hit_at_100": bool(best_rank is not None and best_rank <= 100),
    }


def top10_jaccard(pre_top10: list[str], post_top10: list[str]) -> float | None:
    a, b = set(pre_top10), set(post_top10)
    union = len(a | b)
    if union == 0:
        return None
    return len(a & b) / union


def classify_top10_only_effect(
    *,
    pre_top10: list[str],
    post_top10: list[str],
    targets: list[str],
    pre_hit10: float | None,
    post_hit10: float | None,
) -> dict[str, Any]:
    j = top10_jaccard(pre_top10, post_top10)
    disp = 1.0 - j if j is not None else None
    pre_ranks = _rank_map([(x, 0.0) for x in pre_top10])
    post_ranks = _rank_map([(x, 0.0) for x in post_top10])
    rb = best_target_rank(targets, pre_ranks)
    ra = best_target_rank(targets, post_ranks)
    hit_1_to_0 = (
        pre_hit10 is not None
        and post_hit10 is not None
        and pre_hit10 > 0
        and post_hit10 <= 0
    )
    hit_0_to_1 = (
        pre_hit10 is not None
        and post_hit10 is not None
        and pre_hit10 <= 0
        and post_hit10 > 0
    )
    dropped = rb is not None and ra is None
    entered = rb is None and ra is not None
    worsened = rb is not None and ra is not None and ra >= rb + 5
    improved = rb is not None and ra is not None and ra < rb
    large = bool(disp is not None and disp >= 0.70)
    harmful = bool(hit_1_to_0 or dropped or worsened or (large and (dropped or worsened)))
    useful = bool(hit_0_to_1 or entered or improved)
    return {
        "label_source": "top10_only",
        "top10_jaccard": round(j, 4) if j is not None else None,
        "displacement_magnitude": round(disp, 4) if disp is not None else None,
        "large_displacement": large,
        "rank_before_top10": rb,
        "rank_after_top10": ra,
        "hit_1_to_0": hit_1_to_0,
        "hit_0_to_1": hit_0_to_1,
        "is_harmful_like": harmful,
        "is_useful_like": useful,
    }


def classify_full_rank_effect(
    *,
    pre_snapshot: dict[str, Any],
    post_snapshot: dict[str, Any],
    pre_top10: list[str],
    post_top10: list[str],
) -> dict[str, Any]:
    rb = pre_snapshot.get("best_target_rank")
    ra = post_snapshot.get("best_target_rank")
    sb = pre_snapshot.get("best_target_score")
    sa = post_snapshot.get("best_target_score")
    rank_delta = (ra - rb) if isinstance(rb, int) and isinstance(ra, int) else None
    score_delta = (sa - sb) if isinstance(sb, (int, float)) and isinstance(sa, (int, float)) else None

    pre_hit10 = bool(pre_snapshot.get("hit_at_10"))
    post_hit10 = bool(post_snapshot.get("hit_at_10"))
    hit_1_to_0 = pre_hit10 and not post_hit10
    hit_0_to_1 = not pre_hit10 and post_hit10

    pre_hit100 = bool(pre_snapshot.get("hit_at_100"))
    post_hit100 = bool(post_snapshot.get("hit_at_100"))
    dropped_top100 = pre_hit100 and not post_hit100
    entered_top100 = not pre_hit100 and post_hit100

    worsened = isinstance(rank_delta, int) and rank_delta >= 100
    improved = isinstance(rank_delta, int) and rank_delta <= -20

    j = top10_jaccard(pre_top10, post_top10)
    disp = 1.0 - j if j is not None else None
    large = bool(disp is not None and disp >= 0.70)

    harmful = bool(
        hit_1_to_0
        or dropped_top100
        or worsened
        or (large and isinstance(rank_delta, int) and rank_delta >= 20)
    )
    useful = bool(hit_0_to_1 or entered_top100 or improved)

    return {
        "label_source": "full_rank",
        "best_target_rank_before": rb,
        "best_target_rank_after": ra,
        "target_rank_delta": rank_delta,
        "best_target_score_before": sb,
        "best_target_score_after": sa,
        "target_score_delta": score_delta,
        "hit_at_10_before": pre_hit10,
        "hit_at_10_after": post_hit10,
        "hit_at_100_before": pre_hit100,
        "hit_at_100_after": post_hit100,
        "top10_jaccard": round(j, 4) if j is not None else None,
        "displacement_magnitude": round(disp, 4) if disp is not None else None,
        "large_displacement": large,
        "hit_1_to_0": hit_1_to_0,
        "hit_0_to_1": hit_0_to_1,
        "is_harmful_like": harmful,
        "is_useful_like": useful,
    }
