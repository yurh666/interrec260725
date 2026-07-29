"""Cheap pre-screening before full Monte Carlo VOI (question selection).

Does not replace ``select_best_question`` / ``expected_posterior_entropy``; only
decides whether to invoke them on a given turn.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _choice_softmax_fields(
    p: np.ndarray,
    idx_d: np.ndarray,
    p_D_raw: np.ndarray,
) -> dict[str, Any]:
    """Diagnostics from softmax p over hypotheses (global + dominant subset)."""
    M = int(p.shape[0])
    ps_desc = np.sort(p)[::-1]
    top1 = float(ps_desc[0])
    top2 = float(ps_desc[1]) if M > 1 else 0.0
    entropy = float(-np.sum(p * np.log(np.clip(p, 1e-300, 1.0))))
    return {
        "choice_prob_top1": top1,
        "choice_prob_top2": top2,
        "choice_prob_entropy": entropy,
        "dominant_set_indices": [int(x) for x in idx_d.tolist()],
        "dominant_set_probs": [float(x) for x in p_D_raw.tolist()],
    }


def dominant_pairwise_hrv_score(
    H: np.ndarray,
    Sigma: np.ndarray,
    mu: np.ndarray,
    tau: float,
    mass: float = 0.8,
) -> dict[str, Any]:
    """Dominant-set pairwise posterior variance of utility differences (cheap, no MC).

    Returns a dict with ``score`` (float), diagnostics, and ``ok`` (bool).
    When ``ok`` is False, callers should fall back to full VOI.
    """
    out_base: dict[str, Any] = {
        "score": 0.0,
        "ok": False,
        "dominant_set_size": 0,
        "dominant_set_mass": 0.0,
        "degenerate_single_hypothesis": False,
        "degenerate_no_pairs": False,
        "fallback_reason": None,
    }

    if tau <= 0.0 or not np.isfinite(tau):
        out_base["fallback_reason"] = "invalid_tau"
        return out_base

    if H.ndim != 2:
        out_base["fallback_reason"] = "h_bad_ndim"
        return out_base

    M, d_h = H.shape
    if M == 0:
        out_base["fallback_reason"] = "empty_hypothesis"
        return out_base

    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)

    if mu.shape[0] != d_h:
        out_base["fallback_reason"] = "shape_mismatch_mu"
        return out_base

    if Sigma.shape != (d_h, d_h):
        out_base["fallback_reason"] = "shape_mismatch_sigma"
        return out_base

    if not (0.0 < mass <= 1.0):
        out_base["fallback_reason"] = "invalid_mass"
        return out_base

    # Predicted choice logits under mean preference (stable softmax)
    logits = (H @ mu) / tau
    logits_max = float(np.max(logits))
    exp_logits = np.exp(logits - logits_max)
    denom = float(np.sum(exp_logits))
    if denom <= 0.0 or not np.isfinite(denom):
        out_base["fallback_reason"] = "softmax_unstable"
        return out_base

    p = exp_logits / denom
    if not np.all(np.isfinite(p)):
        out_base["fallback_reason"] = "non_finite_probs"
        return out_base

    order = np.argsort(-p)
    p_sorted = p[order]
    csum = np.cumsum(p_sorted)
    k = int(np.searchsorted(csum, mass, side="left") + 1)
    k = max(1, min(k, M))

    idx_d = order[:k]
    H_D = H[idx_d]
    p_D_raw = p[idx_d]
    mass_actual = float(np.sum(p_D_raw))

    choice_fields = _choice_softmax_fields(p, idx_d, p_D_raw)

    if k == 1:
        out = {
            "score": 0.0,
            "ok": True,
            "dominant_set_size": 1,
            "dominant_set_mass": mass_actual,
            "degenerate_single_hypothesis": True,
            "degenerate_no_pairs": True,
            "fallback_reason": None,
        }
        out.update(choice_fields)
        return out

    p_D = p_D_raw / float(np.sum(p_D_raw))
    if not np.all(np.isfinite(p_D)):
        out_base["fallback_reason"] = "dominant_probs_non_finite"
        return out_base

    # Pairwise (h_i - h_j)^T Sigma (h_i - h_j), i < j within dominant set
    n_d = H_D.shape[0]
    acc_num = 0.0
    acc_w = 0.0
    for i in range(n_d):
        for j in range(i + 1, n_d):
            diff = H_D[i] - H_D[j]
            u_ij = float(diff @ Sigma @ diff)
            if not np.isfinite(u_ij):
                out_base["fallback_reason"] = "non_finite_pair_variance"
                return out_base
            w = float(p_D[i] * p_D[j])
            acc_num += w * u_ij
            acc_w += w

    if acc_w <= 0.0:
        out = {
            "score": 0.0,
            "ok": True,
            "dominant_set_size": k,
            "dominant_set_mass": mass_actual,
            "degenerate_single_hypothesis": False,
            "degenerate_no_pairs": True,
            "fallback_reason": None,
        }
        out.update(choice_fields)
        return out

    score = acc_num / acc_w
    if not np.isfinite(score):
        out_base["fallback_reason"] = "non_finite_score"
        return out_base

    out = {
        "score": float(score),
        "ok": True,
        "dominant_set_size": k,
        "dominant_set_mass": mass_actual,
        "degenerate_single_hypothesis": False,
        "degenerate_no_pairs": False,
        "fallback_reason": None,
    }
    out.update(choice_fields)
    return out


def empty_question_selection() -> dict[str, Any]:
    """Same structure as ``select_best_question`` when there are no hypotheses."""
    return {"best_ids": [], "ig": 0.0, "option_probs": [], "candidate_count": 0}
