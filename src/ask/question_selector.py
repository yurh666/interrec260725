from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from src.ask.information_gain import expected_posterior_entropy
from src.belief.belief_state import BeliefState
from src.belief.laplace_update import fast_inv_and_logdet, fast_sample


def select_best_question(
    belief: BeliefState,
    hypotheses_vectorized: list[dict[str, Any]],
    n_options: int = 3,
    tau: float = 0.2,
    none_bias: float = 0.0,
    none_threshold: float = 0.25,
    alpha_none: float = 1.2,
    mc_samples: int = 64,
) -> dict[str, Any]:
    """Select the hypothesis subset that maximises expected IG.

    Shared-state optimisations
    --------------------------
    * inv(Sigma) computed once via Cholesky (scipy, ~0.5ms vs np 400ms).
    * MC samples drawn once via Cholesky transform (~8ms vs 240ms for mvnormal).
    * posterior entropy computed via logdet only, no matrix inversion.
    """
    M = len(hypotheses_vectorized)
    if M == 0:
        return {
            "best_ids": [],
            "ig": 0.0,
            "option_probs": [],
            "candidate_count": 0,
            "profiling": {
                "expected_posterior_entropy_outer_calls": 0,
                "mc_samples": int(mc_samples),
                "hypothesis_count": 0,
                "subset_size_n": 0,
                "candidate_combo_evaluations": 0,
            },
        }

    n = min(n_options, M)

    # Pre-compute shared resources once
    inv_sigma, _, cho_sigma = fast_inv_and_logdet(belief.Sigma, belief.jitter)
    try:
        from scipy import linalg as spla
        chol_L = spla.cholesky(belief.Sigma + np.eye(len(belief.mu)) * belief.jitter, lower=True, check_finite=False)
    except Exception:
        chol_L = np.linalg.cholesky(belief.Sigma + np.eye(len(belief.mu)) * belief.jitter)
    theta_samples = fast_sample(belief.mu, chol_L, mc_samples)

    epe_outer_calls = 0

    def eval_subset(subset: list[dict[str, Any]]) -> dict[str, Any]:
        nonlocal epe_outer_calls
        epe_outer_calls += 1
        vecs = np.stack([h["vector"] for h in subset])
        return expected_posterior_entropy(
            belief, vecs, tau, none_bias, none_threshold, alpha_none, mc_samples,
            inv_sigma=inv_sigma, theta_samples=theta_samples,
        )

    if M <= 8:
        best_ids: list[str] = []
        best_ig = -1e9
        best_result: dict[str, Any] = {}
        n_combos = 0
        for combo in combinations(range(M), n):
            subset = [hypotheses_vectorized[i] for i in combo]
            result = eval_subset(subset)
            n_combos += 1
            if result["ig"] > best_ig:
                best_ig = result["ig"]
                best_ids = [h["hypothesis_id"] for h in subset]
                best_result = result
    else:
        single_igs = [eval_subset([h])["ig"] for h in hypotheses_vectorized]
        selected_indices = [int(np.argmax(single_igs))]
        while len(selected_indices) < n:
            remaining = [i for i in range(M) if i not in selected_indices]
            if not remaining:
                break
            sub = [hypotheses_vectorized[i] for i in selected_indices]
            gains = [eval_subset(sub + [hypotheses_vectorized[j]])["ig"] for j in remaining]
            selected_indices.append(remaining[int(np.argmax(gains))])
        subset = [hypotheses_vectorized[i] for i in selected_indices]
        best_result = eval_subset(subset)
        best_ig = best_result["ig"]
        best_ids = [h["hypothesis_id"] for h in subset]
        n_combos = M + len(selected_indices) * len(selected_indices)

    return {
        "best_ids": best_ids,
        "ig": best_ig,
        "option_probs": best_result.get("option_probs", []),
        "candidate_count": n_combos,
        "profiling": {
            "expected_posterior_entropy_outer_calls": int(epe_outer_calls),
            "mc_samples": int(mc_samples),
            "hypothesis_count": int(M),
            "subset_size_n": int(n),
            "candidate_combo_evaluations": int(n_combos),
        },
    }
