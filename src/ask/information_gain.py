from __future__ import annotations

from typing import Any

import numpy as np

from src.ask.choice_model import marginal_choice_probs, none_logit
from src.belief.belief_state import BeliefState
from src.belief.laplace_update import laplace_choice_update


def expected_posterior_entropy(
    belief: BeliefState,
    option_vectors: np.ndarray,
    tau: float = 0.2,
    none_bias: float = 0.0,
    none_threshold: float = 0.25,
    alpha_none: float = 1.2,
    mc_samples: int = 64,
    laplace_steps: int = 3,
    inv_sigma: np.ndarray | None = None,
    theta_samples: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute expected posterior entropy (and IG) for a given question.

    Optimisations
    -------------
    * `inv_sigma` is pre-computed outside and shared across subset evaluations.
    * `theta_samples` are pre-drawn outside and reused, eliminating the
      expensive multivariate_normal call per subset.
    * Posterior entropy is computed via logdet (Cholesky), avoiding a full
      matrix inversion O(d³) per option.
    """
    n_opts = len(option_vectors)
    nl = none_logit(belief.mu, option_vectors, tau, none_bias, none_threshold)

    # ── Marginal choice probabilities via pre-drawn samples ──────────────────
    if theta_samples is not None:
        from src.ask.choice_model import choice_probs_given_theta
        probs_sum = None
        for theta in theta_samples:
            p = choice_probs_given_theta(theta, option_vectors, tau, nl)
            probs_sum = p.copy() if probs_sum is None else probs_sum + p
        marg_probs = (probs_sum / len(theta_samples)) if probs_sum is not None else np.ones(n_opts + 1) / (n_opts + 1)
    else:
        marg_probs = marginal_choice_probs(belief.mu, belief.Sigma, option_vectors, tau, nl, mc_samples)

    prior_h = belief.entropy()
    expected_post_h = 0.0
    posterior_entropies: list[float] = []

    for i in range(n_opts):
        h_post = laplace_choice_update(
            belief, option_vectors, i, tau, alpha_none, laplace_steps,
            inv_sigma=inv_sigma, entropy_only=True,
        )
        posterior_entropies.append(float(h_post))
        expected_post_h += float(marg_probs[i]) * float(h_post)

    # none branch
    h_none = laplace_choice_update(
        belief, option_vectors, None, tau, alpha_none, laplace_steps,
        inv_sigma=inv_sigma, entropy_only=True,
    )
    posterior_entropies.append(float(h_none))
    expected_post_h += float(marg_probs[n_opts]) * float(h_none)

    ig = prior_h - expected_post_h
    return {
        "ig": float(ig),
        "prior_entropy": float(prior_h),
        "option_probs": marg_probs.tolist(),
        "posterior_entropies": posterior_entropies,
    }
