from __future__ import annotations

import numpy as np

from src.utils.math import softmax


def choice_probs_given_theta(
    theta: np.ndarray,
    option_vectors: np.ndarray,
    tau: float = 0.2,
    none_logit: float | None = None,
) -> np.ndarray:
    """Softmax choice probability over n options (+ optional none).

    Returns probability vector of length n (or n+1 if none_logit is provided).
    """
    logits = (option_vectors @ theta) / max(tau, 1e-8)
    if none_logit is not None:
        logits = np.append(logits, none_logit)
    return softmax(logits)


def marginal_choice_probs(
    mu: np.ndarray,
    Sigma: np.ndarray,
    option_vectors: np.ndarray,
    tau: float = 0.2,
    none_logit: float | None = None,
    mc_samples: int = 128,
) -> np.ndarray:
    """Monte-Carlo estimate of P(option_i) = E_{theta~N(mu,Sigma)}[P(option_i|theta)].

    Returns probability vector of length n_options (+ none if none_logit is set).
    """
    samples = np.random.multivariate_normal(mu, Sigma, size=mc_samples)
    probs_sum = None
    for theta in samples:
        p = choice_probs_given_theta(theta, option_vectors, tau, none_logit)
        if probs_sum is None:
            probs_sum = p.copy()
        else:
            probs_sum += p
    if probs_sum is None:
        n = len(option_vectors) + (1 if none_logit is not None else 0)
        return np.ones(n) / n
    return probs_sum / mc_samples


def none_logit(
    mu: np.ndarray,
    option_vectors: np.ndarray,
    tau: float = 0.2,
    none_bias: float = 0.0,
    none_threshold: float = 0.25,
) -> float:
    sims = option_vectors @ mu
    max_sim = float(np.max(sims)) if len(sims) > 0 else 0.0
    return none_bias + (none_threshold - max_sim) / max(tau, 1e-8)
