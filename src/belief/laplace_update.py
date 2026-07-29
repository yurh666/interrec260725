from __future__ import annotations

import numpy as np
from scipy import linalg as spla

from src.belief.belief_state import BeliefState
from src.utils.math import softmax

# ── scipy.linalg wrappers (scipy BLAS is ~2000x faster than np.linalg on
#    this system due to numpy using scipy_openblas64 with 64-bit int indexing).
# ─────────────────────────────────────────────────────────────────────────────


def _cho_inv(A: np.ndarray) -> tuple[object, np.ndarray]:
    """Return (cho_factor, A^{-1}) using Cholesky. ~2000x faster than np.linalg.inv."""
    c = spla.cho_factor(A, check_finite=False)
    inv_A = spla.cho_solve(c, np.eye(A.shape[0]), check_finite=False)
    return c, inv_A


def _cho_solve_vec(c: object, v: np.ndarray) -> np.ndarray:
    return spla.cho_solve(c, v, check_finite=False)


def _cho_logdet(c: object) -> float:
    """log|A| from its Cholesky factor (c[0] is the lower triangle)."""
    return 2.0 * float(np.sum(np.log(np.diag(c[0]))))


def fast_inv_and_logdet(A: np.ndarray, jitter: float = 1e-6) -> tuple[np.ndarray, float, object]:
    """Compute A^{-1} and log|A| efficiently using Cholesky.

    Returns (inv_A, logdet, cho_factor).
    Adds jitter to diagonal for numerical stability.
    """
    A_reg = A + np.eye(A.shape[0]) * jitter
    c = spla.cho_factor(A_reg, check_finite=False)
    inv_A = spla.cho_solve(c, np.eye(A.shape[0]), check_finite=False)
    logdet = _cho_logdet(c)
    return inv_A, logdet, c


def fast_sample(mu: np.ndarray, chol_L: np.ndarray, n: int) -> np.ndarray:
    """Draw n samples from N(mu, L L^T) without calling multivariate_normal.

    Uses pre-computed lower-triangular Cholesky L.
    ~30x faster than np.random.multivariate_normal on this system.
    """
    Z = np.random.randn(n, len(mu))
    return mu + Z @ chol_L.T


# ─────────────────────────────────────────────────────────────────────────────


def laplace_choice_update(
    belief: BeliefState,
    option_vectors: np.ndarray,
    selected_index: int | None,
    tau: float = 0.2,
    alpha_none: float = 1.2,
    steps: int = 3,
    inv_sigma: np.ndarray | None = None,
    cho_sigma: object | None = None,
    entropy_only: bool = False,
) -> "BeliefState | float":
    """Laplace-approximate posterior after user selects option `selected_index`.

    Parameters
    ----------
    inv_sigma : pre-computed inv(Sigma). Pass from outside for efficiency.
    cho_sigma : scipy cho_factor of Sigma. Used for fast linear solves.
    entropy_only : if True, return H[posterior] as a float — avoids forming
        the full posterior covariance (expensive matrix inversion).
    """
    d = len(belief.mu)
    jitter = belief.jitter

    if selected_index is None:
        Sigma_new = belief.Sigma * float(alpha_none)
        if entropy_only:
            _, logdet, _ = fast_inv_and_logdet(Sigma_new, jitter)
            return float(0.5 * (d * np.log(2 * np.pi * np.e) + logdet))
        new_belief = BeliefState(mu=belief.mu.copy(), Sigma=Sigma_new, jitter=jitter)
        new_belief.ensure_psd(force=True)
        return new_belief

    # Compute inv_sigma if not provided
    if inv_sigma is None:
        inv_sigma, _, cho_sigma = fast_inv_and_logdet(belief.Sigma, jitter)

    x = option_vectors.astype(float)
    theta = belief.mu.copy()
    tau_safe = max(tau, 1e-8)

    for _ in range(steps):
        logits = (x @ theta) / tau_safe
        probs = softmax(logits)
        y = np.zeros(len(x))
        y[selected_index] = 1.0
        grad = ((y - probs) @ x) / tau_safe - inv_sigma @ (theta - belief.mu)

        P_outer = np.outer(probs, probs)
        weighted = x.T @ (np.diag(probs) - P_outer) @ x
        h_neg = inv_sigma + weighted / (tau_safe ** 2)

        # Use scipy cho_factor for the Newton step solve
        try:
            h_neg_c = spla.cho_factor(h_neg + np.eye(d) * jitter, check_finite=False)
            step = spla.cho_solve(h_neg_c, grad, check_finite=False)
        except Exception:
            break
        theta = theta + step
        if np.linalg.norm(step) < 1e-4:
            break

    # Final Hessian at MAP
    logits = (x @ theta) / tau_safe
    probs = softmax(logits)
    P_outer = np.outer(probs, probs)
    weighted = x.T @ (np.diag(probs) - P_outer) @ x
    h_neg_final = inv_sigma + weighted / (tau_safe ** 2) + np.eye(d) * jitter

    if entropy_only:
        try:
            c_h = spla.cho_factor(h_neg_final, check_finite=False)
            logdet_h = _cho_logdet(c_h)
        except Exception:
            _, logdet_h = np.linalg.slogdet(h_neg_final)
        logdet_sigma_new = -logdet_h
        return float(0.5 * (d * np.log(2 * np.pi * np.e) + logdet_sigma_new))

    # Full update: form Sigma_new
    try:
        c_h = spla.cho_factor(h_neg_final, check_finite=False)
        Sigma_new = spla.cho_solve(c_h, np.eye(d), check_finite=False)
    except Exception:
        Sigma_new = belief.Sigma.copy()

    updated = BeliefState(mu=theta, Sigma=Sigma_new, jitter=jitter)
    updated.ensure_psd(force=True)
    return updated
