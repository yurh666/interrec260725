from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg as spla


@dataclass
class BeliefState:
    mu: np.ndarray
    Sigma: np.ndarray
    jitter: float = 1e-6

    def sample(self, n_samples: int) -> np.ndarray:
        self.ensure_psd()
        try:
            L = spla.cholesky(self.Sigma, lower=True, check_finite=False)
            Z = np.random.randn(n_samples, len(self.mu))
            return self.mu + Z @ L.T
        except Exception:
            return np.random.multivariate_normal(self.mu, self.Sigma, size=n_samples)

    def entropy(self) -> float:
        d = len(self.mu)
        try:
            c = spla.cho_factor(self.Sigma + np.eye(d) * self.jitter, check_finite=False)
            logdet = 2.0 * float(np.sum(np.log(np.diag(c[0]))))
        except Exception:
            sign, logdet = np.linalg.slogdet(self.Sigma)
            if sign <= 0:
                logdet = float(np.sum(np.log(np.maximum(np.diag(self.Sigma), self.jitter))))
        return float(0.5 * (d * np.log(2 * np.pi * np.e) + logdet))

    def eigendecompose(self, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        self.ensure_psd()
        try:
            values, vectors = spla.eigh(
                self.Sigma,
                subset_by_index=[max(0, self.Sigma.shape[0] - top_k), self.Sigma.shape[0] - 1],
                check_finite=False,
            )
            order = np.argsort(values)[::-1]
            return values[order], vectors[:, order]
        except Exception:
            values, vectors = np.linalg.eigh(self.Sigma)
            order = np.argsort(values)[::-1][:top_k]
            return values[order], vectors[:, order]

    def ensure_psd(self, force: bool = False) -> None:
        self.Sigma = 0.5 * (self.Sigma + self.Sigma.T)
        if force:
            # Use fast diagonal regularisation instead of eigvalsh
            diag_min = float(np.min(np.diag(self.Sigma)))
            if diag_min < self.jitter:
                self.Sigma = self.Sigma + np.eye(self.Sigma.shape[0]) * (self.jitter - diag_min + self.jitter)

    def copy(self) -> "BeliefState":
        return BeliefState(self.mu.copy(), self.Sigma.copy(), self.jitter)

    def summary(self, top_k: int = 5, include_eigendecomp: bool = True) -> dict[str, object]:
        out = {
            "entropy": self.entropy(),
            "mu_norm": float(np.linalg.norm(self.mu)),
            "sigma_trace": float(np.trace(self.Sigma)),
        }
        if include_eigendecomp:
            eigvals, _ = self.eigendecompose(top_k)
            out["top_eigenvalues"] = [float(x) for x in eigvals]
        return out
