from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / np.sum(exp)


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), eps)
    return float(np.dot(a, b) / denom)


def weighted_average(vectors: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    if len(vectors) == 0:
        raise ValueError("Cannot average an empty vector set.")
    if weights is None:
        return np.mean(vectors, axis=0)
    weights = np.asarray(weights, dtype=float)
    total = float(np.sum(weights))
    if total <= 0:
        return np.mean(vectors, axis=0)
    return np.sum(vectors * weights[:, None], axis=0) / total
