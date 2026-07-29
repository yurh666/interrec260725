from __future__ import annotations

from typing import Any

import numpy as np

from src.ask.choice_model import none_logit
from src.utils.math import softmax


class UserSimulator:
    """Simulated user who selects options based on theta*.

    Supports two modes:
    - deterministic_argmax: select the option with highest softmax probability.
    - stochastic_sample: sample from the softmax distribution.

    The formal experiments use deterministic_argmax (default).
    """

    def __init__(
        self,
        theta_star: np.ndarray,
        choice_mode: str = "deterministic_argmax",
        tau: float = 0.2,
        tau_none: float = 0.2,
        none_bias: float = 0.0,
        none_threshold: float = 0.25,
    ) -> None:
        self.theta_star = theta_star
        self.choice_mode = choice_mode
        self.tau = tau
        self.tau_none = tau_none
        self.none_bias = none_bias
        self.none_threshold = none_threshold
        self.implementation_mode = choice_mode

    def choose(
        self,
        option_vectors: np.ndarray,
        option_ids: list[str],
        include_none: bool = True,
    ) -> dict[str, Any]:
        """Select an option given the vectorized options.

        Returns dict with selected_option_id, option_similarities, option_probs, etc.
        """
        n_opts = len(option_vectors)
        sims = {}
        logits = {}
        for i, oid in enumerate(option_ids):
            sim = float(np.dot(self.theta_star, option_vectors[i]))
            sims[oid] = sim
            logits[oid] = sim / max(self.tau, 1e-8)

        logit_arr = np.array([logits[oid] for oid in option_ids], dtype=float)

        if include_none:
            nl = none_logit(self.theta_star, option_vectors, self.tau_none, self.none_bias, self.none_threshold)
            logit_arr = np.append(logit_arr, nl)
            option_ids_with_none = list(option_ids) + ["none"]
            logits["none"] = float(nl)
        else:
            option_ids_with_none = list(option_ids)

        probs_arr = softmax(logit_arr)
        probs = {oid: float(probs_arr[i]) for i, oid in enumerate(option_ids_with_none)}

        if self.choice_mode == "deterministic_argmax":
            selected_idx = int(np.argmax(probs_arr))
        else:
            selected_idx = int(np.random.choice(len(probs_arr), p=probs_arr))

        selected_option_id = option_ids_with_none[selected_idx]
        selected_vector_idx = selected_idx if selected_option_id != "none" else None

        return {
            "selected_option_id": selected_option_id,
            "selected_vector_index": selected_vector_idx,
            "option_similarities": sims,
            "option_logits": logits,
            "option_probs": probs,
            "selected_by": f"argmax_softmax" if self.choice_mode == "deterministic_argmax" else "sample_softmax",
            "is_none": selected_option_id == "none",
        }
