from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.belief.belief_state import BeliefState
from src.belief.initialization import initialize_belief
from src.choicerec.theta_builder import build_choicerec_theta_star
from src.utils.math import l2_normalize


def build_valid_chunks(future_valid: list[str], max_turns: int = 15) -> list[list[str]]:
    """Chronological contiguous chunks for online valid release."""
    valid = [str(x) for x in future_valid]
    m = len(valid)
    if m == 0:
        return [[] for _ in range(max_turns)]
    if m <= max_turns:
        return [([valid[i]] if i < m else []) for i in range(max_turns)]
    q, r = divmod(m, max_turns)
    chunks: list[list[str]] = []
    idx = 0
    for t in range(max_turns):
        size = q + 1 if t < r else q
        chunks.append(valid[idx : idx + size])
        idx += size
    return chunks


def _concat_chronological(observed: list[str], future_train: list[str]) -> list[str]:
    return [str(x) for x in observed] + [str(x) for x in future_train]


def compute_behavior_mu(
    *,
    user_id: str,
    visible_behavior_history: list[str],
    dataset: Any,
    index: Any,
    belief_cfg: dict[str, Any],
) -> np.ndarray:
    weights = dataset.weights_for_items(user_id, visible_behavior_history)
    ref = initialize_belief(
        visible_behavior_history,
        index,
        weights=weights,
        init_mode=str(belief_cfg.get("init_mode", "weighted_average")),
        sigma0=float(belief_cfg.get("sigma0", 1.0)),
        jitter=float(belief_cfg.get("jitter", 1e-6)),
    )
    return ref.mu.copy()


@dataclass
class ChoiceRecProtocol:
    session: dict[str, Any]
    dataset: Any
    index: Any
    belief_cfg: dict[str, Any]
    max_turns: int = 15
    observed_history: list[str] = field(default_factory=list)
    future_train: list[str] = field(default_factory=list)
    future_valid: list[str] = field(default_factory=list)
    future_test: list[str] = field(default_factory=list)
    visible_behavior_history: list[str] = field(default_factory=list)
    seen_items: set[str] = field(default_factory=set)
    valid_chunks: list[list[str]] = field(default_factory=list)
    revealed_valid: list[str] = field(default_factory=list)
    behavior_mu: np.ndarray | None = None
    choice_residual: np.ndarray | None = None
    behavior_update_count: int = 0

    @classmethod
    def from_session(
        cls,
        session: dict[str, Any],
        dataset: Any,
        index: Any,
        belief_cfg: dict[str, Any],
        max_turns: int = 15,
    ) -> ChoiceRecProtocol:
        observed = [str(x) for x in session.get("observed_history", [])]
        train = [str(x) for x in session.get("future_train", [])]
        valid = [str(x) for x in session.get("future_valid", [])]
        test = [str(x) for x in session.get("future_test", [])]
        visible = _concat_chronological(observed, train)
        proto = cls(
            session=session,
            dataset=dataset,
            index=index,
            belief_cfg=belief_cfg,
            max_turns=max_turns,
            observed_history=observed,
            future_train=train,
            future_valid=valid,
            future_test=test,
            visible_behavior_history=list(visible),
            seen_items=set(visible),
            valid_chunks=build_valid_chunks(valid, max_turns=max_turns),
        )
        proto.behavior_mu = compute_behavior_mu(
            user_id=str(session["user_id"]),
            visible_behavior_history=proto.visible_behavior_history,
            dataset=dataset,
            index=index,
            belief_cfg=belief_cfg,
        )
        dim = index.embeddings.shape[1]
        proto.choice_residual = np.zeros(dim, dtype=float)
        return proto

    @property
    def remaining_future_valid(self) -> list[str]:
        revealed = set(self.revealed_valid)
        return [v for v in self.future_valid if v not in revealed]

    @property
    def policy_exclude_items(self) -> set[str]:
        """Items hidden from recommender / Frontier (unrevealed valid only; test items stay candidates)."""
        return set(self.seen_items) | set(self.remaining_future_valid)

    def build_theta_star(self) -> dict[str, Any]:
        return build_choicerec_theta_star(
            remaining_future_valid=self.remaining_future_valid,
            future_test=self.future_test,
            index=self.index,
            session=self.session,
            dataset=self.dataset,
        )

    def reference_behavior_mu(self) -> np.ndarray:
        return compute_behavior_mu(
            user_id=str(self.session["user_id"]),
            visible_behavior_history=self.visible_behavior_history,
            dataset=self.dataset,
            index=self.index,
            belief_cfg=self.belief_cfg,
        )

    def record_after_choice(self, belief: BeliefState) -> None:
        assert self.behavior_mu is not None and self.choice_residual is not None
        self.choice_residual = belief.mu - self.behavior_mu

    def apply_behavioral_reveal(
        self,
        turn: int,
        belief: BeliefState,
    ) -> tuple[BeliefState, dict[str, Any]]:
        chunk = self.valid_chunks[turn - 1] if 1 <= turn <= self.max_turns else []
        sigma_before = belief.Sigma.copy()
        behavior_mu_before = self.behavior_mu.copy() if self.behavior_mu is not None else None
        mu_before = belief.mu.copy()

        log: dict[str, Any] = {
            "turn": turn,
            "valid_chunk_revealed_after_turn": list(chunk),
            "revealed_valid_before_turn": list(self.revealed_valid),
            "remaining_valid_before_turn": list(self.remaining_future_valid),
            "behavior_history_size_before_reveal": len(self.visible_behavior_history),
            "behavior_mu_before_reveal": (
                behavior_mu_before.tolist() if behavior_mu_before is not None else []
            ),
            "mu_before_behavior_update": mu_before.tolist(),
            "sigma_trace_before_behavior_update": float(np.trace(sigma_before)),
            "choice_residual_norm_before_reveal": float(
                np.linalg.norm(self.choice_residual) if self.choice_residual is not None else 0.0
            ),
        }

        if not chunk:
            log.update(
                {
                    "behavior_update_applied": False,
                    "behavior_history_size_after_reveal": len(self.visible_behavior_history),
                    "seen_item_count": len(self.seen_items),
                }
            )
            return belief, log

        self.visible_behavior_history.extend(chunk)
        self.revealed_valid.extend(chunk)
        self.seen_items.update(chunk)
        self.behavior_update_count += 1

        new_behavior_mu = self.reference_behavior_mu()
        assert self.choice_residual is not None
        new_mu_raw = new_behavior_mu + self.choice_residual
        new_mu = l2_normalize(new_mu_raw.reshape(1, -1))[0]
        new_belief = BeliefState(
            mu=new_mu,
            Sigma=belief.Sigma.copy(),
            jitter=belief.jitter,
        )
        self.behavior_mu = new_behavior_mu
        self.choice_residual = new_mu - new_behavior_mu

        log.update(
            {
                "behavior_update_applied": True,
                "behavior_history_size_after_reveal": len(self.visible_behavior_history),
                "behavior_mu_after_reveal": new_behavior_mu.tolist(),
                "mu_after_behavior_update": new_mu.tolist(),
                "sigma_trace_after_behavior_update": float(np.trace(new_belief.Sigma)),
                "sigma_unchanged": bool(np.allclose(new_belief.Sigma, sigma_before)),
                "choice_residual_norm_after_reveal": float(np.linalg.norm(self.choice_residual)),
                "seen_item_count": len(self.seen_items),
                "mu_reference_no_choice": self.reference_behavior_mu().tolist(),
            }
        )
        return new_belief, log

    def turn_audit_before_choice(self, turn: int, belief: BeliefState) -> dict[str, Any]:
        theta_info = self.build_theta_star()
        return {
            "turn": turn,
            "revealed_valid_before_turn": list(self.revealed_valid),
            "remaining_valid_before_turn": list(self.remaining_future_valid),
            "behavior_history_size": len(self.visible_behavior_history),
            "behavior_mu_before_choice": (
                self.behavior_mu.tolist() if self.behavior_mu is not None else []
            ),
            "choice_residual_norm": float(
                np.linalg.norm(self.choice_residual) if self.choice_residual is not None else 0.0
            ),
            "simulator_remaining_valid_count": int(theta_info.get("remaining_valid_count", 0)),
            "simulator_test_count": int(theta_info.get("future_test_count", 0)),
            "simulator_theta_norm": float(np.linalg.norm(theta_info["theta_star"])),
            "seen_item_count": len(self.seen_items),
        }

    def session_summary(self) -> dict[str, Any]:
        return {
            "protocol": "choicerec",
            "initial_observed_count": len(self.observed_history),
            "initial_future_train_count": len(self.future_train),
            "initial_visible_history_count": len(_concat_chronological(self.observed_history, self.future_train)),
            "future_valid_count": len(self.future_valid),
            "future_test_count": len(self.future_test),
            "valid_chunk_sizes": [len(c) for c in self.valid_chunks],
            "behavior_update_count": self.behavior_update_count,
            "final_revealed_valid_count": len(self.revealed_valid),
        }
