from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class UserSplit:
    user_id: str
    observed_history: list[str]
    future_train: list[str]
    future_valid: list[str]
    future_test: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "observed_history": self.observed_history,
            "future_train": self.future_train,
            "future_valid": self.future_valid,
            "future_test": self.future_test,
        }


def split_sequence(
    user_id: str,
    items: list[str],
    observed_ratio: float = 0.4,
    future_train_ratio: float = 0.7,
    future_valid_ratio: float = 0.1,
    future_test_ratio: float = 0.2,
) -> UserSplit:
    if not items:
        return UserSplit(user_id, [], [], [], [])
    n_total = len(items)
    obs_end = int(n_total * observed_ratio)
    obs_end = max(1, min(obs_end, n_total))
    future = items[obs_end:]
    n_future = len(future)

    train_end = int(n_future * future_train_ratio)
    valid_end = train_end + int(n_future * future_valid_ratio)
    if future_test_ratio > 0 and valid_end >= n_future and n_future > 0:
        valid_end = max(train_end, n_future - 1)

    return UserSplit(
        user_id=str(user_id),
        observed_history=[str(x) for x in items[:obs_end]],
        future_train=[str(x) for x in future[:train_end]],
        future_valid=[str(x) for x in future[train_end:valid_end]],
        future_test=[str(x) for x in future[valid_end:]],
    )


def split_grouped_histories(histories: dict[str, list[str]], cfg: dict[str, float]) -> list[dict[str, Any]]:
    splits = []
    for user_id, items in histories.items():
        split = split_sequence(
            user_id=user_id,
            items=items,
            observed_ratio=float(cfg.get("observed_ratio", 0.4)),
            future_train_ratio=float(cfg.get("future_train_ratio", 0.7)),
            future_valid_ratio=float(cfg.get("future_valid_ratio", 0.1)),
            future_test_ratio=float(cfg.get("future_test_ratio", 0.2)),
        )
        splits.append(split.to_dict())
    return splits
