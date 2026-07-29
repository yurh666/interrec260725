from __future__ import annotations

from typing import Any


def should_ask(ig: float, c_ask: float) -> tuple[bool, float]:
    """VOI = IG - c_ask. Ask iff VOI > 0."""
    voi = ig - c_ask
    return voi > 0, voi
