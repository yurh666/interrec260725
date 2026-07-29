from __future__ import annotations

from typing import Any


def build_visible_context(session: dict[str, Any], item_lookup: dict[str, dict[str, Any]], max_items: int = 20) -> dict[str, Any]:
    observed = session["observed_history"][-max_items:]
    return {
        "user_id": session["user_id"],
        "observed_history": observed,
        "observed_items": [item_lookup.get(str(item_id), {"item_id": str(item_id)}) for item_id in observed],
    }
