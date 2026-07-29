from __future__ import annotations

from typing import Any


class OptionBalancer:
    """Simple neutrality check for generated options.

    In the current (mock/rule_based) milestone, only a length-balance check
    is performed. A real implementation would send options to an LLM neutrality judge.
    """

    implementation_mode = "rule_based"

    def check(self, options: list[dict[str, Any]]) -> bool:
        lengths = [len(o["option_text"]) for o in options if o["option_id"] != "none"]
        if not lengths:
            return True
        avg = sum(lengths) / len(lengths)
        return all(abs(ln - avg) / max(avg, 1) < 0.5 for ln in lengths)
