from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class MemoryCompatibilityState:
    enabled: bool
    mode: str
    effect_strength: float
    retrieve_enabled: bool
    writeback_enabled: bool
    metrics_path_changes: bool


def parse_memory_config(config: Mapping[str, Any] | None) -> MemoryCompatibilityState:
    cfg = dict(config or {})
    enabled = bool(cfg.get("enabled", False))
    mode = str(cfg.get("mode", "compatibility" if enabled else "disabled"))
    effect_strength = float(cfg.get("effect_strength", cfg.get("memory_projection_alpha", 0.0)))
    retrieve_enabled = bool(cfg.get("retrieve_enabled", enabled))
    writeback_enabled = bool(cfg.get("writeback_enabled", enabled))
    compatibility = (not enabled) or mode in {"compatibility", "memory_compatible", "identity"} or effect_strength == 0.0
    return MemoryCompatibilityState(enabled, mode, effect_strength, retrieve_enabled, writeback_enabled, metrics_path_changes=not compatibility)
