from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent_name = cfg.pop("inherits", None)
    if parent_name:
        parent = load_config(path.parent / parent_name)
        cfg = deep_update(parent, cfg)
    return cfg


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg_out = deepcopy(config)

    def _scrub_llm_key(obj: dict[str, Any]) -> None:
        llm = obj.get("llm")
        if isinstance(llm, dict) and llm.get("api_key"):
            llm["api_key"] = "__redacted__"

    _scrub_llm_key(cfg_out)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_out, f, sort_keys=False, allow_unicode=True)
