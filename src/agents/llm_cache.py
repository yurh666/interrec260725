"""Deterministic disk cache for LLM raw responses (reproducible reruns)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def payload_hex_key(namespace: str, payload: Any) -> str:
    """SHA256 hex for (namespace, canonical payload) without touching disk."""
    body = f"{namespace}:{canonical_json(payload)}"
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def llm_cache_filename(namespace: str, payload: Any) -> str:
    """Stable file name (hex) from namespace + canonical payload."""
    return payload_hex_key(namespace, payload) + ".txt"


class LLMRawResponseCache:
    def __init__(self, root: str | Path | None, enabled: bool) -> None:
        self.root = Path(root) if root else None
        self.enabled = bool(enabled) and self.root is not None

    def path_for(self, namespace: str, payload: Any) -> Path | None:
        if not self.enabled or self.root is None:
            return None
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root / llm_cache_filename(namespace, payload)

    def get(self, namespace: str, payload: Any) -> tuple[str | None, str | None]:
        """Return (raw_text, hex_key) or (None, hex_key) if miss."""
        hex_key = payload_hex_key(namespace, payload)
        p = self.path_for(namespace, payload)
        if p is None or not p.exists():
            return None, hex_key
        return p.read_text(encoding="utf-8"), hex_key

    def put(self, namespace: str, payload: Any, raw: str) -> Path | None:
        p = self.path_for(namespace, payload)
        if p is None:
            return None
        p.write_text(raw, encoding="utf-8")
        return p
