"""Write ``embedding_build_log.json`` (offline embedding audit)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir
from src.utils.time import now_iso


def embedding_build_manifest_path(processed_dir: str | Path) -> Path:
    return Path(processed_dir) / "embedding_build_log.json"


def dump_embedding_manifest(
    processed_dir: str | Path,
    embedding_runtime_inner: dict[str, Any],
    *,
    embedding_fallback_to_mock: bool,
    embedding_mode_requested: str,
    extra_top_level: dict[str, Any] | None = None,
) -> Path:
    """Persist JSON next to vectors; emits one stdout line mirroring persisted ``embedding_runtime``."""

    ensure_dir(Path(processed_dir))
    out = embedding_build_manifest_path(processed_dir)
    payload: dict[str, Any] = {
        "timestamp": now_iso(),
        "embedding_fallback_to_mock": bool(embedding_fallback_to_mock),
        "embedding_runtime": embedding_runtime_inner,
        "embedding_mode_requested": embedding_mode_requested,
    }
    if extra_top_level:
        payload.update(extra_top_level)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    line = {"embedding_runtime": embedding_runtime_inner, "embedding_fallback_to_mock": bool(embedding_fallback_to_mock)}
    print(json.dumps(line, ensure_ascii=False))
    print(f"[embed] audit file: {out}", flush=True)
    return out
