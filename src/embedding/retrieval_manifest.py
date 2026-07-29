"""Offline retrieval artifact metadata (CPU dense search; no runtime ST/torch)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir
from src.utils.time import now_iso

RUNTIME_LEXICAL_VS_DENSE_AUDIT = (
    "BM25 is a lightweight lexical retrieval helper built at runtime from item text metadata.\n\n"
    "Dense semantic embeddings are fully precomputed offline and loaded from disk."
)


def retrieval_manifest_path(processed_dir: str | Path) -> Path:
    return Path(processed_dir) / "retrieval_index_manifest.json"


def write_retrieval_manifest(
    processed_dir: str | Path,
    *,
    n_items: int,
    dim: int,
    embedding_file: str = "item_embeddings.npy",
    id_map_file: str = "item_id_to_index.json",
    faiss_index_file: str | None = None,
) -> Path:
    """Describe how `EmbeddingIndex` loads vectors (numpy matmul on CPU)."""

    out = retrieval_manifest_path(processed_dir)
    ensure_dir(out.parent)
    payload: dict[str, Any] = {
        "timestamp": now_iso(),
        "dense_retrieval_backend": "numpy_inner_product",
        "note": "Runtime uses L2-normalized query · item matrix multiply on CPU; no FAISS required.",
        "runtime_lexical_vs_dense_audit": RUNTIME_LEXICAL_VS_DENSE_AUDIT,
        "embedding_file": embedding_file,
        "item_id_map": id_map_file,
        "shape": [int(n_items), int(dim)],
        "faiss_index_file": faiss_index_file,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[embed] retrieval manifest: {out}", flush=True)
    return out


def try_build_faiss_flat_ip_index(
    processed_dir: str | Path,
    embeddings: Any,
) -> str | None:
    """Optional CPU FAISS flat inner-product index (same geometry as np.dot). Returns filename or None."""

    try:
        import faiss  # type: ignore
        import numpy as np
    except ImportError:
        print("[embed] INFO: faiss not installed — skip FAISS index (numpy backend only).", flush=True)
        return None

    x = np.asarray(embeddings, dtype=np.float32)
    faiss.normalize_L2(x)
    d = x.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(x)
    fname = "item_faiss_flat_ip.index"
    out = Path(processed_dir) / fname
    faiss.write_index(index, str(out))
    print(f"[embed] wrote FAISS IndexFlatIP (CPU): {out}", flush=True)
    return fname
