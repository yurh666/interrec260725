from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from src.embedding.st_device import probe_torch_cuda, resolve_sentence_transformer_device_string
from src.utils.io import save_numpy, save_pickle, write_json
from src.utils.math import l2_normalize


def build_item_text(row: pd.Series | dict[str, Any]) -> str:
    get = row.get
    title = get("title", "") or get("track_name", "") or get("artist_name", "") or get("item_id", "")
    return "\n".join(
        [
            f"title: {title}",
            f"artist: {get('artist_name', '')}",
            f"category: {get('category', '') or get('genre', '')}",
            f"tags: {get('tags', '')}",
            f"description: {get('description', '')}",
        ]
    )


@dataclass
class EncodedItems:
    embeddings: np.ndarray
    item_ids: list[str]
    item_id_to_index: dict[str, int]
    mode: str


class ItemEncoder:
    """Item text → dense vectors.

    SentenceTransformer / BGE modes honor ``embedding_device`` (config: embedding.device).

    Each successful ``fit_transform`` sets ``self.embedding_runtime`` (inner dict matching
    the manifest schema consumed by ``embedding_build_log.dump_embedding_manifest``).
    """

    def __init__(
        self,
        mode: str = "tfidf_svd",
        dim: int = 128,
        normalize: bool = True,
        model_name: str | None = None,
        *,
        embedding_device: str = "auto",
        st_batch_size: int = 32,
    ) -> None:
        self.mode = mode
        self.dim = dim
        self.normalize = normalize
        self.model_name = model_name
        self.embedding_device = embedding_device or "auto"
        self.st_batch_size = max(1, int(st_batch_size))
        self.model: Any = None
        self.implementation_mode = mode
        self.embedding_runtime: dict[str, Any] | None = None

    def fit_transform(self, items: pd.DataFrame) -> EncodedItems:
        texts = [build_item_text(row) for _, row in items.fillna("").iterrows()]
        item_ids = items["item_id"].astype(str).tolist()
        if self.mode == "tfidf_svd":
            embeddings = self._tfidf_svd(texts)
        elif self.mode in {"sentence_transformer", "bge"}:
            embeddings = self._sentence_transformer(texts)
        elif self.mode == "mock":
            embeddings = self._mock(texts)
        else:
            raise ValueError(f"Unsupported embedding mode: {self.mode}")
        if self.normalize:
            embeddings = l2_normalize(embeddings, axis=1)

        encoded = EncodedItems(
            embeddings=embeddings.astype(np.float32),
            item_ids=item_ids,
            item_id_to_index={item_id: idx for idx, item_id in enumerate(item_ids)},
            mode=self.implementation_mode,
        )
        return encoded

    def _set_runtime(
        self,
        *,
        device_field: str,
        embedding_mode: str,
        batch_size: int | None,
        model_name_logged: str | None,
        sentence_transformer_device_requested: str | None = None,
    ) -> None:
        cuda_ok, gpu_name = probe_torch_cuda()
        actual_attr = getattr(self.model, "device", None)
        self.embedding_runtime = {
            "device": device_field,
            "torch_cuda_available": cuda_ok,
            "gpu_name": gpu_name,
            "batch_size": batch_size,
            "embedding_mode": embedding_mode,
            "model_name": model_name_logged,
            "sentence_transformer_device_requested": sentence_transformer_device_requested,
            "sentence_transformer_model_device_actual": str(actual_attr) if actual_attr is not None else None,
        }

        try:
            import json

            print(f"[embedding_runtime] {json.dumps({'embedding_runtime': self.embedding_runtime}, ensure_ascii=False)}", flush=True)
        except Exception:
            pass

    def _tfidf_svd(self, texts: list[str]) -> np.ndarray:
        n_samples = len(texts)
        max_components = max(1, min(self.dim, n_samples - 1))
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=1)),
                ("svd", TruncatedSVD(n_components=max_components, random_state=42)),
            ]
        )
        emb = pipeline.fit_transform(texts)
        self.model = pipeline
        if emb.shape[1] < self.dim:
            emb = np.pad(emb, ((0, 0), (0, self.dim - emb.shape[1])))
        arr = emb[:, : self.dim]
        self._set_runtime(
            device_field="cpu_sklearn_pipeline",
            embedding_mode="tfidf_svd",
            batch_size=None,
            model_name_logged=None,
        )
        return arr

    def _sentence_transformer(self, texts: list[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer

        mname = self.model_name or "sentence-transformers/all-MiniLM-L6-v2"
        resolved = resolve_sentence_transformer_device_string(self.embedding_device)

        print(
            f"[embed][SentenceTransformer] model={mname!r} preference={self.embedding_device!r} "
            f"resolved_device={resolved!r} batch_size={self.st_batch_size}",
            flush=True,
        )

        model = SentenceTransformer(mname, device=resolved)
        self.model = model

        embeddings = model.encode(
            texts,
            batch_size=self.st_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        actual_dev = getattr(model, "device", resolved)
        self._set_runtime(
            device_field=str(actual_dev),
            embedding_mode=str(self.mode),
            batch_size=self.st_batch_size,
            model_name_logged=mname,
            sentence_transformer_device_requested=self.embedding_device,
        )
        rt = dict(self.embedding_runtime or {})
        print(
            "[embed][device] AFTER_LOAD actual_model.device="
            f"{actual_dev} torch_cuda_available={rt.get('torch_cuda_available')} gpu_name={rt.get('gpu_name')!r}",
            flush=True,
        )
        return embeddings

    def _mock(self, texts: list[str]) -> np.ndarray:
        rng = np.random.default_rng(42)
        self.implementation_mode = "mock"
        arr = rng.normal(size=(len(texts), self.dim))
        self._set_runtime(
            device_field="numpy_rng_cpu",
            embedding_mode="mock",
            batch_size=None,
            model_name_logged=None,
        )
        return arr

    def save(self, encoded: EncodedItems, output_dir: str | Path) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_numpy(encoded.embeddings, out / "item_embeddings.npy")
        write_json(encoded.item_id_to_index, out / "item_id_to_index.json")
        save_pickle(
            {
                "item_ids": encoded.item_ids,
                "item_id_to_index": encoded.item_id_to_index,
                "mode": encoded.mode,
                "model": self.model,
            },
            out / "item_index.pkl",
        )
