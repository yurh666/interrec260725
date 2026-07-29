from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from interrec_noapi.surface_writer.local_inference import generate_surface_text
from interrec_noapi.surface_writer.v2_cache import lookup_cache, make_cache_key


@dataclass(frozen=True)
class SurfaceWriterResult:
    question: str
    options: list[str]
    mode: str
    fallback_reason: str | None = None
    api_calls: int = 0
    local_model_loaded: bool = False
    latency_sec: float | None = None


class SurfaceWriter:
    """Natural-language surface writer for already-built semantic options.

    The writer is intentionally side-effect free: it never changes representative
    item ids, option vectors, selected indices, belief state, rankings, or metrics.
    """

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        cfg = dict(config or {})
        self.mode = str(cfg.get("mode", "local_small_model"))
        self._repo_root = Path(__file__).resolve().parents[2]
        self.local_model_path = cfg.get("local_model_path")
        self.inference_config_path = cfg.get("inference_config_path")
        self.base_model_path = cfg.get("base_model_path")
        self.fallback_mode = str(cfg.get("fallback_mode", "mock_template"))
        self.allow_fallback_if_model_load_failed = bool(cfg.get("allow_fallback_if_model_load_failed", True))
        self.api_enabled = bool(cfg.get("api_enabled", False))
        self.cache_dir = cfg.get("cache_dir")
        self.validate_cached_text = bool(cfg.get("validate_cached_text", True))
        self.v2_adapter_path = cfg.get("v2_adapter_path")
        self.v2_base_model_path = cfg.get("v2_base_model_path")
        self.v2_inference_config_path = cfg.get("v2_inference_config_path")
        self.cache_path = cfg.get("cache_path")
        self.cache_implementation_mode = str(cfg.get("cache_implementation_mode") or cfg.get("implementation_mode") or "")
        self.cache_run_id = cfg.get("cache_run_id")
        self.lookup_priority = tuple(cfg.get("lookup_priority") or ("event_cache_key", "canonical_unique_key"))
        self.debug_cache_key = bool(cfg.get("debug_cache_key", False))
        self.allow_model_generation_on_miss = bool(cfg.get("allow_model_generation_on_miss", False))
        self.require_cache_hit = bool(cfg.get("require_cache_hit", False))
        self.allow_mock = bool(cfg.get("allow_mock", True))
        self.api_calls = 0
        self.last_audit: dict[str, Any] = {}
        self._inference_cfg: dict[str, Any] = {}
        inf_path = self._resolve_project_path(self.inference_config_path)
        if inf_path and inf_path.exists():
            import json

            self._inference_cfg = json.loads(inf_path.read_text())
        if self.base_model_path:
            self._inference_cfg["base_model_path"] = self.base_model_path

    def _resolve_project_path(self, raw: Any) -> Path | None:
        if raw is None or str(raw).strip() == "":
            return None
        p = Path(str(raw))
        if p.is_absolute():
            return p
        for base in (Path.cwd(), self._repo_root):
            cand = base / p
            if cand.exists():
                return cand.resolve()
        return (Path.cwd() / p).resolve()

    def write_question(
        self,
        semantic_options: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any] | None = None,
        *,
        option_id_order: Sequence[str] | None = None,
    ) -> SurfaceWriterResult:
        context = dict(context or {})
        mode = self.mode
        fallback_reason: str | None = None
        local_loaded = False
        latency_sec: float | None = None

        if mode == "hybrid_surface_cache_strict":
            from interrec_noapi.surface_writer.hybrid_cache import CacheMissError, lookup_hybrid_cache_strict

            cache_path = Path(str(self.cache_path)) if self.cache_path else None
            impl = self.cache_implementation_mode or str(context.get("implementation_mode") or "")
            if self.cache_run_id and not context.get("run_id"):
                context = {**context, "run_id": self.cache_run_id, "cache_run_id": self.cache_run_id}
            if cache_path and cache_path.exists():
                try:
                    cached = lookup_hybrid_cache_strict(
                        cache_path,
                        str(context.get("dataset") or ""),
                        semantic_options,
                        impl,
                        context=context,
                        lookup_priority=self.lookup_priority,
                        debug=self.debug_cache_key,
                    )
                    order = list(option_id_order or [str(o.get("option_id") or o.get("hypothesis_id") or "") for o in semantic_options])
                    order = [x for x in order if x]
                    by_id = {str(o["option_id"]): str(o["option_text"]) for o in cached["options"]}
                    texts = [by_id.get(oid, "") for oid in order]
                    if all(texts) and cached.get("question_text"):
                        self.last_audit = {
                            "cache_hit": True,
                            "cache_miss": False,
                            "lookup_method": cached.get("lookup_method"),
                            "lookup_key": cached.get("lookup_key"),
                            "dedup_key": cached.get("dedup_key"),
                            "source": cached.get("source"),
                            "debug": cached.get("debug"),
                        }
                        return SurfaceWriterResult(
                            question=str(cached["question_text"]),
                            options=texts,
                            mode=mode,
                            fallback_reason=None,
                            api_calls=0,
                            local_model_loaded=False,
                            latency_sec=0.0,
                        )
                except CacheMissError as exc:
                    self.last_audit = {"cache_hit": False, "cache_miss": True, "error": str(exc)}
                    if self.require_cache_hit or not self.allow_model_generation_on_miss:
                        raise
                    fallback_reason = str(exc)
            else:
                fallback_reason = "hybrid_cache_path_missing"
            if self.require_cache_hit or str(self.fallback_mode or "none") in {"none", ""} or not self.allow_model_generation_on_miss:
                raise RuntimeError(fallback_reason or "hybrid_cache_miss")
            mode = self.fallback_mode
            fallback_reason = fallback_reason or "hybrid_cache_miss"

        if mode == "distilled_model_cache":
            cache_dir = Path(str(self.cache_dir)) if self.cache_dir else None
            if cache_dir and cache_dir.exists():
                key = make_cache_key(
                    str(context.get("dataset") or ""),
                    str(context.get("episode_id") or context.get("session_id") or ""),
                    context.get("turn") or 0,
                    semantic_options,
                    str(context.get("semantic_backend_mode") or ""),
                )
                cached = lookup_cache(cache_dir, key)
                if cached and cached.get("question_text") and cached.get("options"):
                    order = list(option_id_order or [str(o.get("option_id") or o.get("hypothesis_id") or "") for o in semantic_options])
                    order = [x for x in order if x]
                    by_id = {str(o["option_id"]): str(o["option_text"]) for o in cached["options"]}
                    texts = [by_id.get(oid, "") for oid in order]
                    if all(texts) and cached.get("question_text"):
                        self.last_audit = {"cache_hit": True, "cache_key": key, "source": cached.get("source")}
                        return SurfaceWriterResult(
                            question=str(cached["question_text"]),
                            options=texts,
                            mode=mode,
                            fallback_reason=None,
                            api_calls=0,
                            local_model_loaded=True,
                            latency_sec=0.0,
                        )
            mode = str(self.fallback_mode or "local_v2_model")
            fallback_reason = "cache_miss"

        if mode == "local_v2_model":
            adapter = Path(str(self.v2_adapter_path or self.local_model_path)) if (self.v2_adapter_path or self.local_model_path) else None
            if adapter and adapter.exists():
                import json as _json
                import sys
                from pathlib import Path as _P

                root = _P(__file__).resolve().parents[3]
                sys.path.insert(0, str(root / "training" / "surface_writer_v2_sft"))
                from infer_v2_writer import infer_v2_fields  # noqa: WPS433
                from v2_utils import wrap_to_surface_output  # noqa: WPS433

                cfg = {}
                cfg_path = self.v2_inference_config_path or self.inference_config_path
                if cfg_path and _P(str(cfg_path)).exists():
                    cfg = _json.loads(_P(str(cfg_path)).read_text())
                base = str(self.v2_base_model_path or self.base_model_path or cfg.get("base_model_path") or "")
                sample = {
                    "input": {
                        "task": "Generate concise user-facing text fields for a preference question.",
                        "dataset": context.get("dataset"),
                        "question_intent": "Ask the user to choose which preference direction best matches them.",
                        "options": [
                            {
                                "option_id": str(o.get("option_id") or o.get("hypothesis_id")),
                                "role": str(o.get("role") or ""),
                                "cluster_phrase": str(o.get("cluster_phrase") or o.get("text_description") or "")[:200],
                                "representative_item_names": list(o.get("representative_item_names") or [])[:3],
                                "metadata_summary": "",
                                "distinction_from_other_options": str(o.get("distinction_from_other_options") or "")[:200],
                            }
                            for o in semantic_options
                        ],
                        "style_constraints": {"objective": True, "balanced_options": True},
                    }
                }
                parsed, audit = infer_v2_fields(adapter, base, sample, cfg)
                self.last_audit = dict(audit or {})
                self.last_audit["cache_hit"] = False
                latency_sec = audit.get("generation_latency_sec")
                if parsed:
                    order = list(option_id_order or [str(o.get("option_id") or o.get("hypothesis_id") or "") for o in semantic_options])
                    wrapped = wrap_to_surface_output(parsed, [x for x in order if x])
                    texts = [o["option_text"] for o in wrapped["options"]]
                    if wrapped["question_text"] and all(texts):
                        return SurfaceWriterResult(
                            question=wrapped["question_text"],
                            options=texts,
                            mode=mode,
                            fallback_reason=fallback_reason,
                            api_calls=0,
                            local_model_loaded=True,
                            latency_sec=latency_sec,
                        )
            if self.allow_fallback_if_model_load_failed:
                mode = "mock_template"
                fallback_reason = fallback_reason or "local_v2_failed"
            else:
                raise RuntimeError(fallback_reason or "local_v2_failed")

        if mode == "local_small_model":
            adapter = self._resolve_project_path(self.local_model_path)
            if adapter is None or not adapter.exists():
                mode = self.fallback_mode
                fallback_reason = "local_model_path_missing"
            else:
                gen = generate_surface_text(adapter, semantic_options, context, self._inference_cfg)
                self.last_audit = dict(gen)
                latency_sec = gen.get("latency_sec")
                if gen.get("ok"):
                    q = str(gen.get("question_text") or "").strip()
                    parsed_opts = gen.get("options") or []
                    by_id = {str(o.get("option_id")): str(o.get("option_text") or "").strip() for o in parsed_opts if isinstance(o, dict)}
                    order = list(option_id_order or [str(o.get("option_id") or o.get("hypothesis_id") or "") for o in semantic_options])
                    order = [x for x in order if x]
                    texts = [by_id.get(oid, "") for oid in order]
                    if q and all(texts):
                        local_loaded = True
                        mode = "local_small_model"
                        self.last_audit["schema_valid"] = True
                        self.last_audit["option_id_match"] = list(by_id.keys())[: len(order)] == order
                        return SurfaceWriterResult(
                            question=q,
                            options=texts,
                            mode=mode,
                            fallback_reason=None,
                            api_calls=0,
                            local_model_loaded=True,
                            latency_sec=latency_sec,
                        )
                err = str(gen.get("error") or "local_generation_failed")
                if self.allow_fallback_if_model_load_failed:
                    mode = self.fallback_mode
                    fallback_reason = err
                else:
                    raise RuntimeError(err)

        if mode == "api":
            if not self.api_enabled:
                mode = self.fallback_mode
                fallback_reason = "api_writer_not_enabled"
            else:
                raise RuntimeError("API surface writer is disabled in noAPI smoke/default runs")

        if mode not in {"mock_template", "template"}:
            mode = "mock_template"
            fallback_reason = fallback_reason or "unsupported_surface_writer_mode"

        result = self._template(semantic_options, context, mode=mode, fallback_reason=fallback_reason)
        return SurfaceWriterResult(
            question=result.question,
            options=result.options,
            mode=result.mode,
            fallback_reason=result.fallback_reason,
            api_calls=result.api_calls,
            local_model_loaded=local_loaded,
            latency_sec=latency_sec,
        )

    def _template(
        self,
        semantic_options: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any],
        mode: str,
        fallback_reason: str | None,
    ) -> SurfaceWriterResult:
        cluster_phrase = str(context.get("cluster_phrase") or context.get("question_hint") or "these recommendations")
        question = f"Which option best matches your preference for {cluster_phrase}?"
        options: list[str] = []
        for i, opt in enumerate(semantic_options, start=1):
            text = opt.get("option_text") or opt.get("text") or opt.get("label")
            reps = opt.get("representative_item_names") or opt.get("representative_items") or []
            if not text and reps:
                text = ", ".join(str(x) for x in list(reps)[:3])
            options.append(str(text or f"Option {i}"))
        return SurfaceWriterResult(
            question=question,
            options=options,
            mode=mode,
            fallback_reason=fallback_reason,
            api_calls=self.api_calls,
            local_model_loaded=False,
        )
