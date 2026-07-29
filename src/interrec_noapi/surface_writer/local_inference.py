from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

IM_END = "<" + "|im_end|>"
SYSTEM_PROMPT = (
    "You are a surface writer for a recommendation system. "
    "Given semantic option context, output JSON only with question_text and options "
    "(each option has option_id and option_text). "
    "Do not change option_ids or option order."
)

_MODEL_CACHE: dict[str, Any] = {"loaded": False, "error": None}


def _resolve_base_model_path(adapter_dir: Path, inference_cfg: Mapping[str, Any] | None) -> str | None:
    inference_cfg = dict(inference_cfg or {})
    search_roots = [Path.cwd(), adapter_dir.resolve().parents[2]]

    def _exists(raw: str) -> str | None:
        p = os.path.expandvars(os.path.expanduser(raw.strip()))
        if not p or p.startswith("${"):
            return None
        cand = Path(p)
        if cand.is_absolute() and cand.exists():
            return str(cand)
        for root in search_roots:
            rel = root / p
            if rel.exists():
                return str(rel.resolve())
        if cand.exists():
            return str(cand.resolve())
        return None

    for key in ("base_model_path", "base_model_name_or_path"):
        hit = _exists(str(inference_cfg.get(key) or ""))
        if hit:
            return hit
    for env_key in ("INTERREC_BASE_MODEL_PATH",):
        hit = _exists(os.environ.get(env_key, ""))
        if hit:
            return hit
    hit = _exists("models/Qwen2.5-1.5B-Instruct")
    if hit:
        return hit
    adapter_cfg_path = adapter_dir / "adapter_config.json"
    if adapter_cfg_path.exists():
        raw = json.loads(adapter_cfg_path.read_text())
        p = os.path.expandvars(os.path.expanduser(str(raw.get("base_model_name_or_path") or "").strip()))
        if p.startswith("/") and not Path(p).exists():
            p = ""
        if p and "/" in p and not p.startswith("/"):
            hit = _exists(p)
            if hit:
                return hit
        if p:
            hit = _exists(p)
            if hit:
                return hit
        hub = p if "/" in p and not p.startswith("/") else "Qwen/Qwen2.5-1.5B-Instruct"
        cache_guess = Path.home() / ".cache/huggingface/hub"
        if cache_guess.exists():
            slug = f"models--{hub.replace('/', '--')}"
            snap_root = cache_guess / slug / "snapshots"
            if snap_root.exists():
                snaps = sorted(snap_root.iterdir())
                if snaps:
                    return str(snaps[-1])
    return None


def _build_user_payload(
    semantic_options: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    options_semantic = []
    for opt in semantic_options:
        options_semantic.append(
            {
                "option_id": str(opt.get("option_id") or opt.get("hypothesis_id") or ""),
                "role": str(opt.get("role") or ""),
                "bound_cluster_id": str(opt.get("bound_cluster_id") or ""),
                "representative_item_ids": [str(x) for x in (opt.get("representative_item_ids") or opt.get("representative_items") or [])],
                "representative_item_names": [str(x) for x in (opt.get("representative_item_names") or [])],
                "cluster_phrase": str(opt.get("cluster_phrase") or opt.get("text_description") or "")[:200],
                "distinction_from_other_options": str(opt.get("distinction_from_other_options") or "")[:300],
            }
        )
    return {
        "task": "Generate a concise user-facing preference question and options from semantic option context.",
        "dataset": str(context.get("dataset") or ""),
        "memory_summary": str(context.get("memory_summary") or ""),
        "ranking_frontier_summary": str(context.get("ranking_frontier_summary") or ""),
        "clusters": context.get("clusters") or [],
        "options_semantic": options_semantic,
        "constraints": {
            "do_not_change_option_order": True,
            "do_not_change_option_ids": True,
            "output_json_only": True,
        },
    }


def format_chat_prompt(user_payload: dict[str, Any]) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n"
        f"<|im_start|>user\n{json.dumps(user_payload, ensure_ascii=False)}\n"
        f"<|im_start|>assistant\n"
    )


def parse_model_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    for marker in (IM_END, "<|im_end|>", "<|endoftext|>"):
        if marker and marker in text:
            text = text.split(marker, 1)[0].strip()
            break
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _cache_key(adapter_dir: Path, inference_cfg: Mapping[str, Any] | None, base_path: str | None) -> str:
    cfg = dict(inference_cfg or {})
    return "|".join(
        [
            str(adapter_dir.resolve()),
            str(base_path or ""),
            str(cfg.get("device", "cpu")),
            str(bool(cfg.get("load_in_4bit", False))),
            str(cfg.get("max_new_tokens", 512)),
        ]
    )


def load_local_model(adapter_dir: Path, inference_cfg: Mapping[str, Any] | None = None) -> tuple[Any, Any, str | None]:
    inference_cfg = dict(inference_cfg or {})
    base_path = _resolve_base_model_path(adapter_dir, inference_cfg)
    cache_key = _cache_key(adapter_dir, inference_cfg, base_path)
    if _MODEL_CACHE.get("key") == cache_key and _MODEL_CACHE.get("loaded"):
        return _MODEL_CACHE["model"], _MODEL_CACHE["tokenizer"], None
    if _MODEL_CACHE.get("key") == cache_key and _MODEL_CACHE.get("error"):
        return None, None, str(_MODEL_CACHE["error"])

    if not base_path:
        err = "base_model_missing_on_cpu"
        _MODEL_CACHE.update({"key": cache_key, "loaded": False, "error": err})
        return None, None, err

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        err = f"inference_deps_missing:{exc}"
        _MODEL_CACHE.update({"key": cache_key, "loaded": False, "error": err})
        return None, None, err

    device = str(inference_cfg.get("device", "cpu"))
    load_in_4bit = bool(inference_cfg.get("load_in_4bit", False))

    try:
        t_load0 = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            compute_dtype = getattr(torch, str(inference_cfg.get("bnb_4bit_compute_dtype", "float16")), torch.float16)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=str(inference_cfg.get("bnb_4bit_quant_type", "nf4")),
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            base = AutoModelForCausalLM.from_pretrained(
                base_path,
                trust_remote_code=True,
                quantization_config=bnb_config,
                device_map={"": device} if device.startswith("cuda") else "auto",
            )
            model = PeftModel.from_pretrained(base, str(adapter_dir))
        else:
            dtype = getattr(torch, str(inference_cfg.get("inference_dtype", "float32")), torch.float32)
            base = AutoModelForCausalLM.from_pretrained(
                base_path,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map=None,
            )
            model = PeftModel.from_pretrained(base, str(adapter_dir))
            model.to(device)

        model.eval()
        load_time = round(time.perf_counter() - t_load0, 4)
        _MODEL_CACHE.update(
            {
                "key": cache_key,
                "loaded": True,
                "error": None,
                "model": model,
                "tokenizer": tokenizer,
                "base_path": base_path,
                "device": device,
                "load_in_4bit": load_in_4bit,
                "load_time_sec": load_time,
            }
        )
        return model, tokenizer, None
    except Exception as exc:  # noqa: BLE001
        err = f"local_model_load_failed:{type(exc).__name__}:{exc}"
        if "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError":
            err = f"oom:{exc}"
        _MODEL_CACHE.update({"key": cache_key, "loaded": False, "error": err})
        return None, None, err


def generate_surface_text(
    adapter_dir: Path,
    semantic_options: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any] | None,
    inference_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    context = dict(context or {})
    inference_cfg = dict(inference_cfg or {})
    t0 = time.perf_counter()
    model, tokenizer, err = load_local_model(adapter_dir, inference_cfg)
    if err or model is None or tokenizer is None:
        return {
            "ok": False,
            "error": err or "model_not_loaded",
            "latency_sec": round(time.perf_counter() - t0, 4),
        }

    user_payload = _build_user_payload(semantic_options, context)
    prompt = format_chat_prompt(user_payload)
    max_new_tokens = int(inference_cfg.get("max_new_tokens", 512))
    device = str(inference_cfg.get("device") or _MODEL_CACHE.get("device") or "cpu")

    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    inputs = tokenizer(prompt, return_tensors="pt")
    dev = getattr(model, "device", None) or next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    t_gen0 = time.perf_counter()
    try:
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    except torch.cuda.OutOfMemoryError as exc:
        return {"ok": False, "error": f"oom:{exc}", "latency_sec": round(time.perf_counter() - t0, 4), "oom": True}
    gen_latency = round(time.perf_counter() - t_gen0, 4)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False)
    parsed = parse_model_json(text)
    latency = round(time.perf_counter() - t0, 4)
    peak_mb = None
    if device.startswith("cuda") and torch.cuda.is_available():
        peak_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)

    if not parsed:
        return {
            "ok": False,
            "error": "json_parse_failed",
            "raw": text[:500],
            "latency_sec": latency,
            "generation_latency_sec": gen_latency,
            "peak_cuda_memory_mb": peak_mb,
        }

    return {
        "ok": True,
        "question_text": str(parsed.get("question_text") or "").strip(),
        "options": parsed.get("options") or [],
        "raw": text[:500],
        "latency_sec": latency,
        "generation_latency_sec": gen_latency,
        "peak_cuda_memory_mb": peak_mb,
        "base_model_path": _MODEL_CACHE.get("base_path"),
        "load_time_sec": _MODEL_CACHE.get("load_time_sec"),
        "device": _MODEL_CACHE.get("device"),
    }
