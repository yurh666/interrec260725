"""Optional strict JSONL emitter for frontier_clustered_v32 runs (default OFF).

Moves huge string fields to run-local sidecar files so each physical line remains
one json.loads-parseable object. Only active when ``frontier_clustered_v32.strict_jsonl_full_log``."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

SIDE_SUBDIR = Path("artifacts") / "frontier_clustered_v32_raw_outputs"
PREVIEW_MIN = 500
PREVIEW_MAX = 1000


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _preview(s: str, n: int) -> str:
    t = str(s or "")
    return t if len(t) <= n else t[:n]


def _safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def offload_frontier_v32_turn_for_jsonl(
    turn_log: dict[str, Any],
    *,
    run_dir: Path,
    preview_chars: int = 800,
    write_sidecar: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (sanitized_turn_copy, sidecar_meta). Does not mutate input."""

    pc = max(PREVIEW_MIN, min(PREVIEW_MAX, int(preview_chars)))
    meta: dict[str, Any] = {}
    out = copy.deepcopy(turn_log)
    run_dir = Path(run_dir)
    side_root = run_dir / SIDE_SUBDIR
    side_root.mkdir(parents=True, exist_ok=True)

    rid = str(turn_log.get("run_id") or run_dir.name or "run")
    uid = str(turn_log.get("user_id") or "user")
    tn = int(turn_log.get("turn") or 0)

    def _safe_filename_part(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("._") or "x"

    def _pack_block(blk: dict[str, Any]) -> dict[str, Any]:
        sid = f"{_safe_filename_part(rid)}_{_safe_filename_part(uid)}_t{tn}"
        path = side_root / f"{sid}.json"
        raw_out = str(blk.get("raw_generation_output") or "")
        pack: dict[str, Any] = {
            "run_id": rid,
            "user_id": uid,
            "turn": tn,
            "raw_generation_output": raw_out,
            "raw_generation_chars": len(raw_out),
        }
        cq = blk.get("choice_question")
        if isinstance(cq, dict):
            pack["choice_question"] = copy.deepcopy(cq)
        # Light prompt payload if present inside hypothesis audit (debug)
        hga = blk.get("hypothesis_generator_audit")
        if isinstance(hga, dict) and hga.get("last_user_payload_preview") is not None:
            pack["hypothesis_generator_audit_excerpt"] = {
                k: hga[k]
                for k in list(hga.keys())[:20]
                if k not in ("raw_prompt", "raw_messages")
            }
        h = _sha256_text(_safe_json_dumps(pack))
        trunc = len(raw_out) > pc
        blk["raw_generation_output_preview"] = _preview(raw_out, pc)
        blk["raw_generation_output_sha256"] = h
        blk["raw_generation_output_truncated"] = bool(trunc)
        blk["raw_generation_output_size_chars"] = int(len(raw_out))
        blk.pop("raw_generation_output", None)
        if write_sidecar:
            path.write_text(_safe_json_dumps(pack) + "\n", encoding="utf-8")
            blk["raw_generation_output_path"] = str(path.relative_to(run_dir))
            meta["sidecar_path"] = str(path)
            meta["sidecar_sha256"] = h
        else:
            blk["raw_generation_output_path"] = None
            blk["raw_generation_output_sidecar_skipped"] = True
        # Keep choice_question keys but drop huge option blobs in log only
        cq2 = blk.get("choice_question")
        if isinstance(cq2, dict):
            qtxt = str(cq2.get("final_question_text") or cq2.get("question_text") or "")
            blk["choice_question"] = {
                "final_question_text_preview": _preview(qtxt, pc),
                "final_question_text_size_chars": len(qtxt),
                "options_count": len(cq2.get("options") or [])
                if isinstance(cq2.get("options"), list)
                else None,
                "generation_extra_keys": cq2.get("generation_extra_keys"),
            }
        return blk

    gen = out.get("frontier_clustered_v32_generation")
    if isinstance(gen, dict):
        raw = str(gen.get("raw_generation_output") or "")
        if raw:
            gen = _pack_block(gen)
            out["frontier_clustered_v32_generation"] = gen

    cl = out.get("frontier_clustered_v32_closed_loop")
    if isinstance(cl, dict):
        cq = cl.get("choice_question")
        if isinstance(cq, dict):
            qtxt = str(cq.get("final_question_text") or cq.get("question_text") or "")
            cl["choice_question"] = {
                "final_question_text_preview": _preview(qtxt, min(pc, 400)),
                "final_question_text_size_chars": len(qtxt),
            }

    # strip raw_assistant_output if ever present at top-level
    for k in ("raw_assistant_output",):
        if k in out and isinstance(out[k], str) and len(out[k]) > pc:
            blob = out.pop(k)
            sp = side_root / f"{rid}_{uid}_t{tn}_assistant.txt"
            sp.write_text(blob, encoding="utf-8")
            out[f"{k}_path"] = str(sp.relative_to(run_dir))
            out[f"{k}_sha256"] = _sha256_text(blob)
            out[f"{k}_preview"] = _preview(blob, pc)

    return out, meta


def try_roundtrip_json(obj: Any) -> tuple[bool, str]:
    try:
        json.loads(_safe_json_dumps(obj))
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}:{exc}"


def emergency_strip_for_jsonl(obj: Any) -> dict[str, Any]:
    """Drop known-heavy subtrees so a turn row can still be logged."""

    out = copy.deepcopy(obj)
    if not isinstance(out, dict):
        return {"type": "warning", "scope": "full_log_emergency_strip", "payload": str(obj)[:400]}
    blk = out.get("frontier_clustered_v32_generation")
    if isinstance(blk, dict):
        blk_light = {
            "enabled": blk.get("enabled"),
            "mode": blk.get("mode"),
            "fallback_used": blk.get("fallback_used"),
            "fallback_reason": blk.get("fallback_reason"),
            "latency_ms": blk.get("latency_ms"),
            "api_call_count": blk.get("api_call_count"),
            "schema_validation_result": blk.get("schema_validation_result"),
            "telemetry_note": "emergency_strip_dropped_generation_payload_jsonl",
        }
        out["frontier_clustered_v32_generation"] = blk_light
    return out


def scrub_unescaped_physical_newlines_nested(obj: Any) -> None:
    """In-place replacement of literal newline / CR inside strings (belt-and-suspenders)."""

    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if isinstance(v, str):
                if "\n" in v or "\r" in v:
                    obj[k] = re.sub(r"[\r\n]+", "\\n", v)
            elif isinstance(v, (dict, list)):
                scrub_unescaped_physical_newlines_nested(v)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, str) and ("\n" in item or "\r" in item):
                obj[i] = re.sub(r"[\r\n]+", "\\n", item)
            elif isinstance(item, (dict, list)):
                scrub_unescaped_physical_newlines_nested(item)
