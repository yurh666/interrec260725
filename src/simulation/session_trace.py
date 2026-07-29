"""Turn-level session traces for deterministic cluster-bound v2 replay."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

TRACE_VERSION = "cb_v2_trace_v1"


def to_jsonable(obj: Any) -> Any:
    """Recursively convert runtime objects to JSON-serializable values."""

    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return [to_jsonable(v) for v in sorted(obj, key=lambda x: str(x))]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj))
    if hasattr(obj, "tolist"):
        try:
            return to_jsonable(obj.tolist())
        except Exception:
            pass
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return to_jsonable(obj.item())
        except Exception:
            pass
    return str(obj)


def write_trace_json(path: Path, payload: Any) -> None:
    """Write trace payload with canonical JSON-safe conversion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = to_jsonable(payload)
    path.write_text(json.dumps(safe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def payload_hash(obj: Any) -> str:
    return hashlib.sha256(_canonical_json(obj).encode("utf-8")).hexdigest()


def belief_summary_hash(belief_summary: dict[str, Any] | None) -> str | None:
    if not belief_summary:
        return None
    slim = {
        "entropy": belief_summary.get("entropy"),
        "sigma_trace": belief_summary.get("sigma_trace"),
        "mu": [round(float(x), 6) for x in (belief_summary.get("mu") or [])[:32]],
    }
    return payload_hash(slim)


def serialize_hyp_vectorized(hyp_vectorized: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return to_jsonable(hyp_vectorized)  # type: ignore[return-value]


def normalize_option_writer_rows(
    opts: Any,
    hypotheses: list[dict[str, Any]],
    *,
    include_none: bool = True,
) -> list[dict[str, Any]]:
    """Coerce trace option_writer_output (strings or dicts) to option_id/option_text rows."""

    hid_order = [str(h["hypothesis_id"]) for h in hypotheses]
    if not opts:
        core: list[dict[str, Any]] = [
            {"option_id": hid, "option_text": str(h.get("text_description") or "")}
            for hid, h in zip(hid_order, hypotheses)
        ]
    elif isinstance(opts, list) and opts and isinstance(opts[0], str):
        texts = [str(x) for x in opts]
        core = []
        for i, hid in enumerate(hid_order):
            core.append(
                {
                    "option_id": hid,
                    "option_text": texts[i] if i < len(texts) else "",
                }
            )
        if include_none:
            none_text = "None of the above fits well."
            if len(texts) > len(hid_order):
                none_text = texts[len(hid_order)]
            elif texts and "none of the above" in texts[-1].lower():
                none_text = texts[-1]
            core.append({"option_id": "none", "option_text": none_text})
        return core
    elif isinstance(opts, list) and opts and isinstance(opts[0], dict):
        by_id = {
            str(o.get("option_id") or ""): str(o.get("option_text") or "")
            for o in opts
            if isinstance(o, dict)
        }
        core = [{"option_id": hid, "option_text": by_id.get(hid, "")} for hid in hid_order]
        if include_none and "none" in by_id:
            core.append({"option_id": "none", "option_text": by_id["none"]})
    else:
        core = [
            {"option_id": hid, "option_text": str(h.get("text_description") or "")}
            for hid, h in zip(hid_order, hypotheses)
        ]

    if include_none and not any(o.get("option_id") == "none" for o in core):
        core.append({"option_id": "none", "option_text": "None of the above fits well."})
    return core


def deserialize_hyp_vectorized(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        h = copy.deepcopy(row)
        if "vector" in h and h["vector"] is not None:
            h["vector"] = np.asarray(h["vector"], dtype=np.float64)
        out.append(h)
    return out


def _llm_cache_keys_from_turn(turn_log: dict[str, Any]) -> dict[str, Any]:
    keys: dict[str, Any] = {}
    bridge = turn_log.get("llm_semantic_bridge") or {}
    for comp in (
        "direction_translator",
        "hypothesis_generator",
        "option_writer",
    ):
        blk = bridge.get(comp)
        if isinstance(blk, dict):
            keys[comp] = {
                "cache_key_sha256": blk.get("cache_key_sha256"),
                "cache_key_version": blk.get("cache_key_version"),
                "payload_hash": blk.get("payload_hash"),
                "prompt_hash": blk.get("prompt_hash"),
                "cache_hit": blk.get("cache_hit"),
                "cache_miss": blk.get("cache_miss"),
            }
    gen = turn_log.get("frontier_clustered_v32_generation") or {}
    cb2 = gen.get("cluster_bound_generation_v2") if isinstance(gen, dict) else {}
    if isinstance(cb2, dict) and cb2:
        keys["cluster_bound_generation_v2"] = {
            "cache_key_sha256": cb2.get("cache_key_sha256"),
            "cache_key_version": cb2.get("cache_key_version"),
            "payload_hash": cb2.get("payload_hash"),
            "prompt_hash": cb2.get("prompt_hash"),
            "cache_hit": cb2.get("cache_hit"),
            "cache_miss": cb2.get("cache_miss"),
        }
    return keys


def build_turn_trace_record(
    *,
    seed: int,
    session_id: str,
    turn_log: dict[str, Any],
    hyp_vectorized: list[dict[str, Any]] | None,
    translated: list[dict[str, Any]] | None,
    recommendation_after: list[str] | None = None,
) -> dict[str, Any]:
    """Build one turn trace dict from runtime turn_log + vectorized hypotheses."""

    gen = turn_log.get("frontier_clustered_v32_generation") or {}
    cb2 = gen.get("cluster_bound_generation_v2") if isinstance(gen, dict) else {}
    q = turn_log.get("question") or {}
    uc = turn_log.get("user_choice") or {}
    rec = turn_log.get("recommendation") or {}
    sel = copy.deepcopy(turn_log.get("question_selection") or {})
    if "best_ids" not in sel and sel.get("best_question_ids"):
        sel["best_ids"] = list(sel.get("best_question_ids"))

    parsed = list(gen.get("parsed_hypotheses") or []) if isinstance(gen, dict) else []
    bound_ids = [
        h.get("bound_cluster_id")
        for h in parsed
        if isinstance(h, dict)
    ]

    ow_out = None
    bridge = turn_log.get("llm_semantic_bridge") or {}
    ow_audit = bridge.get("option_writer") or {}
    if q.get("options"):
        ow_out = [
            {"option_id": str(h.get("hypothesis_id") or ""), "option_text": str(h.get("text_description") or "")}
            for h in (turn_log.get("hypotheses") or [])
        ]

    return {
        "turn_id": int(turn_log.get("turn") or 0),
        "belief_before_hash": belief_summary_hash(turn_log.get("belief_before")),
        "recommendation_top10_before": list(rec.get("recommended_items") or [])[:10],
        "pre_gate_audit": turn_log.get("frontier_clustered_v32_pre_gate"),
        "oos_skip_audit": turn_log.get("frontier_clustered_v32_oos_skip"),
        "compensation_audit": turn_log.get("frontier_clustered_v32_compensation"),
        "cluster_bound_generation_v2_input_hash": (
            cb2.get("payload_hash") if isinstance(cb2, dict) else None
        ),
        "cluster_bound_generation_v2_parsed_output": (
            copy.deepcopy(parsed) if parsed else None
        ),
        "cluster_bound_generation_v2_block": (
            copy.deepcopy(cb2) if isinstance(cb2, dict) and cb2 else None
        ),
        "selected_bound_cluster_ids": bound_ids,
        "representative_items_by_hypothesis": {
            str(h.get("hypothesis_id") or h.get("option_id") or ""): list(
                h.get("representative_items") or h.get("representative_item_ids") or []
            )
            for h in parsed
            if isinstance(h, dict)
        },
        "generated_option_hypotheses": copy.deepcopy(parsed),
        "hypotheses_log": copy.deepcopy(turn_log.get("hypotheses") or []),
        "hyp_vectorized": serialize_hyp_vectorized(hyp_vectorized or []),
        "option_writer_input_hash": ow_audit.get("payload_hash") if isinstance(ow_audit, dict) else None,
        "option_writer_output": copy.deepcopy(q.get("options") or ow_out),
        "question_text": str(q.get("question_text") or ""),
        "options_text": list(q.get("options") or []),
        "user_choice": copy.deepcopy(uc),
        "selected_option_id": str(uc.get("selected_option_id") or "") or None,
        "selected_option_role": _choice_role(parsed, uc.get("selected_option_id")),
        "selected_bound_cluster_id": _choice_bound(parsed, uc.get("selected_option_id")),
        "belief_update_executed": bool(turn_log.get("belief_update_executed")),
        "do_ask": bool(turn_log.get("do_ask")),
        "question_selection": copy.deepcopy(sel),
        "recommendation_top10_after": list(recommendation_after or [])[:10],
        "target_rank_audit": (turn_log.get("full_rank_effect_audit") or {}).get("best_target_rank"),
        "cache_keys": _llm_cache_keys_from_turn(turn_log),
        "llm_response_hashes": {
            comp: (blk or {}).get("raw_storage_path") or (blk or {}).get("cache_key_sha256")
            for comp, blk in (
                ("direction_translator", bridge.get("direction_translator")),
                ("option_writer", bridge.get("option_writer")),
                ("cluster_bound_generation_v2", cb2 if isinstance(cb2, dict) else None),
            )
            if isinstance(blk, dict)
        },
        "frontier_clustered_v32_generation": copy.deepcopy(gen) if gen else None,
        "translated_directions": copy.deepcopy(translated) if translated else None,
    }


def _choice_role(parsed: list[dict[str, Any]], selected_oid: Any) -> str | None:
    oid = str(selected_oid or "")
    for h in parsed:
        if str(h.get("hypothesis_id") or h.get("option_id") or "") == oid:
            return str(h.get("role") or "") or None
    return None


def _choice_bound(parsed: list[dict[str, Any]], selected_oid: Any) -> Any:
    oid = str(selected_oid or "")
    for h in parsed:
        if str(h.get("hypothesis_id") or h.get("option_id") or "") == oid:
            return h.get("bound_cluster_id")
    return None


class TraceReplayMissError(RuntimeError):
    """Raised when trace replay cannot find required turn data."""


class SessionTraceController:
    """Record or replay per-session turn traces."""

    def __init__(
        self,
        *,
        mode: str,
        trace_path: Path,
        seed: int,
        session_id: str,
        run_id: str,
        fail_fast_on_miss: bool = False,
    ) -> None:
        self.mode = str(mode)
        self.trace_path = trace_path
        self.seed = int(seed)
        self.session_id = str(session_id)
        self.run_id = str(run_id)
        self.fail_fast_on_miss = bool(fail_fast_on_miss)
        self._turns: list[dict[str, Any]] = []
        self._missing: list[dict[str, Any]] = []
        self._missing_fields: list[dict[str, Any]] = []
        self._consumed_turn_ids: set[int] = set()
        self._loaded: dict[str, Any] | None = None
        self._loaded_ok = False
        if self.is_replay:
            if trace_path.is_file():
                self._loaded = json.loads(trace_path.read_text(encoding="utf-8"))
                self._loaded_ok = True
            else:
                self._missing.append(
                    {"reason": "trace_file_missing", "path": str(trace_path)}
                )
        elif self.is_record:
            trace_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_cfg(
        cls,
        cfg: dict[str, Any],
        *,
        run_dir: Path | None,
        session: dict[str, Any],
        run_id: str,
    ) -> SessionTraceController | None:
        trace_cfg = (cfg.get("simulation") or {}).get("trace") or {}
        if not trace_cfg.get("enabled"):
            return None
        cache_raw = (cfg.get("llm") or {}).get("cache") or {}
        cache_cfg = cache_raw if isinstance(cache_raw, dict) else {}
        cache_mode = str(cache_cfg.get("mode", "normal"))
        mode = str(trace_cfg.get("mode") or cache_mode)
        if mode not in ("trace_cache_fill", "trace_replay", "record", "replay"):
            if cache_mode == "trace_cache_fill":
                mode = "trace_cache_fill"
            elif cache_mode == "trace_replay":
                mode = "trace_replay"
            else:
                return None

        seed = int(cfg.get("seed", 42))
        session_id = str(session.get("episode_id", session["user_id"]))
        root = Path(str(trace_cfg.get("root") or ""))
        if not root.is_absolute():
            repo = Path(__file__).resolve().parents[2]
            root = repo / root
        if mode in ("trace_replay", "replay"):
            trace_path = root / str(seed) / f"{session_id}.json"
        else:
            trace_path = root / str(seed) / f"{session_id}.json"
        return cls(
            mode=mode,
            trace_path=trace_path,
            seed=seed,
            session_id=session_id,
            run_id=run_id,
            fail_fast_on_miss=bool(trace_cfg.get("fail_fast_on_miss", False)),
        )

    @property
    def is_record(self) -> bool:
        return self.mode in ("trace_cache_fill", "record")

    @property
    def is_replay(self) -> bool:
        return self.mode in ("trace_replay", "replay")

    @property
    def trace_loaded(self) -> bool:
        return bool(self._loaded_ok and self._loaded)

    def mark_turn_consumed(self, turn_id: int) -> None:
        self._consumed_turn_ids.add(int(turn_id))

    def record_missing_field(self, turn_id: int, field: str, reason: str = "missing") -> None:
        self._missing_fields.append(
            {"turn_id": int(turn_id), "field": str(field), "reason": reason}
        )

    def turn_was_consumed(self, turn_id: int) -> bool:
        return int(turn_id) in self._consumed_turn_ids

    def consumption_summary(self) -> dict[str, Any]:
        turns = (self._loaded or {}).get("turns") or []
        return {
            "trace_loaded": self.trace_loaded,
            "trace_path": str(self.trace_path),
            "n_turns_in_trace": len(turns),
            "trace_consumed_turns": len(self._consumed_turn_ids),
            "consumed_turn_ids": sorted(self._consumed_turn_ids),
            "trace_missing": list(self._missing),
            "trace_missing_fields": list(self._missing_fields),
        }

    def record_turn(
        self,
        turn_log: dict[str, Any],
        *,
        hyp_vectorized: list[dict[str, Any]] | None,
        translated: list[dict[str, Any]] | None = None,
        recommendation_after: list[str] | None = None,
    ) -> None:
        if not self.is_record:
            return
        rec = build_turn_trace_record(
            seed=self.seed,
            session_id=self.session_id,
            turn_log=turn_log,
            hyp_vectorized=hyp_vectorized,
            translated=translated,
            recommendation_after=recommendation_after,
        )
        self._turns.append(rec)

    def finalize(self) -> Path:
        """Write session trace JSON; raises on failure (caller may catch)."""
        if not self.is_record:
            return self.trace_path
        payload = {
            "trace_version": TRACE_VERSION,
            "seed": self.seed,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "n_turns": len(self._turns),
            "turns": self._turns,
        }
        write_trace_json(self.trace_path, payload)
        return self.trace_path

    def finalize_error_path(self) -> Path:
        return self.trace_path.with_suffix(".finalize_error.json")

    def write_finalize_error_report(self, exc: BaseException) -> Path:
        """Persist finalize failure without losing turn count context."""
        report = {
            "trace_version": TRACE_VERSION,
            "seed": self.seed,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "n_turns_buffered": len(self._turns),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        err_path = self.finalize_error_path()
        write_trace_json(err_path, report)
        return err_path

    def get_turn(self, turn_id: int) -> dict[str, Any] | None:
        if not self.is_replay or not self._loaded:
            return None
        for t in self._loaded.get("turns") or []:
            if int(t.get("turn_id") or 0) == int(turn_id):
                return t
        return None

    def require_turn(self, turn_id: int) -> dict[str, Any]:
        t = self.get_turn(turn_id)
        if t is None:
            self._missing.append({"turn_id": turn_id, "reason": "trace_missing"})
            if self.fail_fast_on_miss:
                raise TraceReplayMissError(
                    f"trace missing session={self.session_id} turn={turn_id}"
                )
        return t or {}

    def replay_hyp_vectorized(self, turn_id: int) -> list[dict[str, Any]] | None:
        t = self.get_turn(turn_id)
        if not t:
            return None
        rows = t.get("hyp_vectorized") or []
        if not rows:
            return None
        return deserialize_hyp_vectorized(rows)

    def replay_frontier_block(self, turn_id: int) -> dict[str, Any] | None:
        t = self.get_turn(turn_id)
        if not t:
            return None
        blk = t.get("frontier_clustered_v32_generation")
        return copy.deepcopy(blk) if blk else None

    def replay_translated(self, turn_id: int) -> list[dict[str, Any]] | None:
        t = self.get_turn(turn_id)
        if not t:
            return None
        tr = t.get("translated_directions")
        return copy.deepcopy(tr) if tr else None

    def replay_policy(self, turn_id: int) -> tuple[bool, dict[str, Any]] | None:
        t = self.get_turn(turn_id)
        if not t:
            return None
        sel = copy.deepcopy(t.get("question_selection") or {})
        if "best_ids" not in sel and "best_question_ids" in sel:
            sel["best_ids"] = list(sel.get("best_question_ids") or [])
        best_ids = list(sel.get("best_ids") or [])
        hv = t.get("hyp_vectorized") or []
        hv_ids = [str(h.get("hypothesis_id") or "") for h in hv if isinstance(h, dict)]
        if hv_ids and best_ids and not any(str(x) in hv_ids for x in best_ids):
            sel["best_ids"] = hv_ids[: len(best_ids)] if len(hv_ids) >= len(best_ids) else hv_ids
            self.record_missing_field(
                turn_id,
                "question_selection.best_ids",
                "hyp_id_mismatch_fallback",
            )
        return bool(t.get("do_ask")), sel

    def replay_selected_hypotheses(
        self, turn_id: int, hyp_vectorized: list[dict[str, Any]], best_ids: list[str]
    ) -> list[dict[str, Any]]:
        ids = {str(x) for x in best_ids}
        picked = [h for h in hyp_vectorized if str(h.get("hypothesis_id") or "") in ids]
        if picked:
            return picked
        t = self.get_turn(turn_id)
        if not t:
            return picked
        hv = self.replay_hyp_vectorized(turn_id) or hyp_vectorized
        ids = {str(x) for x in best_ids}
        picked = [h for h in hv if str(h.get("hypothesis_id") or "") in ids]
        if picked:
            return picked
        if hv:
            self.record_missing_field(
                turn_id,
                "question_selection.best_ids",
                "empty_selected_hyps_fallback_all_vectorized",
            )
            return hv[: len(best_ids)] if best_ids else hv
        return picked

    def replay_oos_skip_audit(self, turn_id: int) -> dict[str, Any] | None:
        t = self.get_turn(turn_id)
        if not t:
            return None
        oos = t.get("oos_skip_audit")
        return copy.deepcopy(oos) if isinstance(oos, dict) and oos else None

    def replay_option_writer_output(
        self,
        turn_id: int,
        hypotheses: list[dict[str, Any]],
        *,
        include_none: bool = True,
    ) -> list[dict[str, Any]]:
        t = self.get_turn(turn_id)
        if not t:
            self.record_missing_field(turn_id, "turn_trace", "trace_missing")
            return normalize_option_writer_rows([], hypotheses, include_none=include_none)
        opts = t.get("option_writer_output")
        if not opts:
            opts = t.get("options_text") or []
        if not opts:
            self.record_missing_field(turn_id, "option_writer_output", "missing")
        return normalize_option_writer_rows(
            copy.deepcopy(opts) if opts else [],
            hypotheses,
            include_none=include_none,
        )

    def replay_user_choice(self, turn_id: int) -> dict[str, Any] | None:
        t = self.get_turn(turn_id)
        if not t:
            return None
        uc = t.get("user_choice")
        return copy.deepcopy(uc) if isinstance(uc, dict) else None

    def missing_report(self) -> list[dict[str, Any]]:
        return list(self._missing)


def trace_root_from_cfg(cfg: dict[str, Any]) -> Path:
    trace_cfg = (cfg.get("simulation") or {}).get("trace") or {}
    root = Path(str(trace_cfg.get("root") or "experiments/trace_cache"))
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2] / root
    return root


def count_trace_files(root: Path) -> int:
    if not root.is_dir():
        return 0
    n = 0
    for p in root.rglob("*.json"):
        if p.name.endswith(".json") and p.parent.name.isdigit():
            n += 1
    return n
