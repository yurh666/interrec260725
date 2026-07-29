from __future__ import annotations

import csv
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.utils.config import save_config
from src.logging.full_log_sidecar import (
    emergency_strip_for_jsonl,
    offload_frontier_v32_turn_for_jsonl,
    scrub_unescaped_physical_newlines_nested,
    try_roundtrip_json,
)
from src.utils.io import append_jsonl, ensure_dir, write_json
from src.utils.time import now_iso


def _aggregate_llm_semantic_bridge(all_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll up llm_semantic_bridge across all turns."""

    comps = ("direction_translator", "hypothesis_generator", "option_writer")
    by_c: dict[str, Any] = {c: {"implementation_mode_counts": {}} for c in comps}

    extras: dict[str, int] = {
        "direction_translator_cache_hits": 0,
        "hypothesis_generator_cache_hits": 0,
        "option_writer_cache_hits": 0,
        "option_writer_turns_skipped": 0,
    }

    turns = 0
    fallback_any = False
    mock_any = False

    for ep in all_results:
        for row in ep.get("turns", []) or []:
            turns += 1
            bridge = row.get("llm_semantic_bridge") or {}
            for c in comps:
                block = bridge.get(c)
                if not isinstance(block, dict):
                    continue
                if block.get("skipped"):
                    if c == "option_writer":
                        extras["option_writer_turns_skipped"] += 1
                    continue
                mode = str(block.get("implementation_mode", "unknown"))
                cnts = by_c[c]["implementation_mode_counts"]
                cnts[mode] = cnts.get(mode, 0) + 1
                if "fallback" in mode or "rule_based" in mode:
                    fallback_any = True
                if "mock" in mode:
                    mock_any = True
                if block.get("cache_hit"):
                    extras[f"{c}_cache_hits"] = extras.get(f"{c}_cache_hits", 0) + 1

    return {
        "n_turn_rows": turns,
        "by_component": by_c,
        "cache_hits_detail": {
            k: v for k, v in extras.items() if k.endswith("_cache_hits")
        },
        "option_writer_turns_skipped": extras["option_writer_turns_skipped"],
        "any_real_llm_fallback_contamination": fallback_any,
        "any_mock_semantic_layer": mock_any,
    }


def _aggregate_frontier_clustered_v32_generation(all_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Telemetry rollup for frontier_clustered_v32_generation blocks (stored on turn rows only)."""

    blocks = 0
    schema_hard_pass = 0
    fallback_turns = 0
    hyp_nonempty_nf = 0
    internal_jargon_hits = 0
    mojibake_hits = 0
    malformed_schema = 0
    modes: dict[str, int] = {}
    parsed_nonempty = 0

    for ep in all_results:
        for row in ep.get("turns", []) or []:
            if not isinstance(row, dict):
                continue
            blk = row.get("frontier_clustered_v32_generation")
            if not isinstance(blk, dict):
                continue
            blocks += 1
            mk = str(blk.get("mode") or "?")
            modes[mk] = modes.get(mk, 0) + 1

            ph = blk.get("parsed_hypotheses")
            if isinstance(ph, list) and len(ph) > 0:
                parsed_nonempty += 1

            sch = blk.get("schema_validation_result") if isinstance(blk.get("schema_validation_result"), dict) else {}
            details = sch.get("hard_errors_detail") or []
            internal_jargon_hits += sum(1 for e in details if "internal_jargon" in str(e).lower())
            mojibake_hits += sum(1 for e in details if "mojibake" in str(e).lower())
            hard_ok = bool(sch.get("schema_hard_pass"))
            if hard_ok:
                schema_hard_pass += 1

            sess_h_like = blk.get("parsed_hypotheses")
            if not blk.get("fallback_used") and isinstance(sess_h_like, list) and len(sess_h_like) > 0:
                hyp_nonempty_nf += 1

            fb = blk.get("fallback_used")
            if fb:
                fallback_turns += 1
            if fb and not hard_ok:
                malformed_schema += 1

    return {
        "generation_blocks_seen": blocks,
        "schema_hard_pass_count": schema_hard_pass,
        "fallback_count": fallback_turns,
        "parsed_hypotheses_nonempty_turns": parsed_nonempty,
        "hypotheses_nonempty_without_fallback_turns": hyp_nonempty_nf,
        "internal_jargon_hard_error_signals": internal_jargon_hits,
        "mojibake_hard_error_signals": mojibake_hits,
        "schema_fail_then_fallback_approx": malformed_schema,
        "mode_histogram": modes,
    }


class RunLogger:
    """Creates and manages the per-run experiment directory.

    Directory layout::

        experiments/runs/{run_id}/
            config.yaml
            environment.txt
            git_commit.txt
            run_summary.json
            full_log.jsonl
            metrics.csv
            metrics_by_turn.csv
            artifacts/
            figures/
    """

    def __init__(self, run_id: str, base_dir: str | Path = "experiments/runs") -> None:
        self.run_id = run_id
        self.run_dir = Path(base_dir) / run_id
        ensure_dir(self.run_dir)
        ensure_dir(self.run_dir / "artifacts")
        ensure_dir(self.run_dir / "figures")
        self._turn_rows: list[dict[str, Any]] = []
        self._errors: list[dict[str, Any]] = []
        self._fc32_strict_jsonl: bool = False
        self._fc32_jsonl_preview_chars: int = 800
        self._fc32_jsonl_self_check: bool = True
        self._fc32_write_sidecar: bool = True

    def configure_full_log_policy(self, cfg: dict[str, Any]) -> None:
        """Honor ``frontier_clustered_v32.strict_jsonl_full_log`` (+ optional previews)."""

        fc = (((cfg.get("ask") or {}).get("hypothesis_generation") or {}).get("frontier_clustered_v32") or {})
        self._fc32_strict_jsonl = bool(fc.get("strict_jsonl_full_log", False))
        self._fc32_jsonl_preview_chars = int(fc.get("jsonl_raw_preview_chars", 800))
        self._fc32_jsonl_self_check = bool(fc.get("jsonl_self_check_after_write", True)) if self._fc32_strict_jsonl else False
        perf = cfg.get("performance_optimizations") or {}
        self._fc32_write_sidecar = not (
            bool(perf.get("enabled")) and bool(perf.get("reduce_sidecar_logging"))
        )

    # ----- Setup -----

    def write_config(self, cfg: dict[str, Any]) -> None:
        save_config(cfg, self.run_dir / "config.yaml")

    def write_environment(self) -> None:
        lines = [
            f"run_id: {self.run_id}",
            f"timestamp: {now_iso()}",
            f"python: {platform.python_version()}",
            f"platform: {platform.platform()}",
        ]
        try:
            import sys
            pip_output = subprocess.check_output(
                [sys.executable, "-m", "pip", "list", "--format=freeze"],
                stderr=subprocess.DEVNULL,
                timeout=30,
            ).decode()
            lines.append("packages:")
            lines += [f"  {l}" for l in pip_output.strip().splitlines()]
        except Exception:
            lines.append("packages: <unavailable>")
        (self.run_dir / "environment.txt").write_text("\n".join(lines), encoding="utf-8")

    def write_git_info(self, project_dir: str | Path = ".") -> None:
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(project_dir),
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            status = subprocess.check_output(
                ["git", "status", "--short"],
                cwd=str(project_dir),
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            commit = "git_not_available"
            status = ""
        (self.run_dir / "git_commit.txt").write_text(
            f"commit: {commit}\nstatus:\n{status}", encoding="utf-8"
        )

    def log_embedding_build_audit(self, manifest: dict[str, Any]) -> None:
        """Append offline embedding audit (from ``embedding_build_log.json``) to ``full_log.jsonl``."""
        record: dict[str, Any] = {
            "type": "embedding_build_audit",
            "embedding_runtime": manifest.get("embedding_runtime"),
            "embedding_fallback_to_mock": bool(manifest.get("embedding_fallback_to_mock", False)),
        }
        for k in ("timestamp", "embedding_mode_requested", "build_status", "primary_error", "primary_embedding_error_on_failure"):
            if k in manifest:
                record[k] = manifest[k]
        append_jsonl([record], self.run_dir / "full_log.jsonl")

    def log_embedding_audit_missing(self, processed_dir: str | Path) -> None:
        append_jsonl(
            [
                {
                    "type": "embedding_build_audit",
                    "missing": True,
                    "note": f"No embedding_build_log.json under {processed_dir}; audit fields unknown.",
                }
            ],
            self.run_dir / "full_log.jsonl",
        )

    def log_run_warning(self, scope: str, message: str) -> None:
        """Prominent JSONL WARNING row + stderr (embedding_fallback_to_mock etc.)."""
        append_jsonl(
            [{"type": "warning", "scope": scope, "message": message, "timestamp": now_iso()}],
            self.run_dir / "full_log.jsonl",
        )
        print(f"[run][WARNING:{scope}] {message}", file=sys.stderr)

    # ----- Mock component tracking -----

    def write_mock_warning(self, impl_modes: dict[str, str]) -> None:
        """Create MOCK_WARNING.md if any component is in mock mode.

        This file is written at run startup so it is immediately visible.
        It is intentionally eye-catching to remind that the run is NOT
        a production-quality experiment.
        """
        mock_items = {k: v for k, v in impl_modes.items() if "mock" in str(v).lower()}
        real_items = {k: v for k, v in impl_modes.items() if "mock" not in str(v).lower()}

        lines = ["# ⚠️  MOCK COMPONENTS ACTIVE — NOT A PRODUCTION RUN\n"]

        if mock_items:
            lines += [
                "This run used **rule-based / mock** replacements for the following modules.",
                "Results are for development/debugging purposes only.",
                "**These modules must be replaced before reporting paper results.**",
                "",
                "| Module | Status | What is mocked |",
                "|--------|--------|----------------|",
            ]
            mock_descriptions = {
                "llm": "LLM API not called — direction translation and hypothesis generation use rule-based templates",
                "direction_translator": "Directions described by item IDs only (no semantic LLM translation)",
                "hypothesis_generator": "Hypotheses generated by fixed rules, not LLM reasoning",
                "option_writer": "Option text templated, not polished by LLM",
                "user_simulator": "User choice is deterministic argmax (no stochastic simulation)",
                "item_encoder": "Item embeddings are random vectors (no TF-IDF/BGE encoding)",
            }
            for k, v in mock_items.items():
                desc = mock_descriptions.get(k, f"using mode: {v}")
                lines.append(f"| `{k}` | **MOCK** | {desc} |")

            lines += [
                "",
                "## What needs to change for paper results",
                "",
                "| Module | Required replacement |",
                "|--------|---------------------|",
                "| `llm` | Set `llm.mode: openai` and provide `OPENAI_API_KEY` |",
                "| `item_encoder` | Set `embedding.mode: bge` and run on GPU |",
                "| `user_simulator` | Set `simulation.user_choice_mode: stochastic_sample` |",
                "",
            ]
        else:
            lines = [
                "# ✅  All modules using real implementations",
                "",
                "No mock components detected in this run.",
                "",
                "| Module | Status |",
                "|--------|--------|",
            ]
            for k, v in real_items.items():
                lines.append(f"| `{k}` | ✅ `{v}` |")

        (self.run_dir / "MOCK_WARNING.md").write_text("\n".join(lines), encoding="utf-8")

    # ----- Per-turn logging -----

    def rebuild_turn_rows_from_full_log(self, top_k: int = 10) -> None:
        """Recompute ``_turn_rows`` by scanning ``full_log.jsonl`` (used after --resume)."""

        import json

        self._turn_rows.clear()
        path = self.run_dir / "full_log.jsonl"
        if not path.is_file():
            return
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(o, dict):
                    continue
                if o.get("type") in ("embedding_build_audit", "warning", "error"):
                    continue
                if "turn" not in o or "metrics" not in o:
                    continue
                m = o.get("metrics", {})
                self._turn_rows.append(
                    {
                        "method": o.get("method", "interrec"),
                        "dataset": o.get("dataset", ""),
                        "seed": o.get("seed", 42),
                        "user_id": o.get("user_id", ""),
                        "turn": o.get("turn", 0),
                        f"HitRate@{top_k}": m.get(f"HitRate@{top_k}", ""),
                        f"NDCG@{top_k}": m.get(f"NDCG@{top_k}", ""),
                        f"MRR@{top_k}": m.get(f"MRR@{top_k}", ""),
                        "entropy": o.get("belief_after", {}).get("entropy", ""),
                        "preference_error": m.get("preference_error", ""),
                        "ask": o.get("question_selection", {}).get("ask", False),
                        "IG": o.get("question_selection", {}).get("IG", ""),
                    }
                )

    def log_turn(self, turn_log: dict[str, Any]) -> None:
        """Append one turn record to full_log.jsonl and buffer metrics_by_turn."""
        payload = turn_log
        if self._fc32_strict_jsonl:
            payload, _meta = offload_frontier_v32_turn_for_jsonl(
                turn_log,
                run_dir=self.run_dir,
                preview_chars=self._fc32_jsonl_preview_chars,
                write_sidecar=self._fc32_write_sidecar,
            )
            scrub_unescaped_physical_newlines_nested(payload)

        ok, err = True, ""
        if self._fc32_strict_jsonl and self._fc32_jsonl_self_check:
            ok, err = try_roundtrip_json(payload)
            if not ok:
                scrub_unescaped_physical_newlines_nested(payload)
                ok, err = try_roundtrip_json(payload)
            if not ok:
                payload = emergency_strip_for_jsonl(payload)
                ok, err2 = try_roundtrip_json(payload)
                err = err + ";" + err2 if err2 else err
                append_jsonl(
                    [
                        {
                            "type": "warning",
                            "scope": "full_log_strict_json_roundtrip",
                            "message": err[:2000],
                            "timestamp": now_iso(),
                            "run_id": self.run_id,
                        },
                    ],
                    self.run_dir / "full_log.jsonl",
                )

        append_jsonl([payload], self.run_dir / "full_log.jsonl")
        m = turn_log.get("metrics", {})
        top_k = 10
        self._turn_rows.append(
            {
                "method": turn_log.get("method", "interrec"),
                "dataset": turn_log.get("dataset", ""),
                "seed": turn_log.get("seed", 42),
                "user_id": turn_log.get("user_id", ""),
                "turn": turn_log.get("turn", 0),
                f"HitRate@{top_k}": m.get(f"HitRate@{top_k}", ""),
                f"NDCG@{top_k}": m.get(f"NDCG@{top_k}", ""),
                f"MRR@{top_k}": m.get(f"MRR@{top_k}", ""),
                "entropy": turn_log.get("belief_after", {}).get("entropy", ""),
                "preference_error": m.get("preference_error", ""),
                "ask": turn_log.get("question_selection", {}).get("ask", False),
                "IG": turn_log.get("question_selection", {}).get("IG", ""),
            }
        )

    def log_error(self, user_id: str, tb: str) -> None:
        self._errors.append({"user_id": user_id, "error": tb})
        append_jsonl([{"type": "error", "user_id": user_id, "traceback": tb}], self.run_dir / "full_log.jsonl")

    # ----- Summary / metrics flush -----

    def write_metrics(self, aggregate: dict[str, Any], all_results: list[dict[str, Any]]) -> None:
        """Write metrics.csv, metrics_by_turn.csv, and run_summary.json."""
        # metrics.csv (one row per run)
        mc_path = self.run_dir / "metrics.csv"
        with mc_path.open("w", newline="", encoding="utf-8") as f:
            if aggregate:
                writer = csv.DictWriter(f, fieldnames=list(aggregate.keys()))
                writer.writeheader()
                writer.writerow(aggregate)

        # metrics_by_turn.csv
        mbt_path = self.run_dir / "metrics_by_turn.csv"
        if self._turn_rows:
            with mbt_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(self._turn_rows[0].keys()))
                writer.writeheader()
                writer.writerows(self._turn_rows)

        # run_summary.json
        summary = {
            "run_id": self.run_id,
            "timestamp": now_iso(),
            "n_sessions": len(all_results),
            "n_turns": len(self._turn_rows),
            "errors": len(self._errors),
            "aggregate_metrics": aggregate,
            "llm_semantic_bridge_aggregate": _aggregate_llm_semantic_bridge(all_results),
            "frontier_clustered_v32_aggregate": _aggregate_frontier_clustered_v32_generation(all_results),
        }
        write_json(summary, self.run_dir / "run_summary.json")
