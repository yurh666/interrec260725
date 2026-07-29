"""Write CACHE_NEUTRALITY_AUDIT.md + neutrality_cache_stats.json after a run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.neutrality_cache_stats import load_json, write_json


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / max(len(xs), 1))


def write_neutrality_audit_md(run_dir: Path) -> Path | None:
    """Persist stats JSON and markdown audit. Returns path to markdown or None if no LLM neutrality."""

    run_dir = Path(run_dir)
    stats_path = run_dir / "neutrality_cache_stats.json"
    write_json(stats_path)
    stats = load_json(stats_path)
    jc, jh, jm = int(stats.get("judge_calls", 0)), int(stats.get("judge_cache_hits", 0)), int(stats.get("judge_cache_misses", 0))
    rc, rh, rm = int(stats.get("rewrite_calls", 0)), int(stats.get("rewrite_cache_hits", 0)), int(stats.get("rewrite_cache_misses", 0))

    if jc + rc == 0:
        return None

    lat_path = run_dir / "latency_profile.jsonl"
    rows = _read_jsonl(lat_path)

    def _live_mean(comp: str) -> float:
        xs = [
            float(r["latency_sec"])
            for r in rows
            if r.get("event") == "structured_parse"
            and r.get("component") == comp
            and not r.get("cache_hit")
        ]
        return _mean(xs) if xs else 0.0

    mj = _live_mean("neutrality_judge")
    mr = _live_mean("neutrality_rewrite")

    apis_saved = jh + rh
    wall_saved_est = jh * mj + rh * mr

    j_ratio = jh / max(jc, 1)
    r_ratio = rh / max(rc, 1)
    noop = int(stats.get("retries_skipped_noop_rewrite", 0))

    md = [
        "# Neutrality disk cache & dedup audit",
        "",
        "## Summary counters",
        f"- judge: calls={jc}, hits={jh}, misses={jm}, hit_ratio={j_ratio:.4f}",
        f"- rewrite: calls={rc}, hits={rh}, misses={rm}, hit_ratio={r_ratio:.4f}",
        f"- retries_skipped_noop_rewrite: {noop}",
        "",
        "## Cross-session dedup / reuse",
        "Disk cache keys are canonical JSON (sorted ids, normalized whitespace on texts).",
        f"- **Dedup reuse (judge)**: {jh} cached judge responses reused (same canonical MCQ wording).",
        f"- **Dedup reuse (rewrite)**: {rh} cached rewrite responses reused.",
        f"- **Total API calls avoided (sum of hits)**: {apis_saved}",
        "",
        "## Estimated wall time saved (rough)",
        f"- Mean live `neutrality_judge` latency (this run): {mj:.3f}s",
        f"- Mean live `neutrality_rewrite` latency (this run): {mr:.3f}s",
        f"- **Estimated seconds saved** ≈ hits_j×mean_j + hits_r×mean_r = **{wall_saved_est:.1f}s**",
        "",
        "## Prompt compaction (neutrality templates only)",
        "- Instruction text shortened; JSON schema requirements unchanged (balanced / favored_option_id / critique; rewrite array shape).",
        "- Token/latency delta vs pre-change baseline: re-run `scripts/measure_neutrality_prompt_bloat.py` and compare to archived numbers if needed.",
        "",
        "## Warm cache",
        "- Run once: `users 1–20` (e.g. `--run-id-suffix _warmup`) then full run reuses `.cache/interrec_llm` namespaces `neutrality_judge` / `neutrality_rewrite`.",
        "",
    ]
    out = run_dir / "CACHE_NEUTRALITY_AUDIT.md"
    out.write_text("\n".join(md), encoding="utf-8")
    return out
