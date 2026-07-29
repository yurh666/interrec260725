from __future__ import annotations

from pathlib import Path
from typing import Any

from src.embedding.retrieval_manifest import RUNTIME_LEXICAL_VS_DENSE_AUDIT
from src.utils.io import read_json
from src.utils.time import now_iso


def generate_run_report(run_dir: str | Path, cfg: dict[str, Any]) -> str:
    run_dir = Path(run_dir)
    summary_path = run_dir / "run_summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    run_id = summary.get("run_id", run_dir.name)
    aggregate = summary.get("aggregate_metrics", {})
    impl_modes = cfg.get("_impl_modes", {})

    mock_list = [(k, v) for k, v in impl_modes.items() if "mock" in str(v).lower()]
    real_list = [(k, v) for k, v in impl_modes.items() if "mock" not in str(v).lower()]

    _mock_impact = {
        "llm": "LLM not called — direction translation & hypothesis are rule-based templates; results DO NOT reflect paper method quality",
        "direction_translator": "Directions use item IDs only — no semantic LLM translation",
        "hypothesis_generator": "Hypotheses from fixed rules — no LLM reasoning",
        "option_writer": "Option text is templated, not LLM-polished",
        "user_simulator": "User choice is deterministic argmax",
        "item_encoder": "Random embeddings — retrieval results are meaningless",
    }

    lines: list[str] = []

    # ── Prominent mock banner ─────────────────────────────────────────────────
    processed_dir = Path(cfg.get("dataset", {}).get("processed_dir", "data/processed"))
    emb_audit_path = processed_dir / "embedding_build_log.json"
    emb_audit: dict[str, Any] = {}
    if emb_audit_path.exists():
        emb_audit = read_json(emb_audit_path)

    emb_fb = bool(emb_audit.get("embedding_fallback_to_mock", False))
    emb_rt = emb_audit.get("embedding_runtime") or {}

    if emb_fb:
        lines += [
            f"> ## ⚠️  EMBEDDING FALLBACK TO MOCK",
            f">",
            f"> **`embedding_fallback_to_mock: true`** in `{emb_audit_path}` — item vectors may be **random**.",
            f"> **Do not use these results for paper / production metrics without rebuilding embeddings.**",
            f">",
        ]

    if mock_list:
        lines += [
            f"> # ⚠️  WARNING: THIS RUN USED MOCK COMPONENTS",
            f">",
            f"> **These results are for development / debug purposes only.**",
            f"> **Do NOT use for paper reporting without replacing mock modules.**",
            f">",
        ]
        for k, v in mock_list:
            desc = _mock_impact.get(k, f"mode={v}")
            lines.append(f"> - **`{k}`** is MOCK: {desc}")
        lines += [
            f">",
            f"> See `MOCK_WARNING.md` in the run directory for the full checklist.",
            f"",
        ]

    lines += [
        f"# InterRec Run Report",
        f"",
        f"**run_id:** `{run_id}`",
        f"**timestamp:** {summary.get('timestamp', now_iso())}",
        f"**dataset:** {cfg.get('dataset', {}).get('name', 'unknown')}",
        f"**method:** interrec",
        f"**seed:** {cfg.get('seed', 42)}",
        f"",
        f"## Component Implementation Status",
        f"",
        f"| Module | Status | Notes |",
        f"|--------|--------|-------|",
    ]
    for k, v in mock_list:
        desc = _mock_impact.get(k, "")
        lines.append(f"| `{k}` | ⚠️ **MOCK** | {desc} |")
    for k, v in real_list:
        lines.append(f"| `{k}` | ✅ real (`{v}`) | — |")

    lines += [f"", f"## Offline embedding build audit", f""]
    if emb_audit:
        lines += [
            f"- **path:** `{emb_audit_path}`",
            f"- **embedding_fallback_to_mock:** `{emb_fb}`",
            f"- **build_status:** `{emb_audit.get('build_status', 'n/a')}`",
            f"- **embedding_mode_requested:** `{emb_audit.get('embedding_mode_requested', 'n/a')}`",
            f"",
            f"| Field | Value |",
            f"|--------|-------|",
        ]
        for key in ("device", "torch_cuda_available", "gpu_name", "batch_size", "embedding_mode", "model_name"):
            val = emb_rt.get(key, "")
            lines.append(f"| `{key}` | `{val}` |")
        lines += [f""]
    else:
        lines += [
            f"No `{emb_audit_path}` found — embedding device / fallback were not recorded for this processed dir.",
            f"",
        ]

    ret_path = processed_dir / "retrieval_index_manifest.json"
    lines += [f"## Offline retrieval artifact", f""]
    if ret_path.exists():
        rmeta = read_json(ret_path)
        lines += [
            f"- **path:** `{ret_path}`",
            f"- **dense_retrieval_backend:** `{rmeta.get('dense_retrieval_backend', 'n/a')}`",
            f"- **shape:** `{rmeta.get('shape', [])}`",
            f"- **faiss_index_file:** `{rmeta.get('faiss_index_file')}`",
            f"",
            f"### Dense vs lexical retrieval (audit)",
            f"",
        ]
        audit_note = rmeta.get("runtime_lexical_vs_dense_audit") or RUNTIME_LEXICAL_VS_DENSE_AUDIT
        for para in str(audit_note).strip().split("\n\n"):
            if para.strip():
                lines.append(para.strip())
                lines.append("")
    else:
        lines += [
            f"`{ret_path}` missing — rerun `scripts/build_item_embeddings.py` after pulling latest code.",
            f"",
            f"### Dense vs lexical retrieval (audit)",
            f"",
        ]
        for para in RUNTIME_LEXICAL_VS_DENSE_AUDIT.strip().split("\n\n"):
            if para.strip():
                lines.append(para.strip())
                lines.append("")

    llm_agg = summary.get("llm_semantic_bridge_aggregate") or {}
    lines += [
        f"## LLM semantic bridge (aggregated across turns)",
        f"",
        f"- **any_real_llm_fallback_contamination:** `{llm_agg.get('any_real_llm_fallback_contamination', False)}`",
        f"- **any_mock_semantic_layer:** `{llm_agg.get('any_mock_semantic_layer', False)}`",
        f"- **n_turn_rows:** `{llm_agg.get('n_turn_rows', 0)}`",
        f"",
        f"| component | implementation_mode_counts |",
        f"|-----------|---------------------------|",
    ]
    for comp, blk in (llm_agg.get("by_component") or {}).items():
        modes = blk.get("implementation_mode_counts") or {}
        lines.append(f"| `{comp}` | `{modes}` |")
    ck = llm_agg.get("cache_hits_detail") or {}
    if ck:
        lines += [f"", f"Cache hits per component headers: `{ck}`", ""]
    if llm_agg.get("any_real_llm_fallback_contamination"):
        lines = [
            f"> ## ⚠️  LLM parse / validity fallback detected",
            f"> Rule-based substitutions were recorded for ≥1 semantic-bridge calls in this run.",
            f"> Inspect `full_log.jsonl` `llm_semantic_bridge` blocks per turn.",
            f"",
        ] + lines
    cfg_llm_mode = cfg.get("llm", {}).get("mode", "mock")
    if cfg_llm_mode == "mock":
        lines = [
            f"> **Config:** `llm.mode: mock` — this run does **NOT** invoke a real semantic LLM.",
            f"",
        ] + lines

    lines += [
        f"",
        f"## Data Split",
        f"- observed_ratio: {cfg.get('dataset', {}).get('observed_ratio', 0.4)} (40%)",
        f"- future_train_ratio: {cfg.get('dataset', {}).get('future_train_ratio', 0.7)}",
        f"- future_valid_ratio: {cfg.get('dataset', {}).get('future_valid_ratio', 0.1)}",
        f"- future_test_ratio: {cfg.get('dataset', {}).get('future_test_ratio', 0.2)}",
        f"",
        f"## Core Metrics",
        f"",
    ]

    if mock_list:
        lines += [
            f"> ⚠️ The metrics below were produced with mock modules.",
            f"> They reflect pipeline correctness, NOT method effectiveness.",
            f"",
        ]

    lines += [
        f"| Metric | Value |",
        f"|--------|-------|",
    ]
    for k, v in aggregate.items():
        if k not in ("method", "dataset", "seed"):
            lines.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")

    lines += [
        f"",
        f"## Checklist: To get paper-quality results",
        f"",
        f"- [ ] Set `llm.mode: openai` and provide `OPENAI_API_KEY` in environment",
        f"- [ ] Set `embedding.mode: bge` (requires GPU with ≥8GB VRAM)",
        f"- [ ] Set `simulation.user_choice_mode: stochastic_sample`",
        f"- [ ] Use `configs/shared_experiment_config.yaml` for cross-method alignment",
        f"- [ ] Run multiple seeds (0, 1, 2) and report mean ± std",
        f"",
        f"## Files in this run directory",
        f"",
        f"| File | Description |",
        f"|------|-------------|",
        f"| `MOCK_WARNING.md` | Mock component checklist (generated at run start) |",
        f"| `full_log.jsonl` | Per-turn log + startup `embedding_build_audit` row from `embedding_build_log.json` |",
        f"| `metrics.csv` | Aggregate metrics |",
        f"| `metrics_by_turn.csv` | Turn-level metrics |",
        f"| `config.yaml` | Full configuration used for this run |",
        f"| `environment.txt` | Python / package versions |",
    ]

    report = "\n".join(lines)
    report_path = run_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    return str(report_path)
