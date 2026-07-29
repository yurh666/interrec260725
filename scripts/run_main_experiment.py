#!/usr/bin/env python3
"""Run the InterRec main experiment or BM25 baseline.

Usage
-----
From the interrec/ project root::

    # InterRec (paper default: openai LLM + CPU runtime; pre-built embeddings)
    python scripts/run_main_experiment.py --config configs/default.yaml

    # 无 API Key 的本地调试（mock LLM）
    python scripts/run_main_experiment.py --config configs/debug_llm_mock.yaml

    # BM25 baseline only
    python scripts/run_main_experiment.py --config configs/default.yaml --method bm25

    # Override split and max_users
    python scripts/run_main_experiment.py --split test --max-users 100

The script will NOT run if:
  - data/processed/interactions.csv does not exist  → run preprocess_dataset.py
  - data/processed/item_embeddings.npy does not exist → run build_item_embeddings.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import InterRecDataset
from src.embedding.bm25_index import BM25Index
from src.embedding.index import EmbeddingIndex
from src.logging.report_generator import generate_run_report
from src.logging.run_logger import RunLogger
from src.logging.full_log_rebuild import load_episodes_from_full_log
from src.simulation.experiment_runner import _aggregate_metrics, run_bm25_baseline, run_interrec_experiment
from src.utils.io import read_json
from src.utils.config import load_config, save_config
from src.utils.seed import set_seed
from src.utils.time import make_run_id


def _mock_warning_impl_modes(cfg: dict[str, Any]) -> dict[str, str]:
    """Static preview for MOCK_WARNING.md when ``llm.mode == mock``."""

    return {
        "item_encoder": str(cfg.get("embedding", {}).get("mode", "tfidf_svd")),
        "llm": "mock_llm",
        "direction_translator": "mock",
        "hypothesis_generator": "mock",
        "option_writer": "mock_template",
        "user_simulator": str(cfg.get("simulation", {}).get("user_choice_mode", "deterministic_argmax")),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--method", default="interrec", choices=["interrec", "bm25"])
    p.add_argument("--split", default=None, help="Override simulation.split (train/valid/test)")
    p.add_argument("--max-users", type=int, default=None)
    p.add_argument(
        "--user-start",
        type=int,
        default=None,
        help="1-based inclusive index into dataset.sessions (pair with --user-end; for sharding)",
    )
    p.add_argument(
        "--user-end",
        type=int,
        default=None,
        help="1-based inclusive index into dataset.sessions",
    )
    p.add_argument(
        "--run-id-suffix",
        type=str,
        default="",
        help="Appended to run_id after seed (unique shard label; parallel runs)",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Use this exact run directory name (default: timestamped auto id). For resume.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Append diagnostics/latency; requires --run-id and existing run_dir. "
        "Combine with --user-start/--user-end to continue a shard. "
        "Rebuilds metrics from full_log.jsonl at end.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.split:
        cfg.setdefault("simulation", {})["split"] = args.split

    if (args.user_start is not None) ^ (args.user_end is not None):
        print("[run] ERROR: --user-start and --user-end must be set together.", file=sys.stderr)
        sys.exit(1)
    if args.user_start is not None:
        if args.max_users is not None:
            print("[run] ERROR: cannot combine --max-users with --user-start/--user-end.", file=sys.stderr)
            sys.exit(1)
        cfg.setdefault("simulation", {})["user_slice_start"] = int(args.user_start)
        cfg.setdefault("simulation", {})["user_slice_end"] = int(args.user_end)
    if args.max_users is not None:
        cfg.setdefault("simulation", {})["max_users"] = args.max_users
    if args.seed is not None:
        cfg["seed"] = args.seed

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    if args.resume and not args.run_id:
        print("[run] ERROR: --resume requires --run-id (existing run directory name).", file=sys.stderr)
        sys.exit(1)

    dataset_name = cfg.get("dataset", {}).get("name", "unknown")
    method = args.method
    suffix = args.run_id_suffix or ""
    run_id = (
        str(args.run_id).strip()
        if args.run_id
        else make_run_id(dataset_name, method, seed, extra_suffix=suffix)
    )
    run_dir = Path("experiments/runs") / run_id
    print(f"[run] run_id = {run_id}")

    if args.resume and not run_dir.is_dir():
        print(f"[run] ERROR: resume requested but {run_dir} does not exist.", file=sys.stderr)
        sys.exit(1)

    logger = RunLogger(run_id, base_dir="experiments/runs")
    logger.configure_full_log_policy(cfg)

    from src.utils.neutrality_cache_stats import reset as neutrality_cache_stats_reset

    neutrality_cache_stats_reset()

    lat_path = (logger.run_dir / "latency_profile.jsonl").resolve()
    diag_path = (logger.run_dir / "diagnostics.jsonl").resolve()
    if args.resume:
        print(f"[run] resume: appending diagnostics / latency (no truncate)", flush=True)
    else:
        lat_path.write_text("", encoding="utf-8")
        diag_path.write_text("", encoding="utf-8")
    os.environ["INTERREC_LATENCY_PROFILE_PATH"] = str(lat_path)
    print(f"[run] latency_profile -> {lat_path}", flush=True)
    os.environ["INTERREC_DIAGNOSTICS_JSONL"] = str(diag_path)
    print(f"[run] diagnostics -> {diag_path}", flush=True)

    processed_dir = Path(cfg["dataset"].get("processed_dir", "data/processed"))
    for required in ["interactions.csv", "items.csv", "sessions.json"]:
        if not (processed_dir / required).exists():
            print(f"[run] ERROR: {processed_dir / required} not found.")
            print("       Run: python scripts/preprocess_dataset.py --config <config>")
            sys.exit(1)

    emb_path = processed_dir / "item_embeddings.npy"
    if not emb_path.exists():
        print(f"[run] ERROR: {emb_path} not found.")
        print("       Run: python scripts/build_item_embeddings.py --config <config>")
        sys.exit(1)

    emb_build_log = processed_dir / "embedding_build_log.json"
    if emb_build_log.exists():
        emb_manifest = read_json(emb_build_log)
        cfg["_embedding_build_manifest"] = emb_manifest
        if not args.resume:
            logger.log_embedding_build_audit(emb_manifest)
        if emb_manifest.get("embedding_fallback_to_mock"):
            msg = (
                "embedding_fallback_to_mock=true (item vectors may be random mock). "
                "Do not use for paper metrics without rebuilding embeddings."
            )
            logger.log_run_warning("embedding_fallback_to_mock", msg)
            print(f"[run] ERROR: {msg}", file=sys.stderr)
    else:
        if not args.resume:
            logger.log_embedding_audit_missing(processed_dir)
        print(
            f"[run] WARNING: {emb_build_log} not found — run lacks offline embedding audit record.",
            file=sys.stderr,
        )

    ret_man = processed_dir / "retrieval_index_manifest.json"
    if not ret_man.exists():
        print(
            f"[run] WARNING: {ret_man} not found — re-run build_item_embeddings.py to emit retrieval manifest.",
            file=sys.stderr,
        )

    if not args.resume:
        logger.write_config(cfg)
        logger.write_environment()
        logger.write_git_info(project_dir=Path(__file__).resolve().parent.parent)

    if cfg.get("llm", {}).get("mode") == "mock" and not args.resume:
        logger.write_mock_warning(_mock_warning_impl_modes(cfg))

    print("[run] Loading dataset...")
    dataset = InterRecDataset.load(processed_dir)
    print(f"      {len(dataset.sessions)} sessions, {len(dataset.items)} items")
    sim_dbg = cfg.get("simulation", {})
    us, ue = sim_dbg.get("user_slice_start"), sim_dbg.get("user_slice_end")
    if us is not None and ue is not None:
        n_sl = len(dataset.sessions[int(us) - 1 : int(ue)])
        print(f"[run] User slice (1-based sessions): {us}-{ue} → {n_sl} sessions in this run")

    print("[run] Loading precomputed dense embedding index...")
    # BOOK_ONLY_EXPERIMENTAL: support configurable embedding filename via
    # embedding.precomputed_path config key.  When not set, falls back to
    # the default "item_embeddings.npy" — behaviour is unchanged for
    # LastFM / MovieLens / Yelp.
    _emb_cfg = cfg.get("embedding", {})
    _precomputed_path = _emb_cfg.get("precomputed_path", None)
    if _precomputed_path:
        import os as _os
        _emb_filename = _os.path.basename(str(_precomputed_path))
        print(f"[run] [BOOK_ONLY_EXPERIMENTAL] Using precomputed embedding: {_emb_filename}")
    else:
        _emb_filename = "item_embeddings.npy"
    index = EmbeddingIndex.load(processed_dir, embedding_filename=_emb_filename)

    print("[run] Building lightweight BM25 lexical index (CPU-only)...")
    bm25 = BM25Index.build(dataset.items)

    if method == "bm25":
        print("[run] Running BM25 baseline...")
        agg = run_bm25_baseline(cfg, dataset, index, logger, bm25)
    else:
        print("[run] Running InterRec...")
        agg = run_interrec_experiment(cfg, run_dir, dataset, index, logger, bm25)
        if cfg.get("logging", {}).get("generate_report", False):
            report_path = generate_run_report(run_dir, cfg)
            print(f"[run] Report: {report_path}")

        try:
            nmode = str(((cfg.get("ask") or {}).get("neutrality") or {}).get("mode", ""))
            if nmode == "llm":
                from src.utils.neutrality_cache_audit import write_neutrality_audit_md

                p = write_neutrality_audit_md(logger.run_dir)
                if p:
                    print(f"[run] neutrality audit -> {p}", flush=True)
        except Exception as exc:
            print(f"[run] neutrality audit skipped: {exc}", flush=True)

    if method == "interrec" and args.resume:
        top_k = int(cfg.get("recommendation", {}).get("top_k", 10))
        episodes = load_episodes_from_full_log(logger.run_dir)
        agg = _aggregate_metrics(episodes, top_k)
        logger.rebuild_turn_rows_from_full_log(top_k)
        logger.write_metrics(agg, episodes)
        print(f"[run] resume: rebuilt metrics from full_log ({len(episodes)} episodes)", flush=True)

    save_config(cfg, run_dir / "config.yaml")
    print("[run] Aggregate metrics:")
    for k, v in agg.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"[run] Outputs in: {run_dir}")


if __name__ == "__main__":
    main()
