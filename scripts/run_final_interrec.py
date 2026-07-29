#!/usr/bin/env python3
"""Unified entry for InterRec Final (seed42) across Book / MovieLens / LastFM.

All datasets share the same Python pipeline (``run_main_experiment.py`` →
``run_interrec_experiment`` → ``run_session``). Only the YAML config differs.

Examples::

    python3 scripts/run_final_interrec.py --dataset book
    python3 scripts/run_final_interrec.py --dataset movielens --max-users 10
    python3 scripts/run_final_interrec.py --list-datasets
    python3 scripts/verify_final_unified_equivalence.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config  # noqa: E402

DATASETS = ("book", "movielens", "lastfm")
RUN_CONFIGS = {
    ds: ROOT / "configs/final_unified/runs" / f"{ds}_seed42.yaml" for ds in DATASETS
}
LEGACY_CONFIGS = {
    ds: ROOT / "configs/runs" / f"{ds}_full_integrated_distill_surface_seed42.yaml" for ds in DATASETS
}
DEFAULT_RUN_ID = {ds: f"{ds}_full_integrated_distill_surface_seed42" for ds in DATASETS}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run InterRec Unified Final (seed42)")
    p.add_argument("--dataset", choices=[*DATASETS, "all"], help="Dataset to run")
    p.add_argument("--list-datasets", action="store_true", help="Print registry and exit")
    p.add_argument("--dry-run", action="store_true", help="Print command only")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-id", default=None, help="Override run directory name")
    p.add_argument("--run-id-suffix", default="")
    p.add_argument("--max-users", type=int, default=None)
    p.add_argument("--user-start", type=int, default=None)
    p.add_argument("--user-end", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--split", default=None)
    return p.parse_args()


def _print_registry() -> None:
    print("InterRec Unified Final (seed42)\n")
    print(f"Code entry : {ROOT / 'scripts/run_final_interrec.py'}")
    print(f"Pipeline   : {ROOT / 'scripts/run_main_experiment.py'}")
    print(f"Results    : {ROOT / 'experiments/final_unified_seed42/'}\n")
    for ds in DATASETS:
        cfg = RUN_CONFIGS[ds]
        resolved = load_config(cfg)
        print(f"[{ds}]")
        print(f"  unified config : {cfg.relative_to(ROOT)}")
        print(f"  legacy config  : {LEGACY_CONFIGS[ds].relative_to(ROOT)}")
        print(f"  data           : {resolved.get('dataset', {}).get('processed_dir') or resolved.get('dataset', {}).get('data_root')}")
        print(f"  max_turns      : {resolved.get('simulation', {}).get('max_turns')}")
        print(f"  c_ask          : {resolved.get('ask', {}).get('c_ask')}")
        fc = (resolved.get("ask", {}).get("hypothesis_generation", {}).get("frontier_clustered_v32") or {})
        print(f"  when_to_ask    : {fc.get('when_to_ask', '(entropy gate family)')}")
        print(f"  embedding dim  : {resolved.get('embedding', {}).get('dim')}")
        print(f"  surface_writer : {resolved.get('surface_writer', {}).get('mode')}")
        print()


def _run_one(dataset: str, args: argparse.Namespace) -> int:
    cfg = RUN_CONFIGS[dataset]
    if not cfg.is_file():
        print(f"[error] missing config: {cfg}", file=sys.stderr)
        return 1
    run_id = args.run_id or DEFAULT_RUN_ID[dataset]
    if args.run_id_suffix:
        run_id = f"{run_id}_{args.run_id_suffix}"
    cmd = [
        sys.executable,
        str(ROOT / "scripts/run_main_experiment.py"),
        "--config",
        str(cfg),
        "--seed",
        str(args.seed),
        "--run-id",
        run_id,
    ]
    if args.split:
        cmd.extend(["--split", args.split])
    if args.max_users is not None:
        cmd.extend(["--max-users", str(args.max_users)])
    if args.user_start is not None:
        cmd.extend(["--user-start", str(args.user_start)])
    if args.user_end is not None:
        cmd.extend(["--user-end", str(args.user_end)])
    if args.resume:
        cmd.append("--resume")
    print(f"[run_final_interrec] dataset={dataset} config={cfg.relative_to(ROOT)} run_id={run_id}")
    if args.dry_run:
        print(" ".join(cmd))
        return 0
    return subprocess.call(cmd, cwd=str(ROOT))


def main() -> int:
    args = parse_args()
    if args.list_datasets:
        _print_registry()
        return 0
    if not args.dataset:
        print("[error] specify --dataset {book,movielens,lastfm,all} or --list-datasets", file=sys.stderr)
        return 1
    if args.dataset == "all":
        rc = 0
        for ds in DATASETS:
            code = _run_one(ds, args)
            if code != 0:
                rc = code
        return rc
    return _run_one(args.dataset, args)


if __name__ == "__main__":
    raise SystemExit(main())
