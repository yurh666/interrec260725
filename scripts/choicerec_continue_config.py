#!/usr/bin/env python3
"""Emit a continue config: only episode_ids not yet complete (turn>=max_turns) in full_log."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.in_depth.scripts.common import load_sessions  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def done_episode_ids(full_log: Path, min_turns: int = 15) -> set[str]:
    turns: dict[str, int] = {}
    if not full_log.is_file():
        return set()
    for line in full_log.open():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("turn"):
            continue
        eid = str(row.get("episode_id") or row.get("user_id"))
        turns[eid] = max(turns.get(eid, 0), int(row.get("turn", 0)))
    return {eid for eid, n in turns.items() if n >= min_turns}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--full-log", type=Path, required=True)
    parser.add_argument("--dataset", choices=["book", "movielens", "lastfm"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-turns", type=int, default=15)
    args = parser.parse_args()

    cfg = load_config(str(args.base_config))
    done = done_episode_ids(args.full_log, min_turns=args.min_turns)
    all_ids = [str(s.get("episode_id") or s["user_id"]) for s in load_sessions(args.dataset)]
    remaining = [eid for eid in all_ids if eid not in done]
    cfg.setdefault("simulation", {})["episode_ids"] = remaining
    cfg.setdefault("simulation", {})["max_users"] = None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(
        f"[continue] dataset={args.dataset} done={len(done)} remaining={len(remaining)} -> {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
