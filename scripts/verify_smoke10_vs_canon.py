#!/usr/bin/env python3
"""Compare 10-session smoke runs against final_results_seed42 per-session metrics."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.in_depth.scripts.common import (
    future_test_set,
    load_sessions,
    session_metrics_from_turns,
)

CANON_ROOT = ROOT / "experiments/final_results_seed42"
KEYS = ("HitRate@10", "PR@10", "AR@10", "avg_ask")


def load_turns(log_path: Path) -> dict[str, list[dict]]:
    eps: dict[str, list] = defaultdict(list)
    with log_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("type") or "turn" not in r:
                continue
            sid = str(r.get("episode_id") or r.get("user_id"))
            eps[sid].append(r)
    for sid in eps:
        eps[sid].sort(key=lambda x: int(x["turn"]))
    return eps


def backend_audit(turns: list[dict]) -> dict[str, int]:
    c: dict[str, int] = defaultdict(int)
    for t in turns:
        im = (t.get("implementation_modes") or {}).get("hypothesis_generator") or "?"
        c[str(im)] += 1
        gen = t.get("frontier_clustered_v32_generation") or {}
        cb = gen.get("cluster_bound_generation_v2") or {}
        bt = cb.get("backend_type") or gen.get("mode") or "none"
        c[f"cb:{bt}"] += 1
        hg = (t.get("llm_semantic_bridge") or {}).get("hypothesis_generator") or {}
        c[f"resolved:{hg.get('hypothesis_generation_mode_resolved')}"] += 1
    return dict(c)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=("movielens", "lastfm"))
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args()
    ds = args.dataset
    canon_log = CANON_ROOT / ds / "full_log.jsonl"
    new_log = args.run_dir / "full_log.jsonl"
    if not new_log.is_file():
        print(f"MISSING {new_log}")
        return 1
    if not canon_log.is_file():
        print(f"MISSING canon {canon_log}")
        return 1

    lookup = {str(s.get("episode_id") or s.get("user_id")): s for s in load_sessions(ds)}
    old_eps = load_turns(canon_log)
    new_eps = load_turns(new_log)
    finished = sorted(
        sid
        for sid, turns in new_eps.items()
        if len(turns) >= 15 and sid in lookup and sid in old_eps and len(old_eps[sid]) >= 15
    )
    print(f"=== {ds}: paired finished sessions {len(finished)} ===")
    if not finished:
        print("No complete 15-turn pairs yet.")
        return 2

    mismatches = []
    for sid in finished:
        rel = future_test_set(lookup[sid])
        mo = session_metrics_from_turns(old_eps[sid], rel, ks=(5, 10))
        mn = session_metrics_from_turns(new_eps[sid], rel, ks=(5, 10))
        row = {k: (mo[k], mn[k]) for k in KEYS}
        if any(a != b for a, b in row.values()):
            mismatches.append((sid, row))

    same = len(finished) - len(mismatches)
    print(f"Session metrics identical ({', '.join(KEYS)}): {same}/{len(finished)}")
    if mismatches:
        print("Mismatches (canon → smoke):")
        for sid, row in mismatches[:20]:
            print(f"  {sid}: " + ", ".join(f"{k} {a}→{b}" for k, (a, b) in row.items() if a != b))

    # Path sanity: must not be contrastive_v2 fallback
    audit = backend_audit([t for sid in finished for t in new_eps[sid]])
    print("Smoke hypothesis path counts:", audit)
    bad = sum(v for k, v in audit.items() if "contrastive_v2" in k and "resolved" in k)
    if bad:
        print(f"FAIL: {bad} turns resolved to mock_contrastive_v2 (integrated path not active)")
        return 3
    if not any("integrated" in k for k in audit):
        print("WARN: no integrated_representative_item in cluster_bound telemetry (check logs)")

    ok = len(mismatches) == 0 and bad == 0
    print("OVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
