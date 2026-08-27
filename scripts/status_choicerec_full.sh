#!/usr/bin/env bash
# ChoiceRec 三数据集 full — 瞬时进度 + 十二主表指标（只打印一次）。
#
# 用法:
#   bash scripts/status_choicerec_full.sh
#
set -euo pipefail

ROOT="/home/yrh666/interrec-final-github"
cd "$ROOT"

TMUX_CANDIDATES=(choicerec_pipeline choicerec_book choicerec_ml_wait choicerec_full)

echo "══════════════════════════════════════════════════════════════"
echo " ChoiceRec full · 瞬时快照 · $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "══════════════════════════════════════════════════════════════"
echo

# ── tmux ──
active_tmux=""
for s in "${TMUX_CANDIDATES[@]}"; do
  if tmux has-session -t "$s" 2>/dev/null; then
    echo "tmux     : ✅ session「${s}」运行中"
    active_tmux="${active_tmux:+$active_tmux, }$s"
  fi
done
if [[ -z "$active_tmux" ]]; then
  echo "tmux     : ⏸ 无活跃 pipeline session"
fi
echo

python3 - "$ROOT" <<'PY'
import json
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT))

from experiments.in_depth.scripts.common import (
    DATASET_ORDER,
    load_full_log_episodes,
    load_sessions,
    session_metrics_from_turns,
)

RUNS = {
    "lastfm": ("choicerec_lastfm_full_seed42", 1648),
    "movielens": ("choicerec_movielens_full_seed42", 4731),
    "book": ("choicerec_book_full_seed42", 2107),
}

# runner logs to pick live tqdm (newest matching total wins)
RUNNER_LOGS = {
    "book": [
        ROOT / "experiments/runs/choicerec_book_full_runner.log",
        ROOT / "experiments/runs/choicerec_book_then_ml_pipeline.log",
    ],
    "movielens": [
        ROOT / "experiments/runs/choicerec_movielens_full_runner.log",
        ROOT / "experiments/runs/choicerec_after_book_ml_wait.log",
        ROOT / "experiments/runs/choicerec_book_then_ml_pipeline.log",
    ],
    "lastfm": [
        ROOT / "experiments/runs/choicerec_full_serial.log",
    ],
}

MAIN12 = (
    "HitRate@5", "HitRate@10",
    "Recall@5", "Recall@10",
    "NDCG@5", "NDCG@10",
    "MRR@5", "MRR@10",
    "PR@5", "PR@10",
    "AR@5", "AR@10",
)
MAX_TURN = 15


def log_is_live(path: Path, max_age_sec: int = 180) -> bool:
    if not path.is_file():
        return False
    import time
    return (time.time() - path.stat().st_mtime) <= max_age_sec


def last_tqdm(log_paths: list[Path], expected_total: int | None = None) -> tuple[int, int, str | None]:
    best = (0, 0, None)
    for p in log_paths:
        if not p.is_file():
            continue
        text = p.read_text(errors="replace")
        for line in text.replace("\r", "\n").splitlines():
            if "Running InterRec episodes:" not in line:
                continue
            m = re.search(r"(\d+)/(\d+)", line)
            if not m:
                continue
            done, total = int(m.group(1)), int(m.group(2))
            if expected_total and total != expected_total:
                continue
            if done >= best[0] or (done == best[0] and total >= best[1]):
                best = (done, total, str(p.relative_to(ROOT)))
    return best


def aggregate_metrics(dataset: str, log_path: Path) -> dict:
    if not log_path.is_file():
        return {"n_log": 0, "n_full_turn": 0, "means": {}, "avg_ask": None, "latest": None}
    lookup = {
        str(s.get("episode_id") or s.get("user_id")): s
        for s in load_sessions(dataset)
    }
    episodes = load_full_log_episodes(log_path)
    rows = []
    ask_sum = 0.0
    n_full = 0
    latest = None
    for eid, turns in episodes.items():
        latest = eid
        sess = lookup.get(str(eid))
        if not sess:
            continue
        rel = set(str(x) for x in sess.get("future_test", []))
        m = session_metrics_from_turns(turns, rel, ks=(5, 10))
        rows.append({k: float(m[k]) for k in MAIN12})
        ask_sum += float(m.get("avg_ask", 0.0))
        if turns and max(int(t.get("turn", 0)) for t in turns) >= MAX_TURN:
            n_full += 1
    means = {}
    if rows:
        for k in MAIN12:
            means[k] = sum(r[k] for r in rows) / len(rows)
    return {
        "n_log": len(episodes),
        "n_full_turn": n_full,
        "means": means,
        "avg_ask": ask_sum / len(rows) if rows else None,
        "latest": latest,
    }


print("── 三数据集进度（full_log 为准）──")
print(f"{'':1}{'dataset':<11} {'done':>6} {'total':>6} {'pct':>7}  {'full15':>7}  {'state':<8}  latest_session")
print("-" * 78)

all_aggs = {}
live_runner_lines = []

for ds in DATASET_ORDER:
    rid, total = RUNS[ds]
    run_dir = ROOT / "experiments/runs" / rid
    log = run_dir / "full_log.jsonl"
    marker = (run_dir / "completed.marker").is_file()
    agg = aggregate_metrics(ds, log)
    all_aggs[ds] = agg
    n = agg["n_log"]
    pct = 100.0 * n / total if total else 0.0

    if marker or n >= total:
        flag, state = "✅", "done"
        n_display = n
        pct = 100.0 * n / total if total else 0.0
    else:
        t_done, t_total, src = last_tqdm(RUNNER_LOGS.get(ds, []), expected_total=total)
        n_display = n
        if t_total == total and t_done > n:
            n_display = max(n, t_done)
        pct = 100.0 * n_display / total if total else 0.0
        if t_total == total and t_done > 0 and t_done < total and src and log_is_live(ROOT / src):
            flag, state = "▶", "running"
            live_runner_lines.append(f"  live tqdm ({ds}): {t_done}/{t_total}  ← {src}")
        elif n_display > 0:
            flag, state = "◐", "partial"
        else:
            flag, state = "·", "pending"

    latest = agg["latest"] or "—"
    print(
        f"{flag} {ds:<11} {n_display:6d} {total:6d} {pct:6.1f}%  "
        f"{agg['n_full_turn']:7d}  {state:<8}  {latest}"
    )

if live_runner_lines:
    print()
    print("── 当前 runner tqdm ──")
    for line in live_runner_lines:
        print(line)

print()
print("── 十二主表指标（partial/full · target=future_test）──")
header = f"{'metric':<14}" + "".join(f"{ds:>12}" for ds in DATASET_ORDER)
print(header)
print("-" * len(header))

for k in MAIN12:
    parts = [f"{k:<14}"]
    for ds in DATASET_ORDER:
        v = all_aggs[ds]["means"].get(k)
        parts.append(f"{v:12.4f}" if v is not None else f"{'—':>12}")
    print("".join(parts))

print()
print(f"{'avg_ask':<14}", end="")
for ds in DATASET_ORDER:
    v = all_aggs[ds]["avg_ask"]
    print(f"{v:12.3f}" if v is not None else f"{'—':>12}", end="")
print()

print()
print("说明:")
print("  · LastFM done 后指标仍从 full_log 读取；ML partial 会随续跑增加")
print("  · Book 完成后 pipeline 自动跑: bash scripts/run_choicerec_movielens_resume.sh")
print("  · 重新查看: bash scripts/status_choicerec_full.sh")
PY
