#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
for pair in \
  "book:smoke_book_distill_n2_seed42.yaml" \
  "movielens:smoke_movielens_distill_n2_seed42.yaml" \
  "lastfm:smoke_lastfm_distill_n2_seed42.yaml"; do
  ds="${pair%%:*}"
  cfg="${pair##*:}"
  echo "=== smoke $ds ==="
  python3 scripts/run_main_experiment.py --config "configs/runs/$cfg"
done
echo "[done] all smokes"
