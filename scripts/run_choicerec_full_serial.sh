#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUNS=(
  "choicerec_lastfm_full_seed42:configs/choicerec/lastfm_full_seed42.yaml"
  "choicerec_movielens_full_seed42:configs/choicerec/movielens_full_seed42.yaml"
  "choicerec_book_full_seed42:configs/choicerec/book_full_seed42.yaml"
)

for entry in "${RUNS[@]}"; do
  run_id="${entry%%:*}"
  cfg="${entry#*:}"
  run_dir="experiments/runs/${run_id}"
  if [[ -f "${run_dir}/completed.marker" ]]; then
    echo "[skip] ${run_id} already completed"
    continue
  fi
  echo "[run] ${run_id} cfg=${cfg}"
  resume_flag=()
  if [[ -f "${run_dir}/full_log.jsonl" ]]; then
    resume_flag=(--resume)
  fi
  python3 scripts/run_main_experiment.py --config "$cfg" --run-id "$run_id" "${resume_flag[@]}"
  touch "${run_dir}/completed.marker"
  echo "[done] ${run_id}"
done

echo "[all] ChoiceRec serial full runs completed"
