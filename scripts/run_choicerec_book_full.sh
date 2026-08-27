#!/usr/bin/env bash
# ChoiceRec: Book full now; ML resume later from choicerec_movielens_full_seed42/full_log.jsonl
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUN_ID="${1:-choicerec_book_full_seed42}"
BASE_CFG="configs/choicerec/book_full_seed42.yaml"
RUN_DIR="experiments/runs/${RUN_ID}"
CONT_CFG="${RUN_DIR}/continue_config.yaml"

mkdir -p "$RUN_DIR"
if [[ -f "${RUN_DIR}/full_log.jsonl" ]]; then
  python3 scripts/choicerec_continue_config.py \
    --base-config "$BASE_CFG" \
    --full-log "${RUN_DIR}/full_log.jsonl" \
    --dataset book \
    --out "$CONT_CFG"
  CFG="$CONT_CFG"
  RESUME=(--resume)
else
  CFG="$BASE_CFG"
  RESUME=()
fi

echo "[run] ChoiceRec Book full run_id=${RUN_ID} config=${CFG}"
python3 scripts/run_main_experiment.py --config "$CFG" --run-id "$RUN_ID" "${RESUME[@]}"
touch "${RUN_DIR}/completed.marker"
echo "[done] ${RUN_ID}"
