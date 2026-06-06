#!/usr/bin/env bash
# Fair comparison: same test essays (from unbalanced stratified split) for two checkpoints.
# Usage:
#   export CSV=efcamdat_full_with_corrected.csv
#   ./scripts/compare_fair_eval.sh checkpoints/baseline_dual/best_model.pt checkpoints/balanced_cefr/best_model.pt
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT"
CSV="${CSV:-$ROOT/efcamdat_full_with_corrected.csv}"
SPLIT="${SPLIT:-$ROOT/splits/test_indices.json}"
BASE_CKPT="${1:?baseline checkpoint path}"
BAL_CKPT="${2:?balanced (or other) checkpoint path}"

mkdir -p "$ROOT/splits"
if [[ ! -f "$SPLIT" ]]; then
  echo "Creating $SPLIT (run once; reuse for all fair evals)..."
  python -m cambridge_exp.export_test_split --csv "$CSV" --out "$SPLIT"
fi

echo "=== $BASE_CKPT ==="
python -m cambridge_exp.eval_cli -c "$BASE_CKPT" --csv "$CSV" --test-indices "$SPLIT"

echo "=== $BAL_CKPT ==="
python -m cambridge_exp.eval_cli -c "$BAL_CKPT" --csv "$CSV" --test-indices "$SPLIT"
