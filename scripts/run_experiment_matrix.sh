#!/usr/bin/env bash
# Example matrix: adjust paths, then `bash scripts/run_experiment_matrix.sh`
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT"
CSV="${CSV:-$ROOT/efcamdat_full_with_corrected.csv}"
# Safer on ~8GB GPUs (validation OOM); override: export TRAIN_FLAGS=""
TRAIN_FLAGS="${TRAIN_FLAGS:---eval-batch-size 8}"

python -m cambridge_exp.train_cli $TRAIN_FLAGS --csv "$CSV" --checkpoint-dir "$ROOT/checkpoints/exp_baseline_dual"
python -m cambridge_exp.train_cli $TRAIN_FLAGS --csv "$CSV" --dual-mode raw_only --checkpoint-dir "$ROOT/checkpoints/exp_ablation_raw"
python -m cambridge_exp.train_cli $TRAIN_FLAGS --csv "$CSV" --balance-cefr --checkpoint-dir "$ROOT/checkpoints/exp_balanced_cefr"
python -m cambridge_exp.train_cli $TRAIN_FLAGS --csv "$CSV" --balance-cefr --dual-mode raw_only \
  --checkpoint-dir "$ROOT/checkpoints/exp_balanced_raw_ablation"

echo "Done. Compare val metrics printed at end of each run."
