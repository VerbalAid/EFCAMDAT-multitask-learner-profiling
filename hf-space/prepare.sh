#!/usr/bin/env bash
# Sync backend code from repo root into hf-space/ before pushing to Hugging Face.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$(cd "$(dirname "$0")" && pwd)"

echo "Syncing api/ and cambridge_exp/ → $DEST"
rsync -a --delete \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  "$ROOT/api/" "$DEST/api/"
rsync -a --delete \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  "$ROOT/cambridge_exp/" "$DEST/cambridge_exp/"

CKPT="$ROOT/checkpoints/baseline_dual/best_model.pt"
if [[ -f "$CKPT" ]]; then
  echo "Copying checkpoint → model/best_model.pt"
  cp "$CKPT" "$DEST/model/best_model.pt"
else
  echo "No local checkpoint at $CKPT — copy best_model.pt to hf-space/model/ manually."
fi

echo "Done. Deploy with: bash scripts/upload_hf_space.sh"
