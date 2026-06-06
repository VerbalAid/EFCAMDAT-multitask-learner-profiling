#!/usr/bin/env bash
# Usage: bash scripts/watch_training.sh [path/to/train.log]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="${1:-$ROOT/checkpoints/baseline_dual/train_resume.log}"
exec tail -f "$LOG"
