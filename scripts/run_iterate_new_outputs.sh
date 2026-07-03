#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
COUNT="${COUNT:-4}"
CONCURRENT="${CONCURRENT:-1}"
LOG_PATH="${LOG_PATH:-$ROOT_DIR/new_outputs/iterate_new_outputs_runner.log}"
ANALYZE_ONLY="${ANALYZE_ONLY:-0}"

mkdir -p "$(dirname "$LOG_PATH")"

cmd=(
  "$PYTHON_BIN"
  auto_iterate_until_sci.py
  --count "$COUNT"
  --concurrent "$CONCURRENT"
  --gpu "$GPU_ID"
)

if [[ "$ANALYZE_ONLY" == "1" ]]; then
  cmd+=(--analyze_only)
fi

echo "[$(date '+%F %T')] start: ${cmd[*]}" | tee -a "$LOG_PATH"
exec "${cmd[@]}" 2>&1 | tee -a "$LOG_PATH"
