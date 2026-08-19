#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/ssd/hhw/depth-processing}"
GPU_IDS="${GPU_IDS:-0,6,7}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PID_PATH="$PROJECT_ROOT/logs/lingbot_attention/run.pid"
LOG_PATH="$PROJECT_ROOT/logs/lingbot_attention/run.log"

cd "$PROJECT_ROOT"
mkdir -p logs/lingbot_attention

if [[ -f "$PID_PATH" ]]; then
  existing_pid="$(cat "$PID_PATH")"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "LingBot attention build is already running with PID $existing_pid"
    exit 0
  fi
fi

nohup env \
  PROJECT_ROOT="$PROJECT_ROOT" \
  GPU_IDS="$GPU_IDS" \
  BATCH_SIZE="$BATCH_SIZE" \
  "$PROJECT_ROOT/scripts/run_lingbot_attention_gpus.sh" \
  >"$LOG_PATH" 2>&1 </dev/null &

run_pid=$!
echo "$run_pid" >"$PID_PATH"
echo "Started LingBot attention build PID $run_pid on GPUs $GPU_IDS; log: $LOG_PATH"
