#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/ssd/hhw/depth-processing}"
WORKERS="${WORKERS:-8}"
FFMPEG_THREADS="${FFMPEG_THREADS:-2}"
LOG_PATH="$PROJECT_ROOT/viewer/data/build.log"
PID_PATH="$PROJECT_ROOT/viewer/data/build.pid"

cd "$PROJECT_ROOT"
mkdir -p viewer/data

if [[ -f "$PID_PATH" ]]; then
  existing_pid="$(cat "$PID_PATH")"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "Viewer build is already running with PID $existing_pid"
    exit 0
  fi
fi

nohup .venv/bin/python scripts/build_video_viewer.py \
  --project-root "$PROJECT_ROOT" \
  --workers "$WORKERS" \
  --ffmpeg-threads "$FFMPEG_THREADS" \
  >"$LOG_PATH" 2>&1 </dev/null &

build_pid=$!
echo "$build_pid" >"$PID_PATH"
echo "Started viewer build PID $build_pid; log: $LOG_PATH"
