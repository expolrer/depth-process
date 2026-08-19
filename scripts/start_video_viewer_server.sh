#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/ssd/hhw/depth-processing}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"
PID_PATH="$PROJECT_ROOT/viewer/data/server.pid"
LOG_PATH="$PROJECT_ROOT/viewer/data/server.log"

cd "$PROJECT_ROOT"
mkdir -p viewer/data

if [[ -f "$PID_PATH" ]]; then
  existing_pid="$(cat "$PID_PATH")"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "Viewer server is already running with PID $existing_pid on port $PORT"
    exit 0
  fi
fi

if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)$PORT$"; then
  echo "Port $PORT is already in use" >&2
  exit 1
fi

nohup python3 viewer/serve_viewer.py \
  --host "$HOST" \
  --port "$PORT" \
  --directory "$PROJECT_ROOT/viewer" \
  >"$LOG_PATH" 2>&1 </dev/null &

server_pid=$!
echo "$server_pid" >"$PID_PATH"
sleep 1
if ! kill -0 "$server_pid" 2>/dev/null; then
  cat "$LOG_PATH" >&2
  exit 1
fi
echo "Viewer server PID $server_pid: http://$HOST:$PORT"
