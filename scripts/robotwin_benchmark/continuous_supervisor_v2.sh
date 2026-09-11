#!/usr/bin/env bash
set -uo pipefail

root=/ssd/depth-model
scheduler="$root/scheduler/continuous_v2"
mkdir -p "$scheduler"

while true; do
  complete=0
  for gpu in 4 5 6; do
    [[ -f "$scheduler/gpu${gpu}.done" ]] && complete=$((complete + 1))
  done
  printf '%s RUNNING workers_complete=%s/3 allowed_gpus=4,5,6\n' \
    "$(date --iso-8601=seconds)" "$complete" >"$scheduler/overall.status"
  ((complete == 3)) && break
  sleep 60
done

printf '%s COMPLETE workers=3 allowed_gpus=4,5,6\n' \
  "$(date --iso-8601=seconds)" >"$scheduler/overall.status"
