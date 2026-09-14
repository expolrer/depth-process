#!/usr/bin/env bash
set -euo pipefail

root="${DEPTH_MODEL_ROOT:-/ssd/hhw/depth-model}"
repo="$root/repos/official-act-rgbd"
queue="$root/scheduler/official_act_clean8"
deadline="$(date -d '2026-09-15 07:00:00 +0800' +%s)"
mkdir -p "$queue/logs"

declare -A preferred=(
  [0]=ACT0_RGB
  [2]=ACT1_EARLY_RGBD
  [3]=ACT2_DUAL_SHARED
  [4]=ACT3_DUAL_PER_VIEW
  [5]=ACT4_XYZMAP
  [6]=ACT5_POINT_TOKENS
  [7]=ACT6_LINGBOT_DEPTH
)

for gpu in 4 5 6 7; do
  nohup "$repo/scripts/eight_arch_train_worker.sh" "$gpu" long "${preferred[$gpu]}" \
    >>"$queue/logs/worker_gpu${gpu}.log" 2>&1 &
  printf 'started long worker GPU%s pid=%s preferred=%s\n' "$gpu" "$!" "${preferred[$gpu]}"
done

if (( $(date +%s) < deadline )); then
  for gpu in 0 2 3; do
    nohup "$repo/scripts/eight_arch_train_worker.sh" "$gpu" temporary "${preferred[$gpu]}" "$deadline" \
      >>"$queue/logs/worker_gpu${gpu}.log" 2>&1 &
    printf 'started temporary worker GPU%s pid=%s preferred=%s deadline=%s\n' \
      "$gpu" "$!" "${preferred[$gpu]}" "2026-09-15 07:00:00+08:00"
  done
fi
