#!/usr/bin/env bash
set -euo pipefail

root="${DEPTH_MODEL_ROOT:-/ssd/hhw/depth-model}"
repo="$root/repos/official-act-rgbd"
state="$root/scheduler/post_pi05_act5_eval"
mkdir -p "$state"

if [[ -s "$state/coordinator.pid" ]]; then
  pid="$(cat "$state/coordinator.pid")"
  if [[ -r "/proc/$pid/cmdline" ]] && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q 'pi05_then_act5_gpu7_eval_gpu4_6.sh'; then
    printf 'post-pi0.5 coordinator already running pid=%s\n' "$pid"
    exit 0
  fi
fi

nohup "$repo/scripts/pi05_then_act5_gpu7_eval_gpu4_6.sh" >>"$state/coordinator.log" 2>&1 &
pid="$!"
sleep 1
kill -0 "$pid"
printf '%s\n' "$pid" > "$state/coordinator.pid"
printf 'post_pi05_coordinator_pid=%s\n' "$pid"
