#!/usr/bin/env bash
set -euo pipefail

root="${DEPTH_MODEL_ROOT:-/ssd/hhw/depth-model}"
repo="$root/repos/official-act-rgbd"
state="$root/scheduler/official_pi05_stack_blocks_two"
mkdir -p "$state"

if pgrep -f "$repo/scripts/wait_act_then_pi05.sh" >/dev/null 2>&1; then
  printf 'pi0.5 successor coordinator already running\n'
  exit 0
fi

nohup "$repo/scripts/wait_act_then_pi05.sh" >>"$state/coordinator.log" 2>&1 &
pid=$!
sleep 2
kill -0 "$pid"
printf '%s\n' "$pid" >"$state/coordinator.pid"
printf 'pi05_successor_pid=%s\n' "$pid"
