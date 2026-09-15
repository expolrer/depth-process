#!/usr/bin/env bash
set -euo pipefail

root="${DEPTH_MODEL_ROOT:-/ssd/hhw/depth-model}"
repo="$root/repos/official-act-rgbd"
state="$root/scheduler/official_pi05_stack_blocks_two"
mkdir -p "$state"

for gpu in 4 5 6 7; do
  if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader \
    | grep -q "^$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader),"; then
    printf 'GPU%s already has a compute process; refusing to launch pi0.5\n' "$gpu" >&2
    exit 1
  fi
done

if [[ -s "$state/coordinator.pid" ]]; then
  old_pid="$(cat "$state/coordinator.pid")"
  if [[ -r "/proc/$old_pid/cmdline" ]]; then
    old_cmd="$(tr '\0' ' ' < "/proc/$old_pid/cmdline")"
    if [[ "$old_cmd" == *"$repo/scripts/wait_act_then_pi05.sh"* ]]; then
      kill -TERM "$old_pid"
      for _ in {1..20}; do
        kill -0 "$old_pid" 2>/dev/null || break
        sleep 0.25
      done
      kill -0 "$old_pid" 2>/dev/null && { printf 'Old coordinator did not exit\n' >&2; exit 1; }
    fi
  fi
fi

if pgrep -f "$repo/scripts/prepare_and_train_pi05_stack.sh" >/dev/null 2>&1; then
  printf 'pi0.5 GPU4-7 worker already running\n'
  exit 0
fi

nohup "$repo/scripts/prepare_and_train_pi05_stack.sh" >>"$state/worker_gpu4_7.log" 2>&1 &
pid=$!
sleep 2
kill -0 "$pid"
printf '%s\n' "$pid" >"$state/worker_gpu4_7.pid"
printf 'pi05_gpu4_7_worker_pid=%s\n' "$pid"
