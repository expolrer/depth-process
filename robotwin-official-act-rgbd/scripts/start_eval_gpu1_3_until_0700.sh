#!/usr/bin/env bash
set -euo pipefail

root="${DEPTH_MODEL_ROOT:-/ssd/hhw/depth-model}"
repo="$root/repos/official-act-rgbd"
state="$root/scheduler/official_act_eval_until_0700"
mkdir -p "$state"
deadline="$(date -d "$(date +%F) 07:00:00" +%s)"
if (( $(date +%s) >= deadline )); then
  deadline="$(date -d "tomorrow 07:00:00" +%s)"
fi

declare -A preferred=([1]=ACT0_RGB [2]=ACT1_EARLY_RGBD [3]=ACT2_DUAL_SHARED)
for gpu in 1 2 3; do
  uuid="$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)"
  if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fxq "$uuid"; then
    printf 'GPU%s already has a compute process; refusing temporary evaluation launch\n' "$gpu" >&2
    exit 1
  fi
done

for gpu in 1 2 3; do
  pid_file="$state/gpu${gpu}.pid"
  if [[ -s "$pid_file" ]]; then
    pid="$(cat "$pid_file")"
    if [[ -r "/proc/$pid/cmdline" ]] && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q "temporary_official_eval_supervisor.sh $gpu "; then
      printf 'GPU%s supervisor already running pid=%s\n' "$gpu" "$pid"
      continue
    fi
  fi
  nohup "$repo/scripts/temporary_official_eval_supervisor.sh" "$gpu" "${preferred[$gpu]}" "$deadline" \
    >>"$state/gpu${gpu}.log" 2>&1 &
  printf '%s\n' "$!" > "$pid_file"
  printf 'GPU%s supervisor_pid=%s preferred=%s\n' "$gpu" "$!" "${preferred[$gpu]}"
done

sleep 3
for gpu in 1 2 3; do
  pid="$(cat "$state/gpu${gpu}.pid")"
  kill -0 "$pid"
done
printf 'deadline_epoch=%s deadline_local=%s\n' "$deadline" "$(date -d "@$deadline" --iso-8601=seconds)"
