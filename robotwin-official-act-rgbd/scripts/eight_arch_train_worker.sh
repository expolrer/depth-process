#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 3 ]]; then
  printf 'Usage: %s GPU long|temporary PREFERRED_VARIANT [DEADLINE_EPOCH]\n' "$0" >&2
  exit 2
fi

gpu="$1"
mode="$2"
preferred="$3"
deadline="${4:-0}"
root="${DEPTH_MODEL_ROOT:-/ssd/hhw/depth-model}"
repo="$root/repos/official-act-rgbd"
queue="$root/scheduler/official_act_official6000"
target_epochs=6000
task=stack_blocks_two
task_config=depth_master_clean
suffix=official6000_seed0
variants=(
  ACT0_RGB
  ACT1_EARLY_RGBD
  ACT2_DUAL_SHARED
  ACT3_DUAL_PER_VIEW
  ACT4_XYZMAP
  ACT5_POINT_TOKENS
  ACT6_LINGBOT_DEPTH
  ACT7_DEPTH_TRANSFORMER
)

mkdir -p "$queue/claims" "$queue/done" "$queue/logs"

is_complete() {
  local variant="$1"
  local marker="$root/experiments/OfficialACTRGBD/$variant/$task/$suffix/training_complete.json"
  [[ -s "$marker" ]] || return 1
  "$root/envs/aloha/bin/python" - "$marker" "$target_epochs" <<'PY'
import json, sys
state = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if state.get("status") == "complete" and int(state.get("epoch", 0)) >= int(sys.argv[2]) else 1)
PY
}

reap_stale_claims() {
  local claim owner
  for claim in "$queue"/claims/*; do
    [[ -d "$claim" ]] || continue
    owner="$(cat "$claim/worker_pid" 2>/dev/null || true)"
    if [[ -n "$owner" ]] && ! kill -0 "$owner" 2>/dev/null; then
      rm -rf "$claim"
    fi
  done
}

claim_one() {
  local candidate
  for candidate in "$preferred" "${variants[@]}"; do
    [[ -n "$candidate" ]] || continue
    [[ -e "$queue/done/$candidate" ]] && continue
    if is_complete "$candidate"; then
      touch "$queue/done/$candidate"
      continue
    fi
    if mkdir "$queue/claims/$candidate" 2>/dev/null; then
      printf '%s\n' "$$" > "$queue/claims/$candidate/worker_pid"
      printf '%s\n' "$gpu" > "$queue/claims/$candidate/gpu"
      printf '%s\n' "$mode" > "$queue/claims/$candidate/mode"
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

stop_at_deadline() {
  local wrapper_pid="$1"
  local train_pid
  train_pid="$(pgrep -P "$wrapper_pid" | head -n 1 || true)"
  if [[ -n "$train_pid" ]]; then
    kill -TERM "$train_pid" 2>/dev/null || true
  else
    kill -TERM "$wrapper_pid" 2>/dev/null || true
  fi
}

while true; do
  if [[ "$mode" == "temporary" ]] && (( $(date +%s) >= deadline )); then
    printf '%s GPU%s temporary worker reached deadline\n' "$(date --iso-8601=seconds)" "$gpu"
    exit 0
  fi

  reap_stale_claims
  variant="$(claim_one || true)"
  if [[ -z "$variant" ]]; then
    if [[ "$mode" == "temporary" ]]; then
      sleep 30
    else
      sleep 300
    fi
    continue
  fi

  log="$queue/logs/${variant}_gpu${gpu}.log"
  printf '%s GPU%s starts/resumes %s\n' "$(date --iso-8601=seconds)" "$gpu" "$variant" | tee -a "$log"
  if [[ "$mode" == "temporary" ]]; then
    OFFICIAL_ACT_TEMP_GPU_DEADLINE_EPOCH="$deadline" \
      "$repo/scripts/run_train.sh" "$variant" "$task" "$task_config" "$gpu" "$suffix" >>"$log" 2>&1 &
  else
    "$repo/scripts/run_train.sh" "$variant" "$task" "$task_config" "$gpu" "$suffix" >>"$log" 2>&1 &
  fi
  child="$!"

  deadline_hit=0
  while kill -0 "$child" 2>/dev/null; do
    if [[ "$mode" == "temporary" ]] && (( $(date +%s) >= deadline )); then
      deadline_hit=1
      stop_at_deadline "$child"
      break
    fi
    sleep 15
  done
  wait "$child"
  status="$?"

  if (( deadline_hit )); then
    wait "$child" 2>/dev/null || true
    rm -rf "$queue/claims/$variant"
    printf '%s GPU%s stopped %s at deadline; checkpoint retained\n' "$(date --iso-8601=seconds)" "$gpu" "$variant" | tee -a "$log"
    exit 0
  fi

  if (( status == 0 )) && is_complete "$variant"; then
    touch "$queue/done/$variant"
    rm -rf "$queue/claims/$variant"
    printf '%s GPU%s completed %s\n' "$(date --iso-8601=seconds)" "$gpu" "$variant" | tee -a "$log"
  else
    rm -rf "$queue/claims/$variant"
    printf '%s GPU%s %s exited %s; retrying from checkpoint\n' "$(date --iso-8601=seconds)" "$gpu" "$variant" "$status" | tee -a "$log"
    sleep 30
  fi
done
