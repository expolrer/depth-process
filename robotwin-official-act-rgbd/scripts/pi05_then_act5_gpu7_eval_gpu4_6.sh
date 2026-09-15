#!/usr/bin/env bash
set -euo pipefail

root="${DEPTH_MODEL_ROOT:-/ssd/hhw/depth-model}"
repo="$root/repos/official-act-rgbd"
pi_state="$root/scheduler/official_pi05_stack_blocks_two"
train_queue="$root/scheduler/official_act_official6000"
eval_queue="$root/scheduler/official_act_official6000_eval"
state="$root/scheduler/post_pi05_act5_eval"
variants=(ACT0_RGB ACT1_EARLY_RGBD ACT2_DUAL_SHARED ACT3_DUAL_PER_VIEW ACT4_XYZMAP ACT5_POINT_TOKENS ACT6_LINGBOT_DEPTH ACT7_DEPTH_TRANSFORMER)

mkdir -p "$state" "$train_queue/claims" "$train_queue/done" "$eval_queue/claims" "$eval_queue/done" "$eval_queue/logs"
exec 9>"$state/coordinator.lock"
flock -n 9 || { printf 'post-pi0.5 coordinator is already running\n'; exit 0; }
printf '%s\n' "$$" > "$state/coordinator.pid"

pi05_complete() {
  [[ -s "$pi_state/training_complete.json" ]] || return 1
  python3 - "$pi_state/training_complete.json" <<'PY'
import json, sys
state = json.load(open(sys.argv[1], encoding="utf-8"))
ok = state.get("status") == "complete" and int(state.get("final_step", 0)) >= 20000
raise SystemExit(0 if ok else 1)
PY
}

gpu_is_free() {
  local gpu="$1" uuid
  uuid="$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)"
  ! nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -Fxq "$uuid"
}

reap_stale_claims() {
  local queue="$1" claim owner
  for claim in "$queue"/claims/*; do
    [[ -d "$claim" ]] || continue
    owner="$(cat "$claim/worker_pid" 2>/dev/null || true)"
    if [[ -z "$owner" || ! -r "/proc/$owner/cmdline" ]]; then
      rm -rf "$claim"
    fi
  done
}

process_alive() {
  local pid_file="$1" needle="$2" pid cmd
  [[ -s "$pid_file" ]] || return 1
  pid="$(cat "$pid_file")"
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
  [[ "$cmd" == *"$needle"* ]]
}

start_act5_worker() {
  rm -f "$train_queue/done/ACT5_POINT_TOKENS"
  rm -rf "$train_queue/claims/ACT5_POINT_TOKENS"
  nohup "$repo/scripts/eight_arch_train_worker.sh" 7 long ACT5_POINT_TOKENS \
    >>"$state/act5_gpu7.log" 2>&1 &
  printf '%s\n' "$!" > "$state/act5_gpu7.pid"
  printf '%s GPU7 resumed ACT5_POINT_TOKENS pid=%s\n' "$(date --iso-8601=seconds)" "$!"
}

start_eval_worker() {
  local gpu="$1" preferred_variant="$2" pid_file="$state/eval_gpu${gpu}.pid"
  nohup "$repo/scripts/official_eval_worker.sh" "$gpu" "$preferred_variant" \
    >>"$state/eval_gpu${gpu}.log" 2>&1 &
  printf '%s\n' "$!" > "$pid_file"
  printf '%s GPU%s evaluation worker pid=%s preferred=%s\n' \
    "$(date --iso-8601=seconds)" "$gpu" "$!" "$preferred_variant"
}

printf '%s waiting for pi0.5 step 20000 completion\n' "$(date --iso-8601=seconds)"
until pi05_complete; do
  sleep 60
done
printf '%s pi0.5 complete; waiting for GPU4-7 release\n' "$(date --iso-8601=seconds)"
until gpu_is_free 4 && gpu_is_free 5 && gpu_is_free 6 && gpu_is_free 7; do
  sleep 15
done

reap_stale_claims "$train_queue"
reap_stale_claims "$eval_queue"
act5_complete="$root/experiments/OfficialACTRGBD/ACT5_POINT_TOKENS/stack_blocks_two/official6000_seed0/training_complete.json"
if [[ ! -s "$act5_complete" ]] && ! process_alive "$state/act5_gpu7.pid" 'eight_arch_train_worker.sh 7 long ACT5_POINT_TOKENS'; then
  start_act5_worker
fi

# Prevent the legacy GPU0-3 coordinator from launching a duplicate pool.
mkdir -p "$eval_queue/launched"
declare -A preferred=([4]=ACT0_RGB [5]=ACT1_EARLY_RGBD [6]=ACT2_DUAL_SHARED)
for gpu in 4 5 6; do
  pid_file="$state/eval_gpu${gpu}.pid"
  if ! process_alive "$pid_file" "official_eval_worker.sh $gpu "; then
    start_eval_worker "$gpu" "${preferred[$gpu]}"
  fi
done
printf '%s\n' "$(date --iso-8601=seconds)" > "$state/launched_at"

while true; do
  done_count=0
  for variant in "${variants[@]}"; do
    [[ -e "$eval_queue/done/$variant" ]] && ((done_count += 1))
  done
  if [[ -s "$act5_complete" && "$done_count" -eq "${#variants[@]}" ]]; then
    break
  fi
  if [[ ! -s "$act5_complete" ]] && ! process_alive "$state/act5_gpu7.pid" 'eight_arch_train_worker.sh 7 long ACT5_POINT_TOKENS'; then
    reap_stale_claims "$train_queue"
    start_act5_worker
  fi
  for gpu in 4 5 6; do
    if ! process_alive "$state/eval_gpu${gpu}.pid" "official_eval_worker.sh $gpu "; then
      reap_stale_claims "$eval_queue"
      start_eval_worker "$gpu" "${preferred[$gpu]}"
    fi
  done
  sleep 60
done

for pid_file in "$state"/act5_gpu7.pid "$state"/eval_gpu{4,5,6}.pid; do
  [[ -s "$pid_file" ]] || continue
  pid="$(cat "$pid_file")"
  kill -TERM "$pid" 2>/dev/null || true
done
printf '{"status":"complete","completed_at":"%s"}\n' "$(date --iso-8601=seconds)" > "$state/workflow_complete.json"
