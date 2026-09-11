#!/usr/bin/env bash
set -uo pipefail

gpu="$1"
initial_wrapper_pid="$2"
initial_a4_task="$3"
initial_a4_config="$4"
old_worker_pid="$5"

root=/ssd/depth-model
scheduler="$root/scheduler/continuous_v2"
log="$root/logs/scheduler/gpu${gpu}_v2.log"
status="$scheduler/gpu${gpu}.status"
mkdir -p "$scheduler/phase1/claims" "$scheduler/phase1/done" \
  "$scheduler/phase2/claims" "$scheduler/phase2/done" "$root/logs/scheduler"
exec >>"$log" 2>&1

stamp() { date --iso-8601=seconds; }

set_status() {
  printf '%s %s\n' "$(stamp)" "$*" | tee "$status"
}

pid_is_active() {
  local pid="$1"
  local state
  [[ -r "/proc/$pid/stat" ]] || return 1
  state="$(awk '{print $3}' "/proc/$pid/stat")"
  [[ "$state" != Z ]]
}

run_retry() {
  local label="$1"
  shift
  local attempt rc
  for attempt in 1 2 3; do
    set_status "RUNNING gpu=$gpu task=$label attempt=$attempt"
    "$@"
    rc=$?
    if ((rc == 0)); then
      set_status "FINISHED gpu=$gpu task=$label"
      return 0
    fi
    printf '%s task=%s attempt=%s rc=%s\n' "$(stamp)" "$label" "$attempt" "$rc"
    sleep 60
  done
  set_status "RETRY_LATER gpu=$gpu task=$label"
  return 1
}

custom_eval() {
  run_retry "$1-eval-$2" bash "$root/scripts/run_custom_eval.sh" "$1" "$2" "$3" "$gpu"
}

custom_pipeline() {
  local variant="$1" task="$2" config="$3"
  run_retry "$variant-train-$task" bash "$root/scripts/run_custom_job.sh" \
    "$variant" "$task" "$gpu" 30000 || return 1
  custom_eval "$variant" "$task" "$config"
}

execute_spec() {
  local spec="$1"
  local op variant task config
  IFS='|' read -r op variant task config <<<"$spec"
  case "$op" in
    eval) custom_eval "$variant" "$task" "$config" ;;
    pipeline) custom_pipeline "$variant" "$task" "$config" ;;
    *) printf '%s unknown_spec=%s\n' "$(stamp)" "$spec"; return 2 ;;
  esac
}

run_shared_phase() {
  local phase="$1"
  shift
  local -a jobs=("$@")
  local spec job_id progress
  while true; do
    progress=0
    for spec in "${jobs[@]}"; do
      job_id="${spec//|/_}"
      [[ -f "$scheduler/$phase/done/$job_id" ]] && continue
      if mkdir "$scheduler/$phase/claims/$job_id" 2>/dev/null; then
        if execute_spec "$spec"; then
          touch "$scheduler/$phase/done/$job_id"
        fi
        rmdir "$scheduler/$phase/claims/$job_id" 2>/dev/null || true
        progress=1
        break
      fi
    done

    if ((progress == 0)); then
      local done_count
      done_count="$(find "$scheduler/$phase/done" -mindepth 1 -maxdepth 1 -type f | wc -l)"
      if ((done_count == ${#jobs[@]})); then
        set_status "PHASE_COMPLETE gpu=$gpu phase=$phase jobs=$done_count"
        return 0
      fi
      set_status "WAITING_SHARED_QUEUE gpu=$gpu phase=$phase completed=$done_count/${#jobs[@]}"
      sleep 60
    fi
  done
}

phase1_jobs=(
  'eval|A1|pick_dual_bottles|depth_master_clean'
  'eval|A1|handover_mic|depth_master_clean'
  'eval|A1|place_a2b_left|depth_master_clean'
  'eval|A1|place_a2b_right|depth_master_clean'
  'eval|A1|stack_blocks_two|depth_master_clean'
  'eval|A1|pick_diverse_bottles|depth_master_clutter'
  'eval|A2|pick_dual_bottles|depth_master_clean'
  'eval|A2|handover_mic|depth_master_clean'
  'eval|A2|place_a2b_left|depth_master_clean'
  'eval|A2|place_a2b_right|depth_master_clean'
  'eval|A2|stack_blocks_two|depth_master_clean'
  'eval|A2|pick_diverse_bottles|depth_master_clutter'
  'eval|A5|pick_diverse_bottles|depth_master_clutter'
  'pipeline|A5|pick_dual_bottles|depth_master_clean'
  'pipeline|A5|handover_mic|depth_master_clean'
  'pipeline|A5|place_a2b_right|depth_master_clean'
  'pipeline|A5|stack_blocks_two|depth_master_clean'
)

phase2_jobs=(
  'pipeline|A3|pick_dual_bottles|depth_master_clean'
  'pipeline|A3|handover_mic|depth_master_clean'
  'pipeline|A3|place_a2b_left|depth_master_clean'
  'pipeline|A3|place_a2b_right|depth_master_clean'
  'pipeline|A3|stack_blocks_two|depth_master_clean'
  'pipeline|A3|pick_diverse_bottles|depth_master_clutter'
)

while pid_is_active "$initial_wrapper_pid"; do
  set_status "WAITING_EXISTING gpu=$gpu task=A4-train-$initial_a4_task pid=$initial_wrapper_pid"
  sleep 60
done

if kill -0 "$old_worker_pid" 2>/dev/null; then
  kill -KILL "$old_worker_pid" 2>/dev/null || true
fi

while ! run_retry "A4-eval-$initial_a4_task" bash "$root/scripts/run_a4_eval.sh" \
  "$initial_a4_task" "$initial_a4_config" "$gpu"; do
  sleep 300
done

run_shared_phase phase1 "${phase1_jobs[@]}"

while [[ ! -f "$scheduler/a3_tokens.done" ]]; do
  set_status "WAITING_A3_PRECOMPUTE gpu=$gpu"
  if mkdir "$scheduler/a3_precompute.claim" 2>/dev/null; then
    if run_retry A3-token-precompute bash "$root/scripts/precompute_a3_tokens_v2.sh"; then
      touch "$scheduler/a3_tokens.done"
    else
      rmdir "$scheduler/a3_precompute.claim" 2>/dev/null || true
      sleep 300
    fi
  else
    sleep 60
  fi
done

run_shared_phase phase2 "${phase2_jobs[@]}"
set_status "COMPLETE gpu=$gpu"
touch "$scheduler/gpu${gpu}.done"
