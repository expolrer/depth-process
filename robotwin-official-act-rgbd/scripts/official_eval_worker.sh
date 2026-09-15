#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 2 ]]; then
  printf 'Usage: %s GPU PREFERRED_VARIANT\n' "$0" >&2
  exit 2
fi

gpu="$1"
preferred="$2"
root="${DEPTH_MODEL_ROOT:-/ssd/hhw/depth-model}"
repo="$root/repos/official-act-rgbd"
queue="$root/scheduler/official_act_official6000_eval"
task=stack_blocks_two
task_config=depth_master_clean
suffix=official6000_seed0
seed_file="$root/repos/RoboTwin/eval_result/stack_blocks_two/ACT/depth_master_clean/A0-seed0/2026-09-10 22:28:35/valid_seeds.txt"
variants=(ACT0_RGB ACT1_EARLY_RGBD ACT2_DUAL_SHARED ACT3_DUAL_PER_VIEW ACT4_XYZMAP ACT5_POINT_TOKENS ACT6_LINGBOT_DEPTH ACT7_DEPTH_TRANSFORMER)

mkdir -p "$queue/claims" "$queue/done" "$queue/logs"
test "$(wc -l < "$seed_file")" -ge 100

is_ready() {
  local candidate="$1"
  local output="$root/experiments/OfficialACTRGBD/$candidate/$task/$suffix"
  [[ -s "$output/training_complete.json" && -s "$output/artifact_manifest.json" ]]
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
    [[ -e "$queue/done/$candidate" ]] && continue
    is_ready "$candidate" || continue
    if mkdir "$queue/claims/$candidate" 2>/dev/null; then
      printf '%s\n' "$$" > "$queue/claims/$candidate/worker_pid"
      printf '%s\n' "$gpu" > "$queue/claims/$candidate/gpu"
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

all_done() {
  local candidate
  for candidate in "${variants[@]}"; do
    [[ -e "$queue/done/$candidate" ]] || return 1
  done
}

while true; do
  reap_stale_claims
  variant="$(claim_one || true)"
  if [[ -z "$variant" ]]; then
    if all_done; then
      printf '%s GPU%s all official evaluations complete; worker exits\n' "$(date --iso-8601=seconds)" "$gpu"
      exit 0
    fi
    sleep 300
    continue
  fi
  checkpoint="$root/experiments/OfficialACTRGBD/$variant/$task/$suffix"
  log="$queue/logs/${variant}_gpu${gpu}.log"
  printf '%s GPU%s starts official evaluation %s\n' "$(date --iso-8601=seconds)" "$gpu" "$variant" | tee -a "$log"
  OFFICIAL_ACT_PLATFORM=server56_h100 "$repo/scripts/run_eval.sh" \
    "$variant" "$task" "$task_config" "$gpu" "$checkpoint" "$seed_file" 100 >>"$log" 2>&1
  status="$?"
  if ((status == 0)); then
    touch "$queue/done/$variant"
    rm -rf "$queue/claims/$variant"
    printf '%s GPU%s completed official evaluation %s\n' "$(date --iso-8601=seconds)" "$gpu" "$variant" | tee -a "$log"
  else
    rm -rf "$queue/claims/$variant"
    printf '%s GPU%s evaluation %s failed with %s; retrying\n' "$(date --iso-8601=seconds)" "$gpu" "$variant" "$status" | tee -a "$log"
    sleep 60
  fi
done
