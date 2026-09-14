#!/usr/bin/env bash
set -euo pipefail

root="${DEPTH_MODEL_ROOT:-/ssd/hhw/depth-model}"
repo="$root/repos/official-act-rgbd"
eval_queue="$root/scheduler/official_act_official6000_eval"
state="$root/scheduler/official_pi05_stack_blocks_two"
variants=(ACT0_RGB ACT1_EARLY_RGBD ACT2_DUAL_SHARED ACT3_DUAL_PER_VIEW ACT4_XYZMAP ACT5_POINT_TOKENS ACT6_LINGBOT_DEPTH ACT7_DEPTH_TRANSFORMER)

mkdir -p "$state"
exec 9>"$state/coordinator.lock"
flock -n 9 || { printf 'pi0.5 successor coordinator is already running\n'; exit 0; }

printf '%s waiting for all official ACT evaluations\n' "$(date --iso-8601=seconds)"
while true; do
  ready=1
  for variant in "${variants[@]}"; do
    [[ -e "$eval_queue/done/$variant" ]] || ready=0
  done
  ((ready)) && break
  sleep 60
done

printf '%s all ACT evaluations complete; waiting for ACT processes to exit\n' "$(date --iso-8601=seconds)"
while pgrep -f 'official-act-rgbd/scripts/official_eval_worker.sh|official-act-rgbd/train.py|script/eval_policy.py.*OfficialACTRGBD' >/dev/null 2>&1; do
  sleep 30
done

printf '%s starting official RoboTwin pi0.5 RGB-only four-GPU full fine-tuning\n' "$(date --iso-8601=seconds)"
exec "$repo/scripts/prepare_and_train_pi05_stack.sh"
