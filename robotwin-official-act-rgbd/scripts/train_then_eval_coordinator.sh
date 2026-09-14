#!/usr/bin/env bash
set -euo pipefail

root="${DEPTH_MODEL_ROOT:-/ssd/hhw/depth-model}"
repo="$root/repos/official-act-rgbd"
train_root="$root/experiments/OfficialACTRGBD"
queue="$root/scheduler/official_act_official6000_eval"
variants=(ACT0_RGB ACT1_EARLY_RGBD ACT2_DUAL_SHARED ACT3_DUAL_PER_VIEW ACT4_XYZMAP ACT5_POINT_TOKENS ACT6_LINGBOT_DEPTH ACT7_DEPTH_TRANSFORMER)

mkdir -p "$queue/logs"
while true; do
  ready=1
  for variant in "${variants[@]}"; do
    output="$train_root/$variant/stack_blocks_two/official6000_seed0"
    [[ -s "$output/training_complete.json" && -s "$output/artifact_manifest.json" ]] || ready=0
  done
  ((ready)) && break
  sleep 60
done

if mkdir "$queue/launched" 2>/dev/null; then
  declare -A preferred=([0]=ACT0_RGB [1]=ACT1_EARLY_RGBD [2]=ACT2_DUAL_SHARED [3]=ACT3_DUAL_PER_VIEW)
  for gpu in 0 1 2 3; do
    nohup "$repo/scripts/official_eval_worker.sh" "$gpu" "${preferred[$gpu]}" \
      >>"$queue/logs/worker_gpu${gpu}.log" 2>&1 &
    printf 'started evaluation worker GPU%s pid=%s preferred=%s\n' "$gpu" "$!" "${preferred[$gpu]}"
  done
fi
