#!/usr/bin/env bash
set -euo pipefail

root=/ssd/hhw/depth-model
repo="$root/repos/official-act-rgbd"
output="$root/experiments/OfficialACTRGBD/_smoke/ACT0_RGB/stack_blocks_two"
CUDA_VISIBLE_DEVICES=2 TORCH_HOME="$root/models/torch" \
  "$root/envs/aloha/bin/python" "$repo/train.py" \
  --variant ACT0_RGB \
  --task stack_blocks_two \
  --task-config depth_master_clean \
  --project-root "$root" \
  --official-act-root "$root/repos/RoboTwin/policy/ACT" \
  --output "$output" \
  --epochs 1 \
  --batch-size 8 \
  --workers 1 \
  --validation-frames 1 \
  --validate-every 1 \
  --save-every 1 \
  --no-resume \
  --device cuda

test -s "$output/policy_best.ckpt"
test -s "$output/policy_last.ckpt"
test -s "$output/training_last.pt"
test -s "$output/dataset_stats.pkl"
test -s "$output/metrics.jsonl"
printf 'TRAIN_CLI_SMOKE_OK %s\n' "$output"
