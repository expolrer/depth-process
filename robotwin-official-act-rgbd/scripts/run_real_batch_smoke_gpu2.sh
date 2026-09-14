#!/usr/bin/env bash
set -euo pipefail

root=/ssd/hhw/depth-model
repo="$root/repos/official-act-rgbd"
CUDA_VISIBLE_DEVICES=2 TORCH_HOME="$root/models/torch" \
  "$root/envs/aloha/bin/python" "$repo/smoke_real_batch.py" \
  --project-root "$root" \
  --official-act-root "$root/repos/RoboTwin/policy/ACT" \
  --variant ACT2_DUAL_SHARED \
  --task stack_blocks_two \
  --device cuda \
  --output "$root/manifests/official_act_rgbd_real_batch_smoke.json"
