#!/usr/bin/env bash
set -euo pipefail

root=/ssd/hhw/depth-model
repo="$root/repos/official-act-rgbd"
CUDA_VISIBLE_DEVICES=2 TORCH_HOME="$root/models/torch" \
  "$root/envs/a3-eval/bin/python" "$repo/smoke_frontends.py" \
  --official-act-root "$root/repos/RoboTwin/policy/ACT" \
  --device cuda \
  --height 224 \
  --width 224 \
  --variants ACT6_LINGBOT_DEPTH \
  --lingbot-repo "$root/repos/lingbot-depth" \
  --lingbot-checkpoint "$root/models/lingbot-depth-v0.5/model.pt" \
  --lingbot-vendor "$root/repos/depth-processing-vendor" \
  --output "$root/manifests/official_act_rgbd_lingbot_a3_eval_frontend_smoke.json"
