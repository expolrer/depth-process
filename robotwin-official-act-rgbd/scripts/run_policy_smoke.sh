#!/usr/bin/env bash
set -euo pipefail

root="${DEPTH_MODEL_ROOT:-/ssd/hhw/depth-model}"
repo="$root/repos/official-act-rgbd"
official="$root/repos/RoboTwin/policy/ACT"
python="$root/envs/aloha/bin/python"

"$python" "$repo/smoke_policy.py" \
  --official-act-root "$official" \
  --device cpu \
  --output "$root/manifests/official_act_rgbd_policy_smoke.json"
