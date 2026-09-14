#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  printf 'Usage: %s VARIANT TASK TASK_CONFIG GPU [OUTPUT_SUFFIX]\n' "$0" >&2
  exit 2
fi

variant="$1"
task="$2"
config="$3"
gpu="$4"
suffix="${5:-seed0}"
platform="${OFFICIAL_ACT_PLATFORM:-server56_h100}"
root="${DEPTH_MODEL_ROOT:-/ssd/hhw/depth-model}"
repo="$root/repos/official-act-rgbd"
if [[ "$variant" == "ACT6_LINGBOT_DEPTH" ]]; then
  python="$root/envs/dp3/bin/python"
else
  python="$root/envs/aloha/bin/python"
fi
output="$root/experiments/OfficialACTRGBD/$variant/$task/$suffix"

case "$platform" in
  server56_h100|server56)
    [[ "$gpu" =~ ^[4-7]$ ]] || { printf 'server56 H100 formal jobs require physical GPU4-7; GPU0 is forbidden\n' >&2; exit 2; }
    ;;
  *)
    printf 'Training platform must be server56_h100, got %s\n' "$platform" >&2
    exit 2
    ;;
esac
"$python" "$repo/scripts/workflow_guard.py" train "$variant" "$task"
mkdir -p "$output"
CUDA_VISIBLE_DEVICES="$gpu" TORCH_HOME="$root/models/torch" \
  "$python" "$repo/train.py" \
  --variant "$variant" \
  --task "$task" \
  --task-config "$config" \
  --project-root "$root" \
  --official-act-root "$root/repos/RoboTwin/policy/ACT" \
  --output "$output" \
  --lingbot-repo "$root/repos/lingbot-depth" \
  --lingbot-checkpoint "$root/models/lingbot-depth-v0.5/model.pt" \
  --lingbot-vendor "$root/repos/depth-processing-vendor" \
  --device cuda
"$python" "$repo/scripts/artifact_manifest.py" create "$output" --repo "$repo"
