#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  printf 'Usage: %s VARIANT TASK TASK_CONFIG GPU CHECKPOINT_DIR SEED_FILE [ROLLOUTS]\n' "$0" >&2
  exit 2
fi

variant="$1"
task="$2"
config="$3"
gpu="$4"
checkpoint_dir="$5"
seed_file="$6"
requested_rollouts="${7:-}"
platform="${OFFICIAL_ACT_PLATFORM:-server56_h100}"
root="${DEPTH_MODEL_ROOT:-/ssd/hhw/depth-model}"
robotwin="$root/repos/RoboTwin"
python="$root/envs/robotwin/bin/python"
isolation="$root/runtime/sapien-gpu-isolation/libsapien_gpu_redirect.so"
setting="${variant}-controlled-seed0"
runtime_env=(XFORMERS_DISABLED=1)
python_path=".:$root/repos/official-act-rgbd"
if [[ "$variant" == "ACT6_LINGBOT_DEPTH" ]]; then
  robotwin="$root/repos/RoboTwin-A3"
  python="$root/envs/a3-eval/bin/python"
  runtime_env=(ROBOTWIN_SKIP_PLANNERS=1)
  python_path=".:$root/repos/official-act-rgbd:$root/repos/depth-processing-vendor:$root/repos/lingbot-depth:$root/repos/RoboTwin/policy/ACT"
fi

test -s "$checkpoint_dir/policy_best.ckpt"
test -s "$seed_file"
"$python" "$root/repos/official-act-rgbd/scripts/workflow_guard.py" eval "$variant" "$task"
planned_rollouts=$("$python" -c '
import json, sys
plan = json.load(open(sys.argv[1], encoding="utf-8"))
stage = next(item for item in plan["stages"] if item["id"] == plan["current_stage"])
print(stage["eval_episodes"])
' "$root/repos/official-act-rgbd/execution_plan.json")
if [[ -n "$requested_rollouts" && "$requested_rollouts" != "$planned_rollouts" ]]; then
  printf 'Requested rollouts %s do not match planned rollouts %s\n' "$requested_rollouts" "$planned_rollouts" >&2
  exit 2
fi
rollouts="$planned_rollouts"
"$python" "$root/repos/official-act-rgbd/scripts/artifact_manifest.py" verify "$checkpoint_dir"
cd "$robotwin"
common_env=(CUDA_VISIBLE_DEVICES="$gpu" TORCH_HOME="$root/models/torch" PYTHONPATH="$python_path:${PYTHONPATH:-}")
case "$platform" in
  server56_h100|server56)
    [[ "$gpu" =~ ^[4-7]$ ]] || { printf 'server56 formal jobs require physical GPU4-7; GPU0 is forbidden\n' >&2; exit 2; }
    test -r "$isolation"
    platform_env=(ROBOTWIN_PHYSICAL_GPU="$gpu" ROBOTWIN_DRM_RENDER_INDEX="$((128 + gpu))" LD_PRELOAD="$isolation")
    ;;
  autodl_4090d|autodl)
    [[ "$gpu" =~ ^[0-7]$ ]] || { printf 'autodl 4090D logical GPU must be 0-7\n' >&2; exit 2; }
    platform_env=(ROBOTWIN_PHYSICAL_GPU="$gpu")
    ;;
  *)
    printf 'Evaluation platform must be server56_h100 or autodl_4090d, got %s\n' "$platform" >&2
    exit 2
    ;;
esac
env "${common_env[@]}" "${platform_env[@]}" "${runtime_env[@]}" "$python" script/eval_policy.py \
  --config policy/OfficialACTRGBD/deploy_policy.yml \
  --overrides \
  --policy_name OfficialACTRGBD \
  --variant "$variant" \
  --task_name "$task" \
  --task_config "$config" \
  --ckpt_setting "$setting" \
  --checkpoint "$checkpoint_dir/policy_best.ckpt" \
  --seed 0 \
  --seed_list_path "$seed_file" \
  --test_num "$rollouts"
