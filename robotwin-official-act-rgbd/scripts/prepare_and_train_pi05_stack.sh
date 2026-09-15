#!/usr/bin/env bash
set -euo pipefail

root="${DEPTH_MODEL_ROOT:-/ssd/hhw/depth-model}"
repo="$root/repos/official-act-rgbd"
robotwin="$root/repos/RoboTwin"
pi05="$robotwin/policy/pi05"
python=/ssd/hhw/openpi-hzh/.venv/bin/python
raw="$pi05/processed_data/stack_blocks_two-depth_master_clean-50"
lerobot_home="$root/datasets/lerobot"
repo_id=robotwin/stack_blocks_two_rgb_50
dataset="$lerobot_home/$repo_id"
state="$root/scheduler/official_pi05_stack_blocks_two"
checkpoint="$root/experiments/OfficialPi05/checkpoints/pi05_robotwin_stack_blocks_two_full_4gpu/pi05_stack_blocks_two_rgb_full_seed0"

mkdir -p "$state" "$lerobot_home"
exec 9>"$state/worker.lock"
flock -n 9 || { printf 'pi0.5 successor worker is already running\n'; exit 0; }

export PYTHONPATH="$pi05:$pi05/src:$pi05/packages/openpi-client/src"
export HF_LEROBOT_HOME="$lerobot_home"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

if [[ ! -s "$state/raw_rgb_complete" ]]; then
  cd "$pi05"
  "$python" scripts/process_data.py stack_blocks_two depth_master_clean 50
  test "$(find "$raw" -mindepth 1 -maxdepth 1 -type d -name 'episode_*' | wc -l)" -eq 50
  printf 'complete\n' > "$state/raw_rgb_complete"
fi

if [[ ! -s "$state/lerobot_rgb_complete" ]]; then
  cd "$pi05"
  "$python" examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py \
    --raw_dir "$raw" --repo_id "$repo_id"
  test -s "$dataset/meta/info.json"
  "$python" - "$dataset/meta/info.json" <<'PY'
import json, sys
info = json.load(open(sys.argv[1], encoding="utf-8"))
features = info.get("features", {})
depth_keys = [key for key in features if "depth" in key.lower()]
if depth_keys:
    raise SystemExit(f"RGB-only dataset unexpectedly contains depth features: {depth_keys}")
required = {
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
    "observation.state",
    "action",
}
missing = sorted(required - set(features))
if missing:
    raise SystemExit(f"Missing required features: {missing}")
if int(info.get("total_episodes", 0)) != 50:
    raise SystemExit(f"Expected 50 episodes, got {info.get('total_episodes')}")
print("RGB-only LeRobot dataset verified", info.get("total_frames"), "frames")
PY
  printf 'complete\n' > "$state/lerobot_rgb_complete"
fi

asset="$root/experiments/OfficialPi05/assets/pi05_robotwin_stack_blocks_two_full_4gpu/$repo_id/norm_stats.json"
if [[ ! -s "$asset" ]]; then
  cd "$pi05"
  env JAX_PLATFORMS=cpu "$python" "$repo/scripts/pi05_stack_blocks_two_full.py" norm-stats
  test -s "$asset"
fi

while [[ ! -s "$state/training_complete.json" ]]; do
  resume=()
  if find "$checkpoint" -mindepth 1 -maxdepth 1 -type d -regextype posix-extended -regex '.*/[0-9]+' -print -quit 2>/dev/null | grep -q .; then
    resume=(--resume)
  fi
  printf '%s starts pi0.5 full fine-tuning resume=%s\n' "$(date --iso-8601=seconds)" "${resume[*]:-false}"
  set +e
  cd "$pi05"
  env CUDA_VISIBLE_DEVICES=4,5,6,7 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    JAX_COMPILATION_CACHE_DIR="$root/cache/jax" \
    "$python" "$repo/scripts/pi05_stack_blocks_two_full.py" train "${resume[@]}"
  status_code=$?
  set -e
  final_step="$(find "$checkpoint" -mindepth 1 -maxdepth 1 -type d -regextype posix-extended -regex '.*/[0-9]+' -printf '%f\n' 2>/dev/null | sort -n | tail -1)"
  if [[ "$status_code" -eq 0 && "${final_step:-0}" -ge 20000 ]]; then
    "$python" - "$state/training_complete.json" "$checkpoint" "$final_step" <<'PY'
import datetime, json, sys
payload = {
    "status": "complete",
    "completed_at": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
    "checkpoint_dir": sys.argv[2],
    "final_step": int(sys.argv[3]),
    "full_finetune": True,
    "depth_used": False,
    "gpus": [4, 5, 6, 7],
}
open(sys.argv[1], "w", encoding="utf-8").write(json.dumps(payload, indent=2) + "\n")
PY
    break
  fi
  printf '%s pi0.5 training exited %s at step %s; retrying from latest checkpoint\n' \
    "$(date --iso-8601=seconds)" "$status_code" "${final_step:-none}" >&2
  sleep 60
done
