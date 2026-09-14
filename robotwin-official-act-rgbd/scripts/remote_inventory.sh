#!/usr/bin/env bash
set -euo pipefail

root=/ssd/hhw/depth-model
printf 'TIME %s\n' "$(date --iso-8601=seconds)"
printf 'ROOT '
test -d "$root" && printf 'OK\n' || printf 'MISSING\n'
printf 'OFFICIAL_ACT '
test -d "$root/repos/RoboTwin/policy/ACT" && printf 'OK\n' || printf 'MISSING\n'
printf 'ALOHA_PYTHON '
test -x "$root/envs/aloha/bin/python" && printf 'OK\n' || printf 'MISSING\n'
printf 'ROBOTWIN_PYTHON '
test -x "$root/envs/robotwin/bin/python" && printf 'OK\n' || printf 'MISSING\n'

sha256sum \
  "$root/repos/RoboTwin/policy/ACT/detr/models/detr_vae.py" \
  "$root/repos/RoboTwin/policy/ACT/detr/models/backbone.py" \
  "$root/repos/RoboTwin/policy/ACT/detr/models/transformer.py" \
  "$root/repos/RoboTwin/policy/ACT/act_policy.py" \
  "$root/repos/RoboTwin/policy/ACT/utils.py" \
  "$root/repos/RoboTwin/policy/ACT/process_data.py"

nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
printf '%s\n' '--- PROJECT PROCESSES ---'
ps -eo pid,etimes,args --sort=pid | grep -E 'RGBDACTv2|OfficialACTRGBD|continuous_v2|action_step_diagnostic' | grep -v grep || true
printf '%s\n' '--- CURRENT RGBD INTEGRATION ---'
ls -l "$root/repos/RoboTwin/RGBDACTv2.py" "$root/repos/RoboTwin/policy/RGBDACTv2/deploy_policy.yml" 2>/dev/null || true
(
  cd "$root/repos/RoboTwin"
  PYTHONPATH=".:./policy:$root/repos/rgbd-act-v2" "$root/envs/robotwin/bin/python" - <<'PY'
import RGBDACTv2
print("RGBDACTv2_IMPORT", RGBDACTv2.__file__)
PY
)
(
  cd "$root/repos/RoboTwin"
  PYTHONPATH=".:./policy:$root/repos/official-act-rgbd" "$root/envs/robotwin/bin/python" - <<'PY'
import OfficialACTRGBD
print("OFFICIAL_ACT_RGBD_IMPORT", OfficialACTRGBD.__file__)
for name in ("get_model", "eval", "reset_model"):
    assert hasattr(OfficialACTRGBD, name), name
PY
)
