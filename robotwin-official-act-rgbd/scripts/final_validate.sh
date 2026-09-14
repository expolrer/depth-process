#!/usr/bin/env bash
set -euo pipefail

root=/ssd/hhw/depth-model
repo="$root/repos/official-act-rgbd"
official="$root/repos/RoboTwin/policy/ACT"

while IFS= read -r script; do
  bash -n "$script"
done < <(find "$repo/scripts" -maxdepth 1 -type f -name '*.sh' | sort)

"$root/envs/aloha/bin/python" -m compileall -q "$repo"
"$root/envs/aloha/bin/python" "$repo/train.py" --help >/dev/null

for manifest in \
  "$root/manifests/official_act_rgbd_preflight.json" \
  "$root/manifests/official_act_rgbd_parity.json" \
  "$root/manifests/official_act_rgbd_frontends.json" \
  "$root/manifests/official_act_rgbd_data_contract.json" \
  "$root/manifests/official_act_rgbd_policy_smoke.json" \
  "$root/manifests/official_act_rgbd_real_batch_smoke.json" \
  "$root/manifests/official_act_rgbd_lingbot_frontend_smoke.json" \
  "$root/manifests/official_act_rgbd_lingbot_a3_eval_frontend_smoke.json"; do
  test -s "$manifest"
done

"$root/envs/aloha/bin/python" - "$root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
names = (
    "official_act_rgbd_preflight.json",
    "official_act_rgbd_parity.json",
    "official_act_rgbd_frontends.json",
    "official_act_rgbd_data_contract.json",
    "official_act_rgbd_policy_smoke.json",
    "official_act_rgbd_real_batch_smoke.json",
    "official_act_rgbd_lingbot_frontend_smoke.json",
    "official_act_rgbd_lingbot_a3_eval_frontend_smoke.json",
)
rows = {}
for name in names:
    payload = json.loads((root / "manifests" / name).read_text())
    rows[name] = bool(payload.get("passed"))
print(json.dumps(rows, indent=2))
if not all(rows.values()):
    raise SystemExit(1)
PY

cd "$root/repos/RoboTwin"
PYTHONPATH=".:./policy:$repo:$official" "$root/envs/robotwin/bin/python" - <<'PY'
import OfficialACTRGBD
assert all(hasattr(OfficialACTRGBD, name) for name in ("get_model", "eval", "reset_model"))
print("ROBOTWIN_IMPORT_OK", OfficialACTRGBD.__file__)
PY

cd "$root/repos/RoboTwin-A3"
ROBOTWIN_SKIP_PLANNERS=1 \
PYTHONPATH=".:./policy:$repo:$root/repos/depth-processing-vendor:$root/repos/lingbot-depth:$official" \
  "$root/envs/a3-eval/bin/python" - <<'PY'
import OfficialACTRGBD
assert all(hasattr(OfficialACTRGBD, name) for name in ("get_model", "eval", "reset_model"))
print("A3_EVAL_IMPORT_OK", OfficialACTRGBD.__file__)
PY

printf 'FINAL_VALIDATION_OK\n'
