#!/usr/bin/env bash
set -euo pipefail

root="${DEPTH_MODEL_ROOT:-/ssd/hhw/depth-model}"
repo="${OFFICIAL_ACT_RGBD_REPO:-$root/repos/official-act-rgbd}"
official="${ROBOTWIN_ACT_ROOT:-$root/repos/RoboTwin/policy/ACT}"
python="${ACT_PYTHON:-$root/envs/aloha/bin/python}"
manifest="$root/manifests/official_act_rgbd_preflight.json"

test -x "$python"
test -d "$official/detr/models"
test -f "$repo/verify_parity.py"
mkdir -p "$(dirname "$manifest")"

"$python" -m compileall -q "$repo"
"$python" "$repo/verify_parity.py" \
  --official-act-root "$official" \
  --output "$root/manifests/official_act_rgbd_parity.json"
"$python" "$repo/smoke_frontends.py" \
  --official-act-root "$official" \
  --device cpu \
  --output "$root/manifests/official_act_rgbd_frontends.json"
"$python" "$repo/verify_data_contract.py" \
  --project-root "$root" \
  --official-act-root "$official" \
  --output "$root/manifests/official_act_rgbd_data_contract.json"

"$python" - "$root" "$manifest" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
manifest = Path(sys.argv[2])
tasks = {
    "pick_dual_bottles": "depth_master_clean",
    "stack_blocks_two": "depth_master_clean",
    "handover_mic": "depth_master_clean",
    "place_a2b_left": "depth_master_clean",
    "place_a2b_right": "depth_master_clean",
    "pick_diverse_bottles": "depth_master_clutter",
}
rows = {}
for task, config in tasks.items():
    master = root / "datasets/master" / task / config / "data"
    processed = root / "datasets/act_processed" / f"sim-{task}" / f"{config}-50"
    rows[task] = {
        "config": config,
        "master_episodes": len(list(master.glob("episode*.hdf5"))),
        "official_act_episodes": len(list(processed.glob("episode_*.hdf5"))),
    }
passed = all(row["master_episodes"] == 50 and row["official_act_episodes"] == 50 for row in rows.values())
payload = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "passed": passed,
    "datasets": rows,
    "parity": json.loads((root / "manifests/official_act_rgbd_parity.json").read_text()),
    "frontends": json.loads((root / "manifests/official_act_rgbd_frontends.json").read_text()),
    "data_contract": json.loads((root / "manifests/official_act_rgbd_data_contract.json").read_text()),
}
manifest.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
if not passed:
    raise SystemExit(1)
PY
