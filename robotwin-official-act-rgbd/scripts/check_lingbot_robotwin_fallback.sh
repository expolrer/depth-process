#!/usr/bin/env bash
set -u

root=/ssd/hhw/depth-model
python="$root/envs/robotwin/bin/python"
for mode in repo_only repo_with_vendor; do
  printf '=== %s ===\n' "$mode"
  "$python" - "$root" "$mode" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
mode = sys.argv[2]
paths = [str(root / "repos/lingbot-depth")]
if mode == "repo_with_vendor":
    paths.insert(0, str(root / "repos/depth-processing-vendor"))
sys.path[:0] = paths
try:
    from mdm.model.v2 import MDMModel
    print("OK", MDMModel.__name__)
except Exception as error:
    print("FAILED", type(error).__name__, str(error))
PY
done
