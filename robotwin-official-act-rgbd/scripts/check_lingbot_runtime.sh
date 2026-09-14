#!/usr/bin/env bash
set -u

root=/ssd/hhw/depth-model
for environment in aloha dp3 robotwin; do
  python="$root/envs/$environment/bin/python"
  printf '=== %s ===\n' "$environment"
  "$python" - "$root" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path[:0] = [
    str(root / "repos/depth-processing-vendor"),
    str(root / "repos/lingbot-depth"),
    str(root / "repos/RoboTwin/policy/ACT"),
]
try:
    import torch
    import torchvision
    from detr.models import build_ACT_model
    from mdm.model.v2 import MDMModel
    print("OK", sys.version.split()[0], torch.__version__, torchvision.__version__, MDMModel.__name__)
except Exception as error:
    print("FAILED", type(error).__name__, str(error))
    raise
PY
done
