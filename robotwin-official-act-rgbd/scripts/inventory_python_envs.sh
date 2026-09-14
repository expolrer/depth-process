#!/usr/bin/env bash
set -u

roots=(/ssd/hhw/depth-model/envs /root/miniconda3/envs /opt/conda/envs)
for root in "${roots[@]}"; do
  [[ -d "$root" ]] || continue
  for python in "$root"/*/bin/python; do
    [[ -x "$python" ]] || continue
    environment="$(dirname "$(dirname "$python")")"
    printf '=== %s ===\n' "$environment"
    timeout 15 "$python" - <<'PY'
import importlib.util
import sys

def version(name):
    if importlib.util.find_spec(name) is None:
        return "missing"
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "present"))
    except Exception as error:
        return f"broken:{type(error).__name__}:{error}"

print("python", sys.version.split()[0])
for package in ("torch", "torchvision", "xformers", "sapien", "cv2", "h5py", "huggingface_hub"):
    print(package, version(package))
PY
  done
done
