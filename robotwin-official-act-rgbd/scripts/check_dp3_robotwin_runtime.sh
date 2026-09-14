#!/usr/bin/env bash
set -euo pipefail

root=/ssd/hhw/depth-model
cd "$root/repos/RoboTwin"
PYTHONPATH=".:./policy:$root/repos/official-act-rgbd:$root/repos/depth-processing-vendor:$root/repos/lingbot-depth" \
  "$root/envs/dp3/bin/python" - <<'PY'
import cv2
import h5py
import numpy
import sapien
import torch
import yaml
from envs import CONFIGS_PATH
from envs.stack_blocks_two import stack_blocks_two
from mdm.model.v2 import MDMModel
print("OK", torch.__version__, sapien.__version__, CONFIGS_PATH, stack_blocks_two.__name__, MDMModel.__name__)
PY
