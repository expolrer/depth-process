#!/usr/bin/env bash
set -euo pipefail

repo=/ssd/hhw/depth-model/repos/official-act-rgbd
mkdir -p \
  "$repo/official_act_rgbd" \
  "$repo/integration/RoboTwin/OfficialACTRGBD" \
  "$repo/scripts" \
  /ssd/hhw/depth-model/manifests \
  /ssd/hhw/depth-model/experiments/OfficialACTRGBD
