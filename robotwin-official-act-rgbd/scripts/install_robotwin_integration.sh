#!/usr/bin/env bash
set -euo pipefail

root="${DEPTH_MODEL_ROOT:-/ssd/hhw/depth-model}"
source_dir="$root/repos/official-act-rgbd/integration/RoboTwin/OfficialACTRGBD"
robotwin="$root/repos/RoboTwin"
target_dir="$robotwin/policy/OfficialACTRGBD"
a3_target_dir="$root/repos/RoboTwin-A3/policy/OfficialACTRGBD"

test -f "$source_dir/__init__.py"
test -f "$source_dir/deploy_policy.yml"
mkdir -p "$target_dir"
install -m 0644 "$source_dir/__init__.py" "$target_dir/__init__.py"
install -m 0644 "$source_dir/deploy_policy.yml" "$target_dir/deploy_policy.yml"
if [[ -d "$root/repos/RoboTwin-A3/policy" ]]; then
  mkdir -p "$a3_target_dir"
  install -m 0644 "$source_dir/__init__.py" "$a3_target_dir/__init__.py"
  install -m 0644 "$source_dir/deploy_policy.yml" "$a3_target_dir/deploy_policy.yml"
fi
printf 'Installed OfficialACTRGBD policy entry in RoboTwin and RoboTwin-A3\n'
