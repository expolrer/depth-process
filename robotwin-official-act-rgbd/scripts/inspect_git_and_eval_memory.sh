#!/usr/bin/env bash
set -euo pipefail

root=/ssd/hhw/depth-model
repo="$root/repos/official-act-rgbd"

printf '%s\n' '--- NEW REPOSITORY ---'
if git -C "$repo" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git -C "$repo" status --short
  git -C "$repo" remote -v || true
  git -C "$repo" branch --show-current
else
  printf 'NOT_A_GIT_REPOSITORY %s\n' "$repo"
fi

printf '%s\n' '--- AVAILABLE GIT REPOSITORIES ---'
find "$root/repos" -maxdepth 3 -type d -name .git -printf '%h\n' | sort

printf '%s\n' '--- GIT REMOTES ---'
while IFS= read -r candidate; do
  printf '[%s]\n' "$candidate"
  git -C "$candidate" remote -v 2>/dev/null || true
done < <(find "$root/repos" -maxdepth 3 -type d -name .git -printf '%h\n' | sort)

printf '%s\n' '--- GPU INVENTORY ---'
nvidia-smi --query-gpu=index,name,memory.total,memory.used,compute_cap --format=csv,noheader
printf '%s\n' '--- COMPUTE PROCESSES ---'
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader

printf '%s\n' '--- CHECKPOINT SIZES ---'
find "$root/experiments" -type f \( -name 'policy_best.ckpt' -o -name 'policy_best.pt' \) -printf '%s %p\n' \
  | sort -nr | head -n 20

