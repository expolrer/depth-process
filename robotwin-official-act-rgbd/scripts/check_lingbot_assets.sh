#!/usr/bin/env bash
set -euo pipefail

root=/ssd/hhw/depth-model
for path in \
  "$root/repos/lingbot-depth" \
  "$root/repos/depth-processing-vendor" \
  "$root/models/lingbot-depth-v0.5/model.pt"; do
  if [[ -e "$path" ]]; then
    du -sh "$path"
  else
    printf 'MISSING %s\n' "$path"
  fi
done
find "$root" -maxdepth 5 -type f \( -iname 'model.pt' -o -iname '*v0.5*.pt' \) -printf '%s %p\n' | sort -n
