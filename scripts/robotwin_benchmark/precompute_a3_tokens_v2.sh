#!/usr/bin/env bash
set -euo pipefail

root=/ssd/depth-model
repo="$root/repos/depth-policy-benchmark"
status="$root/manifests/A3_tokens.status"
output_root="$root/datasets/lingbot_tokens"
mkdir -p "$root/logs/A3_tokens" "$output_root"

count_tokens() {
  find "$output_root" -type f -name '*.lingbot_tokens.hdf5' 2>/dev/null | wc -l
}

if [[ "$(count_tokens)" -eq 300 ]]; then
  printf '%s COMPLETE files=300\n' "$(date --iso-8601=seconds)" >"$status"
  exit 0
fi

printf '%s RUNNING shards=3 gpu=4-6\n' "$(date --iso-8601=seconds)" >"$status"
pids=()
for shard in 0 1 2; do
  gpu=$((shard + 4))
  CUDA_VISIBLE_DEVICES="$gpu" XFORMERS_DISABLED=1 \
    PYTHONPATH="$repo:$root/repos/depth-processing-vendor:$root/repos/lingbot-depth" \
    "$root/envs/dp3/bin/python" "$repo/precompute_lingbot_tokens.py" \
    --shard-index "$shard" --num-shards 3 --batch-size 4 --resolution-level 0 \
    >"$root/logs/A3_tokens/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

count="$(count_tokens)"
if [[ "$count" -ne 300 ]]; then
  printf '%s FAILED files=%s expected=300\n' "$(date --iso-8601=seconds)" "$count" >"$status"
  exit 1
fi
printf '%s COMPLETE files=%s\n' "$(date --iso-8601=seconds)" "$count" >"$status"
