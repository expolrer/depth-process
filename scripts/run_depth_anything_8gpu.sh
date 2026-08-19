#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/ssd/hhw/depth-processing}"
INPUT_ROOT="${INPUT_ROOT:-${PROJECT_ROOT}/outputs/extracted}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/processed}"
DA_REPO="${DA_REPO:-/ssd/hhw/Depth-Anything-V2}"
CHECKPOINT="${CHECKPOINT:-${DA_REPO}/checkpoints/depth_anything_v2_vits.pth}"
PYTHON_BIN="${PYTHON_BIN:-/ssd/hhw/openpi-hzh/.venv/bin/python}"
BATCH_SIZE="${BATCH_SIZE:-8}"
INPUT_SIZE="${INPUT_SIZE:-392}"
OVERWRITE="${OVERWRITE:-0}"
LOG_DIR="${PROJECT_ROOT}/logs/depth_anything_v2"

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}/reports"
pids=()
extra_args=()
if [[ "${OVERWRITE}" == "1" ]]; then
  extra_args+=(--overwrite)
fi
for gpu in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/scripts/process_depth_anything_v2.py" \
      --input-root "${INPUT_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --repo "${DA_REPO}" \
      --checkpoint "${CHECKPOINT}" \
      --shard-index "${gpu}" \
      --num-shards 8 \
      --batch-size "${BATCH_SIZE}" \
      --input-size "${INPUT_SIZE}" \
      "${extra_args[@]}" \
      >"${LOG_DIR}/gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "At least one Depth-Anything-V2 shard failed; inspect ${LOG_DIR}" >&2
  exit "${status}"
fi
