#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/ssd/hhw/depth-processing}"
INPUT_ROOT="${INPUT_ROOT:-${PROJECT_ROOT}/outputs/extracted}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/processed}"
LINGBOT_REPO="${LINGBOT_REPO:-/ssd/hhw/lingbot-depth}"
CHECKPOINT="${CHECKPOINT:-${LINGBOT_REPO}/checkpoint/lingbot-vla-v2-depth/model.pt}"
PYTHON_BIN="${PYTHON_BIN:-/ssd/hhw/openpi-hzh/.venv/bin/python}"
VENDOR_DIR="${VENDOR_DIR:-${PROJECT_ROOT}/vendor}"
BATCH_SIZE="${BATCH_SIZE:-2}"
RESOLUTION_LEVEL="${RESOLUTION_LEVEL:-6}"
OVERWRITE="${OVERWRITE:-0}"
LOG_DIR="${PROJECT_ROOT}/logs/lingbot_v05"

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}/reports"

pids=()
extra_args=()
if [[ "${OVERWRITE}" == "1" ]]; then
  extra_args+=(--overwrite)
fi
for gpu in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${VENDOR_DIR}" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/process_lingbot.py" \
      --input-root "${INPUT_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --repo "${LINGBOT_REPO}" \
      --checkpoint "${CHECKPOINT}" \
      --vendor "${VENDOR_DIR}" \
      --shard-index "${gpu}" \
      --num-shards 8 \
      --batch-size "${BATCH_SIZE}" \
      --resolution-level "${RESOLUTION_LEVEL}" \
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
  echo "At least one LingBot-Depth shard failed; inspect ${LOG_DIR}" >&2
  exit "${status}"
fi

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/merge_lingbot_reports.py" \
  --reports-dir "${OUTPUT_ROOT}/reports" \
  --output "${OUTPUT_ROOT}/lingbot_v05_summary.json"
