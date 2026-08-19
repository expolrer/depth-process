#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/ssd/hhw/depth-processing}"
INPUT_ROOT="${INPUT_ROOT:-${PROJECT_ROOT}/outputs/extracted}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/processed}"
LINGBOT_REPO="${LINGBOT_REPO:-/ssd/hhw/lingbot-depth}"
CHECKPOINT="${CHECKPOINT:-${LINGBOT_REPO}/checkpoint/lingbot-vla-v2-depth/model.pt}"
PYTHON_BIN="${PYTHON_BIN:-/ssd/hhw/openpi-hzh/.venv/bin/python}"
VENDOR_DIR="${VENDOR_DIR:-${PROJECT_ROOT}/vendor}"
GPU_IDS="${GPU_IDS:-0,6,7}"
BATCH_SIZE="${BATCH_SIZE:-4}"
RESOLUTION_LEVEL="${RESOLUTION_LEVEL:-6}"
OVERWRITE="${OVERWRITE:-0}"
ALLOW_BUSY_GPUS="${ALLOW_BUSY_GPUS:-0}"
LOG_DIR="${PROJECT_ROOT}/logs/lingbot_attention"

IFS=',' read -r -a gpu_ids <<<"${GPU_IDS}"
num_shards="${#gpu_ids[@]}"
if [[ "$num_shards" -lt 1 ]]; then
  echo "GPU_IDS must contain at least one GPU" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}/reports"
rm -f "${OUTPUT_ROOT}/reports"/lingbot_attention_shard_*.json

if [[ "${ALLOW_BUSY_GPUS}" != "1" ]]; then
  for gpu in "${gpu_ids[@]}"; do
    used_mib="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    if [[ "${used_mib}" -gt 2048 ]]; then
      echo "GPU ${gpu} already uses ${used_mib} MiB; refusing to start" >&2
      exit 1
    fi
  done
fi

pids=()
extra_args=()
if [[ "${OVERWRITE}" == "1" ]]; then
  extra_args+=(--overwrite)
fi
for shard_index in "${!gpu_ids[@]}"; do
  gpu="${gpu_ids[$shard_index]}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${VENDOR_DIR}" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/process_lingbot_attention.py" \
      --input-root "${INPUT_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --repo "${LINGBOT_REPO}" \
      --checkpoint "${CHECKPOINT}" \
      --vendor "${VENDOR_DIR}" \
      --shard-index "${shard_index}" \
      --num-shards "${num_shards}" \
      --batch-size "${BATCH_SIZE}" \
      --resolution-level "${RESOLUTION_LEVEL}" \
      "${extra_args[@]}" \
      >"${LOG_DIR}/gpu_${gpu}.log" 2>&1 &
  shard_pid="$!"
  pids+=("${shard_pid}")
  echo "Started attention shard ${shard_index}/${num_shards} on GPU ${gpu}, PID ${shard_pid}"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "At least one LingBot attention shard failed; inspect ${LOG_DIR}" >&2
  exit "${status}"
fi

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/merge_lingbot_attention_reports.py" \
  --reports-dir "${OUTPUT_ROOT}/reports" \
  --output "${OUTPUT_ROOT}/lingbot_attention_summary.json"
