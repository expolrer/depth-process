#!/usr/bin/env bash
set -euo pipefail

ROOT=/ssd/hhw/depth-processing
PYTHON=/ssd/hhw/openpi-hzh/.venv/bin/python
export PYTHONPATH="$ROOT/vendor/lerobot_act_compat:/root/.cache/uv/archive-v0/iDys0lhm1IZhRSQL"
SCRIPT="$ROOT/scripts/compute_act_approved_roi_attention.py"
INDEX="$ROOT/outputs/act_no_prompt/index/index.jsonl"
TRACKS="$ROOT/outputs/target_tracks_required/track_index.jsonl"
CHECKPOINT="$ROOT/outputs/act_no_prompt/shared_act_depth_only/checkpoint_009000.pt"
BENCHMARK="$ROOT/outputs/act_no_prompt/final_depth_only/benchmark_report.json"
OUTPUT="$ROOT/outputs/act_approved_roi_attention"

COMMON=(
  --root "$ROOT"
  --index "$INDEX"
  --tracks "$TRACKS"
  --checkpoint "$CHECKPOINT"
  --benchmark-report "$BENCHMARK"
  --output-dir "$OUTPUT"
  --batch-size 32
  --num-workers 4
  --overwrite
)

mkdir -p "$OUTPUT/logs"
CUDA_VISIBLE_DEVICES=6 "$PYTHON" "$SCRIPT" "${COMMON[@]}" --device cuda:0 \
  --methods raw_aligned rgb_guided temporal_rgb_guided lingbot_v05 zero_depth \
  >"$OUTPUT/logs/gpu6.log" 2>&1 &
PID6=$!

CUDA_VISIBLE_DEVICES=7 "$PYTHON" "$SCRIPT" "${COMMON[@]}" --device cuda:0 \
  --methods depth_anything_v2_fused lingbot_v05_sensor_fused ai_consensus_fused spatially_shuffled_raw \
  >"$OUTPUT/logs/gpu7.log" 2>&1 &
PID7=$!

status=0
wait "$PID6" || status=$?
wait "$PID7" || status=$?
if [[ "$status" -ne 0 ]]; then
  echo "A worker failed; inspect $OUTPUT/logs" >&2
  exit "$status"
fi

"$PYTHON" "$SCRIPT" "${COMMON[@]}" --summarize-only
