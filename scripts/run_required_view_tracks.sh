#!/usr/bin/env bash
set -euo pipefail

cd /ssd/hhw/depth-processing
export PYTHONPATH=/ssd/hhw/depth-processing/models/sam2
PY=/root/miniconda3/envs/qwen35vl/bin/python
mkdir -p outputs/target_tracks_required

CUDA_VISIBLE_DEVICES=2 "$PY" scripts/track_required_views_sam2.py --role head --device cuda:0 > outputs/target_tracks_required/head.log 2>&1 &
head_pid=$!
CUDA_VISIBLE_DEVICES=3 "$PY" scripts/track_required_views_sam2.py --role wrist --device cuda:0 > outputs/target_tracks_required/wrist.log 2>&1 &
wrist_pid=$!

wait "$head_pid"
wait "$wrist_pid"
.venv/bin/python scripts/merge_required_view_tracks.py
