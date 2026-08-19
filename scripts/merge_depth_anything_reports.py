#!/usr/bin/env python3
"""Merge Depth-Anything-V2 shard reports into compact quality statistics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "p99": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def summarize_frames(frames: list[dict[str, object]]) -> dict[str, object]:
    successful = [frame for frame in frames if bool(frame.get("calibration_ok"))]
    return {
        "frames": len(frames),
        "fit_success_frames": len(successful),
        "fit_success_fraction": len(successful) / len(frames) if frames else 0.0,
        "metric_rmse_m": distribution(
            [float(frame["metric_depth_rmse_m"]) for frame in successful]
        ),
        "inverse_depth_rmse": distribution(
            [float(frame["inverse_depth_rmse"]) for frame in successful]
        ),
        "raw_valid_fraction": distribution(
            [float(frame["raw_valid_fraction"]) for frame in frames]
        ),
        "output_valid_fraction": distribution(
            [float(frame["output_valid_fraction"]) for frame in frames]
        ),
        "filled_fraction": distribution([float(frame["filled_fraction"]) for frame in frames]),
        "failed_examples": [
            {
                "sequence": frame.get("sequence"),
                "frame_index": frame.get("frame_index"),
                "reason": frame.get("error")
                or (
                    "nonpositive_affine_scale"
                    if float(frame.get("scale", 0.0)) <= 0
                    else "metric_rmse_above_quality_threshold"
                ),
            }
            for frame in frames
            if not bool(frame.get("calibration_ok"))
        ][:100],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.reports_dir.glob("depth_anything_v2_shard_*.json"))
    if not paths:
        raise SystemExit(f"No Depth-Anything shard reports in {args.reports_dir}")
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    all_frames = [frame for shard in shards for frame in shard["frames"]]
    by_sequence: dict[str, list[dict[str, object]]] = defaultdict(list)
    for frame in all_frames:
        by_sequence[str(frame["sequence"])].append(frame)

    summary = {
        "method": "depth_anything_v2_fused",
        "checkpoint": shards[0]["checkpoint"],
        "num_shards": len(shards),
        "selected_frames": sum(int(shard["selected_frames"]) for shard in shards),
        "processed_frames": sum(int(shard["processed_frames"]) for shard in shards),
        "resumed_frames": sum(int(shard["resumed_frames"]) for shard in shards),
        "wall_time_upper_bound_seconds": max(float(shard["elapsed_seconds"]) for shard in shards),
        "overall": summarize_frames(all_frames),
        "sequences": {
            sequence: summarize_frames(frames) for sequence, frames in sorted(by_sequence.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2))
    print(f"Summary: {args.output}")


if __name__ == "__main__":
    main()
