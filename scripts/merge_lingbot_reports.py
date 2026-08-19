#!/usr/bin/env python3
"""Merge per-GPU LingBot-Depth inference reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.reports_dir.glob("lingbot_v05_shard_*.json"))
    if not paths:
        raise SystemExit(f"No shard reports found in {args.reports_dir}")
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    stats_rows = [shard["stats"] for shard in shards]
    valid = sum(int(row["valid_pixels"]) for row in stats_rows)
    pixels = sum(int(row["pixels"]) for row in stats_rows)
    value_sum = sum(float(row["value_sum"]) for row in stats_rows)
    squared_sum = sum(float(row["value_squared_sum"]) for row in stats_rows)
    mean = value_sum / valid if valid else None
    variance = max(0.0, squared_sum / valid - mean * mean) if mean is not None else None
    summary = {
        "method": "lingbot_v05",
        "checkpoint": shards[0]["checkpoint"],
        "num_shards": len(shards),
        "selected_frames": sum(int(row["selected_frames"]) for row in shards),
        "processed_frames": sum(int(row["processed_frames"]) for row in shards),
        "resumed_frames": sum(int(row["resumed_frames"]) for row in shards),
        "wall_time_upper_bound_seconds": max(float(row["elapsed_seconds"]) for row in shards),
        "aggregate_gpu_seconds": sum(float(row["elapsed_seconds"]) for row in shards),
        "stats": {
            "frames": sum(int(row["frames"]) for row in stats_rows),
            "pixels": pixels,
            "valid_pixels": valid,
            "valid_fraction": valid / pixels if pixels else 0.0,
            "mean_m": mean,
            "std_m": math.sqrt(variance) if variance is not None else None,
            "min_m": min(float(row["min_m"]) for row in stats_rows if row["min_m"] is not None),
            "max_m": max(float(row["max_m"]) for row in stats_rows if row["max_m"] is not None),
        },
        "shards": shards,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Merged {len(shards)} reports: {args.output}")


if __name__ == "__main__":
    main()

