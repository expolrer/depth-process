#!/usr/bin/env python3
"""Merge LingBot attention shard reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.reports_dir.glob("lingbot_attention_shard_*.json"))
    if not paths:
        raise SystemExit("No LingBot attention shard reports found")
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    first = shards[0]
    report = {
        "method": "lingbot_attention",
        "checkpoint": first["checkpoint"],
        "definition": first["definition"],
        "num_shards": len(shards),
        "batch_size": first["batch_size"],
        "resolution_level": first["resolution_level"],
        "layers": first["layers"],
        "query_grid": first["query_grid"],
        "selected_frames": sum(item["selected_frames"] for item in shards),
        "processed_frames": sum(item["processed_frames"] for item in shards),
        "resumed_frames": sum(item["resumed_frames"] for item in shards),
        "wall_time_upper_bound_seconds": max(item["elapsed_seconds"] for item in shards),
        "aggregate_gpu_seconds": sum(item["elapsed_seconds"] for item in shards),
        "shards": shards,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("selected_frames", "processed_frames", "resumed_frames", "wall_time_upper_bound_seconds")}, indent=2))


if __name__ == "__main__":
    main()
