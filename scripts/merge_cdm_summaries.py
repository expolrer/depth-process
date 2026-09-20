#!/usr/bin/env python3
"""Merge per-camera CDM reports produced by parallel workers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    frames = [frame for report in reports for frame in report.get("frames", [])]
    frames.sort(key=lambda row: (str(row["sequence"]), int(row["frame_index"])))
    merged = {
        "method": "camera_depth_models",
        "camera_config": reports[0]["camera_config"],
        "cameras": sorted({camera for report in reports for camera in report.get("cameras", [])}),
        "processed_frames": sum(int(report.get("processed_frames", 0)) for report in reports),
        "resumed_frames": sum(int(report.get("resumed_frames", 0)) for report in reports),
        "skipped_unconfigured_sequences": sorted(
            {sequence for report in reports for sequence in report.get("skipped_unconfigured_sequences", [])}
        ),
        "elapsed_seconds": max(float(report.get("elapsed_seconds", 0.0)) for report in reports),
        "outputs": sorted({name for report in reports for name in report.get("outputs", [])}),
        "frames": frames,
        "source_reports": [str(path) for path in args.inputs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"Merged {len(frames)} frames -> {args.output}")


if __name__ == "__main__":
    main()
