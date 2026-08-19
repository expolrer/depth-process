#!/usr/bin/env python3
"""Export synchronized RGB, original 16-bit depth, intrinsics, and manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depth_pipeline.rosbag_extract import extract_camera, resolve_bags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path, help="ROS1 bag files or directories")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cameras", nargs="+", default=["cam_h", "cam_l", "cam_r"])
    parser.add_argument("--max-sync-delta-ms", type=float, default=80.0)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    bags = resolve_bags(args.inputs)
    if not bags:
        raise SystemExit("No .bag files found")
    args.output_root.mkdir(parents=True, exist_ok=True)

    report = []
    for bag_path in bags:
        for camera in args.cameras:
            report.append(
                extract_camera(
                    bag_path=bag_path,
                    output_root=args.output_root,
                    camera=camera,
                    max_sync_delta_ms=args.max_sync_delta_ms,
                    workers=args.workers,
                )
            )
    report_path = args.output_root / "extraction_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Exported {sum(row['paired_frames'] for row in report)} RGB-D pairs")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
