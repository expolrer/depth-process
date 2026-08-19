#!/usr/bin/env python3
"""Run RGB-guided spatial refinement and optical-flow temporal stabilization."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depth_pipeline.classical import (
    RunningDepthStats,
    encode_depth_mm,
    rgb_guided_refine,
    sanitize_depth,
    temporal_stabilize,
)


def read_manifest(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def process_sequence(
    manifest_path: Path,
    input_root: Path,
    output_root: Path,
    min_depth_m: float,
    max_depth_m: float,
) -> dict[str, object]:
    relative = manifest_path.parent.relative_to(input_root)
    destination = output_root / relative
    guided_dir = destination / "rgb_guided"
    temporal_dir = destination / "temporal_rgb_guided"
    guided_dir.mkdir(parents=True, exist_ok=True)
    temporal_dir.mkdir(parents=True, exist_ok=True)

    raw_stats = RunningDepthStats()
    guided_stats = RunningDepthStats()
    temporal_stats = RunningDepthStats()
    rows = read_manifest(manifest_path)
    previous_rgb = None
    previous_depth = None
    started = time.perf_counter()

    for row in rows:
        rgb = cv2.imread(str(row["rgb_path"]), cv2.IMREAD_COLOR)
        depth_path = row.get("depth_aligned_rgb_mm_path", row["depth_raw_mm_path"])
        depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth_mm is None:
            raise RuntimeError(f"Failed to decode frame {row['frame_index']} from {manifest_path}")
        depth = sanitize_depth(depth_mm, min_depth_m=min_depth_m, max_depth_m=max_depth_m)
        guided = rgb_guided_refine(rgb, depth)
        temporal = temporal_stabilize(previous_rgb, previous_depth, rgb, guided)

        name = f"{int(row['frame_index']):06d}.png"
        if not cv2.imwrite(str(guided_dir / name), encode_depth_mm(guided, max_depth_m)):
            raise RuntimeError(f"Failed to write {guided_dir / name}")
        if not cv2.imwrite(str(temporal_dir / name), encode_depth_mm(temporal, max_depth_m)):
            raise RuntimeError(f"Failed to write {temporal_dir / name}")

        raw_stats.update(depth)
        guided_stats.update(guided)
        temporal_stats.update(temporal)
        previous_rgb = rgb
        previous_depth = temporal

    result = {
        "sequence": str(relative),
        "frames": len(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {
            "rgb_guided": str(guided_dir),
            "temporal_rgb_guided": str(temporal_dir),
        },
        "stats": {
            "raw": raw_stats.as_dict(),
            "rgb_guided": guided_stats.as_dict(),
            "temporal_rgb_guided": temporal_stats.as_dict(),
        },
    }
    (destination / "classical_stats.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sequence-workers", type=int, default=8)
    parser.add_argument("--min-depth-m", type=float, default=0.08)
    parser.add_argument("--max-depth-m", type=float, default=10.0)
    args = parser.parse_args()

    cv2.setNumThreads(1)
    manifests = sorted(args.input_root.glob("*/cam_*/manifest.jsonl"))
    if not manifests:
        raise SystemExit(f"No manifests found below {args.input_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=args.sequence_workers) as executor:
        futures = {
            executor.submit(
                process_sequence,
                manifest,
                args.input_root,
                args.output_root,
                args.min_depth_m,
                args.max_depth_m,
            ): manifest
            for manifest in manifests
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"Completed {result['sequence']}: {result['frames']} frames "
                f"in {result['elapsed_seconds']:.1f}s",
                flush=True,
            )

    results.sort(key=lambda item: str(item["sequence"]))
    summary = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "methods": ["rgb_guided", "temporal_rgb_guided"],
        "sequences": results,
        "total_frames": sum(int(result["frames"]) for result in results),
    }
    summary_path = args.output_root / "classical_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
