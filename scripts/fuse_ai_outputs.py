#!/usr/bin/env python3
"""Create sensor-preserving LingBot and two-model consensus depth products."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np


def load_depth(path: Path, min_depth_m: float, max_depth_m: float) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read {path}")
    depth = image.astype(np.float32) * 0.001
    depth[(depth < min_depth_m) | (depth > max_depth_m)] = 0.0
    return depth


def save_depth(path: Path, depth_m: np.ndarray) -> None:
    encoded = np.rint(depth_m * 1000.0).clip(0, 65535).astype(np.uint16)
    if not cv2.imwrite(str(path), encoded):
        raise RuntimeError(f"Failed to write {path}")


def process_sequence(
    manifest_path: Path,
    input_root: Path,
    processed_root: Path,
    min_depth_m: float,
    max_depth_m: float,
    consensus_relative_tolerance: float,
    consensus_absolute_tolerance_m: float,
) -> dict[str, object]:
    sequence = manifest_path.parent.relative_to(input_root)
    lingbot_dir = processed_root / sequence / "lingbot_v05"
    depth_anything_dir = processed_root / sequence / "depth_anything_v2_fused"
    lingbot_fused_dir = processed_root / sequence / "lingbot_v05_sensor_fused"
    consensus_dir = processed_root / sequence / "ai_consensus_fused"
    lingbot_fused_dir.mkdir(parents=True, exist_ok=True)
    consensus_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    pixels = 0
    raw_valid = 0
    lingbot_filled = 0
    consensus_filled = 0
    disagreements = 0
    started = time.perf_counter()

    for row in rows:
        frame_index = int(row["frame_index"])
        filename = f"{frame_index:06d}.png"
        sensor_path = Path(row.get("depth_aligned_rgb_mm_path", row["depth_raw_mm_path"]))
        raw = load_depth(sensor_path, min_depth_m, max_depth_m)
        lingbot = load_depth(lingbot_dir / filename, min_depth_m, max_depth_m)
        depth_anything = load_depth(depth_anything_dir / filename, min_depth_m, max_depth_m)
        measured = raw > 0
        missing = ~measured

        lingbot_fused = raw.copy()
        lingbot_fill = missing & (lingbot > 0)
        lingbot_fused[lingbot_fill] = lingbot[lingbot_fill]

        both = missing & (lingbot > 0) & (depth_anything > 0)
        tolerance = np.maximum(
            consensus_absolute_tolerance_m,
            consensus_relative_tolerance * np.minimum(lingbot, depth_anything),
        )
        agree = both & (np.abs(lingbot - depth_anything) <= tolerance)
        consensus = raw.copy()
        consensus[agree] = 0.5 * (lingbot[agree] + depth_anything[agree])

        save_depth(lingbot_fused_dir / filename, lingbot_fused)
        save_depth(consensus_dir / filename, consensus)
        pixels += raw.size
        raw_valid += int(np.count_nonzero(measured))
        lingbot_filled += int(np.count_nonzero(lingbot_fill))
        consensus_filled += int(np.count_nonzero(agree))
        disagreements += int(np.count_nonzero(both & ~agree))

    result = {
        "sequence": str(sequence),
        "frames": len(rows),
        "raw_valid_fraction": raw_valid / pixels,
        "lingbot_sensor_fused_valid_fraction": (raw_valid + lingbot_filled) / pixels,
        "consensus_valid_fraction": (raw_valid + consensus_filled) / pixels,
        "lingbot_filled_fraction": lingbot_filled / pixels,
        "consensus_filled_fraction": consensus_filled / pixels,
        "ai_disagreement_fraction": disagreements / pixels,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (processed_root / sequence / "ai_fusion_stats.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min-depth-m", type=float, default=0.08)
    parser.add_argument("--max-depth-m", type=float, default=10.0)
    parser.add_argument("--consensus-relative-tolerance", type=float, default=0.20)
    parser.add_argument("--consensus-absolute-tolerance-m", type=float, default=0.15)
    args = parser.parse_args()

    manifests = sorted(args.input_root.glob("*/cam_*/manifest.jsonl"))
    cv2.setNumThreads(1)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_sequence,
                manifest,
                args.input_root,
                args.processed_root,
                args.min_depth_m,
                args.max_depth_m,
                args.consensus_relative_tolerance,
                args.consensus_absolute_tolerance_m,
            ): manifest
            for manifest in manifests
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"Completed {result['sequence']}: "
                f"LingBot fused={result['lingbot_sensor_fused_valid_fraction']:.3f}, "
                f"consensus={result['consensus_valid_fraction']:.3f}",
                flush=True,
            )

    results.sort(key=lambda item: str(item["sequence"]))
    summary = {
        "methods": ["lingbot_v05_sensor_fused", "ai_consensus_fused"],
        "consensus_relative_tolerance": args.consensus_relative_tolerance,
        "consensus_absolute_tolerance_m": args.consensus_absolute_tolerance_m,
        "sequences": results,
        "total_frames": sum(int(result["frames"]) for result in results),
    }
    path = args.processed_root / "ai_fusion_summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Summary: {path}")


if __name__ == "__main__":
    main()
