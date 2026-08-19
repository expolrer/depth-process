#!/usr/bin/env python3
"""Register native depth images into each camera's raw RGB image plane."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depth_pipeline.geometry import DepthRegistrar, find_frame_transform


def register_sequence(manifest_path: Path, input_root: Path, workers: int) -> dict[str, object]:
    camera_root = manifest_path.parent
    metadata_path = camera_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    source_frame = str(metadata["depth_camera_info"]["frame_id"])
    target_frame = str(metadata["rgb_camera_info"]["frame_id"])
    destination = camera_root / "depth_aligned_rgb_mm"
    destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    if source_frame == target_frame:
        for row in rows:
            row["depth_aligned_rgb_mm_path"] = row["depth_raw_mm_path"]
        mode = "already_aligned"
        aligned_valid_fractions: list[float] = []
    else:
        color_from_depth = find_frame_transform(
            metadata["camera_transforms"], source_frame, target_frame
        )
        registrar = DepthRegistrar(
            metadata["depth_camera_info"], metadata["rgb_camera_info"], color_from_depth
        )
        mode = "tf_intrinsics_projection"

        def process(row: dict[str, object]) -> tuple[int, str, float]:
            frame_index = int(row["frame_index"])
            source_path = Path(str(row["depth_raw_mm_path"]))
            target_path = destination / f"{frame_index:06d}.png"
            depth = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise RuntimeError(f"Failed to read {source_path}")
            aligned = registrar.register(depth)
            if not cv2.imwrite(str(target_path), aligned):
                raise RuntimeError(f"Failed to write {target_path}")
            return frame_index, str(target_path), float(np.mean(aligned > 0))

        results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process, row) for row in rows]
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda item: item[0])
        paths_by_index = {index: path for index, path, _ in results}
        aligned_valid_fractions = [fraction for _, _, fraction in results]
        for row in rows:
            row["depth_aligned_rgb_mm_path"] = paths_by_index[int(row["frame_index"])]

    for row in rows:
        row["depth_sensor_native_mm_path"] = row["depth_raw_mm_path"]
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8"
    )
    report = {
        "sequence": str(camera_root.relative_to(input_root)),
        "frames": len(rows),
        "mode": mode,
        "source_frame": source_frame,
        "target_frame": target_frame,
        "aligned_valid_fraction_mean": (
            float(np.mean(aligned_valid_fractions)) if aligned_valid_fractions else None
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (camera_root / "alignment_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--sequence-workers", type=int, default=6)
    parser.add_argument("--frame-workers", type=int, default=3)
    args = parser.parse_args()
    manifests = sorted(args.input_root.glob("*/cam_*/manifest.jsonl"))
    cv2.setNumThreads(1)
    results = []
    with ThreadPoolExecutor(max_workers=args.sequence_workers) as executor:
        futures = {
            executor.submit(register_sequence, manifest, args.input_root, args.frame_workers): manifest
            for manifest in manifests
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"Completed {result['sequence']}: {result['mode']} "
                f"({result['frames']} frames, {result['elapsed_seconds']:.1f}s)",
                flush=True,
            )
    results.sort(key=lambda item: str(item["sequence"]))
    path = args.input_root / "alignment_summary.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Summary: {path}")


if __name__ == "__main__":
    main()
