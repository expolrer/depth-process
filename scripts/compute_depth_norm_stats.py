#!/usr/bin/env python3
"""Compute metric-depth normalization statistics for all exported methods."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Accumulator:
    frames: int = 0
    pixels: int = 0
    valid_pixels: int = 0
    value_sum: float = 0.0
    squared_sum: float = 0.0
    minimum_m: float = float("inf")
    maximum_m: float = 0.0
    samples: list[np.ndarray] = field(default_factory=list)

    def update(self, depth_m: np.ndarray, sample_stride: int) -> None:
        valid = np.isfinite(depth_m) & (depth_m > 0)
        values = depth_m[valid].astype(np.float64)
        self.frames += 1
        self.pixels += depth_m.size
        self.valid_pixels += values.size
        if values.size:
            self.value_sum += float(values.sum())
            self.squared_sum += float(np.square(values).sum())
            self.minimum_m = min(self.minimum_m, float(values.min()))
            self.maximum_m = max(self.maximum_m, float(values.max()))
        sampled = depth_m[::sample_stride, ::sample_stride]
        sampled = sampled[np.isfinite(sampled) & (sampled > 0)]
        if sampled.size:
            self.samples.append(sampled.astype(np.float32))

    def merge(self, other: "Accumulator") -> None:
        self.frames += other.frames
        self.pixels += other.pixels
        self.valid_pixels += other.valid_pixels
        self.value_sum += other.value_sum
        self.squared_sum += other.squared_sum
        self.minimum_m = min(self.minimum_m, other.minimum_m)
        self.maximum_m = max(self.maximum_m, other.maximum_m)
        self.samples.extend(other.samples)

    def as_dict(self) -> dict[str, object]:
        mean = self.value_sum / self.valid_pixels if self.valid_pixels else None
        variance = (
            max(0.0, self.squared_sum / self.valid_pixels - mean * mean)
            if mean is not None
            else None
        )
        sample = np.concatenate(self.samples) if self.samples else np.array([], dtype=np.float32)
        quantiles = (
            {str(value): float(np.percentile(sample, value)) for value in (1, 5, 50, 95, 99)}
            if sample.size
            else {}
        )
        return {
            "frames": self.frames,
            "pixels": self.pixels,
            "valid_pixels": self.valid_pixels,
            "valid_fraction": self.valid_pixels / self.pixels if self.pixels else 0.0,
            "mean_m": mean,
            "std_m": math.sqrt(variance) if variance is not None else None,
            "min_m": self.minimum_m if self.valid_pixels else None,
            "max_m": self.maximum_m if self.valid_pixels else None,
            "sampled_quantiles_m": quantiles,
            "sample_count": int(sample.size),
        }


def load_depth(path: Path, min_depth_m: float, max_depth_m: float) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read {path}")
    depth = image.astype(np.float32) * 0.001
    depth[(depth < min_depth_m) | (depth > max_depth_m)] = 0.0
    return depth


def sequence_stats(
    manifest_path: Path,
    input_root: Path,
    processed_root: Path,
    methods: list[str],
    min_depth_m: float,
    max_depth_m: float,
    sample_stride: int,
) -> tuple[str, dict[str, Accumulator]]:
    sequence = manifest_path.parent.relative_to(input_root)
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    accumulators = {method: Accumulator() for method in methods}
    for row in rows:
        frame_index = int(row["frame_index"])
        paths = {
            "raw_native": Path(row.get("depth_sensor_native_mm_path", row["depth_raw_mm_path"])),
            "raw_rgb_aligned": Path(
                row.get("depth_aligned_rgb_mm_path", row["depth_raw_mm_path"])
            ),
        }
        for method in methods:
            if method not in paths:
                paths[method] = processed_root / sequence / method / f"{frame_index:06d}.png"
            accumulators[method].update(
                load_depth(paths[method], min_depth_m, max_depth_m), sample_stride
            )
    return str(sequence), accumulators


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--sample-stride", type=int, default=32)
    parser.add_argument("--min-depth-m", type=float, default=0.08)
    parser.add_argument("--max-depth-m", type=float, default=10.0)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "raw_native",
            "raw_rgb_aligned",
            "rgb_guided",
            "temporal_rgb_guided",
            "lingbot_v05",
            "depth_anything_v2_fused",
            "lingbot_v05_sensor_fused",
            "ai_consensus_fused",
        ],
    )
    args = parser.parse_args()

    manifests = sorted(args.input_root.glob("*/cam_*/manifest.jsonl"))
    cv2.setNumThreads(1)
    global_accumulators = {method: Accumulator() for method in args.methods}
    sequence_results: dict[str, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                sequence_stats,
                manifest,
                args.input_root,
                args.processed_root,
                args.methods,
                args.min_depth_m,
                args.max_depth_m,
                args.sample_stride,
            ): manifest
            for manifest in manifests
        }
        for future in as_completed(futures):
            sequence, accumulators = future.result()
            sequence_results[sequence] = {
                method: accumulator.as_dict() for method, accumulator in accumulators.items()
            }
            for method, accumulator in accumulators.items():
                global_accumulators[method].merge(accumulator)
            print(f"Completed norm stats: {sequence}", flush=True)

    report = {
        "unit": "meter",
        "source_encoding": "uint16_png_millimeter",
        "scale_factor_to_meter": 0.001,
        "invalid_value": 0,
        "normalization": "(depth_m - mean_m) / std_m over valid pixels only",
        "valid_range_m": [args.min_depth_m, args.max_depth_m],
        "methods": {
            method: accumulator.as_dict()
            for method, accumulator in global_accumulators.items()
        },
        "sequences": dict(sorted(sequence_results.items())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Stats: {args.output}")


if __name__ == "__main__":
    main()

