#!/usr/bin/env python3
"""Generate per-frame side-by-side depth comparisons and quantitative summaries."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class MetricAccumulator:
    frames: int = 0
    pixels: int = 0
    valid_pixels: int = 0
    filled_pixels: int = 0
    comparison_pixels: int = 0
    absolute_error_sum: float = 0.0
    squared_error_sum: float = 0.0

    def update(self, raw: np.ndarray, output: np.ndarray) -> None:
        raw_valid = raw > 0
        output_valid = output > 0
        comparison = raw_valid & output_valid
        difference = output[comparison].astype(np.float64) - raw[comparison].astype(np.float64)
        self.frames += 1
        self.pixels += raw.size
        self.valid_pixels += int(np.count_nonzero(output_valid))
        self.filled_pixels += int(np.count_nonzero(~raw_valid & output_valid))
        self.comparison_pixels += difference.size
        self.absolute_error_sum += float(np.abs(difference).sum())
        self.squared_error_sum += float(np.square(difference).sum())

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "frames": self.frames,
            "pixels": self.pixels,
            "valid_pixels": self.valid_pixels,
            "filled_pixels": self.filled_pixels,
            "valid_fraction": self.valid_pixels / self.pixels if self.pixels else 0.0,
            "filled_fraction": self.filled_pixels / self.pixels if self.pixels else 0.0,
            "sensor_overlap_pixels": self.comparison_pixels,
            "sensor_absolute_error_sum_m": self.absolute_error_sum,
            "sensor_squared_error_sum_m2": self.squared_error_sum,
            "sensor_mae_m": (
                self.absolute_error_sum / self.comparison_pixels if self.comparison_pixels else None
            ),
            "sensor_rmse_m": (
                (self.squared_error_sum / self.comparison_pixels) ** 0.5
                if self.comparison_pixels
                else None
            ),
        }


def read_depth(path: Path, min_depth_m: float, max_depth_m: float) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 2:
        raise RuntimeError(f"Failed to read depth image: {path}")
    depth = image.astype(np.float32) * 0.001
    depth[(depth < min_depth_m) | (depth > max_depth_m)] = 0.0
    return depth


def colorize_depth(depth_m: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    valid = depth_m > 0
    normalized = np.clip((depth_m - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    image = cv2.applyColorMap(np.rint(normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    image[~valid] = 0
    return image


def labeled_panel(image: np.ndarray, label: str, width: int, height: int) -> np.ndarray:
    panel = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height + 38, width, 3), dtype=np.uint8)
    canvas[:height] = panel
    cv2.putText(
        canvas,
        label,
        (10, height + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def process_sequence(
    manifest_path: Path,
    input_root: Path,
    processed_root: Path,
    comparison_root: Path,
    methods: list[tuple[str, str]],
    panel_width: int,
    panel_height: int,
    vmin: float,
    vmax: float,
    min_depth_m: float,
    max_depth_m: float,
    jpeg_quality: int,
) -> dict[str, object]:
    sequence = manifest_path.parent.relative_to(input_root)
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
    frames_dir = comparison_root / sequence / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    metrics = {method: MetricAccumulator() for method, _ in methods}
    overview_images: dict[int, np.ndarray] = {}
    sample_indices = set(np.linspace(0, max(0, len(rows) - 1), min(6, len(rows)), dtype=int))
    started = time.perf_counter()

    for position, row in enumerate(rows):
        rgb = cv2.imread(str(row["rgb_path"]), cv2.IMREAD_COLOR)
        if rgb is None:
            raise RuntimeError(f"Failed to read RGB image: {row['rgb_path']}")
        native_path = Path(row.get("depth_sensor_native_mm_path", row["depth_raw_mm_path"]))
        sensor_path = Path(row.get("depth_aligned_rgb_mm_path", row["depth_raw_mm_path"]))
        native = read_depth(native_path, min_depth_m, max_depth_m)
        raw = read_depth(sensor_path, min_depth_m, max_depth_m)
        panels = [labeled_panel(rgb, "RGB", panel_width, panel_height)]
        native_label = f"Raw native ({np.mean(native > 0) * 100:.1f}%)"
        panels.append(
            labeled_panel(
                colorize_depth(native, vmin, vmax), native_label, panel_width, panel_height
            )
        )
        raw_label = f"RGB-aligned sensor ({np.mean(raw > 0) * 100:.1f}%)"
        panels.append(
            labeled_panel(colorize_depth(raw, vmin, vmax), raw_label, panel_width, panel_height)
        )

        frame_index = int(row["frame_index"])
        for method, label in methods:
            method_path = processed_root / sequence / method / f"{frame_index:06d}.png"
            output = read_depth(method_path, min_depth_m, max_depth_m)
            metrics[method].update(raw, output)
            method_label = f"{label} ({np.mean(output > 0) * 100:.1f}%)"
            panels.append(
                labeled_panel(
                    colorize_depth(output, vmin, vmax),
                    method_label,
                    panel_width,
                    panel_height,
                )
            )

        comparison = cv2.hconcat(panels)
        comparison_path = frames_dir / f"{frame_index:06d}.jpg"
        if not cv2.imwrite(
            str(comparison_path), comparison, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        ):
            raise RuntimeError(f"Failed to write {comparison_path}")
        if position in sample_indices:
            overview_images[position] = comparison

    overview = cv2.vconcat([overview_images[index] for index in sorted(overview_images)])
    overview_path = comparison_root / sequence / "overview.jpg"
    cv2.imwrite(str(overview_path), overview, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    result = {
        "sequence": str(sequence),
        "frames": len(rows),
        "comparison_frames": str(frames_dir),
        "overview": str(overview_path),
        "color_scale_m": [vmin, vmax],
        "metrics": {method: accumulator.as_dict() for method, accumulator in metrics.items()},
        "elapsed_seconds": time.perf_counter() - started,
    }
    (comparison_root / sequence / "comparison_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def parse_method(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Methods must be DIRECTORY=LABEL")
    directory, label = value.split("=", 1)
    return directory, label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--method",
        action="append",
        type=parse_method,
        dest="methods",
        help="Processed directory and display label, e.g. lingbot_v05=LingBot v0.5",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--panel-width", type=int, default=424)
    parser.add_argument("--panel-height", type=int, default=240)
    parser.add_argument("--vmin", type=float, default=0.2)
    parser.add_argument("--vmax", type=float, default=4.0)
    parser.add_argument("--min-depth-m", type=float, default=0.08)
    parser.add_argument("--max-depth-m", type=float, default=10.0)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    args = parser.parse_args()

    methods = args.methods or [
        ("rgb_guided", "RGB guided"),
        ("temporal_rgb_guided", "Temporal"),
        ("lingbot_v05", "LingBot v0.5"),
        ("depth_anything_v2_fused", "Depth Anything V2"),
    ]
    manifests = sorted(args.input_root.glob("*/cam_*/manifest.jsonl"))
    if not manifests:
        raise SystemExit("No extraction manifests found")
    args.output_root.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_sequence,
                manifest,
                args.input_root,
                args.processed_root,
                args.output_root,
                methods,
                args.panel_width,
                args.panel_height,
                args.vmin,
                args.vmax,
                args.min_depth_m,
                args.max_depth_m,
                args.jpeg_quality,
            ): manifest
            for manifest in manifests
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"Completed {result['sequence']}: {result['frames']} comparisons "
                f"in {result['elapsed_seconds']:.1f}s",
                flush=True,
            )

    results.sort(key=lambda item: str(item["sequence"]))
    global_metrics = {}
    for method, _ in methods:
        rows = [result["metrics"][method] for result in results]
        pixels = sum(int(row["pixels"]) for row in rows)
        valid_pixels = sum(int(row["valid_pixels"]) for row in rows)
        filled_pixels = sum(int(row["filled_pixels"]) for row in rows)
        overlap = sum(int(row["sensor_overlap_pixels"]) for row in rows)
        absolute_error = sum(float(row["sensor_absolute_error_sum_m"]) for row in rows)
        squared_error = sum(float(row["sensor_squared_error_sum_m2"]) for row in rows)
        global_metrics[method] = {
            "frames": sum(int(row["frames"]) for row in rows),
            "valid_fraction": valid_pixels / pixels if pixels else 0.0,
            "filled_fraction": filled_pixels / pixels if pixels else 0.0,
            "sensor_overlap_pixels": overlap,
            "sensor_mae_m": absolute_error / overlap if overlap else None,
            "sensor_rmse_m": (squared_error / overlap) ** 0.5 if overlap else None,
        }
    summary = {
        "input_root": str(args.input_root),
        "processed_root": str(args.processed_root),
        "methods": [{"directory": method, "label": label} for method, label in methods],
        "total_frames": sum(int(result["frames"]) for result in results),
        "global_metrics": global_metrics,
        "sequences": results,
    }
    summary_path = args.output_root / "comparison_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
