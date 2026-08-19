#!/usr/bin/env python3
"""Depth-Anything-V2 RGB prior with per-frame metric sensor calibration."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as functional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from depth_pipeline.fusion import calibrate_relative_inverse_depth, fuse_sensor_and_prior


@dataclass(frozen=True)
class Task:
    global_index: int
    sequence: str
    frame_index: int
    rgb_path: Path
    depth_path: Path
    output_path: Path


def discover(input_root: Path, output_root: Path) -> list[Task]:
    tasks: list[Task] = []
    global_index = 0
    for manifest in sorted(input_root.glob("*/cam_*/manifest.jsonl")):
        sequence = str(manifest.parent.relative_to(input_root))
        destination = output_root / sequence / "depth_anything_v2_fused"
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            frame_index = int(row["frame_index"])
            tasks.append(
                Task(
                    global_index,
                    sequence,
                    frame_index,
                    Path(row["rgb_path"]),
                    Path(row.get("depth_aligned_rgb_mm_path", row["depth_raw_mm_path"])),
                    destination / f"{frame_index:06d}.png",
                )
            )
            global_index += 1
    return tasks


def transform_image(model, rgb_bgr: np.ndarray, input_size: int) -> torch.Tensor:
    tensor, _ = model.image2tensor(rgb_bgr, input_size)
    return tensor[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--input-size", type=int, default=392)
    parser.add_argument("--min-depth-m", type=float, default=0.08)
    parser.add_argument("--max-depth-m", type=float, default=10.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo))
    from depth_anything_v2.dpt import DepthAnythingV2

    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("Invalid shard index")
    tasks = [
        task
        for task in discover(args.input_root, args.output_root)
        if task.global_index % args.num_shards == args.shard_index
    ]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    for task in tasks:
        task.output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0")
    config = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
    model = DepthAnythingV2(**config)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model = model.to(device).eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    cv2.setNumThreads(1)

    reports: list[dict[str, object]] = []
    pending = [task for task in tasks if args.overwrite or not task.output_path.exists()]
    resumed = len(tasks) - len(pending)
    started = time.perf_counter()
    processed = 0

    for batch_start in range(0, len(pending), args.batch_size):
        task_batch = pending[batch_start : batch_start + args.batch_size]
        rgb_images = [cv2.imread(str(task.rgb_path), cv2.IMREAD_COLOR) for task in task_batch]
        depth_images = [cv2.imread(str(task.depth_path), cv2.IMREAD_UNCHANGED) for task in task_batch]
        if any(image is None for image in rgb_images) or any(depth is None for depth in depth_images):
            raise RuntimeError("Failed to decode an input frame")

        image_tensors = [transform_image(model, image, args.input_size) for image in rgb_images]
        image_batch = torch.stack(image_tensors).to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            relative = model(image_batch)
            relative = functional.interpolate(
                relative[:, None],
                size=depth_images[0].shape,
                mode="bilinear",
                align_corners=True,
            )[:, 0]
        priors = relative.float().cpu().numpy()

        for task, depth_mm, prior in zip(task_batch, depth_images, priors):
            sensor = depth_mm.astype(np.float32) * 0.001
            sensor[(sensor < args.min_depth_m) | (sensor > args.max_depth_m)] = 0.0
            try:
                calibrated, fit = calibrate_relative_inverse_depth(
                    prior,
                    sensor,
                    min_depth_m=args.min_depth_m,
                    max_depth_m=args.max_depth_m,
                )
            except ValueError as exc:
                calibrated = np.zeros_like(sensor)
                fit = {"calibration_ok": False, "error": str(exc)}
            if not bool(fit.get("calibration_ok")):
                calibrated.fill(0.0)
            fused, fill_mask = fuse_sensor_and_prior(sensor, calibrated)
            encoded = np.rint(fused * 1000.0).clip(0, 65535).astype(np.uint16)
            if not cv2.imwrite(str(task.output_path), encoded):
                raise RuntimeError(f"Failed to write {task.output_path}")
            fit.update(
                sequence=task.sequence,
                frame_index=task.frame_index,
                raw_valid_fraction=float(np.mean(sensor > 0)),
                output_valid_fraction=float(np.mean(fused > 0)),
                filled_fraction=float(np.mean(fill_mask)),
            )
            reports.append(fit)
            processed += 1

        if processed % (args.batch_size * 25) == 0 or processed == len(pending):
            elapsed = time.perf_counter() - started
            print(
                f"shard={args.shard_index} processed={processed}/{len(pending)} "
                f"resumed={resumed} rate={processed / max(elapsed, 1e-6):.2f} frame/s",
                flush=True,
            )

    report_dir = args.output_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"depth_anything_v2_shard_{args.shard_index:02d}.json"
    summary = {
        "method": "depth_anything_v2_fused",
        "checkpoint": str(args.checkpoint),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "selected_frames": len(tasks),
        "processed_frames": processed,
        "resumed_frames": resumed,
        "elapsed_seconds": time.perf_counter() - started,
        "fit_success_frames": sum(bool(row.get("calibration_ok")) for row in reports),
        "frames": reports,
    }
    report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
