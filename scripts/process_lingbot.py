#!/usr/bin/env python3
"""Multi-GPU-shardable LingBot-Depth v0.5 batch inference."""

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


@dataclass(frozen=True)
class FrameTask:
    global_index: int
    sequence: str
    frame_index: int
    rgb_path: Path
    depth_path: Path
    output_path: Path


class Stats:
    def __init__(self) -> None:
        self.frames = 0
        self.pixels = 0
        self.valid_pixels = 0
        self.value_sum = 0.0
        self.value_squared_sum = 0.0
        self.minimum_m = float("inf")
        self.maximum_m = 0.0

    def update(self, depth_m: np.ndarray) -> None:
        values = depth_m[np.isfinite(depth_m) & (depth_m > 0)].astype(np.float64)
        self.frames += 1
        self.pixels += depth_m.size
        self.valid_pixels += values.size
        if values.size:
            self.value_sum += float(values.sum())
            self.value_squared_sum += float(np.square(values).sum())
            self.minimum_m = min(self.minimum_m, float(values.min()))
            self.maximum_m = max(self.maximum_m, float(values.max()))

    def as_dict(self) -> dict[str, float | int | None]:
        mean = self.value_sum / self.valid_pixels if self.valid_pixels else None
        variance = (
            max(0.0, self.value_squared_sum / self.valid_pixels - mean * mean)
            if mean is not None
            else None
        )
        return {
            "frames": self.frames,
            "pixels": self.pixels,
            "valid_pixels": self.valid_pixels,
            "valid_fraction": self.valid_pixels / self.pixels if self.pixels else 0.0,
            "value_sum": self.value_sum,
            "value_squared_sum": self.value_squared_sum,
            "mean_m": mean,
            "std_m": variance**0.5 if variance is not None else None,
            "min_m": self.minimum_m if self.valid_pixels else None,
            "max_m": self.maximum_m if self.valid_pixels else None,
        }


def discover_tasks(input_root: Path, output_root: Path) -> list[FrameTask]:
    tasks = []
    global_index = 0
    for manifest_path in sorted(input_root.glob("*/cam_*/manifest.jsonl")):
        relative = manifest_path.parent.relative_to(input_root)
        destination = output_root / relative / "lingbot_v05"
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            frame_index = int(row["frame_index"])
            tasks.append(
                FrameTask(
                    global_index=global_index,
                    sequence=str(relative),
                    frame_index=frame_index,
                    rgb_path=Path(row["rgb_path"]),
                    depth_path=Path(row.get("depth_aligned_rgb_mm_path", row["depth_raw_mm_path"])),
                    output_path=destination / f"{frame_index:06d}.png",
                )
            )
            global_index += 1
    return tasks


def batches(items: list[FrameTask], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def load_batch(tasks: list[FrameTask], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images = []
    depths = []
    for task in tasks:
        rgb_bgr = cv2.imread(str(task.rgb_path), cv2.IMREAD_COLOR)
        depth_mm = cv2.imread(str(task.depth_path), cv2.IMREAD_UNCHANGED)
        if rgb_bgr is None or depth_mm is None:
            raise RuntimeError(f"Failed to decode {task.rgb_path} or {task.depth_path}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        images.append(torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0))
        depths.append(torch.from_numpy(depth_mm.astype(np.float32) * 0.001))
    return torch.stack(images).to(device, non_blocking=True), torch.stack(depths).to(
        device, non_blocking=True
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--resolution-level", type=int, default=6)
    parser.add_argument("--min-depth-m", type=float, default=0.08)
    parser.add_argument("--max-depth-m", type=float, default=10.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("shard-index must be in [0, num-shards)")
    sys.path.insert(0, str(args.vendor))
    sys.path.insert(0, str(args.repo))
    from mdm.model.v2 import MDMModel

    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0")
    tasks = [
        task
        for task in discover_tasks(args.input_root, args.output_root)
        if task.global_index % args.num_shards == args.shard_index
    ]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        raise SystemExit("No frames selected")
    for task in tasks:
        task.output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint on shard {args.shard_index}: {args.checkpoint}", flush=True)
    model = MDMModel.from_pretrained(args.checkpoint).to(device).eval()
    stats = Stats()
    processed = 0
    resumed = 0
    started = time.perf_counter()

    pending = []
    for task in tasks:
        if task.output_path.exists() and not args.overwrite:
            existing = cv2.imread(str(task.output_path), cv2.IMREAD_UNCHANGED)
            if existing is not None:
                stats.update(existing.astype(np.float32) * 0.001)
                resumed += 1
                continue
        pending.append(task)

    for batch_index, task_batch in enumerate(batches(pending, args.batch_size), start=1):
        images, depths = load_batch(task_batch, device)
        output = model.infer(
            images,
            depth_in=depths,
            resolution_level=args.resolution_level,
            apply_mask=False,
            use_fp16=True,
        )["depth"]
        predictions = output.float().cpu().numpy()
        for task, prediction in zip(task_batch, predictions):
            prediction = np.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)
            prediction[
                (prediction < args.min_depth_m) | (prediction > args.max_depth_m)
            ] = 0.0
            encoded = np.rint(prediction * 1000.0).clip(0, 65535).astype(np.uint16)
            if not cv2.imwrite(str(task.output_path), encoded):
                raise RuntimeError(f"Failed to write {task.output_path}")
            stats.update(prediction)
            processed += 1
        if batch_index % 25 == 0 or processed == len(pending):
            elapsed = time.perf_counter() - started
            rate = processed / elapsed if elapsed else 0.0
            print(
                f"shard={args.shard_index} processed={processed}/{len(pending)} "
                f"resumed={resumed} rate={rate:.2f} frame/s",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    report = {
        "method": "lingbot_v05",
        "checkpoint": str(args.checkpoint),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "batch_size": args.batch_size,
        "resolution_level": args.resolution_level,
        "selected_frames": len(tasks),
        "processed_frames": processed,
        "resumed_frames": resumed,
        "elapsed_seconds": elapsed,
        "stats": stats.as_dict(),
    }
    reports_dir = args.output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"lingbot_v05_shard_{args.shard_index:02d}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
