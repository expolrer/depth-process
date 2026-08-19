#!/usr/bin/env python3
"""Extract LingBot RGB-D ViT attention maps for every aligned RGB-D frame."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F


ATTENTION_METHODS = (
    "lingbot_cross_attention",
    "lingbot_depth_token_attention",
)


@dataclass(frozen=True)
class FrameTask:
    global_index: int
    sequence: str
    frame_index: int
    rgb_path: Path
    depth_path: Path
    cross_overlay_path: Path
    cross_raw_path: Path
    depth_overlay_path: Path
    depth_raw_path: Path


class AttentionStats:
    def __init__(self) -> None:
        self.frames = 0
        self.token_values = 0
        self.cross_sum = 0.0
        self.cross_squared_sum = 0.0
        self.cross_mass_sum = 0.0
        self.depth_sum = 0.0
        self.depth_squared_sum = 0.0
        self.depth_mass_sum = 0.0

    def update(self, cross: np.ndarray, depth: np.ndarray) -> None:
        cross64 = cross.astype(np.float64)
        depth64 = depth.astype(np.float64)
        self.frames += 1
        self.token_values += cross.size
        self.cross_sum += float(cross64.sum())
        self.cross_squared_sum += float(np.square(cross64).sum())
        self.cross_mass_sum += float(cross64.sum())
        self.depth_sum += float(depth64.sum())
        self.depth_squared_sum += float(np.square(depth64).sum())
        self.depth_mass_sum += float(depth64.sum())

    @staticmethod
    def _summary(value_sum: float, squared_sum: float, count: int) -> dict[str, float]:
        mean = value_sum / count if count else 0.0
        variance = max(0.0, squared_sum / count - mean * mean) if count else 0.0
        return {"mean_probability": mean, "std_probability": variance**0.5}

    def as_dict(self) -> dict[str, object]:
        return {
            "frames": self.frames,
            "token_values": self.token_values,
            "cross_attention": {
                **self._summary(self.cross_sum, self.cross_squared_sum, self.token_values),
                "mean_rgb_attention_mass": self.cross_mass_sum / self.frames if self.frames else 0.0,
            },
            "depth_token_attention": {
                **self._summary(self.depth_sum, self.depth_squared_sum, self.token_values),
                "mean_depth_attention_mass": self.depth_mass_sum / self.frames if self.frames else 0.0,
            },
        }


def discover_tasks(input_root: Path, output_root: Path) -> list[FrameTask]:
    tasks: list[FrameTask] = []
    global_index = 0
    for manifest_path in sorted(input_root.glob("*/cam_*/manifest.jsonl")):
        relative = manifest_path.parent.relative_to(input_root)
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            frame_index = int(row["frame_index"])
            tasks.append(
                FrameTask(
                    global_index=global_index,
                    sequence=relative.as_posix(),
                    frame_index=frame_index,
                    rgb_path=Path(row["rgb_path"]),
                    depth_path=Path(
                        row.get("depth_aligned_rgb_mm_path", row["depth_raw_mm_path"])
                    ),
                    cross_overlay_path=(
                        output_root
                        / relative
                        / "lingbot_cross_attention"
                        / f"{frame_index:06d}.jpg"
                    ),
                    cross_raw_path=(
                        output_root
                        / relative
                        / "lingbot_cross_attention_raw"
                        / f"{frame_index:06d}.png"
                    ),
                    depth_overlay_path=(
                        output_root
                        / relative
                        / "lingbot_depth_token_attention"
                        / f"{frame_index:06d}.jpg"
                    ),
                    depth_raw_path=(
                        output_root
                        / relative
                        / "lingbot_depth_token_attention_raw"
                        / f"{frame_index:06d}.png"
                    ),
                )
            )
            global_index += 1
    return tasks


def batched(items: list[FrameTask], batch_size: int) -> Iterable[list[FrameTask]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def flatten_transformer_blocks(backbone: torch.nn.Module) -> list[torch.nn.Module]:
    blocks: list[torch.nn.Module] = []
    for candidate in backbone.blocks:
        if hasattr(candidate, "attn") and hasattr(candidate, "norm1"):
            blocks.append(candidate)
            continue
        for child in candidate.children():
            if hasattr(child, "attn") and hasattr(child, "norm1"):
                blocks.append(child)
    if not blocks:
        raise RuntimeError("No Transformer attention blocks found")
    return blocks


class LingBotAttentionCollector:
    """Recompute selected exact attention rows from hooked Transformer block inputs."""

    def __init__(
        self,
        backbone: torch.nn.Module,
        layer_indices: list[int],
        query_grid: int,
    ) -> None:
        blocks = flatten_transformer_blocks(backbone)
        resolved = sorted({index % len(blocks) for index in layer_indices})
        if not resolved:
            raise ValueError("At least one attention layer is required")
        self.layer_indices = resolved
        self.layer_names = [f"{index}/{len(blocks) - 1}" for index in resolved]
        self.query_grid = query_grid
        self.num_register_tokens = int(getattr(backbone, "num_register_tokens", 0))
        self.token_rows = 0
        self.token_cols = 0
        self.query_indices: torch.Tensor | None = None
        self.query_valid: torch.Tensor | None = None
        self.cross_maps: list[torch.Tensor] = []
        self.depth_maps: list[torch.Tensor] = []
        self.handles = [
            blocks[index].register_forward_pre_hook(self._make_hook(index)) for index in resolved
        ]

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def configure(self, depths: torch.Tensor, token_rows: int, token_cols: int) -> None:
        self.token_rows = token_rows
        self.token_cols = token_cols
        rows = torch.linspace(
            0,
            token_rows - 1,
            steps=min(self.query_grid, token_rows),
            device=depths.device,
        ).round().long()
        cols = torch.linspace(
            0,
            token_cols - 1,
            steps=min(self.query_grid, token_cols),
            device=depths.device,
        ).round().long()
        grid_rows, grid_cols = torch.meshgrid(rows, cols, indexing="ij")
        self.query_indices = (grid_rows * token_cols + grid_cols).flatten()
        valid_grid = F.interpolate(
            (depths > 0.01).float().unsqueeze(1),
            size=(token_rows, token_cols),
            mode="area",
        ).flatten(1)
        query_valid = valid_grid[:, self.query_indices] > 0.05
        empty = query_valid.sum(dim=1) == 0
        if empty.any():
            query_valid[empty] = True
        self.query_valid = query_valid
        self.cross_maps.clear()
        self.depth_maps.clear()

    def _make_hook(self, layer_index: int):
        def hook(block: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            if self.query_indices is None or self.query_valid is None:
                raise RuntimeError("Attention collector was not configured")
            hidden = inputs[0]
            if not isinstance(hidden, torch.Tensor):
                raise TypeError("Attention extraction requires unmasked batched token tensors")
            batch_size, sequence_length, channels = hidden.shape
            token_count = self.token_rows * self.token_cols
            rgb_start = 1 + self.num_register_tokens
            rgb_end = rgb_start + token_count
            depth_start = rgb_end
            depth_end = depth_start + token_count
            if sequence_length != depth_end:
                raise ValueError(
                    f"Unexpected token sequence at layer {layer_index}: "
                    f"got {sequence_length}, expected {depth_end}"
                )

            attention = block.attn
            normalized = block.norm1(hidden)
            qkv = (
                attention.qkv(normalized)
                .reshape(
                    batch_size,
                    sequence_length,
                    3,
                    attention.num_heads,
                    channels // attention.num_heads,
                )
                .permute(2, 0, 3, 1, 4)
            )
            queries, keys, _values = qkv.unbind(0)
            sampled_queries = queries[:, :, depth_start + self.query_indices, :]
            logits = torch.matmul(sampled_queries, keys.transpose(-2, -1)) * attention.scale
            probabilities = logits.float().softmax(dim=-1)
            valid = self.query_valid[:, None, :, None].float()
            valid_count = valid.sum(dim=2).clamp_min(1.0) * attention.num_heads
            cross = (probabilities[..., rgb_start:rgb_end] * valid).sum(dim=(1, 2))
            cross = cross / valid_count[:, 0]

            cls_logits = torch.matmul(queries[:, :, :1, :], keys.transpose(-2, -1))
            cls_probabilities = (cls_logits * attention.scale).float().softmax(dim=-1)
            depth = cls_probabilities[..., depth_start:depth_end].mean(dim=(1, 2))
            self.cross_maps.append(cross.detach())
            self.depth_maps.append(depth.detach())

        return hook

    def result(self) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self.cross_maps) != len(self.layer_indices):
            raise RuntimeError(
                f"Captured {len(self.cross_maps)} layers, expected {len(self.layer_indices)}"
            )
        cross = torch.stack(self.cross_maps, dim=0).mean(dim=0)
        depth = torch.stack(self.depth_maps, dim=0).mean(dim=0)
        return (
            cross.unflatten(1, (self.token_rows, self.token_cols)),
            depth.unflatten(1, (self.token_rows, self.token_cols)),
        )


def load_batch(
    tasks: list[FrameTask], device: torch.device
) -> tuple[list[np.ndarray], list[np.ndarray], torch.Tensor, torch.Tensor]:
    rgb_bgr_items: list[np.ndarray] = []
    depth_mm_items: list[np.ndarray] = []
    image_tensors: list[torch.Tensor] = []
    depth_tensors: list[torch.Tensor] = []
    expected_shape: tuple[int, int] | None = None
    for task in tasks:
        rgb_bgr = cv2.imread(str(task.rgb_path), cv2.IMREAD_COLOR)
        depth_mm = cv2.imread(str(task.depth_path), cv2.IMREAD_UNCHANGED)
        if rgb_bgr is None or depth_mm is None:
            raise RuntimeError(f"Failed to decode {task.rgb_path} or {task.depth_path}")
        if depth_mm.ndim != 2 or rgb_bgr.shape[:2] != depth_mm.shape:
            raise ValueError(f"RGB/depth shape mismatch for {task.sequence}/{task.frame_index}")
        if expected_shape is None:
            expected_shape = depth_mm.shape
        elif depth_mm.shape != expected_shape:
            raise ValueError("All images in a batch must have the same dimensions")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        rgb_bgr_items.append(rgb_bgr)
        depth_mm_items.append(depth_mm)
        image_tensors.append(torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0))
        depth_tensors.append(torch.from_numpy(depth_mm.astype(np.float32) * 0.001))
    images = torch.stack(image_tensors).to(device, non_blocking=True)
    depths = torch.stack(depth_tensors).to(device, non_blocking=True)
    return rgb_bgr_items, depth_mm_items, images, depths


def robust_normalize(score: np.ndarray) -> np.ndarray:
    values = score[np.isfinite(score)]
    if not values.size:
        return np.zeros_like(score, dtype=np.float32)
    low, high = np.percentile(values, [5.0, 99.5])
    if high <= low + 1e-12:
        return np.zeros_like(score, dtype=np.float32)
    normalized = np.clip((score - low) / (high - low), 0.0, 1.0).astype(np.float32)
    return np.power(normalized, 0.72).astype(np.float32)


def render_attention_overlay(
    depth_mm: np.ndarray,
    attention: np.ndarray,
    alpha: float,
    depth_min_m: float,
    depth_max_m: float,
) -> np.ndarray:
    depth_m = depth_mm.astype(np.float32) * 0.001
    valid = np.isfinite(depth_m) & (depth_m > 0)
    span = max(depth_max_m - depth_min_m, 1e-6)
    depth_normalized = np.clip((depth_m - depth_min_m) / span, 0.0, 1.0)
    gray = np.round(218.0 - depth_normalized * 168.0).astype(np.uint8)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    base[~valid] = (12, 14, 15)
    heat = cv2.applyColorMap(np.round(attention * 255.0).astype(np.uint8), cv2.COLORMAP_INFERNO)
    strength = (alpha * (0.12 + 0.88 * attention))[..., None]
    overlay = base.astype(np.float32) * (1.0 - strength) + heat.astype(np.float32) * strength
    return np.clip(overlay, 0, 255).astype(np.uint8)


def output_is_complete(task: FrameTask) -> bool:
    paths = (
        task.cross_overlay_path,
        task.cross_raw_path,
        task.depth_overlay_path,
        task.depth_raw_path,
    )
    return all(path.is_file() and path.stat().st_size > 0 for path in paths)


def write_outputs(
    task: FrameTask,
    depth_mm: np.ndarray,
    cross_score: np.ndarray,
    depth_score: np.ndarray,
    args: argparse.Namespace,
) -> None:
    height, width = depth_mm.shape
    cross_full = cv2.resize(cross_score, (width, height), interpolation=cv2.INTER_CUBIC)
    depth_full = cv2.resize(depth_score, (width, height), interpolation=cv2.INTER_CUBIC)
    cross_normalized = robust_normalize(cross_full)
    depth_normalized = robust_normalize(depth_full)
    cross_raw = np.round(cross_normalized * 65535.0).astype(np.uint16)
    depth_raw = np.round(depth_normalized * 65535.0).astype(np.uint16)
    cross_overlay = render_attention_overlay(
        depth_mm,
        cross_normalized,
        args.overlay_alpha,
        args.depth_min_m,
        args.depth_max_m,
    )
    depth_overlay = render_attention_overlay(
        depth_mm,
        depth_normalized,
        args.overlay_alpha,
        args.depth_min_m,
        args.depth_max_m,
    )
    for path in (
        task.cross_overlay_path,
        task.cross_raw_path,
        task.depth_overlay_path,
        task.depth_raw_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(task.cross_overlay_path), cross_overlay, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
    ):
        raise RuntimeError(f"Failed to write {task.cross_overlay_path}")
    if not cv2.imwrite(str(task.cross_raw_path), cross_raw):
        raise RuntimeError(f"Failed to write {task.cross_raw_path}")
    if not cv2.imwrite(
        str(task.depth_overlay_path), depth_overlay, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
    ):
        raise RuntimeError(f"Failed to write {task.depth_overlay_path}")
    if not cv2.imwrite(str(task.depth_raw_path), depth_raw):
        raise RuntimeError(f"Failed to write {task.depth_raw_path}")


def parse_layers(value: str) -> list[int]:
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("layers must be comma-separated integers") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--resolution-level", type=int, default=6)
    parser.add_argument("--layers", type=parse_layers, default=[-4, -3, -2, -1])
    parser.add_argument("--query-grid", type=int, default=8)
    parser.add_argument("--overlay-alpha", type=float, default=0.78)
    parser.add_argument("--depth-min-m", type=float, default=0.2)
    parser.add_argument("--depth-max-m", type=float, default=4.0)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("shard-index must be in [0, num-shards)")
    if args.query_grid < 1:
        raise SystemExit("query-grid must be positive")
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

    print(f"Loading LingBot checkpoint on shard {args.shard_index}: {args.checkpoint}", flush=True)
    model = MDMModel.from_pretrained(args.checkpoint).to(device).eval()
    collector = LingBotAttentionCollector(
        model.encoder.backbone,
        layer_indices=args.layers,
        query_grid=args.query_grid,
    )
    print(
        f"Attention layers={collector.layer_names} query_grid={args.query_grid}x{args.query_grid}",
        flush=True,
    )

    pending = tasks if args.overwrite else [task for task in tasks if not output_is_complete(task)]
    resumed = len(tasks) - len(pending)
    processed = 0
    stats = AttentionStats()
    started = time.perf_counter()
    num_tokens = int(
        model.num_tokens_range[0]
        + (args.resolution_level / 9) * (model.num_tokens_range[1] - model.num_tokens_range[0])
    )
    try:
        for batch_index, task_batch in enumerate(batched(pending, args.batch_size), start=1):
            _rgb_items, depth_mm_items, images, depths = load_batch(task_batch, device)
            aspect_ratio = images.shape[-1] / images.shape[-2]
            token_rows = round((num_tokens / aspect_ratio) ** 0.5)
            token_cols = round((num_tokens * aspect_ratio) ** 0.5)
            collector.configure(depths, token_rows, token_cols)
            with torch.inference_mode():
                features, cls_token = model.infer_feat(
                    images,
                    depth_in=depths,
                    resolution_level=args.resolution_level,
                    use_fp16=True,
                    enable_depth_mask=False,
                )
                del features, cls_token
                cross_maps, depth_maps = collector.result()
            cross_items = cross_maps.float().cpu().numpy()
            depth_items = depth_maps.float().cpu().numpy()
            for task, depth_mm, cross_score, depth_score in zip(
                task_batch,
                depth_mm_items,
                cross_items,
                depth_items,
            ):
                write_outputs(task, depth_mm, cross_score, depth_score, args)
                stats.update(cross_score, depth_score)
                processed += 1
            if batch_index % 10 == 0 or processed == len(pending):
                elapsed = time.perf_counter() - started
                rate = processed / elapsed if elapsed else 0.0
                print(
                    f"shard={args.shard_index} processed={processed}/{len(pending)} "
                    f"resumed={resumed} rate={rate:.2f} frame/s",
                    flush=True,
                )
    finally:
        collector.close()

    elapsed = time.perf_counter() - started
    report = {
        "method": "lingbot_attention",
        "checkpoint": str(args.checkpoint),
        "definition": {
            "cross_attention": "sampled valid depth queries to RGB keys",
            "depth_token_attention": "CLS query to depth keys",
            "aggregation": "mean over selected layers, heads, and sampled queries",
            "normalization": "per-frame 5th to 99.5th percentile, gamma 0.72",
        },
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "batch_size": args.batch_size,
        "resolution_level": args.resolution_level,
        "layers": collector.layer_names,
        "query_grid": args.query_grid,
        "selected_frames": len(tasks),
        "processed_frames": processed,
        "resumed_frames": resumed,
        "elapsed_seconds": elapsed,
        "stats": stats.as_dict(),
    }
    reports_dir = args.output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"lingbot_attention_shard_{args.shard_index:02d}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
