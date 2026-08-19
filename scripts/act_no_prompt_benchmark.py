#!/usr/bin/env python3
"""Train one prompt-free ACT policy and compare all depth-processing methods."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy


CAMERAS = ("cam_h", "cam_l", "cam_r")
RGB_KEYS = tuple(f"observation.images.rgb_{camera[-1]}" for camera in CAMERAS)
DEPTH_KEYS = tuple(f"observation.images.depth_{camera[-1]}" for camera in CAMERAS)
IMAGE_KEYS = RGB_KEYS + DEPTH_KEYS
NORMAL_METHODS = (
    "raw_aligned",
    "rgb_guided",
    "temporal_rgb_guided",
    "lingbot_v05",
    "depth_anything_v2_fused",
    "lingbot_v05_sensor_fused",
    "ai_consensus_fused",
)
CONTROLS = ("zero_depth", "spatially_shuffled_raw")


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def compute_stats(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    train_rows = [row for row in rows if not row["is_eval"]]
    state = np.asarray([row["state"] for row in train_rows], dtype=np.float32)
    action = np.asarray([row["action"] for row in train_rows], dtype=np.float32)
    return {
        "state_mean": state.mean(axis=0).tolist(),
        "state_std": np.maximum(state.std(axis=0), 1e-4).tolist(),
        "action_mean": action.mean(axis=0).tolist(),
        "action_std": np.maximum(action.std(axis=0), 1e-4).tolist(),
    }


def normalize_image(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor - 0.5) / 0.5


def load_rgb(path: str, width: int, height: int) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR))
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).float().div_(255.0)
    return normalize_image(tensor)


def shuffle_depth_blocks(depth: np.ndarray, seed: int, rows: int = 8, cols: int = 10) -> np.ndarray:
    height, width = depth.shape
    crop_height = height - height % rows
    crop_width = width - width % cols
    source = depth[:crop_height, :crop_width]
    block_h = crop_height // rows
    block_w = crop_width // cols
    blocks = source.reshape(rows, block_h, cols, block_w).transpose(0, 2, 1, 3)
    flat = blocks.reshape(rows * cols, block_h, block_w)
    permutation = np.random.default_rng(seed).permutation(rows * cols)
    shuffled = flat[permutation].reshape(rows, cols, block_h, block_w).transpose(0, 2, 1, 3)
    output = depth.copy()
    output[:crop_height, :crop_width] = shuffled.reshape(crop_height, crop_width)
    return output


def load_depth(
    row: dict[str, Any],
    camera: str,
    method: str,
    width: int,
    height: int,
    min_depth_m: float,
    max_depth_m: float,
) -> torch.Tensor:
    if method == "zero_depth":
        normalized = np.zeros((height, width), dtype=np.float32)
    else:
        source_method = "raw_aligned" if method == "spatially_shuffled_raw" else method
        image = cv2.imread(row["depth_paths"][source_method][camera], cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 2:
            raise RuntimeError(f"Failed to read depth for {row['sequence']} {camera} {source_method}")
        depth_m = image.astype(np.float32) * 0.001
        depth_m = cv2.resize(depth_m, (width, height), interpolation=cv2.INTER_NEAREST)
        valid = (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
        normalized = np.zeros_like(depth_m, dtype=np.float32)
        normalized[valid] = np.clip(
            (depth_m[valid] - min_depth_m) / (max_depth_m - min_depth_m), 0.0, 1.0
        )
        if method == "spatially_shuffled_raw":
            token = f"{row['sequence']}:{row['frame_position']}:{camera}".encode("utf-8")
            normalized = shuffle_depth_blocks(normalized, zlib.crc32(token))
    tensor = torch.from_numpy(normalized.copy()).unsqueeze(0).repeat(3, 1, 1)
    return normalize_image(tensor)


class ACTDepthDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        stats: dict[str, list[float]],
        split: str,
        chunk_size: int,
        width: int,
        height: int,
        min_depth_m: float,
        max_depth_m: float,
        input_mode: str,
        method: str | None = None,
        max_frames: int | None = None,
    ) -> None:
        self.rows = rows
        self.chunk_size = chunk_size
        self.width = width
        self.height = height
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.input_mode = input_mode
        self.method = method
        self.state_mean = torch.tensor(stats["state_mean"], dtype=torch.float32)
        self.state_std = torch.tensor(stats["state_std"], dtype=torch.float32)
        self.action_mean = torch.tensor(stats["action_mean"], dtype=torch.float32)
        self.action_std = torch.tensor(stats["action_std"], dtype=torch.float32)
        self.by_sequence: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            self.by_sequence[row["sequence"]].append(index)

        selected = [index for index, row in enumerate(rows) if bool(row["is_eval"]) == (split == "eval")]
        if max_frames is not None:
            selected = selected[:max_frames]
        if split == "train":
            self.entries = [(index, method_name) for index in selected for method_name in NORMAL_METHODS]
        else:
            if method is None:
                raise ValueError("Evaluation dataset requires a fixed method")
            self.entries = [(index, method) for index in selected]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, item: int) -> dict[str, Any]:
        row_index, method = self.entries[item]
        row = self.rows[row_index]
        sequence_indices = self.by_sequence[row["sequence"]]
        position = int(row["frame_position"])
        action_rows = []
        padding = []
        for offset in range(self.chunk_size):
            target_position = min(position + offset, len(sequence_indices) - 1)
            action_rows.append(self.rows[sequence_indices[target_position]]["action"])
            padding.append(position + offset >= len(sequence_indices))

        sample: dict[str, Any] = {
            "observation.state": (torch.tensor(row["state"]) - self.state_mean) / self.state_std,
            "action": (torch.tensor(action_rows) - self.action_mean) / self.action_std,
            "action_is_pad": torch.tensor(padding, dtype=torch.bool),
            "row_index": row_index,
            "sequence": row["sequence"],
            "frame_position": int(row["frame_position"]),
            "method": method,
        }
        if self.input_mode == "rgb_depth":
            for key, camera in zip(RGB_KEYS, CAMERAS, strict=True):
                sample[key] = load_rgb(row["rgb_paths"][camera], self.width, self.height)
        for key, camera in zip(DEPTH_KEYS, CAMERAS, strict=True):
            sample[key] = load_depth(
                row,
                camera,
                method,
                self.width,
                self.height,
                self.min_depth_m,
                self.max_depth_m,
            )
        return sample


def active_image_keys(input_mode: str) -> tuple[str, ...]:
    return IMAGE_KEYS if input_mode == "rgb_depth" else DEPTH_KEYS


def make_config(chunk_size: int, width: int, height: int, input_mode: str) -> ACTConfig:
    inputs = {"observation.state": PolicyFeature(FeatureType.STATE, (14,))}
    inputs.update(
        {
            key: PolicyFeature(FeatureType.VISUAL, (3, height, width))
            for key in active_image_keys(input_mode)
        }
    )
    return ACTConfig(
        input_features=inputs,
        output_features={"action": PolicyFeature(FeatureType.ACTION, (14,))},
        chunk_size=chunk_size,
        n_action_steps=chunk_size,
        pretrained_backbone_weights=None,
        use_vae=True,
        device="cuda",
    )


def load_backbone(policy: ACTPolicy, path: Path | None) -> None:
    if path is None:
        return
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
    else:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload.get("model", payload)
    filtered = {
        key: value
        for key, value in state.items()
        if key.startswith("model.backbone.") and key in policy.state_dict()
    }
    if not filtered:
        raise RuntimeError(f"No ACT backbone tensors found in {path}")
    policy.load_state_dict(filtered, strict=False)
    print(f"Loaded {len(filtered)} backbone tensors from {path}", flush=True)


def distributed_context() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(args: argparse.Namespace) -> None:
    rank, local_rank, world_size, device = distributed_context()
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    rows = load_rows(args.index)
    stats = compute_stats(rows)
    dataset = ACTDepthDataset(
        rows,
        stats,
        "train",
        args.chunk_size,
        args.width,
        args.height,
        args.min_depth_m,
        args.max_depth_m,
        args.input_mode,
    )
    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    config = make_config(args.chunk_size, args.width, args.height, args.input_mode)
    policy = ACTPolicy(config)
    load_backbone(policy, args.backbone_checkpoint)
    policy.to(device)
    model: torch.nn.Module = policy
    if world_size > 1:
        model = DistributedDataParallel(policy, device_ids=[local_rank], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=args.lr, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    epoch = 0
    if args.resume and args.resume.is_file():
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        policy.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        step = int(payload["step"])
        epoch = int(payload.get("epoch", 0))
        if rank == 0:
            print(f"Resumed {args.resume} at step {step}", flush=True)

    if rank == 0:
        (args.output_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
        run_config = vars(args).copy()
        run_config = {key: str(value) if isinstance(value, Path) else value for key, value in run_config.items()}
        run_config.update(
            {
                "prompt_used": False,
                "state_dim": 14,
                "action_dim": 14,
                "image_keys": list(active_image_keys(args.input_mode)),
                "methods": list(NORMAL_METHODS),
                "world_size": world_size,
            }
        )
        (args.output_dir / "train_config.json").write_text(json.dumps(run_config, indent=2) + "\n")

    model.train()
    started = time.time()
    running_loss = 0.0
    while step < args.steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            if step >= args.steps:
                break
            tensor_batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
                if isinstance(value, torch.Tensor) and key not in {"row_index", "frame_position"}
            }
            if args.state_dropout_prob > 0:
                drop = torch.rand(tensor_batch["observation.state"].shape[0], device=device)
                tensor_batch["observation.state"][drop < args.state_dropout_prob] = 0
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
                loss, loss_dict = model(tensor_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()
            step += 1
            running_loss += float(loss.detach())

            if rank == 0 and step % args.log_every == 0:
                record = {
                    "step": step,
                    "epoch": epoch,
                    "loss": running_loss / args.log_every,
                    "l1_loss": float(loss_dict["l1_loss"]),
                    "elapsed_seconds": time.time() - started,
                }
                print(json.dumps(record), flush=True)
                with (args.output_dir / "train_log.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
                running_loss = 0.0

            if rank == 0 and (step % args.save_every == 0 or step == args.steps):
                atomic_checkpoint(
                    args.output_dir / "latest.pt",
                    {
                        "model": policy.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "epoch": epoch,
                        "stats": stats,
                    },
                )
        epoch += 1

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def closest_grid(token_count: int, aspect: float) -> tuple[int, int]:
    candidates = [(height, token_count // height) for height in range(1, token_count + 1) if token_count % height == 0]
    return min(candidates, key=lambda pair: abs(pair[1] / pair[0] - aspect))


class MetricAccumulator:
    def __init__(self) -> None:
        self.absolute_sum = 0.0
        self.squared_sum = 0.0
        self.count = 0
        self.first_absolute_sum = 0.0
        self.first_squared_sum = 0.0
        self.first_count = 0
        self.depth_mass_sum = 0.0
        self.depth_entropy_sum = 0.0
        self.frames = 0

    def update_actions(self, prediction: torch.Tensor, target: torch.Tensor, padding: torch.Tensor) -> None:
        error = prediction - target
        valid = (~padding).unsqueeze(-1).expand_as(error)
        selected = error[valid]
        self.absolute_sum += float(selected.abs().sum())
        self.squared_sum += float(selected.square().sum())
        self.count += selected.numel()
        first = error[:, 0]
        self.first_absolute_sum += float(first.abs().sum())
        self.first_squared_sum += float(first.square().sum())
        self.first_count += first.numel()

    def update_attention(self, mass: torch.Tensor, entropy: torch.Tensor) -> None:
        self.depth_mass_sum += float(mass.sum())
        self.depth_entropy_sum += float(entropy.sum())
        self.frames += mass.numel()

    def as_dict(self) -> dict[str, float | int]:
        return {
            "frames": self.frames,
            "chunk_mae_rad": self.absolute_sum / max(self.count, 1),
            "chunk_rmse_rad": math.sqrt(self.squared_sum / max(self.count, 1)),
            "first_step_mae_rad": self.first_absolute_sum / max(self.first_count, 1),
            "first_step_rmse_rad": math.sqrt(self.first_squared_sum / max(self.first_count, 1)),
            "depth_attention_mass": self.depth_mass_sum / max(self.frames, 1),
            "depth_attention_entropy": self.depth_entropy_sum / max(self.frames, 1),
        }


def write_attention_panel(
    path: Path,
    rgb_batch: list[torch.Tensor],
    attention_maps: list[torch.Tensor],
    sample_index: int,
    method: str,
) -> None:
    panels = []
    for camera, rgb_tensor, attention in zip(CAMERAS, rgb_batch, attention_maps, strict=True):
        rgb = ((rgb_tensor[sample_index].float().cpu() * 0.5 + 0.5).clamp(0, 1) * 255).byte()
        rgb = rgb.permute(1, 2, 0).numpy()
        heat = attention[sample_index].float().cpu().numpy()
        heat = (heat - heat.min()) / max(float(heat.max() - heat.min()), 1e-12)
        heat = cv2.resize(heat, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
        color = cv2.applyColorMap(np.rint(heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        overlay = cv2.addWeighted(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 0.58, color, 0.42, 0)
        cv2.putText(overlay, f"{method} {camera}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(overlay)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.hconcat(panels), [cv2.IMWRITE_JPEG_QUALITY, 92])


@torch.no_grad()
def evaluate_method(
    policy: ACTPolicy,
    rows: list[dict[str, Any]],
    stats: dict[str, list[float]],
    args: argparse.Namespace,
    method: str,
    device: torch.device,
) -> dict[str, Any]:
    dataset = ACTDepthDataset(
        rows,
        stats,
        "eval",
        args.chunk_size,
        args.width,
        args.height,
        args.min_depth_m,
        args.max_depth_m,
        args.input_mode,
        method=method,
        max_frames=args.max_eval_frames,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: tuple[torch.Tensor, torch.Tensor]) -> None:
        captured["weights"] = output[1].detach()

    handle = policy.model.decoder.layers[-1].multihead_attn.register_forward_hook(hook)
    action_mean = torch.tensor(stats["action_mean"], device=device)
    action_std = torch.tensor(stats["action_std"], device=device)
    global_metrics = MetricAccumulator()
    sequence_metrics: dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
    visualizations = 0
    visualization_counts: dict[str, int] = defaultdict(int)
    sequence_count = len({row["sequence"] for row in dataset.rows if row["is_eval"]})
    visualization_quota = max(1, math.ceil(args.visualizations_per_method / max(sequence_count, 1)))
    policy.eval()

    for batch in loader:
        sequences = list(batch["sequence"])
        tensor_batch = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor) and key not in {"row_index", "frame_position"}
        }
        prediction_norm = policy.predict_action_chunk(tensor_batch)
        prediction = prediction_norm * action_std + action_mean
        target = tensor_batch["action"] * action_std + action_mean
        padding = tensor_batch["action_is_pad"]
        global_metrics.update_actions(prediction, target, padding)

        weights = captured["weights"].float()
        if weights.ndim == 4:
            weights = weights.mean(dim=1)
        query_mean = weights.mean(dim=1)
        prefix_tokens = 2
        image_keys = active_image_keys(args.input_mode)
        image_token_count = (query_mean.shape[-1] - prefix_tokens) // len(image_keys)
        grid_h, grid_w = closest_grid(image_token_count, args.width / args.height)
        depth_maps = []
        depth_mass = torch.zeros(query_mean.shape[0], device=query_mean.device)
        entropy = torch.zeros_like(depth_mass)
        depth_image_indices = range(3, 6) if args.input_mode == "rgb_depth" else range(3)
        for image_index in depth_image_indices:
            start = prefix_tokens + image_index * image_token_count
            values = query_mean[:, start : start + image_token_count]
            depth_mass += values.sum(dim=-1)
            probabilities = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            entropy += -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1) / math.log(image_token_count)
            depth_maps.append(values.reshape(-1, grid_h, grid_w))
        entropy /= 3.0
        global_metrics.update_attention(depth_mass, entropy)

        for sample_index, sequence in enumerate(sequences):
            item_metrics = sequence_metrics[sequence]
            item_metrics.update_actions(
                prediction[sample_index : sample_index + 1],
                target[sample_index : sample_index + 1],
                padding[sample_index : sample_index + 1],
            )
            item_metrics.update_attention(depth_mass[sample_index : sample_index + 1], entropy[sample_index : sample_index + 1])

        if visualizations < args.visualizations_per_method:
            if args.input_mode == "rgb_depth":
                rgb_batch = [batch[key] for key in RGB_KEYS]
            else:
                row_indices = [int(value) for value in batch["row_index"]]
                rgb_batch = [
                    torch.stack(
                        [load_rgb(rows[index]["rgb_paths"][camera], args.width, args.height) for index in row_indices]
                    )
                    for camera in CAMERAS
                ]
            for sample_index, sequence in enumerate(sequences):
                if visualizations >= args.visualizations_per_method:
                    break
                if visualization_counts[sequence] >= visualization_quota:
                    continue
                safe_sequence = sequences[sample_index].replace("/", "_")
                frame = int(batch["frame_position"][sample_index])
                write_attention_panel(
                    args.output_dir / "attention" / method / f"{safe_sequence}_{frame:06d}.jpg",
                    rgb_batch,
                    depth_maps,
                    sample_index,
                    method,
                )
                visualizations += 1
                visualization_counts[sequence] += 1

    handle.remove()
    return {
        "method": method,
        "metrics": global_metrics.as_dict(),
        "sequences": {sequence: metric.as_dict() for sequence, metric in sorted(sequence_metrics.items())},
    }


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    rows = load_rows(args.index)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stats = payload.get("stats")
    if stats is None:
        stats = json.loads((args.checkpoint.parent / "stats.json").read_text())
    config = make_config(args.chunk_size, args.width, args.height, args.input_mode)
    policy = ACTPolicy(config)
    policy.load_state_dict(payload["model"])
    policy.to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for method in NORMAL_METHODS + CONTROLS:
        print(f"Evaluating {method}", flush=True)
        result = evaluate_method(policy, rows, stats, args, method, device)
        results.append(result)
        print(json.dumps(result["metrics"]), flush=True)

    zero_mae = next(result["metrics"]["chunk_mae_rad"] for result in results if result["method"] == "zero_depth")
    for result in results:
        result["metrics"]["depth_benefit_vs_zero_mae_rad"] = zero_mae - result["metrics"]["chunk_mae_rad"]
    normal_results = [result for result in results if result["method"] in NORMAL_METHODS]
    ranking = sorted(normal_results, key=lambda result: result["metrics"]["chunk_mae_rad"])
    report = {
        "schema": "act_no_prompt_arm14_benchmark_v1",
        "prompt_used": False,
        "input_mode": args.input_mode,
        "shared_checkpoint": str(args.checkpoint),
        "normal_methods": list(NORMAL_METHODS),
        "negative_controls": list(CONTROLS),
        "metric_scope": "offline held-out 14-DoF arm action prediction; excludes end-effector commands",
        "attention": "last ACT decoder action-query cross-attention; raw metrics are comparable across methods",
        "ranking_by_chunk_mae": [result["method"] for result in ranking],
        "results": results,
    }
    report_path = args.output_dir / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (args.output_dir / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["method"] + list(results[0]["metrics"].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({"method": result["method"], **result["metrics"]})
    print(f"Wrote {report_path}")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--min-depth-m", type=float, default=0.2)
    parser.add_argument("--max-depth-m", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--input-mode", choices=("rgb_depth", "depth_only"), default="rgb_depth")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    add_common_arguments(train_parser)
    train_parser.add_argument("--steps", type=int, default=12000)
    train_parser.add_argument("--lr", type=float, default=1e-5)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--grad-clip", type=float, default=10.0)
    train_parser.add_argument("--seed", type=int, default=20260813)
    train_parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument("--log-every", type=int, default=50)
    train_parser.add_argument("--save-every", type=int, default=1000)
    train_parser.add_argument("--backbone-checkpoint", type=Path)
    train_parser.add_argument("--resume", type=Path)
    train_parser.add_argument("--state-dropout-prob", type=float, default=0.0)

    eval_parser = subparsers.add_parser("evaluate")
    add_common_arguments(eval_parser)
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--device", default="cuda:0")
    eval_parser.add_argument("--max-eval-frames", type=int)
    eval_parser.add_argument("--visualizations-per-method", type=int, default=24)

    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
