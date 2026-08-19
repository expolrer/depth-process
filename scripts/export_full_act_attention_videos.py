#!/usr/bin/env python3
"""Export full-sequence no-prompt ACT depth-attention mosaic videos."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from act_no_prompt_benchmark import (
    CAMERAS,
    CONTROLS,
    DEPTH_KEYS,
    NORMAL_METHODS,
    active_image_keys,
    closest_grid,
    load_depth,
    load_rgb,
    make_config,
)
from lerobot.policies.act.modeling_act import ACTPolicy


SEQUENCE_SLUGS = {
    "A10-A15-G-S-01-TQ_03_01-4_304-leju_claw-20260512172205-200049-8c435b-v003": "leju_claw",
    "A10-A15-G-S-01-TQ_09_01-P4_297-dex_hand-20260629103123-49-9a5c2d-v003": "dex_hand",
    "chengzhong_xianxia_main1": "chengzhong",
    "dajian_xianxia_main1": "dajian",
    "zhoumian_xianxia_main1": "zhoumian",
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


class FullSequenceDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        stats: dict[str, list[float]],
        method: str,
        width: int,
        height: int,
        min_depth_m: float,
        max_depth_m: float,
    ) -> None:
        self.rows = rows
        self.method = method
        self.width = width
        self.height = height
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.state_mean = torch.tensor(stats["state_mean"], dtype=torch.float32)
        self.state_std = torch.tensor(stats["state_std"], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        sample: dict[str, Any] = {
            "observation.state": (
                torch.tensor(row["state"], dtype=torch.float32) - self.state_mean
            )
            / self.state_std,
            "sequence": row["sequence"],
            "frame_position": int(row["frame_position"]),
        }
        for key, camera in zip(DEPTH_KEYS, CAMERAS, strict=True):
            sample[key] = load_depth(
                row,
                camera,
                self.method,
                self.width,
                self.height,
                self.min_depth_m,
                self.max_depth_m,
            )
            sample[f"visual_{camera}"] = load_rgb(
                row["rgb_paths"][camera], self.width, self.height
            )
        return sample


class RawVideoWriter:
    def __init__(self, path: Path, width: int, height: int, fps: float, crf: int) -> None:
        self.path = path
        self.temporary = path.with_suffix(".part.mp4")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary.unlink(missing_ok=True)
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-y",
            str(self.temporary),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable")
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        stderr = self.process.stderr.read().decode("utf-8", errors="replace") if self.process.stderr else ""
        code = self.process.wait()
        if code:
            self.temporary.unlink(missing_ok=True)
            raise RuntimeError(stderr.strip() or f"ffmpeg exited {code}")
        self.temporary.replace(self.path)


def probe_frame_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


def attention_panel(
    batch: dict[str, Any], depth_maps: list[torch.Tensor], sample_index: int, method: str
) -> np.ndarray:
    panels = []
    for camera, attention in zip(CAMERAS, depth_maps, strict=True):
        rgb_tensor = batch[f"visual_{camera}"][sample_index]
        rgb = ((rgb_tensor.float() * 0.5 + 0.5).clamp(0, 1) * 255).byte()
        rgb = rgb.permute(1, 2, 0).numpy()
        heat = attention[sample_index].float().cpu().numpy()
        heat = (heat - heat.min()) / max(float(heat.max() - heat.min()), 1e-12)
        heat = cv2.resize(heat, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
        color = cv2.applyColorMap(np.rint(heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        overlay = cv2.addWeighted(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 0.58, color, 0.42, 0)
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 34), (12, 16, 20), -1)
        cv2.putText(
            overlay,
            f"{method} {camera}",
            (10, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panels.append(overlay)
    return cv2.hconcat(panels)


@torch.no_grad()
def export_method(
    policy: ACTPolicy,
    rows: list[dict[str, Any]],
    stats: dict[str, list[float]],
    method: str,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    sequence_counts: dict[str, int] = {}
    for row in rows:
        sequence_counts[row["sequence"]] = sequence_counts.get(row["sequence"], 0) + 1
    outputs = {
        sequence: args.output_dir / SEQUENCE_SLUGS[sequence] / "attention" / f"{method}.mp4"
        for sequence in sequence_counts
    }
    if all(probe_frame_count(path) == sequence_counts[sequence] for sequence, path in outputs.items()):
        print(f"[{method}] all videos already complete", flush=True)
        return

    dataset = FullSequenceDataset(
        rows,
        stats,
        method,
        args.width,
        args.height,
        args.min_depth_m,
        args.max_depth_m,
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
    policy.eval()
    writer: RawVideoWriter | None = None
    current_sequence: str | None = None
    written = 0
    try:
        for batch in loader:
            tensor_batch = {
                key: batch[key].to(device, non_blocking=True)
                for key in ("observation.state", *DEPTH_KEYS)
            }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
                policy.predict_action_chunk(tensor_batch)
            weights = captured["weights"].float()
            if weights.ndim == 4:
                weights = weights.mean(dim=1)
            query_mean = weights.mean(dim=1)
            image_token_count = (query_mean.shape[-1] - 2) // len(active_image_keys("depth_only"))
            grid_h, grid_w = closest_grid(image_token_count, args.width / args.height)
            depth_maps = []
            for image_index in range(3):
                start = 2 + image_index * image_token_count
                depth_maps.append(
                    query_mean[:, start : start + image_token_count].reshape(-1, grid_h, grid_w)
                )

            for sample_index, sequence in enumerate(batch["sequence"]):
                if sequence != current_sequence:
                    if writer is not None:
                        writer.close()
                    current_sequence = sequence
                    output = outputs[sequence]
                    output.unlink(missing_ok=True)
                    writer = RawVideoWriter(output, args.width * 3, args.height, args.fps, args.crf)
                    print(f"[{method}] writing {SEQUENCE_SLUGS[sequence]}", flush=True)
                if writer is None:
                    raise RuntimeError("video writer was not initialized")
                writer.write(attention_panel(batch, depth_maps, sample_index, method))
                written += 1
                if written % args.log_every == 0:
                    print(f"[{method}] {written}/{len(rows)} frames", flush=True)
        if writer is not None:
            writer.close()
            writer = None
    finally:
        handle.remove()
        if writer is not None:
            if writer.process.stdin is not None and not writer.process.stdin.closed:
                writer.process.stdin.close()
            writer.process.kill()
    print(f"[{method}] complete: {written} frames", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--min-depth-m", type=float, default=0.2)
    parser.add_argument("--max-depth-m", type=float, default=4.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    valid_methods = set(NORMAL_METHODS + CONTROLS)
    invalid = [method for method in args.methods if method not in valid_methods]
    if invalid:
        raise ValueError(f"Unknown methods: {invalid}")
    rows = load_rows(args.index)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stats = payload.get("stats")
    if stats is None:
        stats = json.loads((args.checkpoint.parent / "stats.json").read_text())
    device = torch.device(args.device)
    config = make_config(args.chunk_size, args.width, args.height, "depth_only")
    policy = ACTPolicy(config)
    policy.load_state_dict(payload["model"])
    policy.to(device)
    for method in args.methods:
        export_method(policy, rows, stats, method, args, device)


if __name__ == "__main__":
    main()
