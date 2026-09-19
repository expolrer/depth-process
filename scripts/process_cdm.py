#!/usr/bin/env python3
"""Run camera-specific CDMs and export pure plus sensor-preserving depth."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from depth_pipeline.cdm import (
    CameraDepthSpec,
    fuse_sensor_and_cdm,
    make_inverse_depth_prompt,
    sanitize_sensor_depth,
)


ENCODER_FEATURES = {
    "vits": 64,
    "vitb": 128,
    "vitl": 256,
    "vitg": 384,
}


@dataclass(frozen=True)
class Task:
    sequence: str
    camera_id: str
    frame_index: int
    rgb_path: Path
    depth_path: Path


def parse_camera_config(path: Path) -> dict[str, CameraDepthSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cameras = payload.get("cameras", payload)
    if not isinstance(cameras, dict) or not cameras:
        raise ValueError("Camera config must contain a non-empty 'cameras' mapping")
    result = {}
    for camera_id, row in cameras.items():
        if not isinstance(row, dict):
            raise ValueError(f"Invalid camera configuration for {camera_id}")
        result[camera_id] = CameraDepthSpec(
            model_id=str(row["model_id"]).lower(),
            checkpoint=str(row["checkpoint"]),
            min_depth_m=float(row["min_depth_m"]),
            max_depth_m=float(row["max_depth_m"]),
        )
    return result


def discover(input_root: Path, configured: set[str]) -> tuple[list[Task], list[str]]:
    tasks: list[Task] = []
    skipped: list[str] = []
    for manifest in sorted(input_root.glob("*/cam_*/manifest.jsonl")):
        camera_id = manifest.parent.name
        sequence = str(manifest.parent.relative_to(input_root))
        if camera_id not in configured:
            skipped.append(sequence)
            continue
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            tasks.append(
                Task(
                    sequence=sequence,
                    camera_id=camera_id,
                    frame_index=int(row["frame_index"]),
                    rgb_path=Path(row["rgb_path"]),
                    depth_path=Path(
                        row.get("depth_aligned_rgb_mm_path", row["depth_raw_mm_path"])
                    ),
                )
            )
    return tasks, skipped


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(payload)!r}")
    return {str(key).removeprefix("module."): value for key, value in payload.items()}


def save_depth(path: Path, depth_m: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = np.rint(depth_m * 1000.0).clip(0, 65535).astype(np.uint16)
    if not cv2.imwrite(str(path), encoded):
        raise RuntimeError(f"Failed to write {path}")


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
        raise RuntimeError(f"Failed to write {path}")


def infer_depth(model, rgb_bgr: np.ndarray, sensor_m: np.ndarray, input_size: int) -> np.ndarray:
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    inverse = make_inverse_depth_prompt(sensor_m)
    prediction = model.infer_image(rgb, inverse, input_size=input_size)
    if isinstance(prediction, dict):
        prediction = prediction.get("depth", prediction.get("pred_depth"))
    if torch.is_tensor(prediction):
        prediction = prediction.detach().float().cpu().numpy()
    value = np.asarray(prediction, dtype=np.float32).squeeze()
    if value.shape != sensor_m.shape:
        value = cv2.resize(value, (sensor_m.shape[1], sensor_m.shape[0]), interpolation=cv2.INTER_LINEAR)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--camera-config", type=Path, required=True)
    parser.add_argument("--encoder", default="vitl", choices=("vits", "vitb", "vitl", "vitg"))
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    specs = parse_camera_config(args.camera_config)
    tasks, skipped_sequences = discover(args.input_root, set(specs))
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        raise SystemExit("No frames matched configured cameras")

    sys.path.insert(0, str(args.repo))
    from rgbddepth.dpt import RGBDDepth

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_rows: list[dict[str, object]] = []
    processed = 0
    resumed = 0
    started_all = time.perf_counter()

    for model_id in sorted({spec.model_id for spec in specs.values()}):
        model_specs = [spec for spec in specs.values() if spec.model_id == model_id]
        checkpoints = {spec.checkpoint for spec in model_specs}
        if len(checkpoints) != 1:
            raise ValueError(f"Model {model_id} maps to multiple checkpoints: {checkpoints}")
        checkpoint = Path(checkpoints.pop())
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model = RGBDDepth(encoder=args.encoder, features=ENCODER_FEATURES[args.encoder])
        model.load_state_dict(load_state_dict(checkpoint), strict=True)
        model = model.to(device).eval()

        model_tasks = [task for task in tasks if specs[task.camera_id].model_id == model_id]
        for task in model_tasks:
            pure_path = args.output_root / task.sequence / "cdm_camera_specific" / f"{task.frame_index:06d}.png"
            fused_path = args.output_root / task.sequence / "cdm_sensor_fused" / f"{task.frame_index:06d}.png"
            mask_path = args.output_root / task.sequence / "cdm_fill_mask" / f"{task.frame_index:06d}.png"
            if not args.overwrite and pure_path.is_file() and fused_path.is_file() and mask_path.is_file():
                resumed += 1
                continue
            rgb = cv2.imread(str(task.rgb_path), cv2.IMREAD_COLOR)
            raw_mm = cv2.imread(str(task.depth_path), cv2.IMREAD_UNCHANGED)
            if rgb is None or raw_mm is None:
                raise RuntimeError(f"Failed to decode inputs for {task.sequence}/{task.frame_index}")
            spec = specs[task.camera_id]
            sensor, sensor_valid = sanitize_sensor_depth(
                raw_mm.astype(np.float32) * 0.001,
                min_depth_m=spec.min_depth_m,
                max_depth_m=spec.max_depth_m,
            )
            started = time.perf_counter()
            with torch.inference_mode():
                cdm_depth = infer_depth(model, rgb, sensor, args.input_size)
            fused, fill_mask, _ = fuse_sensor_and_cdm(
                sensor,
                cdm_depth,
                min_depth_m=spec.min_depth_m,
                max_depth_m=spec.max_depth_m,
            )
            elapsed = time.perf_counter() - started
            save_depth(pure_path, cdm_depth)
            save_depth(fused_path, fused)
            save_mask(mask_path, fill_mask)
            output_rows.append(
                {
                    "sequence": task.sequence,
                    "frame_index": task.frame_index,
                    "camera_id": task.camera_id,
                    "model_id": spec.model_id,
                    "sensor_valid_fraction": float(np.mean(sensor_valid)),
                    "cdm_valid_fraction": float(np.mean(cdm_depth > 0)),
                    "fused_valid_fraction": float(np.mean(fused > 0)),
                    "filled_fraction": float(np.mean(fill_mask)),
                    "inference_and_fusion_ms": elapsed * 1000.0,
                }
            )
            processed += 1
            if processed % 50 == 0:
                print(f"processed={processed}/{len(tasks)} resumed={resumed}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "method": "camera_depth_models",
        "camera_config": str(args.camera_config),
        "processed_frames": processed,
        "resumed_frames": resumed,
        "skipped_unconfigured_sequences": skipped_sequences,
        "elapsed_seconds": time.perf_counter() - started_all,
        "outputs": ["cdm_camera_specific", "cdm_sensor_fused", "cdm_fill_mask"],
        "frames": output_rows,
    }
    report_path = args.output_root / "cdm_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
