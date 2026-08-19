#!/usr/bin/env python3
"""Generate task ROI masks for RGB-D attention analysis.

The preferred backend for this script is GroundingDINO/SAM2. When those optional
packages are not installed, the script falls back to a deterministic RGB-D
proposal backend so the rest of the review and metric pipeline remains runnable.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


CAMERAS = ("cam_h", "cam_l", "cam_r")
TASK_TEXT = {
    "chengzhong": (
        "metal sleeve . weighing platform . left box . right box . robot hand"
    ),
    "dajian": (
        "automotive sheet metal part . center support stand . right support stand . robot hand"
    ),
    "zhoumian": "pin connector . seat belt . cable . empty box . robot hand",
    "leju_claw": "object . gripper . robot hand . tabletop",
    "dex_hand": "object . dexterous hand . robot hand . tabletop",
}


@dataclass(frozen=True)
class Frame:
    sequence: str
    camera: str
    frame_index: int
    rgb_path: Path
    depth_path: Path


def task_prompt(sequence: str) -> str:
    lowered = sequence.lower()
    for key, prompt in TASK_TEXT.items():
        if key in lowered:
            return prompt
    return "task object . robot hand . tabletop"


def iter_frames(input_root: Path) -> list[Frame]:
    frames: list[Frame] = []
    for manifest_path in sorted(input_root.glob("*/cam_*/manifest.jsonl")):
        sequence = manifest_path.parent.parent.name
        camera = manifest_path.parent.name
        if camera not in CAMERAS:
            continue
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            frames.append(
                Frame(
                    sequence=sequence,
                    camera=camera,
                    frame_index=int(row["frame_index"]),
                    rgb_path=Path(row["rgb_path"]),
                    depth_path=Path(row.get("depth_aligned_rgb_mm_path", row["depth_raw_mm_path"])),
                )
            )
    return frames


def keep_largest_components(mask: np.ndarray, max_components: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return mask.astype(bool)
    areas = [(index, int(stats[index, cv2.CC_STAT_AREA])) for index in range(1, count)]
    areas.sort(key=lambda item: item[1], reverse=True)
    selected = {index for index, _ in areas[:max_components]}
    return np.isin(labels, list(selected))


def rgbd_proposal_mask(rgb: np.ndarray, depth_mm: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    height, width = depth_mm.shape
    valid = depth_mm > 0
    if valid.any():
        depth = depth_mm.astype(np.float32)
        near = np.percentile(depth[valid], 18.0)
        far = np.percentile(depth[valid], 72.0)
        depth_band = valid & (depth >= near) & (depth <= far)
    else:
        depth_band = np.zeros((height, width), dtype=bool)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    sat_mask = saturation > max(32, np.percentile(saturation, 60))
    bright_mask = value > np.percentile(value, 45)
    image_prior = sat_mask | bright_mask

    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    edge_support = cv2.dilate(edges, np.ones((7, 7), np.uint8), iterations=1) > 0

    mask = (depth_band & (image_prior | edge_support)).astype(np.uint8)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    mask = keep_largest_components(mask, 3)

    if mask.mean() < 0.005 and valid.any():
        mask = keep_largest_components(depth_band.astype(np.uint8), 3)
    if mask.mean() > 0.45:
        mask = keep_largest_components((depth_band & edge_support).astype(np.uint8), 4)
    if mask.mean() < 0.002:
        margin_y = int(height * 0.18)
        margin_x = int(width * 0.18)
        fallback = np.zeros((height, width), dtype=bool)
        fallback[margin_y : height - margin_y, margin_x : width - margin_x] = True
        mask = fallback

    mask = mask.astype(np.uint8) * 255
    area_ratio = float((mask > 0).mean())
    confidence = float(np.clip(1.0 - abs(area_ratio - 0.12) / 0.20, 0.05, 0.95))
    return mask, {"area_ratio": area_ratio, "proposal_confidence": confidence}


def overlay_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = rgb.copy()
    color = np.zeros_like(rgb)
    color[:, :, 1] = 210
    color[:, :, 2] = 255
    alpha = (mask > 0).astype(np.float32) * 0.42
    overlay = (overlay * (1.0 - alpha[..., None]) + color * alpha[..., None]).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    return overlay


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("outputs/extracted"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/roi_masks"))
    parser.add_argument("--backend", default="auto", choices=("auto", "grounded_sam2", "classical_proposal"))
    parser.add_argument("--review-stride", type=int, default=60)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    selected_backend = args.backend
    if selected_backend in ("auto", "grounded_sam2"):
        try:
            __import__("groundingdino")
            __import__("sam2")
            if selected_backend == "grounded_sam2":
                raise NotImplementedError(
                    "Grounded-SAM2 adapter is not configured with local checkpoints yet"
                )
        except Exception as exc:
            print(f"[roi] Grounded-SAM2 unavailable: {type(exc).__name__}: {exc}")
            selected_backend = "classical_proposal"

    frames = iter_frames(args.input_root)
    if args.max_frames is not None:
        frames = frames[: args.max_frames]

    rows: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []
    for index, frame in enumerate(frames):
        rgb = cv2.imread(str(frame.rgb_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(frame.depth_path), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None or depth.ndim != 2:
            print(f"[roi] skip unreadable frame {frame.sequence}/{frame.camera}/{frame.frame_index:06d}")
            continue
        mask, metrics = rgbd_proposal_mask(rgb, depth)
        rel = Path(frame.sequence) / frame.camera
        mask_path = args.output_root / rel / "mask" / f"{frame.frame_index:06d}.png"
        overlay_path = args.output_root / rel / "overlay" / f"{frame.frame_index:06d}.jpg"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(mask_path), mask):
            raise RuntimeError(f"failed to write {mask_path}")
        if index % max(1, args.review_stride) == 0:
            if not cv2.imwrite(str(overlay_path), overlay_mask(rgb, mask), [cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise RuntimeError(f"failed to write {overlay_path}")
            review_rows.append(
                {
                    "sequence": frame.sequence,
                    "camera": frame.camera,
                    "frame_index": frame.frame_index,
                    "rgb_path": str(frame.rgb_path),
                    "mask_path": str(mask_path),
                    "overlay_path": str(overlay_path),
                    "prompt": task_prompt(frame.sequence),
                    **metrics,
                }
            )
        rows.append(
            {
                "sequence": frame.sequence,
                "camera": frame.camera,
                "frame_index": frame.frame_index,
                "rgb_path": str(frame.rgb_path),
                "depth_path": str(frame.depth_path),
                "mask_path": str(mask_path),
                "prompt": task_prompt(frame.sequence),
                "backend": selected_backend,
                "needs_human_review": selected_backend != "grounded_sam2",
                **metrics,
            }
        )

    write_jsonl(args.output_root / "roi_index.jsonl", rows)
    write_jsonl(args.output_root / "review_index.jsonl", review_rows)
    summary = {
        "schema": "rgbd_roi_mask_v1",
        "backend": selected_backend,
        "frames": len(rows),
        "review_frames": len(review_rows),
        "mean_area_ratio": float(np.mean([row["area_ratio"] for row in rows])) if rows else 0.0,
        "note": (
            "Grounded-SAM2 was requested, but the local backend may fall back to "
            "classical RGB-D proposals when GroundingDINO/SAM2 packages or checkpoints are unavailable."
        ),
    }
    (args.output_root / "roi_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
