from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TargetAnnotation:
    episode_index: int
    frame_index: int
    camera: str
    bbox_xyxy: tuple[float, float, float, float]
    bbox_normalized: tuple[float, float, float, float]
    visible: bool
    mask_path: str | None = None
    instance_id: str | None = None
    phase: str | None = None
    confidence: float | None = None


def normalize_box(
    box_xyxy: list[float] | tuple[float, ...], width: int, height: int
) -> tuple[float, float, float, float]:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    return (
        max(0.0, min(1.0, x1 / width)),
        max(0.0, min(1.0, y1 / height)),
        max(0.0, min(1.0, x2 / width)),
        max(0.0, min(1.0, y2 / height)),
    )


def box_to_patch_target(
    bbox_normalized: list[float] | tuple[float, ...],
    grid_height: int,
    grid_width: int,
) -> np.ndarray:
    x1, y1, x2, y2 = [float(value) for value in bbox_normalized]
    xs = (np.arange(grid_width, dtype=np.float32) + 0.5) / grid_width
    ys = (np.arange(grid_height, dtype=np.float32) + 0.5) / grid_height
    target = (ys[:, None] >= y1) & (ys[:, None] <= y2) & (xs[None, :] >= x1) & (xs[None, :] <= x2)
    return target.astype(np.float32)


def phase_for_frame(frame: int, start: int, contact: int, release: int, end: int) -> str:
    if frame < start or frame > end:
        return "outside_event"
    if frame < contact:
        return "approach"
    if frame == contact:
        return "contact"
    if frame < release:
        return "transport"
    if frame == release:
        return "release"
    return "post_release"


class AnnotationIndex:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._items: dict[tuple[int, int, str], TargetAnnotation] = {}
        for row in rows:
            episode = int(row["episode_index"])
            frame = int(row["frame_index"])
            for camera, item in row.get("targets", {}).items():
                if not item.get("bbox_xyxy"):
                    continue
                annotation = TargetAnnotation(
                    episode_index=episode,
                    frame_index=frame,
                    camera=camera,
                    bbox_xyxy=tuple(float(value) for value in item["bbox_xyxy"]),
                    bbox_normalized=tuple(float(value) for value in item["bbox_normalized"]),
                    visible=bool(item.get("visible", True)),
                    mask_path=item.get("mask_path"),
                    instance_id=item.get("instance_id"),
                    phase=item.get("phase"),
                    confidence=item.get("confidence"),
                )
                self._items[(episode, frame, camera)] = annotation

    @classmethod
    def from_jsonl(cls, path: Path) -> AnnotationIndex:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls(rows)

    def get(self, episode_index: int, frame_index: int, camera: str) -> TargetAnnotation | None:
        return self._items.get((int(episode_index), int(frame_index), camera))


def roi_crop_tensor(image: Any, bbox_normalized: Any, output_size: tuple[int, int]) -> Any:
    """Crop a CHW torch tensor without importing torch until this feature is used."""
    from torch.nn import functional

    _, height, width = image.shape
    x1, y1, x2, y2 = [float(value) for value in bbox_normalized]
    left, top = max(0, int(x1 * width)), max(0, int(y1 * height))
    right, bottom = (
        min(width, max(left + 1, int(np.ceil(x2 * width)))),
        min(height, max(top + 1, int(np.ceil(y2 * height)))),
    )
    crop = image[:, top:bottom, left:right].unsqueeze(0)
    return functional.interpolate(
        crop, size=output_size, mode="bilinear", align_corners=False
    ).squeeze(0)


def roi_auxiliary_loss(attention_logits: Any, patch_targets: Any, valid: Any) -> Any:
    """BCE target-localization loss; boxes are supervision and are not needed at inference."""
    from torch.nn import functional

    per_token = functional.binary_cross_entropy_with_logits(
        attention_logits, patch_targets, reduction="none"
    )
    per_sample = per_token.flatten(1).mean(dim=1)
    valid = valid.to(per_sample.dtype)
    return (per_sample * valid).sum() / valid.sum().clamp_min(1.0)
