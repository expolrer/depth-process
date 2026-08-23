from __future__ import annotations

from typing import Any

import numpy as np

from .target_annotations import TargetAnnotation, box_to_patch_target


def add_act_roi_supervision(
    sample: dict[str, Any],
    annotation: TargetAnnotation | None,
    camera: str,
    patch_grid: tuple[int, int],
) -> dict[str, Any]:
    """Add framework-neutral fields consumed by a patched ACT policy/trainer."""
    output = dict(sample)
    prefix = f"observation.target_roi.{camera}"
    if annotation is None or not annotation.visible:
        output[f"{prefix}.bbox"] = np.zeros(4, dtype=np.float32)
        output[f"{prefix}.valid"] = np.float32(0.0)
        output[f"{prefix}.patch_target"] = np.zeros(patch_grid, dtype=np.float32)
        return output
    output[f"{prefix}.bbox"] = np.asarray(annotation.bbox_normalized, dtype=np.float32)
    output[f"{prefix}.valid"] = np.float32(1.0)
    output[f"{prefix}.patch_target"] = box_to_patch_target(
        annotation.bbox_normalized,
        patch_grid[0],
        patch_grid[1],
    )
    return output


def roi_coordinate_token(annotation: TargetAnnotation | None) -> np.ndarray:
    if annotation is None or not annotation.visible:
        return np.zeros(5, dtype=np.float32)
    x1, y1, x2, y2 = annotation.bbox_normalized
    return np.asarray(
        [(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1, 1.0],
        dtype=np.float32,
    )
