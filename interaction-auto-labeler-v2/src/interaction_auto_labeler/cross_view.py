from __future__ import annotations

from typing import Any

import numpy as np


def backproject_depth(
    depth_mm: np.ndarray,
    intrinsics: np.ndarray,
    box_xyxy: list[float],
    stride: int = 4,
) -> np.ndarray:
    if depth_mm.ndim != 2:
        raise ValueError("depth_mm must be a single-channel image")
    x1, y1, x2, y2 = [round(value) for value in box_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(depth_mm.shape[1], x2), min(depth_mm.shape[0], y2)
    ys, xs = np.mgrid[y1:y2:stride, x1:x2:stride]
    depth = depth_mm[y1:y2:stride, x1:x2:stride].astype(np.float64) / 1000.0
    valid = depth > 0
    if not valid.any():
        return np.empty((0, 3), dtype=np.float64)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    z = depth[valid]
    x = (xs[valid] - cx) * z / fx
    y = (ys[valid] - cy) * z / fy
    return np.column_stack((x, y, z))


def transform_points(points: np.ndarray, target_from_source: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.reshape(0, 3)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (target_from_source @ homogeneous.T).T[:, :3]


def project_points(
    points: np.ndarray,
    intrinsics: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, list[float] | None]:
    valid = points[:, 2] > 1e-6 if points.size else np.zeros(0, dtype=bool)
    points = points[valid]
    if not len(points):
        return np.empty((0, 2)), None
    uvw = (intrinsics @ points.T).T
    pixels = uvw[:, :2] / uvw[:, 2:3]
    height, width = image_shape
    inside = (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    )
    pixels = pixels[inside]
    if not len(pixels):
        return pixels, None
    minimum, maximum = pixels.min(axis=0), pixels.max(axis=0)
    return pixels, [float(minimum[0]), float(minimum[1]), float(maximum[0]), float(maximum[1])]


def soft_cross_view_score(features: dict[str, Any]) -> float:
    weights = {
        "appearance_similarity": 0.30,
        "color_similarity": 0.10,
        "depth_consistency": 0.15,
        "contact_sync": 0.25,
        "co_motion_consistency": 0.20,
    }
    available = {key: features.get(key) for key in weights if features.get(key) is not None}
    denominator = sum(weights[key] for key in available)
    if not denominator:
        return 0.0
    return float(
        sum(weights[key] * max(0.0, min(1.0, float(value))) for key, value in available.items())
        / denominator
    )
