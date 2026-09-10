from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.fx <= 0 or self.fy <= 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("camera focal lengths and dimensions must be positive")

    @property
    def matrix(self) -> np.ndarray:
        return np.asarray(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_transform(transform: Any, name: str = "transform") -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"{name} must be a homogeneous transform")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-4):
        raise ValueError(f"{name} rotation determinant must be one")
    return matrix


def invert_transform(transform: Any) -> np.ndarray:
    matrix = validate_transform(transform)
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = matrix[:3, :3].T
    inverse[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return inverse


def backproject_depth(
    depth_m: Any,
    intrinsics: CameraIntrinsics,
    mask: Any | None = None,
    max_depth_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.shape != (intrinsics.height, intrinsics.width):
        raise ValueError("depth shape does not match camera intrinsics")
    valid = np.isfinite(depth) & (depth > 0)
    if max_depth_m is not None:
        valid &= depth <= max_depth_m
    if mask is not None:
        supplied_mask = np.asarray(mask, dtype=bool)
        if supplied_mask.shape != depth.shape:
            raise ValueError("mask shape does not match depth")
        valid &= supplied_mask
    y, x = np.nonzero(valid)
    z = depth[y, x]
    points = np.column_stack(
        ((x - intrinsics.cx) * z / intrinsics.fx, (y - intrinsics.cy) * z / intrinsics.fy, z)
    )
    return points, np.column_stack((x, y)).astype(np.int32)


def transform_points(points: Any, target_from_source: Any) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    transform = validate_transform(target_from_source, "target_from_source")
    return points @ transform[:3, :3].T + transform[:3, 3]


def project_points(
    points: Any,
    intrinsics: CameraIntrinsics,
    clip: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-8)
    index = np.flatnonzero(valid)
    selected = points[valid]
    if not len(selected):
        return np.empty((0, 2), dtype=np.float64), index
    pixels = np.column_stack(
        (
            intrinsics.fx * selected[:, 0] / selected[:, 2] + intrinsics.cx,
            intrinsics.fy * selected[:, 1] / selected[:, 2] + intrinsics.cy,
        )
    )
    if clip:
        inside = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < intrinsics.width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < intrinsics.height)
        )
        pixels, index = pixels[inside], index[inside]
    return pixels, index


def reproject_mask(
    source_depth_m: Any,
    source_mask: Any,
    source_intrinsics: CameraIntrinsics,
    target_from_source: Any,
    target_intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    points, _ = backproject_depth(source_depth_m, source_intrinsics, source_mask)
    target_points = transform_points(points, target_from_source)
    pixels, point_indices = project_points(target_points, target_intrinsics)
    mask = np.zeros((target_intrinsics.height, target_intrinsics.width), dtype=bool)
    z_buffer = np.full(mask.shape, np.inf, dtype=np.float64)
    if not len(pixels):
        return mask, z_buffer
    integer_pixels = np.rint(pixels).astype(int)
    integer_pixels[:, 0] = np.clip(integer_pixels[:, 0], 0, target_intrinsics.width - 1)
    integer_pixels[:, 1] = np.clip(integer_pixels[:, 1], 0, target_intrinsics.height - 1)
    depth = target_points[point_indices, 2]
    order = np.argsort(depth)[::-1]
    for index in order:
        x, y = integer_pixels[index]
        z_buffer[y, x] = depth[index]
        mask[y, x] = True
    return mask, z_buffer


def mask_iou(first: Any, second: Any) -> float:
    first_mask = np.asarray(first, dtype=bool)
    second_mask = np.asarray(second, dtype=bool)
    if first_mask.shape != second_mask.shape:
        raise ValueError("masks must have the same shape")
    intersection = np.count_nonzero(first_mask & second_mask)
    union = np.count_nonzero(first_mask | second_mask)
    return intersection / union if union else 1.0


def reprojection_metrics(projected: Any, target: Any) -> dict[str, float | int]:
    projected_mask = np.asarray(projected, dtype=bool)
    target_mask = np.asarray(target, dtype=bool)
    if projected_mask.shape != target_mask.shape:
        raise ValueError("projected and target masks must have the same shape")
    py, px = np.nonzero(projected_mask)
    ty, tx = np.nonzero(target_mask)
    centroid_error = math.nan
    if len(px) and len(tx):
        centroid_error = float(math.hypot(px.mean() - tx.mean(), py.mean() - ty.mean()))
    return {
        "projected_pixels": len(px),
        "target_pixels": len(tx),
        "iou": mask_iou(projected_mask, target_mask),
        "centroid_error_px": centroid_error,
    }


def fuse_point_clouds(
    clouds: Iterable[Any],
    transforms_world_from_camera: Iterable[Any],
    voxel_size_m: float = 0.005,
) -> tuple[np.ndarray, np.ndarray]:
    if voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be positive")
    clouds = list(clouds)
    transforms = list(transforms_world_from_camera)
    if len(clouds) != len(transforms):
        raise ValueError("one transform is required for every point cloud")
    transformed = [
        transform_points(cloud, transform)
        for cloud, transform in zip(clouds, transforms)
        if len(np.asarray(cloud))
    ]
    if not transformed:
        return np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.int32)
    points = np.concatenate(transformed, axis=0)
    voxel = np.floor(points / voxel_size_m).astype(np.int64)
    _, inverse = np.unique(voxel, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    fused = np.column_stack(
        [np.bincount(inverse, weights=points[:, axis]) / counts for axis in range(3)]
    )
    return fused, counts.astype(np.int32)


def write_ply(path: Path, points: Any) -> None:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
    )
    rows = "".join(f"{x:.7f} {y:.7f} {z:.7f}\n" for x, y, z in points)
    path.write_text(header + rows, encoding="ascii")
