from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import CameraIntrinsics, backproject_depth, transform_points, validate_transform


@dataclass(frozen=True)
class ObjectPose:
    frame_index: int
    world_from_object: tuple[tuple[float, float, float, float], ...]
    confidence: float
    source: str
    metric_translation: bool
    canonical_rotation: bool
    visible: bool = True

    def __post_init__(self) -> None:
        validate_transform(self.world_from_object, "world_from_object")
        if self.frame_index < 0 or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("pose frame and confidence are invalid")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def depth_proxy_pose(
    frame_index: int,
    depth_m: Any,
    mask: Any,
    intrinsics: CameraIntrinsics,
    world_from_camera: Any,
    minimum_points: int = 30,
) -> ObjectPose:
    """Estimate a metric centroid and PCA frame, explicitly not a canonical 6D pose."""
    points, _ = backproject_depth(depth_m, intrinsics, mask)
    if len(points) < minimum_points:
        raise ValueError(f"insufficient valid target depth: {len(points)} < {minimum_points}")
    world_points = transform_points(points, world_from_camera)
    center = np.median(world_points, axis=0)
    distance = np.linalg.norm(world_points - center, axis=1)
    keep = distance <= np.quantile(distance, 0.95)
    filtered = world_points[keep]
    center = filtered.mean(axis=0)
    _, singular, vectors = np.linalg.svd(filtered - center, full_matrices=False)
    rotation = vectors.T
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = center
    spread = float(singular[-1] / max(singular[0], 1e-9)) if len(singular) >= 3 else 0.0
    confidence = min(0.75, 0.4 + 0.25 * min(1.0, len(filtered) / 1000) + 0.1 * spread)
    return ObjectPose(
        frame_index=frame_index,
        world_from_object=tuple(tuple(float(value) for value in row) for row in transform),
        confidence=confidence,
        source="rgbd_centroid_pca_proxy",
        metric_translation=True,
        canonical_rotation=False,
    )


def load_foundationpose_trajectory(path: Path) -> list[ObjectPose]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("poses", payload) if isinstance(payload, dict) else payload
    output = []
    for row in rows:
        matrix = row.get("world_from_object", row.get("pose"))
        output.append(
            ObjectPose(
                frame_index=int(row["frame_index"]),
                world_from_object=tuple(tuple(float(value) for value in line) for line in matrix),
                confidence=float(row.get("confidence", 1.0)),
                source=str(row.get("source", "foundationpose")),
                metric_translation=True,
                canonical_rotation=True,
                visible=bool(row.get("visible", True)),
            )
        )
    return output


def trajectory_quality(poses: Iterable[ObjectPose], fps: float) -> dict[str, Any]:
    poses = sorted((pose for pose in poses if pose.visible), key=lambda item: item.frame_index)
    if fps <= 0:
        raise ValueError("fps must be positive")
    if len(poses) < 2:
        return {
            "pose_count": len(poses),
            "path_length_m": 0.0,
            "max_speed_m_s": 0.0,
            "jump_frames": [],
        }
    translations = np.asarray([pose.world_from_object for pose in poses])[:, :3, 3]
    frame_delta = np.diff([pose.frame_index for pose in poses]) / fps
    displacement = np.linalg.norm(np.diff(translations, axis=0), axis=1)
    speed = displacement / np.maximum(frame_delta, 1e-9)
    median = float(np.median(speed))
    mad = float(np.median(np.abs(speed - median)))
    threshold = median + 8.0 * max(1.4826 * mad, 1e-6)
    return {
        "pose_count": len(poses),
        "path_length_m": float(displacement.sum()),
        "max_speed_m_s": float(speed.max()),
        "jump_threshold_m_s": threshold,
        "jump_frames": [
            poses[index + 1].frame_index for index in np.flatnonzero(speed > threshold)
        ],
        "canonical_rotation": all(pose.canonical_rotation for pose in poses),
    }
