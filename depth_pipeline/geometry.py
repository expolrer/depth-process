"""Camera geometry helpers for registering depth into the RGB image plane."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


def quaternion_xyzw_to_matrix(quaternion: list[float]) -> np.ndarray:
    x, y, z, w = (float(value) for value in quaternion)
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        raise ValueError("Zero-length quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_matrix(translation: list[float], quaternion: list[float]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_xyzw_to_matrix(quaternion)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def find_frame_transform(
    transforms: list[dict[str, Any]], source_frame: str, target_frame: str
) -> np.ndarray:
    """Return a homogeneous transform mapping source-frame points into target-frame coordinates."""
    if source_frame == target_frame:
        return np.eye(4, dtype=np.float64)
    graph: dict[str, list[tuple[str, np.ndarray]]] = {}
    for row in transforms:
        parent = str(row["parent"])
        child = str(row["child"])
        parent_from_child = transform_matrix(
            row["translation_xyz_m"], row["rotation_xyzw"]
        )
        graph.setdefault(child, []).append((parent, parent_from_child))
        graph.setdefault(parent, []).append((child, np.linalg.inv(parent_from_child)))

    queue = deque([(source_frame, np.eye(4, dtype=np.float64))])
    visited = {source_frame}
    while queue:
        frame, frame_from_source = queue.popleft()
        for neighbor, neighbor_from_frame in graph.get(frame, []):
            if neighbor in visited:
                continue
            neighbor_from_source = neighbor_from_frame @ frame_from_source
            if neighbor == target_frame:
                return neighbor_from_source
            visited.add(neighbor)
            queue.append((neighbor, neighbor_from_source))
    raise ValueError(f"No TF chain from {source_frame!r} to {target_frame!r}")


def distort_normalized_points(x: np.ndarray, y: np.ndarray, coefficients: list[float]):
    values = list(coefficients) + [0.0] * (8 - len(coefficients))
    k1, k2, p1, p2, k3, k4, k5, k6 = values[:8]
    radius2 = x * x + y * y
    radius4 = radius2 * radius2
    radius6 = radius4 * radius2
    numerator = 1.0 + k1 * radius2 + k2 * radius4 + k3 * radius6
    denominator = 1.0 + k4 * radius2 + k5 * radius4 + k6 * radius6
    radial = numerator / np.maximum(denominator, 1e-12)
    x_distorted = x * radial + 2.0 * p1 * x * y + p2 * (radius2 + 2.0 * x * x)
    y_distorted = y * radial + p1 * (radius2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return x_distorted, y_distorted


class DepthRegistrar:
    def __init__(
        self,
        depth_camera_info: dict[str, Any],
        color_camera_info: dict[str, Any],
        color_from_depth: np.ndarray,
    ) -> None:
        self.depth_k = np.asarray(depth_camera_info["K"], dtype=np.float64)
        self.color_k = np.asarray(color_camera_info["K"], dtype=np.float64)
        self.color_d = list(color_camera_info.get("D", []))
        self.color_from_depth = np.asarray(color_from_depth, dtype=np.float64)
        self.height = int(depth_camera_info["height"])
        self.width = int(depth_camera_info["width"])
        self.color_height = int(color_camera_info["height"])
        self.color_width = int(color_camera_info["width"])
        pixel_y, pixel_x = np.mgrid[: self.height, : self.width]
        self.ray_x = ((pixel_x - self.depth_k[0, 2]) / self.depth_k[0, 0]).astype(np.float32)
        self.ray_y = ((pixel_y - self.depth_k[1, 2]) / self.depth_k[1, 1]).astype(np.float32)

    def register(self, depth_mm: np.ndarray) -> np.ndarray:
        if depth_mm.shape != (self.height, self.width):
            raise ValueError(f"Unexpected depth shape: {depth_mm.shape}")
        valid = depth_mm > 0
        if not np.any(valid):
            return np.zeros((self.color_height, self.color_width), dtype=np.uint16)
        z_depth = depth_mm[valid].astype(np.float64) * 0.001
        points_depth = np.column_stack(
            (self.ray_x[valid] * z_depth, self.ray_y[valid] * z_depth, z_depth)
        )
        rotation = self.color_from_depth[:3, :3]
        translation = self.color_from_depth[:3, 3]
        points_color = points_depth @ rotation.T + translation
        z_color = points_color[:, 2]
        in_front = z_color > 0.01
        points_color = points_color[in_front]
        z_color = z_color[in_front]
        x = points_color[:, 0] / z_color
        y = points_color[:, 1] / z_color
        if self.color_d and np.any(np.asarray(self.color_d) != 0):
            x, y = distort_normalized_points(x, y, self.color_d)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        z_color = z_color[finite]
        pixel_x = np.rint(self.color_k[0, 0] * x + self.color_k[0, 2]).astype(np.int32)
        pixel_y = np.rint(self.color_k[1, 1] * y + self.color_k[1, 2]).astype(np.int32)
        inside = (
            (pixel_x >= 0)
            & (pixel_x < self.color_width)
            & (pixel_y >= 0)
            & (pixel_y < self.color_height)
        )
        flat_indices = pixel_y[inside] * self.color_width + pixel_x[inside]
        values_mm = np.rint(z_color[inside] * 1000.0).clip(1, 65535).astype(np.uint32)
        output = np.full(self.color_height * self.color_width, np.iinfo(np.uint32).max, np.uint32)
        np.minimum.at(output, flat_indices, values_mm)
        missing = output == np.iinfo(np.uint32).max
        output[missing] = 0
        return output.reshape(self.color_height, self.color_width).astype(np.uint16)
