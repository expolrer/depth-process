#!/usr/bin/env python3
"""Sampled, model-free evidence suite for processed RGB-D sequences.

The script is intentionally CPU-only and single-process. It evaluates four
complementary evidence families without loading a learned model:

1. Spatial/temporal depth quality.
2. Natural-hole recovery against a bidirectional temporal proxy.
3. RGB-feature-anchored cross-view 3D consistency.
4. Geometry inside human-approved task-object masks.

Cross-view scores are explicitly a proxy when a common robot-frame camera
calibration is unavailable. RGB correspondences and raw sensor depth estimate
the per-frame rigid transform; processed depths are evaluated on held-out
correspondences under that fixed transform.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import numpy as np


CAMERAS = ("cam_h", "cam_l", "cam_r")
CAMERA_PAIRS = (("cam_h", "cam_l"), ("cam_h", "cam_r"), ("cam_l", "cam_r"))
METHODS = (
    "raw_aligned",
    "rgb_guided",
    "temporal_rgb_guided",
    "lingbot_v05",
    "depth_anything_v2_fused",
    "lingbot_v05_sensor_fused",
    "ai_consensus_fused",
    "cdm_camera_specific",
    "cdm_sensor_fused",
)
METHOD_LABELS = {
    "raw_aligned": "Raw aligned depth",
    "rgb_guided": "RGB guided",
    "temporal_rgb_guided": "Temporal RGB guided",
    "lingbot_v05": "LingBot-Depth v0.5",
    "depth_anything_v2_fused": "Depth Anything V2 fused",
    "lingbot_v05_sensor_fused": "LingBot v0.5 + sensor fusion",
    "ai_consensus_fused": "AI consensus fused",
    "cdm_camera_specific": "CDM D405 camera-specific",
    "cdm_sensor_fused": "CDM D405 + sensor fusion",
}
MIN_DEPTH_M = 0.08
MAX_DEPTH_M = 10.0
RAW_DEPTH_DIR = "depth_raw_mm"


class MetricLists:
    def __init__(self) -> None:
        self.values: dict[str, list[float]] = defaultdict(list)

    def add(self, **metrics: float | int | None) -> None:
        for name, value in metrics.items():
            if value is None:
                continue
            numeric = float(value)
            if np.isfinite(numeric):
                self.values[name].append(numeric)

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, values in sorted(self.values.items()):
            array = np.asarray(values, dtype=np.float64)
            result[name] = {
                "mean": float(array.mean()),
                "median": float(np.median(array)),
                "p10": float(np.quantile(array, 0.10)),
                "p90": float(np.quantile(array, 0.90)),
                "samples": int(array.size),
            }
        return result


class PixelAccumulator:
    def __init__(self) -> None:
        self.total = 0
        self.valid = 0
        self.error_count = 0
        self.absolute_error_sum = 0.0
        self.squared_error_sum = 0.0
        self.good_05 = 0
        self.good_10 = 0
        self.frames = 0

    def update(self, predictions: np.ndarray, target: np.ndarray, eligible: np.ndarray) -> None:
        count = int(np.count_nonzero(eligible))
        if count == 0:
            return
        self.frames += 1
        self.total += count
        prediction_valid = eligible & valid_depth(predictions)
        self.valid += int(np.count_nonzero(prediction_valid))
        if not np.any(prediction_valid):
            return
        error = np.abs(predictions[prediction_valid] - target[prediction_valid]).astype(np.float64)
        self.error_count += int(error.size)
        self.absolute_error_sum += float(error.sum())
        self.squared_error_sum += float(np.square(error).sum())
        self.good_05 += int(np.count_nonzero(error <= 0.05))
        self.good_10 += int(np.count_nonzero(error <= 0.10))

    def summary(self) -> dict[str, float | int | None]:
        return {
            "frames_with_evidence": self.frames,
            "evaluable_hole_pixels": self.total,
            "recovered_pixels": self.valid,
            "recovery_coverage": self.valid / self.total if self.total else None,
            "mae_m": self.absolute_error_sum / self.error_count if self.error_count else None,
            "rmse_m": math.sqrt(self.squared_error_sum / self.error_count) if self.error_count else None,
            "within_5cm_fraction": self.good_05 / self.error_count if self.error_count else None,
            "within_10cm_fraction": self.good_10 / self.error_count if self.error_count else None,
        }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sequence_from_report(row: dict[str, Any]) -> str:
    return Path(row["manifest"]).parent.parent.name


def evenly_spaced_frames(frame_count: int, count: int) -> list[int]:
    if frame_count <= 2:
        return list(range(frame_count))
    count = min(count, frame_count - 2)
    return sorted(set(int(value) for value in np.linspace(1, frame_count - 2, count)))


def valid_depth(depth: np.ndarray) -> np.ndarray:
    return np.isfinite(depth) & (depth >= MIN_DEPTH_M) & (depth <= MAX_DEPTH_M)


def load_depth(path: Path, scale: float = 1.0) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 3:
        image = image[..., 0]
    if scale != 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    depth = image.astype(np.float32) * 0.001
    depth[~valid_depth(depth)] = 0.0
    return depth


def load_rgb(path: Path, scale: float = 1.0) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    if scale != 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image


def depth_path(root: Path, sequence: str, camera: str, method: str, frame: int) -> Path:
    name = f"{frame:06d}.png"
    if method == "raw_aligned":
        return root / "outputs" / "extracted" / sequence / camera / RAW_DEPTH_DIR / name
    return root / "outputs" / "processed" / sequence / camera / method / name


def rgb_path(root: Path, sequence: str, camera: str, frame: int) -> Path:
    return root / "outputs" / "extracted" / sequence / camera / "rgb_raw" / f"{frame:06d}.jpg"


def robust_median(values: np.ndarray) -> float | None:
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else None


def spatial_quality(rgb: np.ndarray, depth: np.ndarray, raw: np.ndarray) -> dict[str, float | None]:
    valid = valid_depth(depth)
    raw_valid = valid_depth(raw)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    rgb_gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    rgb_gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    rgb_gradient = cv2.magnitude(rgb_gx, rgb_gy)
    flat_threshold = float(np.quantile(rgb_gradient, 0.55))
    flat = rgb_gradient <= flat_threshold

    median = cv2.medianBlur(depth, 5)
    stable = flat & valid & valid_depth(median)
    residual = np.abs(depth - median)
    roughness = robust_median(residual[stable])
    spike_rate = float(np.mean(residual[stable] > 0.08)) if np.any(stable) else None

    safe_depth = depth.copy()
    safe_depth[~valid] = 0.0
    depth_gx = cv2.Sobel(safe_depth, cv2.CV_32F, 1, 0, ksize=3)
    depth_gy = cv2.Sobel(safe_depth, cv2.CV_32F, 0, 1, ksize=3)
    depth_gradient = cv2.magnitude(depth_gx, depth_gy)
    gradient_values = depth_gradient[valid]
    threshold = max(0.04, float(np.quantile(gradient_values, 0.92))) if gradient_values.size else 0.04
    depth_edges = (depth_gradient >= threshold) & valid
    rgb_edges = cv2.Canny(gray, 70, 160) > 0
    kernel = np.ones((3, 3), np.uint8)
    depth_dilated = cv2.dilate(depth_edges.astype(np.uint8), kernel) > 0
    rgb_dilated = cv2.dilate(rgb_edges.astype(np.uint8), kernel) > 0
    precision = float(np.mean(rgb_dilated[depth_edges])) if np.any(depth_edges) else None
    recall = float(np.mean(depth_dilated[rgb_edges])) if np.any(rgb_edges) else None
    edge_f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    overlap = valid & raw_valid
    sensor_errors = np.abs(depth[overlap] - raw[overlap]) if np.any(overlap) else np.empty(0)
    sensor_median = float(np.median(sensor_errors)) if sensor_errors.size else None
    sensor_mae = float(np.mean(sensor_errors)) if sensor_errors.size else None
    sensor_rmse = float(np.sqrt(np.mean(np.square(sensor_errors)))) if sensor_errors.size else None
    return {
        "valid_fraction": float(valid.mean()),
        "flat_region_roughness_m": roughness,
        "isolated_spike_rate": spike_rate,
        "rgb_depth_edge_f1": edge_f1,
        "sensor_preservation_median_ae_m": sensor_median,
        "sensor_overlap_mae_m": sensor_mae,
        "sensor_preservation_rmse_m": sensor_rmse,
    }


def warp_source_to_target(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    source_depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_gray = cv2.cvtColor(source_rgb, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_rgb, cv2.COLOR_BGR2GRAY)
    target_to_source = cv2.calcOpticalFlowFarneback(
        target_gray,
        source_gray,
        None,
        0.5,
        3,
        19,
        3,
        5,
        1.2,
        0,
    )
    height, width = target_gray.shape
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x = grid_x + target_to_source[..., 0]
    map_y = grid_y + target_to_source[..., 1]
    warped_depth = cv2.remap(source_depth, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    warped_gray = cv2.remap(source_gray, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    photometric_error = cv2.absdiff(target_gray, warped_gray).astype(np.float32)
    flow_magnitude = cv2.magnitude(target_to_source[..., 0], target_to_source[..., 1])
    return warped_depth, photometric_error, flow_magnitude


def temporal_residual(
    previous_rgb: np.ndarray,
    current_rgb: np.ndarray,
    previous_depth: np.ndarray,
    current_depth: np.ndarray,
) -> tuple[float | None, float | None, float]:
    warped, photometric, flow = warp_source_to_target(previous_rgb, current_rgb, previous_depth)
    eligible = valid_depth(warped) & valid_depth(current_depth) & (photometric <= 24.0) & (flow <= 80.0)
    if not np.any(eligible):
        return None, None, 0.0
    errors = np.abs(warped[eligible] - current_depth[eligible])
    return float(np.median(errors)), float(np.quantile(errors, 0.90)), float(eligible.mean())


def natural_hole_proxy(
    previous_rgb: np.ndarray,
    current_rgb: np.ndarray,
    next_rgb: np.ndarray,
    previous_raw: np.ndarray,
    current_raw: np.ndarray,
    next_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    previous_warped, previous_photo, previous_flow = warp_source_to_target(
        previous_rgb, current_rgb, previous_raw
    )
    next_warped, next_photo, next_flow = warp_source_to_target(next_rgb, current_rgb, next_raw)
    current_holes = ~valid_depth(current_raw)
    temporal_agreement = np.abs(previous_warped - next_warped) <= 0.06
    eligible = (
        current_holes
        & valid_depth(previous_warped)
        & valid_depth(next_warped)
        & temporal_agreement
        & (previous_photo <= 22.0)
        & (next_photo <= 22.0)
        & (previous_flow <= 80.0)
        & (next_flow <= 80.0)
    )
    proxy = (previous_warped + next_warped) * 0.5
    proxy[~eligible] = 0.0
    return proxy, eligible


def intrinsics_matrix(info: dict[str, Any], scale: float) -> np.ndarray:
    matrix = np.asarray(info["rgb_camera_info"]["K"], dtype=np.float64).copy()
    matrix[0, :] *= scale
    matrix[1, :] *= scale
    matrix[2, 2] = 1.0
    return matrix


def backproject(points_xy: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.rint(points_xy[:, 0]).astype(np.int32)
    y = np.rint(points_xy[:, 1]).astype(np.int32)
    inside = (x >= 0) & (x < depth.shape[1]) & (y >= 0) & (y < depth.shape[0])
    z = np.zeros(x.shape[0], dtype=np.float64)
    z[inside] = depth[y[inside], x[inside]]
    valid = inside & np.isfinite(z) & (z >= MIN_DEPTH_M) & (z <= MAX_DEPTH_M)
    points = np.zeros((x.shape[0], 3), dtype=np.float64)
    points[:, 2] = z
    points[:, 0] = (points_xy[:, 0] - intrinsics[0, 2]) / intrinsics[0, 0] * z
    points[:, 1] = (points_xy[:, 1] - intrinsics[1, 2]) / intrinsics[1, 1] * z
    return points, valid


def rigid_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def dominant_plane_metrics(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    rng: np.random.Generator,
    stride: int = 4,
    iterations: int = 55,
    threshold_m: float = 0.02,
) -> dict[str, float] | None:
    sampled = depth[::stride, ::stride]
    pixel_y, pixel_x = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    valid = valid_depth(sampled) & (sampled <= 4.0)
    if np.count_nonzero(valid) < 120:
        return None
    z = sampled[valid].astype(np.float64)
    x = (pixel_x[valid] - intrinsics[0, 2]) / intrinsics[0, 0] * z
    y = (pixel_y[valid] - intrinsics[1, 2]) / intrinsics[1, 1] * z
    points = np.column_stack((x, y, z))
    if points.shape[0] > 2800:
        points = points[rng.choice(points.shape[0], size=2800, replace=False)]
    best = np.zeros(points.shape[0], dtype=bool)
    for _ in range(iterations):
        sample = points[rng.choice(points.shape[0], size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        offset = -float(normal @ sample[0])
        inliers = np.abs(points @ normal + offset) <= threshold_m
        if np.count_nonzero(inliers) > np.count_nonzero(best):
            best = inliers
    if np.count_nonzero(best) < 100:
        return None
    plane_points = points[best]
    center = plane_points.mean(axis=0)
    _, _, vt = np.linalg.svd(plane_points - center, full_matrices=False)
    normal = vt[-1]
    residual = np.abs((plane_points - center) @ normal)
    return {
        "plane_rmse_m": float(np.sqrt(np.mean(np.square(residual)))),
        "plane_median_residual_m": float(np.median(residual)),
        "plane_inlier_fraction": float(np.mean(best)),
    }


def ransac_rigid(
    source: np.ndarray,
    target: np.ndarray,
    rng: np.random.Generator,
    threshold_m: float = 0.06,
    iterations: int = 220,
) -> tuple[np.ndarray | None, np.ndarray]:
    if source.shape[0] < 8:
        return None, np.zeros(source.shape[0], dtype=bool)
    best = np.zeros(source.shape[0], dtype=bool)
    for _ in range(iterations):
        sample = rng.choice(source.shape[0], size=3, replace=False)
        try:
            transform = rigid_transform(source[sample], target[sample])
        except np.linalg.LinAlgError:
            continue
        residual = np.linalg.norm(transform_points(source, transform) - target, axis=1)
        inliers = residual <= threshold_m
        if np.count_nonzero(inliers) > np.count_nonzero(best):
            best = inliers
    if np.count_nonzero(best) < 8:
        return None, best
    return rigid_transform(source[best], target[best]), best


def feature_correspondences(
    image_a: np.ndarray, image_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    sift = cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.025)
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
    keypoints_a, descriptors_a = sift.detectAndCompute(gray_a, None)
    keypoints_b, descriptors_b = sift.detectAndCompute(gray_b, None)
    diagnostics = {"keypoints_a": len(keypoints_a), "keypoints_b": len(keypoints_b), "ratio_matches": 0, "fundamental_inliers": 0}
    if descriptors_a is None or descriptors_b is None or len(keypoints_a) < 12 or len(keypoints_b) < 12:
        return np.empty((0, 2)), np.empty((0, 2)), diagnostics
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    reverse = matcher.knnMatch(descriptors_b, descriptors_a, k=2)
    reverse_good = {match.queryIdx: match.trainIdx for match, second in reverse if match.distance < 0.72 * second.distance}
    good = [
        match
        for match, second in forward
        if match.distance < 0.72 * second.distance and reverse_good.get(match.trainIdx) == match.queryIdx
    ]
    diagnostics["ratio_matches"] = len(good)
    if len(good) < 12:
        return np.empty((0, 2)), np.empty((0, 2)), diagnostics
    points_a = np.float32([keypoints_a[match.queryIdx].pt for match in good])
    points_b = np.float32([keypoints_b[match.trainIdx].pt for match in good])
    _, mask = cv2.findFundamentalMat(points_a, points_b, cv2.FM_RANSAC, 1.5, 0.995)
    if mask is None:
        return np.empty((0, 2)), np.empty((0, 2)), diagnostics
    selected = mask.ravel().astype(bool)
    diagnostics["fundamental_inliers"] = int(np.count_nonzero(selected))
    return points_a[selected], points_b[selected], diagnostics


def event_balanced(rows: list[dict[str, Any]], metric_names: Iterable[str]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["event_id"], row["role"])].append(row)
    group_means = []
    for members in grouped.values():
        record = {}
        for metric in metric_names:
            values = [float(member[metric]) for member in members if member.get(metric) is not None and np.isfinite(member[metric])]
            record[metric] = float(np.mean(values)) if values else None
        group_means.append(record)
    output: dict[str, Any] = {"event_views": len(group_means)}
    for metric in metric_names:
        values = [record[metric] for record in group_means if record[metric] is not None]
        output[metric] = float(np.mean(values)) if values else None
    return output


def roi_metrics(depth: np.ndarray, raw: np.ndarray, mask: np.ndarray) -> dict[str, float | None]:
    selected = mask > 0
    if not np.any(selected):
        return {}
    valid = valid_depth(depth)
    raw_valid = valid_depth(raw)
    roi_valid = selected & valid
    eroded = cv2.erode(selected.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    dilated = cv2.dilate(selected.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    inner_boundary = selected & ~eroded
    outer_ring = dilated & ~selected
    median_filtered = cv2.medianBlur(depth, 5)
    rough_mask = eroded & valid & valid_depth(median_filtered)
    roughness = robust_median(np.abs(depth[rough_mask] - median_filtered[rough_mask])) if np.any(rough_mask) else None
    values = depth[roi_valid]
    span = float(np.quantile(values, 0.90) - np.quantile(values, 0.10)) if values.size >= 10 else None
    inner_values = depth[inner_boundary & valid]
    outer_values = depth[outer_ring & valid]
    boundary_contrast = (
        float(abs(np.median(inner_values) - np.median(outer_values)))
        if inner_values.size >= 5 and outer_values.size >= 5
        else None
    )
    raw_holes = selected & ~raw_valid
    overlap = selected & raw_valid & valid
    return {
        "roi_valid_fraction": float(np.mean(valid[selected])),
        "roi_raw_hole_fill_fraction": float(np.mean(valid[raw_holes])) if np.any(raw_holes) else None,
        "roi_median_depth_m": float(np.median(values)) if values.size else None,
        "roi_p90_p10_span_m": span,
        "roi_surface_roughness_m": roughness,
        "roi_boundary_valid_fraction": float(np.mean(valid[inner_boundary])) if np.any(inner_boundary) else None,
        "roi_boundary_contrast_m": boundary_contrast,
        "roi_sensor_preservation_median_ae_m": robust_median(np.abs(depth[overlap] - raw[overlap])) if np.any(overlap) else None,
    }


def colorize_depth(depth: np.ndarray, minimum: float = 0.2, maximum: float = 4.0) -> np.ndarray:
    valid = valid_depth(depth)
    normalized = np.clip((depth - minimum) / (maximum - minimum), 0.0, 1.0)
    image = cv2.applyColorMap(np.rint((1.0 - normalized) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    image[~valid] = 0
    return image


def label_panel(image: np.ndarray, label: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 32), (15, 18, 21), -1)
    cv2.putText(output, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def write_representative(
    root: Path,
    output_dir: Path,
    row: dict[str, Any],
    methods: tuple[str, ...],
) -> str:
    sequence = row["sequence"]
    camera = row["camera"]
    frame = int(row["frame_index"])
    rgb = load_rgb(rgb_path(root, sequence, camera, frame), scale=0.5)
    mask = cv2.imread(str(root / row["mask_path"]), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(root / row["mask_path"])
    mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    overlay = rgb.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 220, 255), 2)
    panels = [label_panel(overlay, "RGB + approved target ROI")]
    for method in methods:
        depth = load_depth(depth_path(root, sequence, camera, method, frame), scale=0.5)
        panels.append(label_panel(colorize_depth(depth), METHOD_LABELS[method]))
    canvas = cv2.hconcat(panels)
    slug = f"{sequence}__{row['event_index']:02d}_{row['role']}_{camera}_{frame:06d}.jpg"
    target = output_dir / "representatives" / slug
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), canvas, [cv2.IMWRITE_JPEG_QUALITY, 91])
    return str(target.relative_to(output_dir))


def rank_methods(records: dict[str, dict[str, Any]], metric: str, higher_is_better: bool) -> list[str]:
    available = [(method, record.get(metric)) for method, record in records.items() if record.get(metric) is not None]
    return [method for method, _ in sorted(available, key=lambda item: item[1], reverse=higher_is_better)]


def main() -> None:
    global MIN_DEPTH_M, MAX_DEPTH_M, RAW_DEPTH_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/ssd/hhw/depth-processing"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--spatial-samples-per-view", type=int, default=12)
    parser.add_argument("--temporal-samples-per-view", type=int, default=6)
    parser.add_argument("--geometry-samples-per-sequence", type=int, default=6)
    parser.add_argument("--roi-stride", type=int, default=20)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--min-depth-m", type=float, default=MIN_DEPTH_M)
    parser.add_argument("--max-depth-m", type=float, default=MAX_DEPTH_M)
    parser.add_argument(
        "--raw-depth-dir",
        choices=("depth_raw_mm", "depth_aligned_rgb_mm"),
        default=RAW_DEPTH_DIR,
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS[:7]))
    parser.add_argument("--cameras", nargs="+", choices=CAMERAS, default=list(CAMERAS))
    args = parser.parse_args()
    if args.min_depth_m <= 0 or args.max_depth_m <= args.min_depth_m:
        parser.error("Expected 0 < --min-depth-m < --max-depth-m")
    MIN_DEPTH_M = args.min_depth_m
    MAX_DEPTH_M = args.max_depth_m
    RAW_DEPTH_DIR = args.raw_depth_dir
    args.root = args.root.resolve()
    args.output_dir = (args.output_dir or args.root / "outputs" / "depth_quality_evidence_v1").resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    methods = tuple(dict.fromkeys(args.methods))
    cameras = tuple(dict.fromkeys(args.cameras))
    if len(cameras) < 2:
        raise ValueError("At least two cameras are required for multiview geometry")
    camera_pairs = tuple(combinations(cameras, 2))

    extraction_report = json.loads(
        (args.root / "outputs" / "extracted" / "extraction_report.json").read_text(encoding="utf-8")
    )
    metadata = {(sequence_from_report(row), row["camera"]): row for row in extraction_report}
    sequences = sorted({sequence for sequence, _ in metadata})
    spatial = {method: MetricLists() for method in methods}
    temporal = {method: MetricLists() for method in methods}
    recovery = {method: PixelAccumulator() for method in methods}
    sampled_spatial_frames = 0
    sampled_temporal_frames = 0

    for sequence in sequences:
        for camera in cameras:
            info = metadata[(sequence, camera)]
            frame_count = int(info["paired_frames"])
            spatial_frames = evenly_spaced_frames(frame_count, args.spatial_samples_per_view)
            temporal_frames = evenly_spaced_frames(frame_count - 2, args.temporal_samples_per_view)
            for frame in spatial_frames:
                rgb = load_rgb(rgb_path(args.root, sequence, camera, frame), args.scale)
                raw = load_depth(depth_path(args.root, sequence, camera, "raw_aligned", frame), args.scale)
                for method in methods:
                    depth = raw if method == "raw_aligned" else load_depth(
                        depth_path(args.root, sequence, camera, method, frame), args.scale
                    )
                    spatial[method].add(**spatial_quality(rgb, depth, raw))
                sampled_spatial_frames += 1

            for frame in temporal_frames:
                frame = min(max(frame, 1), frame_count - 2)
                previous_rgb = load_rgb(rgb_path(args.root, sequence, camera, frame - 1), args.scale)
                current_rgb = load_rgb(rgb_path(args.root, sequence, camera, frame), args.scale)
                next_rgb = load_rgb(rgb_path(args.root, sequence, camera, frame + 1), args.scale)
                raw_previous = load_depth(
                    depth_path(args.root, sequence, camera, "raw_aligned", frame - 1), args.scale
                )
                raw_current = load_depth(
                    depth_path(args.root, sequence, camera, "raw_aligned", frame), args.scale
                )
                raw_next = load_depth(
                    depth_path(args.root, sequence, camera, "raw_aligned", frame + 1), args.scale
                )
                proxy, eligible = natural_hole_proxy(
                    previous_rgb,
                    current_rgb,
                    next_rgb,
                    raw_previous,
                    raw_current,
                    raw_next,
                )
                for method in methods:
                    previous = raw_previous if method == "raw_aligned" else load_depth(
                        depth_path(args.root, sequence, camera, method, frame - 1), args.scale
                    )
                    current = raw_current if method == "raw_aligned" else load_depth(
                        depth_path(args.root, sequence, camera, method, frame), args.scale
                    )
                    median_error, p90_error, overlap = temporal_residual(
                        previous_rgb, current_rgb, previous, current
                    )
                    temporal[method].add(
                        flow_compensated_median_residual_m=median_error,
                        flow_compensated_p90_residual_m=p90_error,
                        temporal_evaluable_fraction=overlap,
                    )
                    recovery[method].update(current, proxy, eligible)
                sampled_temporal_frames += 1

    # Parse camera manifests once for timestamp-based cross-view synchronization.
    manifests: dict[tuple[str, str], list[dict[str, Any]]] = {}
    timestamps: dict[tuple[str, str], np.ndarray] = {}
    for key, info in metadata.items():
        rows = read_jsonl(Path(info["manifest"]))
        manifests[key] = rows
        timestamps[key] = np.asarray([row["rgb_bag_timestamp_ns"] for row in rows], dtype=np.int64)

    geometry_rows: list[dict[str, Any]] = []
    planarity_rows: list[dict[str, Any]] = []
    geometry_diagnostics = defaultdict(int)
    for sequence in sequences:
        reference_camera = cameras[0]
        reference_count = int(metadata[(sequence, reference_camera)]["paired_frames"])
        for reference_frame in evenly_spaced_frames(
            reference_count, args.geometry_samples_per_sequence
        ):
            reference_timestamp = int(timestamps[(sequence, reference_camera)][reference_frame])
            synchronized = {reference_camera: reference_frame}
            sync_delta_ms = {reference_camera: 0.0}
            for camera in cameras[1:]:
                values = timestamps[(sequence, camera)]
                index = int(np.searchsorted(values, reference_timestamp))
                candidates = [value for value in (index - 1, index) if 0 <= value < values.size]
                selected = min(
                    candidates,
                    key=lambda value: abs(int(values[value]) - reference_timestamp),
                )
                synchronized[camera] = selected
                sync_delta_ms[camera] = abs(int(values[selected]) - reference_timestamp) / 1e6
            if max(sync_delta_ms.values()) > 50.0:
                geometry_diagnostics["sync_rejected"] += 1
                continue

            rgb_by_camera = {
                camera: load_rgb(rgb_path(args.root, sequence, camera, synchronized[camera]), args.scale)
                for camera in cameras
            }
            depths_by_camera = {
                method: {
                    camera: load_depth(
                        depth_path(args.root, sequence, camera, method, synchronized[camera]), args.scale
                    )
                    for camera in cameras
                }
                for method in methods
            }
            intrinsics = {
                camera: intrinsics_matrix(metadata[(sequence, camera)], args.scale)
                for camera in cameras
            }
            for method in methods:
                per_camera = {
                    camera: dominant_plane_metrics(
                        depths_by_camera[method][camera], intrinsics[camera], rng
                    )
                    for camera in cameras
                }
                accepted = [metrics for metrics in per_camera.values() if metrics is not None]
                if len(accepted) < 2:
                    geometry_diagnostics["planarity_rejected"] += 1
                    continue
                rmse_values = np.asarray(
                    [metrics["plane_rmse_m"] for metrics in accepted], dtype=np.float64
                )
                planarity_rows.append(
                    {
                        "sequence": sequence,
                        "reference_frame": reference_frame,
                        "method": method,
                        "camera_count": len(accepted),
                        "mean_plane_rmse_m": float(rmse_values.mean()),
                        "three_view_plane_rmse_std_m": float(rmse_values.std()),
                        "mean_plane_inlier_fraction": float(
                            np.mean([metrics["plane_inlier_fraction"] for metrics in accepted])
                        ),
                        "per_camera": per_camera,
                    }
                )
                geometry_diagnostics["planarity_accepted"] += 1
            for camera_a, camera_b in camera_pairs:
                points_a, points_b, feature_diag = feature_correspondences(
                    rgb_by_camera[camera_a], rgb_by_camera[camera_b]
                )
                geometry_diagnostics["pair_attempts"] += 1
                geometry_diagnostics["fundamental_inliers"] += feature_diag["fundamental_inliers"]
                if points_a.shape[0] < 12:
                    geometry_diagnostics["feature_rejected"] += 1
                    continue
                raw_points_a, raw_valid_a = backproject(
                    points_a, depths_by_camera["raw_aligned"][camera_a], intrinsics[camera_a]
                )
                raw_points_b, raw_valid_b = backproject(
                    points_b, depths_by_camera["raw_aligned"][camera_b], intrinsics[camera_b]
                )
                raw_valid = raw_valid_a & raw_valid_b
                if np.count_nonzero(raw_valid) < 10:
                    geometry_diagnostics["raw_depth_rejected"] += 1
                    continue
                transform, raw_inliers = ransac_rigid(
                    raw_points_a[raw_valid], raw_points_b[raw_valid], rng
                )
                if transform is None or np.count_nonzero(raw_inliers) < 8:
                    geometry_diagnostics["rigid_fit_rejected"] += 1
                    continue
                geometry_diagnostics["accepted_pairs"] += 1
                for method in methods:
                    method_points_a, method_valid_a = backproject(
                        points_a, depths_by_camera[method][camera_a], intrinsics[camera_a]
                    )
                    method_points_b, method_valid_b = backproject(
                        points_b, depths_by_camera[method][camera_b], intrinsics[camera_b]
                    )
                    method_valid = method_valid_a & method_valid_b
                    if np.count_nonzero(method_valid) < 5:
                        continue
                    residual = np.linalg.norm(
                        transform_points(method_points_a[method_valid], transform)
                        - method_points_b[method_valid],
                        axis=1,
                    )
                    trimmed = residual[residual <= 0.50]
                    if trimmed.size < 5:
                        continue
                    geometry_rows.append(
                        {
                            "sequence": sequence,
                            "reference_frame": reference_frame,
                            "camera_pair": f"{camera_a}-{camera_b}",
                            "method": method,
                            "sync_delta_ms": max(sync_delta_ms[camera_a], sync_delta_ms[camera_b]),
                            "rgb_correspondences": int(points_a.shape[0]),
                            "valid_3d_correspondences": int(np.count_nonzero(method_valid)),
                            "valid_correspondence_fraction": float(np.mean(method_valid)),
                            "median_residual_m": float(np.median(trimmed)),
                            "p90_residual_m": float(np.quantile(trimmed, 0.90)),
                        }
                    )

    geometry_by_method: dict[str, dict[str, Any]] = {}
    for method in methods:
        members = [row for row in geometry_rows if row["method"] == method]
        metrics = MetricLists()
        for row in members:
            metrics.add(
                valid_correspondence_fraction=row["valid_correspondence_fraction"],
                median_residual_m=row["median_residual_m"],
                p90_residual_m=row["p90_residual_m"],
            )
        geometry_by_method[method] = {
            "accepted_pair_frames": len(members),
            **{name: value["mean"] for name, value in metrics.summary().items()},
            "distribution": metrics.summary(),
        }

    planarity_by_method: dict[str, dict[str, Any]] = {}
    for method in methods:
        members = [row for row in planarity_rows if row["method"] == method]
        metrics = MetricLists()
        for row in members:
            metrics.add(
                mean_plane_rmse_m=row["mean_plane_rmse_m"],
                three_view_plane_rmse_std_m=row["three_view_plane_rmse_std_m"],
                mean_plane_inlier_fraction=row["mean_plane_inlier_fraction"],
            )
        distribution = metrics.summary()
        planarity_by_method[method] = {
            "synchronized_frames": len(members),
            **{name: value["mean"] for name, value in distribution.items()},
            "distribution": distribution,
        }

    tracks = read_jsonl(args.root / "outputs" / "target_tracks_required" / "track_index.jsonl")
    selected_tracks: list[dict[str, Any]] = []
    by_event_role: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in tracks:
        if row["camera"] not in cameras:
            continue
        by_event_role[(row["event_id"], row["role"])].append(row)
    for members in by_event_role.values():
        members.sort(key=lambda row: int(row["frame_index"]))
        mandatory = {
            int(members[0]["frame_index"]),
            int(members[-1]["frame_index"]),
            int(members[0]["anchor_frame_index"]),
            int(members[0]["release_frame_index"]),
        }
        start = int(members[0]["frame_index"])
        for row in members:
            frame = int(row["frame_index"])
            if frame in mandatory or (frame - start) % args.roi_stride == 0:
                selected_tracks.append(row)

    roi_rows: list[dict[str, Any]] = []
    for row in selected_tracks:
        sequence = row["sequence"]
        camera = row["camera"]
        frame = int(row["frame_index"])
        mask = cv2.imread(str(args.root / row["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if mask is None or not np.any(mask):
            continue
        raw = load_depth(depth_path(args.root, sequence, camera, "raw_aligned", frame))
        for method in methods:
            depth = raw if method == "raw_aligned" else load_depth(
                depth_path(args.root, sequence, camera, method, frame)
            )
            metrics = roi_metrics(depth, raw, mask)
            if not metrics:
                continue
            roi_rows.append(
                {
                    "method": method,
                    "event_id": row["event_id"],
                    "sequence": sequence,
                    "role": row["role"],
                    "camera": camera,
                    "frame_index": frame,
                    **metrics,
                }
            )

    roi_metric_names = (
        "roi_valid_fraction",
        "roi_raw_hole_fill_fraction",
        "roi_median_depth_m",
        "roi_p90_p10_span_m",
        "roi_surface_roughness_m",
        "roi_boundary_valid_fraction",
        "roi_boundary_contrast_m",
        "roi_sensor_preservation_median_ae_m",
    )
    roi_by_method = {
        method: event_balanced([row for row in roi_rows if row["method"] == method], roi_metric_names)
        for method in methods
    }
    for method in methods:
        for role in ("head", "wrist"):
            roi_by_method[method][role] = event_balanced(
                [row for row in roi_rows if row["method"] == method and row["role"] == role],
                roi_metric_names,
            )

    representatives = []
    seen_sequences = set()
    for row in sorted(selected_tracks, key=lambda item: (item["sequence"], item["event_index"], item["role"])):
        if row["sequence"] in seen_sequences or row["role"] != "wrist":
            continue
        if int(row["frame_index"]) != int(row["anchor_frame_index"]):
            continue
        representatives.append(
            {
                "sequence": row["sequence"],
                "event_id": row["event_id"],
                "camera": row["camera"],
                "frame_index": int(row["frame_index"]),
                "image": write_representative(args.root, args.output_dir, row, methods),
            }
        )
        seen_sequences.add(row["sequence"])

    spatial_summary = {
        method: {name: values["mean"] for name, values in spatial[method].summary().items()}
        | {"distribution": spatial[method].summary()}
        for method in methods
    }
    temporal_summary = {
        method: {name: values["mean"] for name, values in temporal[method].summary().items()}
        | {"distribution": temporal[method].summary()}
        for method in methods
    }
    recovery_summary = {method: recovery[method].summary() for method in methods}

    report = {
        "schema": "depth_quality_evidence_v1",
        "generated_at_unix": time.time(),
        "methods": [{"id": method, "label": METHOD_LABELS[method]} for method in methods],
        "sampling": {
            "sequences": len(sequences),
            "camera_views": len(sequences) * len(cameras),
            "cameras": list(cameras),
            "spatial_frames": sampled_spatial_frames,
            "temporal_triplets": sampled_temporal_frames,
            "roi_frames": len(selected_tracks),
            "scale": args.scale,
            "seed": args.seed,
            "valid_depth_range_m": [MIN_DEPTH_M, MAX_DEPTH_M],
            "raw_depth_dir": RAW_DEPTH_DIR,
        },
        "resource_guard": {
            "gpu_used": False,
            "processes": 1,
            "recommended_launcher": "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 ionice -c3 nice -n 19",
        },
        "no_model_quality": {
            "description": "Coverage, flat-region local residual, spike rate, RGB/depth edge alignment, sensor preservation, and RGB-flow-compensated temporal residual.",
            "spatial_by_method": spatial_summary,
            "temporal_by_method": temporal_summary,
            "rankings": {
                "coverage": rank_methods(spatial_summary, "valid_fraction", True),
                "flat_region_smoothness": rank_methods(spatial_summary, "flat_region_roughness_m", False),
                "edge_alignment": rank_methods(spatial_summary, "rgb_depth_edge_f1", True),
                "temporal_stability": rank_methods(
                    temporal_summary, "flow_compensated_median_residual_m", False
                ),
            },
        },
        "occlusion_recovery": {
            "description": "Natural raw-depth holes evaluated only where previous and next raw frames agree after RGB optical-flow warping.",
            "by_method": recovery_summary,
            "rankings": {
                "within_5cm": rank_methods(recovery_summary, "within_5cm_fraction", True),
                "recovery_coverage": rank_methods(recovery_summary, "recovery_coverage", True),
                "mae": rank_methods(recovery_summary, "mae_m", False),
            },
        },
        "multiview_geometry": {
            "strict_calibrated_reprojection": False,
            "primary_mode": "dominant-surface planarity and residual spread over synchronized head/left-wrist/right-wrist depth",
            "secondary_mode": "synchronized RGB feature correspondences + raw-depth per-frame rigid fit proxy when enough overlap exists",
            "limitation": "The bags expose per-camera internal TF but no common head/left-wrist/right-wrist robot-frame extrinsics. Planarity is rotation-invariant; feature scores are proxy residuals, not calibrated camera reprojection error.",
            "diagnostics": dict(geometry_diagnostics),
            "planarity_by_method": planarity_by_method,
            "ranking_by_plane_rmse": rank_methods(
                planarity_by_method, "mean_plane_rmse_m", False
            ),
            "ranking_by_three_view_spread": rank_methods(
                planarity_by_method, "three_view_plane_rmse_std_m", False
            ),
            "feature_rigid_fit_by_method": geometry_by_method,
            "feature_ranking_by_median_residual": rank_methods(
                geometry_by_method, "median_residual_m", False
            ),
        },
        "task_roi_geometry": {
            "description": "Event-balanced geometry inside final human-approved head and executing-wrist SAM2 masks.",
            "events": len({row["event_id"] for row in selected_tracks}),
            "event_views": len(by_event_role),
            "sampled_track_frames": len(selected_tracks),
            "by_method": roi_by_method,
            "rankings": {
                "roi_coverage": rank_methods(roi_by_method, "roi_valid_fraction", True),
                "roi_surface_smoothness": rank_methods(
                    roi_by_method, "roi_surface_roughness_m", False
                ),
                "roi_boundary_completeness": rank_methods(
                    roi_by_method, "roi_boundary_valid_fraction", True
                ),
            },
        },
        "representatives": representatives,
        "limitations": [
            "Raw sensor depth and temporal consensus are references, not laser-scanner ground truth.",
            "Lower roughness can indicate denoising or over-smoothing; interpret it with edge alignment and sensor preservation.",
            "Natural-hole recovery is evaluated only on temporally observable, bidirectionally consistent pixels.",
            "Cross-view planarity is rotation-invariant; calibrated reprojection remains unavailable because common robot-frame camera extrinsics are absent.",
            "No attention map or trained policy output is used in these four evidence families.",
        ],
        "elapsed_seconds": time.time() - started,
    }

    (args.output_dir / "depth_quality_evidence.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "method_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = [
            "method",
            "valid_fraction",
            "flat_region_roughness_m",
            "rgb_depth_edge_f1",
            "temporal_residual_m",
            "hole_recovery_coverage",
            "hole_recovery_mae_m",
            "hole_within_5cm_fraction",
            "three_view_plane_rmse_m",
            "three_view_plane_rmse_std_m",
            "feature_multiview_median_residual_m",
            "roi_valid_fraction",
            "roi_surface_roughness_m",
            "roi_boundary_valid_fraction",
            "roi_sensor_preservation_median_ae_m",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method in methods:
            writer.writerow(
                {
                    "method": method,
                    "valid_fraction": spatial_summary[method].get("valid_fraction"),
                    "flat_region_roughness_m": spatial_summary[method].get("flat_region_roughness_m"),
                    "rgb_depth_edge_f1": spatial_summary[method].get("rgb_depth_edge_f1"),
                    "temporal_residual_m": temporal_summary[method].get(
                        "flow_compensated_median_residual_m"
                    ),
                    "hole_recovery_coverage": recovery_summary[method].get("recovery_coverage"),
                    "hole_recovery_mae_m": recovery_summary[method].get("mae_m"),
                    "hole_within_5cm_fraction": recovery_summary[method].get(
                        "within_5cm_fraction"
                    ),
                    "three_view_plane_rmse_m": planarity_by_method[method].get(
                        "mean_plane_rmse_m"
                    ),
                    "three_view_plane_rmse_std_m": planarity_by_method[method].get(
                        "three_view_plane_rmse_std_m"
                    ),
                    "feature_multiview_median_residual_m": geometry_by_method[method].get(
                        "median_residual_m"
                    ),
                    "roi_valid_fraction": roi_by_method[method].get("roi_valid_fraction"),
                    "roi_surface_roughness_m": roi_by_method[method].get(
                        "roi_surface_roughness_m"
                    ),
                    "roi_boundary_valid_fraction": roi_by_method[method].get(
                        "roi_boundary_valid_fraction"
                    ),
                    "roi_sensor_preservation_median_ae_m": roi_by_method[method].get(
                        "roi_sensor_preservation_median_ae_m"
                    ),
                }
            )
    with (args.output_dir / "multiview_pair_rows.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        if geometry_rows:
            writer = csv.DictWriter(handle, fieldnames=list(geometry_rows[0]))
            writer.writeheader()
            writer.writerows(geometry_rows)
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "elapsed_seconds": report["elapsed_seconds"],
                "sampling": report["sampling"],
                "geometry_diagnostics": report["multiview_geometry"]["diagnostics"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
