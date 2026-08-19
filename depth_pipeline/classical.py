"""Classical RGB-guided and temporally aligned depth filters."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RunningDepthStats:
    frames: int = 0
    pixels: int = 0
    valid_pixels: int = 0
    value_sum: float = 0.0
    value_squared_sum: float = 0.0
    minimum_m: float = float("inf")
    maximum_m: float = 0.0

    def update(self, depth_m: np.ndarray) -> None:
        valid = np.isfinite(depth_m) & (depth_m > 0)
        values = depth_m[valid].astype(np.float64)
        self.frames += 1
        self.pixels += depth_m.size
        self.valid_pixels += values.size
        if values.size:
            self.value_sum += float(values.sum())
            self.value_squared_sum += float(np.square(values).sum())
            self.minimum_m = min(self.minimum_m, float(values.min()))
            self.maximum_m = max(self.maximum_m, float(values.max()))

    def as_dict(self) -> dict[str, float | int | None]:
        if not self.valid_pixels:
            return {
                "frames": self.frames,
                "pixels": self.pixels,
                "valid_pixels": 0,
                "valid_fraction": 0.0,
                "mean_m": None,
                "std_m": None,
                "min_m": None,
                "max_m": None,
            }
        mean = self.value_sum / self.valid_pixels
        variance = max(0.0, self.value_squared_sum / self.valid_pixels - mean * mean)
        return {
            "frames": self.frames,
            "pixels": self.pixels,
            "valid_pixels": self.valid_pixels,
            "valid_fraction": self.valid_pixels / self.pixels,
            "mean_m": mean,
            "std_m": variance**0.5,
            "min_m": self.minimum_m,
            "max_m": self.maximum_m,
        }


def sanitize_depth(depth_mm: np.ndarray, min_depth_m: float, max_depth_m: float) -> np.ndarray:
    if depth_mm.dtype != np.uint16 or depth_mm.ndim != 2:
        raise ValueError(f"Expected uint16 depth image, got {depth_mm.dtype} {depth_mm.shape}")
    depth_m = depth_mm.astype(np.float32) * 0.001
    invalid = ~np.isfinite(depth_m) | (depth_m < min_depth_m) | (depth_m > max_depth_m)
    depth_m[invalid] = 0.0
    return depth_m


def guided_filter(guidance: np.ndarray, source: np.ndarray, radius: int, epsilon: float) -> np.ndarray:
    """Fast grayscale guided filter from He et al., implemented with box filters."""
    guidance = guidance.astype(np.float32)
    source = source.astype(np.float32)
    kernel = (2 * radius + 1, 2 * radius + 1)
    mean_i = cv2.boxFilter(guidance, -1, kernel, borderType=cv2.BORDER_REFLECT)
    mean_p = cv2.boxFilter(source, -1, kernel, borderType=cv2.BORDER_REFLECT)
    corr_i = cv2.boxFilter(guidance * guidance, -1, kernel, borderType=cv2.BORDER_REFLECT)
    corr_ip = cv2.boxFilter(guidance * source, -1, kernel, borderType=cv2.BORDER_REFLECT)
    variance_i = corr_i - mean_i * mean_i
    covariance_ip = corr_ip - mean_i * mean_p
    a = covariance_ip / (variance_i + epsilon)
    b = mean_p - a * mean_i
    mean_a = cv2.boxFilter(a, -1, kernel, borderType=cv2.BORDER_REFLECT)
    mean_b = cv2.boxFilter(b, -1, kernel, borderType=cv2.BORDER_REFLECT)
    return mean_a * guidance + mean_b


def internal_hole_mask(invalid: np.ndarray, max_hole_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(invalid.astype(np.uint8), 8)
    selected = np.zeros_like(invalid, dtype=bool)
    height, width = invalid.shape
    for label in range(1, count):
        x, y, component_width, component_height, area = stats[label]
        touches_border = (
            x == 0 or y == 0 or x + component_width >= width or y + component_height >= height
        )
        if not touches_border and area <= max_hole_area:
            selected[labels == label] = True
    return selected


def rgb_guided_refine(
    rgb_bgr: np.ndarray,
    depth_m: np.ndarray,
    *,
    max_hole_area: int = 40_000,
    inpaint_radius: float = 4.0,
    guided_radius: int = 7,
    guided_epsilon: float = 2.5e-3,
    valid_blend: float = 0.8,
) -> np.ndarray:
    """Denoise valid pixels and fill bounded holes while following RGB edges."""
    valid = depth_m > 0
    if not np.any(valid):
        return depth_m.copy()

    median = cv2.medianBlur(depth_m, 5)
    plausible = valid & (median > 0) & (np.abs(median - depth_m) < np.maximum(0.08, depth_m * 0.04))
    denoised = depth_m.copy()
    denoised[plausible] = 0.65 * depth_m[plausible] + 0.35 * median[plausible]

    holes = internal_hole_mask(~valid, max_hole_area=max_hole_area)
    valid_values = denoised[valid]
    near, far = np.percentile(valid_values, [1.0, 99.0])
    near = max(float(near), 1e-3)
    far = max(float(far), near + 1e-3)

    inverse_near = 1.0 / near
    inverse_far = 1.0 / far
    inverse = np.zeros_like(denoised, dtype=np.float32)
    inverse[valid] = 1.0 / np.clip(denoised[valid], near, far)
    normalized = (inverse - inverse_far) / (inverse_near - inverse_far)
    normalized[~valid] = 0.0
    inpainted = cv2.inpaint(normalized, holes.astype(np.uint8) * 255, inpaint_radius, cv2.INPAINT_TELEA)

    gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    filtered_inverse = guided_filter(gray, inpainted, guided_radius, guided_epsilon)
    filtered_inverse = np.clip(
        filtered_inverse * (inverse_near - inverse_far) + inverse_far,
        inverse_far,
        inverse_near,
    )
    guided_depth = 1.0 / np.maximum(filtered_inverse, 1e-6)

    output = np.zeros_like(denoised)
    output[holes] = guided_depth[holes]
    output[valid] = valid_blend * denoised[valid] + (1.0 - valid_blend) * guided_depth[valid]
    return output


def temporal_stabilize(
    previous_rgb_bgr: np.ndarray | None,
    previous_depth_m: np.ndarray | None,
    rgb_bgr: np.ndarray,
    depth_m: np.ndarray,
    *,
    alpha: float = 0.65,
    flow_scale: float = 0.25,
) -> np.ndarray:
    """Warp the previous depth into the current frame and blend only stable surfaces."""
    if previous_rgb_bgr is None or previous_depth_m is None:
        return depth_m.copy()

    height, width = depth_m.shape
    small_size = (max(32, int(width * flow_scale)), max(24, int(height * flow_scale)))
    current_gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
    previous_gray = cv2.cvtColor(previous_rgb_bgr, cv2.COLOR_BGR2GRAY)
    current_small = cv2.resize(current_gray, small_size, interpolation=cv2.INTER_AREA)
    previous_small = cv2.resize(previous_gray, small_size, interpolation=cv2.INTER_AREA)

    # Current-to-previous flow lets remap sample previous depth at each current pixel.
    flow_small = cv2.calcOpticalFlowFarneback(
        current_small,
        previous_small,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    flow = cv2.resize(flow_small, (width, height), interpolation=cv2.INTER_LINEAR)
    flow[..., 0] *= width / small_size[0]
    flow[..., 1] *= height / small_size[1]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]
    warped_depth = cv2.remap(
        previous_depth_m,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_gray = cv2.remap(
        previous_gray,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    threshold = np.maximum(0.035, depth_m * 0.025)
    stable = (
        (depth_m > 0)
        & (warped_depth > 0)
        & (np.abs(depth_m - warped_depth) < threshold)
        & (np.abs(current_gray.astype(np.int16) - warped_gray.astype(np.int16)) < 28)
    )
    output = depth_m.copy()
    output[stable] = alpha * depth_m[stable] + (1.0 - alpha) * warped_depth[stable]
    return output


def encode_depth_mm(
    depth_m: np.ndarray, max_depth_m: float, min_depth_m: float = 0.08
) -> np.ndarray:
    clean = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
    clean[(clean < min_depth_m) | (clean > max_depth_m)] = 0.0
    return np.rint(clean * 1000.0).clip(0, 65535).astype(np.uint16)
