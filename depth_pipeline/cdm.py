"""Camera-specific CDM preprocessing and sensor-preserving fusion helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraDepthSpec:
    model_id: str
    checkpoint: str
    min_depth_m: float
    max_depth_m: float

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must not be empty")
        if self.min_depth_m <= 0 or self.max_depth_m <= self.min_depth_m:
            raise ValueError("Expected 0 < min_depth_m < max_depth_m")


def sanitize_sensor_depth(
    depth_m: np.ndarray,
    *,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return range-checked metric depth and the explicit validity mask.

    Validity here only means finite, non-zero and inside the configured sensor
    range. It is not a learned confidence estimate and cannot reject flying
    pixels that happen to fall inside the range.
    """
    sensor = np.asarray(depth_m, dtype=np.float32).copy()
    valid = (
        np.isfinite(sensor)
        & (sensor >= min_depth_m)
        & (sensor <= max_depth_m)
    )
    sensor[~valid] = 0.0
    return sensor, valid


def make_inverse_depth_prompt(depth_m: np.ndarray) -> np.ndarray:
    """Convert valid metric depth to the inverse-depth prompt expected by CDM."""
    depth = np.asarray(depth_m, dtype=np.float32)
    inverse = np.zeros_like(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    inverse[valid] = 1.0 / depth[valid]
    return inverse


def fuse_sensor_and_cdm(
    sensor_depth_m: np.ndarray,
    cdm_depth_m: np.ndarray,
    *,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preserve valid measurements and use CDM only where the sensor is invalid."""
    sensor, sensor_valid = sanitize_sensor_depth(
        sensor_depth_m,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
    )
    cdm = np.asarray(cdm_depth_m, dtype=np.float32).copy()
    cdm_valid = (
        np.isfinite(cdm)
        & (cdm >= min_depth_m)
        & (cdm <= max_depth_m)
    )
    cdm[~cdm_valid] = 0.0
    fill_mask = ~sensor_valid & cdm_valid
    fused = sensor.copy()
    fused[fill_mask] = cdm[fill_mask]
    return fused, fill_mask, sensor_valid
