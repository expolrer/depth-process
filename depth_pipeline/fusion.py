"""Metric calibration and sensor/prior fusion helpers."""

from __future__ import annotations

import cv2
import numpy as np


def robust_affine_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_samples: int = 50_000,
    iterations: int = 8,
    huber_delta: float = 1.5,
) -> tuple[float, float, float]:
    """Fit y = scale*x + shift using deterministic Huber IRLS."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 100:
        raise ValueError(f"Need at least 100 calibration samples, got {x.size}")
    if x.size > max_samples:
        indices = np.linspace(0, x.size - 1, max_samples, dtype=np.int64)
        x = x[indices]
        y = y[indices]

    design = np.column_stack((x, np.ones_like(x)))
    lower = np.percentile(y, 1.0)
    upper = np.percentile(y, 99.0)
    inlier = (y >= lower) & (y <= upper)
    coefficients, *_ = np.linalg.lstsq(design[inlier], y[inlier], rcond=None)

    for _ in range(iterations):
        residual = y - design @ coefficients
        center = np.median(residual)
        mad = np.median(np.abs(residual - center))
        sigma = max(1.4826 * mad, 1e-6)
        normalized = np.abs(residual - center) / (huber_delta * sigma)
        weights = np.ones_like(normalized)
        outside = normalized > 1.0
        weights[outside] = 1.0 / normalized[outside]
        weighted_design = design * np.sqrt(weights[:, None])
        weighted_y = y * np.sqrt(weights)
        coefficients, *_ = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)

    final_residual = y - design @ coefficients
    rmse = float(np.sqrt(np.mean(np.square(final_residual))))
    return float(coefficients[0]), float(coefficients[1]), rmse


def calibrate_relative_inverse_depth(
    relative_prior: np.ndarray,
    sensor_depth_m: np.ndarray,
    *,
    min_depth_m: float = 0.08,
    max_depth_m: float = 10.0,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Calibrate a relative inverse-depth prior against metric sensor samples."""
    prior = relative_prior.astype(np.float32)
    sensor = sensor_depth_m.astype(np.float32)
    valid = (
        np.isfinite(sensor)
        & (sensor >= min_depth_m)
        & (sensor <= max_depth_m)
        & np.isfinite(prior)
        & (prior > 0)
    )

    # Avoid depth discontinuities and the noisiest edge pixels during metric fitting.
    sensor_median = cv2.medianBlur(sensor, 5)
    stable = valid & (sensor_median > 0) & (
        np.abs(sensor - sensor_median) < np.maximum(0.05, sensor * 0.03)
    )
    if np.count_nonzero(stable) < 1000:
        stable = valid

    x = prior[stable]
    y = 1.0 / sensor[stable]
    scale, shift, inverse_rmse = robust_affine_fit(x, y)
    inverse_metric = scale * prior + shift
    calibrated = np.zeros_like(prior, dtype=np.float32)
    plausible = np.isfinite(inverse_metric) & (inverse_metric > 1.0 / max_depth_m)
    calibrated[plausible] = 1.0 / inverse_metric[plausible]
    calibrated[(calibrated < min_depth_m) | (calibrated > max_depth_m)] = 0.0

    prediction_valid = calibrated[valid] > 0
    metric_rmse = (
        float(np.sqrt(np.mean(np.square(calibrated[valid][prediction_valid] - sensor[valid][prediction_valid]))))
        if np.any(prediction_valid)
        else float("inf")
    )
    report = {
        "fit_samples": int(np.count_nonzero(stable)),
        "scale": scale,
        "shift": shift,
        "inverse_depth_rmse": inverse_rmse,
        "metric_depth_rmse_m": metric_rmse,
        "calibration_ok": bool(scale > 0 and np.isfinite(metric_rmse) and metric_rmse < 1.5),
    }
    return calibrated, report


def fuse_sensor_and_prior(
    sensor_depth_m: np.ndarray,
    calibrated_prior_m: np.ndarray,
    *,
    max_hole_area: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep measured depth and fill invalid regions from the calibrated RGB prior."""
    sensor_valid = np.isfinite(sensor_depth_m) & (sensor_depth_m > 0)
    prior_valid = np.isfinite(calibrated_prior_m) & (calibrated_prior_m > 0)
    fill_mask = ~sensor_valid & prior_valid
    if max_hole_area is not None:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(fill_mask.astype(np.uint8), 8)
        selected = np.zeros_like(fill_mask)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] <= max_hole_area:
                selected[labels == label] = True
        fill_mask = selected

    fused = sensor_depth_m.copy()
    fused[fill_mask] = calibrated_prior_m[fill_mask]
    return fused, fill_mask

