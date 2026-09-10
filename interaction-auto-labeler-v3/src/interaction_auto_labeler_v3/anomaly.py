from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Anomaly:
    kind: str
    start_frame: int
    end_frame: int
    severity: float
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _matrix(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError("modality arrays must have shape [time, features]")
    return array


def _robust_threshold(values: np.ndarray, multiplier: float = 8.0) -> float:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return math.inf
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return median + multiplier * max(1.4826 * mad, 1e-9)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    output: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value != previous + 1:
            output.append((start, previous))
            start = value
        previous = value
    output.append((start, previous))
    return output


def detect_anomalies(
    modalities: dict[str, Any],
    timestamps: Any | None = None,
    depth_min_valid_ratio: float = 0.2,
    freeze_min_frames: int = 8,
) -> list[Anomaly]:
    matrices = {name: _matrix(value) for name, value in modalities.items()}
    lengths = {len(value) for value in matrices.values()}
    if timestamps is not None:
        lengths.add(len(np.asarray(timestamps)))
    if not lengths or len(lengths) != 1:
        raise ValueError("all supplied modalities must have the same non-zero length")
    frame_count = next(iter(lengths))
    anomalies: list[Anomaly] = []

    for name, values in matrices.items():
        invalid = ~np.isfinite(values).all(axis=1)
        for start, end in _runs(invalid):
            anomalies.append(Anomaly("non_finite", start, end, 1.0, {"modality": name}))

    if timestamps is not None:
        time = np.asarray(timestamps, dtype=np.float64)
        delta = np.diff(time)
        positive = delta[np.isfinite(delta) & (delta > 0)]
        expected = float(np.median(positive)) if len(positive) else 0.0
        threshold = max(expected * 3.0, expected + 1e-6)
        bad = (~np.isfinite(delta)) | (delta <= 0) | (delta > threshold)
        for start, end in _runs(bad):
            severity = 1.0
            if expected > 0 and np.isfinite(delta[start]) and delta[start] > 0:
                severity = min(1.0, max(0.3, delta[start] / threshold - 1.0))
            anomalies.append(
                Anomaly(
                    "timestamp_discontinuity",
                    start,
                    min(frame_count - 1, end + 1),
                    severity,
                    {"expected_dt": expected, "threshold_dt": threshold},
                )
            )

    for name in ("joint", "action", "eef", "gripper"):
        if name not in matrices:
            continue
        delta = np.linalg.norm(np.diff(matrices[name], axis=0), axis=1)
        threshold = _robust_threshold(delta)
        for start, end in _runs(delta > threshold):
            peak = float(np.nanmax(delta[start : end + 1]))
            severity = min(1.0, peak / max(threshold, 1e-9) - 1.0)
            anomalies.append(
                Anomaly(
                    f"{name}_spike",
                    start,
                    min(frame_count - 1, end + 1),
                    max(0.3, severity),
                    {"peak_delta": peak, "threshold": threshold},
                )
            )

    if "depth_valid_ratio" in matrices:
        ratio = matrices["depth_valid_ratio"][:, 0]
        for start, end in _runs(ratio < depth_min_valid_ratio):
            minimum = float(np.nanmin(ratio[start : end + 1]))
            severity = min(1.0, (depth_min_valid_ratio - minimum) / depth_min_valid_ratio)
            anomalies.append(
                Anomaly(
                    "depth_dropout",
                    start,
                    end,
                    max(0.2, severity),
                    {"minimum_valid_ratio": minimum, "threshold": depth_min_valid_ratio},
                )
            )

    if "visual" in matrices and "action" in matrices:
        visual_delta = np.linalg.norm(
            np.diff(matrices["visual"], axis=0, prepend=matrices["visual"][:1]), axis=1
        )
        action_delta = np.linalg.norm(
            np.diff(matrices["action"], axis=0, prepend=matrices["action"][:1]), axis=1
        )
        visual_threshold = max(float(np.quantile(visual_delta, 0.05)), 1e-9)
        action_threshold = float(np.quantile(action_delta, 0.6))
        frozen = (visual_delta <= visual_threshold) & (action_delta > action_threshold)
        for start, end in _runs(frozen):
            if end - start + 1 >= freeze_min_frames:
                anomalies.append(
                    Anomaly(
                        "visual_freeze_during_action",
                        start,
                        end,
                        min(1.0, (end - start + 1) / max(freeze_min_frames * 2, 1)),
                        {"duration_frames": end - start + 1},
                    )
                )
    return sorted(anomalies, key=lambda item: (item.start_frame, item.kind))


def quality_gate(
    calibrated_reliability: float,
    evidence_coverage: float,
    progress_score: float,
    anomalies: list[Anomaly],
    accept_threshold: float = 0.8,
    minimum_evidence_coverage: float = 0.5,
    reject_non_finite: bool = True,
) -> dict[str, Any]:
    severe = [item for item in anomalies if item.severity >= 0.7]
    anomaly_penalty = min(0.5, sum(item.severity for item in anomalies) * 0.05)
    score = (
        0.55 * calibrated_reliability
        + 0.20 * evidence_coverage
        + 0.25 * progress_score
        - anomaly_penalty
    )
    reasons = []
    if calibrated_reliability < accept_threshold:
        reasons.append("reliability_below_threshold")
    if evidence_coverage < minimum_evidence_coverage:
        reasons.append("insufficient_evidence_coverage")
    if progress_score < 0.65:
        reasons.append("progress_inconsistent")
    if severe:
        reasons.append("severe_sensor_or_motion_anomaly")
    decision = "accept" if not reasons and score >= accept_threshold else "review"
    if reject_non_finite and any(
        item.kind == "non_finite" and item.severity >= 1.0 for item in anomalies
    ):
        decision = "reject"
        reasons.append("non_finite_required_modality")
    return {
        "schema": "embodied_quality_gate_v1",
        "decision": decision,
        "score": float(np.clip(score, 0.0, 1.0)),
        "reasons": list(dict.fromkeys(reasons)),
        "severe_anomaly_count": len(severe),
        "anomalies": [item.to_dict() for item in anomalies],
    }
