from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReliabilityWeights:
    detection: float = 1.0
    track_stability: float = 1.3
    contact: float = 1.5
    co_motion: float = 1.2
    depth_consistency: float = 1.0
    cross_view: float = 1.2
    boundary: float = 0.8
    language_grounding: float = 0.7
    required: tuple[str, ...] = ("track_stability", "contact")

    def values(self) -> dict[str, float]:
        payload = asdict(self)
        payload.pop("required")
        return {key: float(value) for key, value in payload.items()}


def aggregate_reliability(
    features: dict[str, float | None], weights: ReliabilityWeights | None = None
) -> dict[str, Any]:
    """Aggregate available evidence without silently treating missing evidence as success."""
    weights = weights or ReliabilityWeights()
    configured = weights.values()
    available: dict[str, float] = {}
    for name, value in features.items():
        if name not in configured or value is None:
            continue
        score = float(value)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"reliability feature {name!r} must be in [0, 1]")
        if configured[name] > 0:
            available[name] = score
    if not available:
        return {
            "raw_score": 0.0,
            "evidence_coverage": 0.0,
            "available": {},
            "missing_required": list(weights.required),
        }
    denominator = sum(configured[name] for name in available)
    raw_score = sum(available[name] * configured[name] for name in available) / denominator
    total_weight = sum(value for value in configured.values() if value > 0)
    coverage = denominator / total_weight if total_weight else 0.0
    missing_required = [name for name in weights.required if name not in available]
    return {
        "raw_score": float(raw_score),
        "evidence_coverage": float(coverage),
        "available": available,
        "missing_required": missing_required,
    }


@dataclass(frozen=True)
class IsotonicCalibrator:
    upper_bounds: tuple[float, ...]
    probabilities: tuple[float, ...]
    sample_counts: tuple[int, ...]
    schema: str = "isotonic_reliability_calibrator_v1"

    def __post_init__(self) -> None:
        if not self.upper_bounds or len(self.upper_bounds) != len(self.probabilities):
            raise ValueError("calibrator requires equally sized non-empty arrays")
        if len(self.sample_counts) != len(self.upper_bounds):
            raise ValueError("sample_counts length must match calibrator points")
        if any(first > second for first, second in zip(self.upper_bounds, self.upper_bounds[1:])):
            raise ValueError("upper_bounds must be sorted")
        if any(first > second for first, second in zip(self.probabilities, self.probabilities[1:])):
            raise ValueError("calibrated probabilities must be monotonic")

    def predict_one(self, score: float) -> float:
        index = int(np.searchsorted(self.upper_bounds, float(score), side="left"))
        index = min(index, len(self.probabilities) - 1)
        return float(self.probabilities[index])

    def predict(self, scores: Iterable[float]) -> list[float]:
        return [self.predict_one(score) for score in scores]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> IsotonicCalibrator:
        return cls(
            upper_bounds=tuple(float(value) for value in payload["upper_bounds"]),
            probabilities=tuple(float(value) for value in payload["probabilities"]),
            sample_counts=tuple(int(value) for value in payload["sample_counts"]),
            schema=str(payload.get("schema", "isotonic_reliability_calibrator_v1")),
        )


def fit_isotonic_calibrator(scores: Iterable[float], correct: Iterable[bool]) -> IsotonicCalibrator:
    pairs = sorted((float(score), int(bool(label))) for score, label in zip(scores, correct))
    if not pairs:
        raise ValueError("at least one calibration sample is required")
    if any(not math.isfinite(score) or not 0.0 <= score <= 1.0 for score, _ in pairs):
        raise ValueError("calibration scores must be finite and in [0, 1]")

    grouped: list[list[float]] = []
    for score, label in pairs:
        if grouped and grouped[-1][1] == score:
            grouped[-1][2] += label
            grouped[-1][3] += 1
        else:
            grouped.append([score, score, float(label), 1.0])

    blocks: list[list[float]] = []
    for group in grouped:
        blocks.append(group)
        while len(blocks) >= 2:
            previous = blocks[-2][2] / blocks[-2][3]
            current = blocks[-1][2] / blocks[-1][3]
            if previous <= current:
                break
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([left[0], right[1], left[2] + right[2], left[3] + right[3]])
    return IsotonicCalibrator(
        upper_bounds=tuple(float(block[1]) for block in blocks),
        probabilities=tuple(float(block[2] / block[3]) for block in blocks),
        sample_counts=tuple(int(block[3]) for block in blocks),
    )


def selective_risk_curve(
    confidence: Iterable[float],
    correct: Iterable[bool],
    thresholds: Iterable[float] | None = None,
) -> list[dict[str, float | int]]:
    pairs = [(float(score), bool(label)) for score, label in zip(confidence, correct)]
    if not pairs:
        return []
    thresholds = thresholds or np.linspace(0.0, 1.0, 21)
    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        accepted = [label for score, label in pairs if score >= float(threshold)]
        accuracy = sum(accepted) / len(accepted) if accepted else 1.0
        rows.append(
            {
                "threshold": float(threshold),
                "accepted": len(accepted),
                "coverage": len(accepted) / len(pairs),
                "precision": float(accuracy),
                "risk": float(1.0 - accuracy),
            }
        )
    return rows


def choose_threshold(
    curve: Iterable[dict[str, float | int]], minimum_precision: float
) -> dict[str, float | int] | None:
    candidates = [
        row
        for row in curve
        if float(row["precision"]) >= minimum_precision and int(row["accepted"]) > 0
    ]
    return (
        max(candidates, key=lambda row: (float(row["coverage"]), -float(row["threshold"])))
        if candidates
        else None
    )


def save_calibrator(calibrator: IsotonicCalibrator, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(calibrator.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_calibrator(path: Path) -> IsotonicCalibrator:
    return IsotonicCalibrator.from_dict(json.loads(path.read_text(encoding="utf-8")))
