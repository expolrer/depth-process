from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from typing import Any

import numpy as np

from .event_graph import FrameSpan, Segment

LEVELS = ("task", "subtask", "event")


@dataclass(frozen=True)
class BoundaryConfig:
    modality_weights: dict[str, float] = field(
        default_factory=lambda: {
            "scene": 1.0,
            "visual": 0.8,
            "object": 1.2,
            "eef": 0.9,
            "joint": 0.5,
            "action": 0.8,
            "gripper": 1.4,
            "force": 1.2,
            "pause": 0.7,
        }
    )
    level_modalities: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "task": ("scene", "visual", "pause"),
            "subtask": ("visual", "object", "eef", "gripper", "force", "pause"),
            "event": ("object", "eef", "joint", "action", "gripper", "force"),
        }
    )
    threshold_quantiles: dict[str, float] = field(
        default_factory=lambda: {"task": 0.94, "subtask": 0.88, "event": 0.82}
    )
    min_gap_frames: dict[str, int] = field(
        default_factory=lambda: {"task": 90, "subtask": 30, "event": 10}
    )
    smoothing_frames: int = 7
    max_boundaries: dict[str, int | None] = field(
        default_factory=lambda: {"task": None, "subtask": None, "event": None}
    )

    def __post_init__(self) -> None:
        if self.smoothing_frames < 1:
            raise ValueError("smoothing_frames must be positive")
        for level in LEVELS:
            quantile = self.threshold_quantiles[level]
            if not 0.0 < quantile < 1.0:
                raise ValueError("threshold quantiles must be in (0, 1)")
            if self.min_gap_frames[level] < 1:
                raise ValueError("minimum boundary gaps must be positive")


@dataclass(frozen=True)
class BoundaryProposal:
    level: str
    frame_index: int
    score: float
    uncertainty_frames: int
    evidence: dict[str, float]
    source: str = "multimodal_change_point"

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"unsupported boundary level: {self.level}")
        if self.frame_index < 0 or self.uncertainty_frames < 0:
            raise ValueError("boundary frames must be non-negative")
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("boundary score must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_feature_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("each modality must have shape [time, features] and at least two frames")
    values = values.copy()
    for column in range(values.shape[1]):
        finite = np.isfinite(values[:, column])
        replacement = float(np.median(values[finite, column])) if finite.any() else 0.0
        values[~finite, column] = replacement
    return values


def robust_standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1.4826 * mad, float(np.std(values)), 1e-9)
    z = np.maximum(0.0, (values - median) / scale)
    return z / (1.0 + z)


def smooth_signal(values: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if width <= 1:
        return values.copy()
    width = min(width + (width + 1) % 2, len(values) if len(values) % 2 else len(values) - 1)
    if width <= 1:
        return values.copy()
    radius = width // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(padded, kernel, mode="valid")


def modality_change_scores(
    modalities: dict[str, np.ndarray], smoothing_frames: int = 7
) -> dict[str, np.ndarray]:
    if not modalities:
        raise ValueError("at least one modality is required")
    matrices = {name: _as_feature_matrix(value) for name, value in modalities.items()}
    lengths = {len(value) for value in matrices.values()}
    if len(lengths) != 1:
        raise ValueError("all modalities must have the same number of frames")
    output: dict[str, np.ndarray] = {}
    for name, matrix in matrices.items():
        delta = np.linalg.norm(np.diff(matrix, axis=0, prepend=matrix[:1]), axis=1)
        output[name] = smooth_signal(robust_standardize(delta), smoothing_frames)
    return output


def _fuse(
    scores: dict[str, np.ndarray], names: tuple[str, ...], weights: dict[str, float]
) -> tuple[np.ndarray, list[str]]:
    available = [name for name in names if name in scores and weights.get(name, 0.0) > 0]
    if not available:
        length = len(next(iter(scores.values())))
        return np.zeros(length, dtype=np.float64), []
    denominator = sum(float(weights[name]) for name in available)
    fused = sum(scores[name] * float(weights[name]) for name in available) / denominator
    return np.clip(fused, 0.0, 1.0), available


def _peak_uncertainty(scores: np.ndarray, index: int) -> int:
    threshold = scores[index] * 0.8
    left = index
    right = index
    while left > 0 and scores[left - 1] >= threshold:
        left -= 1
    while right + 1 < len(scores) and scores[right + 1] >= threshold:
        right += 1
    return max(index - left, right - index)


def find_peaks(
    scores: np.ndarray,
    quantile: float,
    min_gap_frames: int,
    max_boundaries: int | None = None,
) -> list[int]:
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) < 3 or np.max(scores) <= 0:
        return []
    threshold = float(np.quantile(scores[1:-1], quantile))
    candidates = [
        index
        for index in range(1, len(scores) - 1)
        if scores[index] >= threshold
        and scores[index] >= scores[index - 1]
        and scores[index] >= scores[index + 1]
    ]
    selected: list[int] = []
    for index in sorted(candidates, key=lambda item: (-scores[item], item)):
        if all(abs(index - prior) >= min_gap_frames for prior in selected):
            selected.append(index)
            if max_boundaries is not None and len(selected) >= max_boundaries:
                break
    return sorted(selected)


def propose_hierarchical_boundaries(
    modalities: dict[str, np.ndarray], config: BoundaryConfig | None = None
) -> tuple[list[BoundaryProposal], dict[str, np.ndarray]]:
    config = config or BoundaryConfig()
    per_modality = modality_change_scores(modalities, config.smoothing_frames)
    proposals = []
    level_scores = {}
    for level in LEVELS:
        fused, available = _fuse(
            per_modality, config.level_modalities[level], config.modality_weights
        )
        level_scores[level] = fused
        for index in find_peaks(
            fused,
            config.threshold_quantiles[level],
            config.min_gap_frames[level],
            config.max_boundaries[level],
        ):
            evidence = {
                name: float(per_modality[name][index])
                for name in available
                if per_modality[name][index] > 0
            }
            proposals.append(
                BoundaryProposal(
                    level=level,
                    frame_index=index,
                    score=float(fused[index]),
                    uncertainty_frames=_peak_uncertainty(fused, index),
                    evidence=evidence,
                )
            )
    return proposals, level_scores


def segments_from_boundaries(
    frame_count: int,
    proposals: list[BoundaryProposal],
) -> list[Segment]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    segments: list[Segment] = []
    proposed_cuts = {
        level: sorted(
            {item.frame_index for item in proposals if item.level == level and item.frame_index > 0}
        )
        for level in LEVELS
    }
    by_level: dict[str, list[int]] = {}
    inherited: set[int] = set()
    for level in LEVELS:
        inherited.update(proposed_cuts[level])
        by_level[level] = sorted(inherited)
    parent_level = {"task": None, "subtask": "task", "event": "subtask"}
    spans_by_level: dict[str, list[tuple[str, FrameSpan]]] = {}
    proposal_lookup = {(item.level, item.frame_index): item for item in proposals}
    for level in LEVELS:
        cuts = [0] + [item for item in by_level[level] if item < frame_count] + [frame_count]
        spans_by_level[level] = []
        for index, (start, stop) in enumerate(pairwise(cuts)):
            span = FrameSpan(start, stop - 1)
            segment_id = f"{level}:{index:04d}"
            parent_id = None
            if parent_level[level]:
                parent_id = next(
                    (
                        item_id
                        for item_id, parent_span in spans_by_level[parent_level[level]]
                        if parent_span.contains(span)
                    ),
                    None,
                )
            left_boundary = proposal_lookup.get((level, start))
            right_boundary = proposal_lookup.get((level, stop))
            confidence_values = [
                item.score for item in (left_boundary, right_boundary) if item is not None
            ]
            uncertainty_values = [
                item.uncertainty_frames
                for item in (left_boundary, right_boundary)
                if item is not None
            ]
            segments.append(
                Segment(
                    segment_id=segment_id,
                    level=level,
                    span=span,
                    label=f"unlabeled {level} {index}",
                    parent_segment_id=parent_id,
                    confidence=(
                        sum(confidence_values) / len(confidence_values)
                        if confidence_values
                        else 1.0
                    ),
                    boundary_uncertainty_frames=max(uncertainty_values, default=0),
                )
            )
            spans_by_level[level].append((segment_id, span))
    return segments


def load_modalities_npz(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}
