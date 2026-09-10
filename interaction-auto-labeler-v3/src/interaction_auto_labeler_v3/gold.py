from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _stable_tie_break(row: dict[str, Any], seed: int) -> str:
    episode_id = str(row.get("episode_id", row.get("episode_index", "")))
    return hashlib.sha256(f"{seed}:{episode_id}".encode()).hexdigest()


def stratified_gold_sample(
    rows: Iterable[dict[str, Any]],
    sample_size: int,
    strata: tuple[str, ...] = ("task", "scene", "active_arm"),
    seed: int = 0,
) -> list[dict[str, Any]]:
    rows = list(rows)
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size >= len(rows):
        return sorted(rows, key=lambda row: _stable_tie_break(row, seed))
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(key, "unknown")) for key in strata)].append(row)
    for group in groups.values():
        group.sort(key=lambda row: _stable_tie_break(row, seed))

    allocations = {
        key: min(len(group), max(1, math.floor(sample_size * len(group) / len(rows))))
        for key, group in groups.items()
    }
    while sum(allocations.values()) > sample_size:
        candidates = [key for key, count in allocations.items() if count > 1]
        if not candidates:
            candidates = [key for key, count in allocations.items() if count > 0]
        key = max(candidates, key=lambda item: (allocations[item], len(groups[item]), item))
        allocations[key] -= 1
    while sum(allocations.values()) < sample_size:
        candidates = [key for key in groups if allocations[key] < len(groups[key])]
        key = max(
            candidates,
            key=lambda item: (
                len(groups[item]) - allocations[item],
                -allocations[item],
                item,
            ),
        )
        allocations[key] += 1
    sampled = [row for key, group in groups.items() for row in group[: allocations[key]]]
    random.Random(seed).shuffle(sampled)
    return sampled


def bbox_iou(first: Iterable[float], second: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in first]
    bx1, by1, bx2, by2 = [float(value) for value in second]
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (
        max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        + max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def boundary_metrics(
    predicted: Iterable[int], gold: Iterable[int], tolerance_frames: int
) -> dict[str, float | int]:
    if tolerance_frames < 0:
        raise ValueError("tolerance_frames must be non-negative")
    predicted = sorted(int(value) for value in predicted)
    unmatched = {int(value) for value in gold}
    errors = []
    true_positive = 0
    for value in predicted:
        candidates = [item for item in unmatched if abs(item - value) <= tolerance_frames]
        if not candidates:
            continue
        match = min(candidates, key=lambda item: abs(item - value))
        unmatched.remove(match)
        true_positive += 1
        errors.append(abs(match - value))
    precision = true_positive / len(predicted) if predicted else 0.0
    gold_count = true_positive + len(unmatched)
    recall = true_positive / gold_count if gold_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": len(predicted) - true_positive,
        "false_negative": len(unmatched),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_absolute_error_frames": sum(errors) / len(errors) if errors else 0.0,
    }


def calibration_metrics(
    confidence: Iterable[float], correct: Iterable[bool], bins: int = 10
) -> dict[str, float]:
    pairs = [(float(score), bool(label)) for score, label in zip(confidence, correct)]
    if not pairs:
        return {"brier": 0.0, "ece": 0.0}
    brier = sum((score - float(label)) ** 2 for score, label in pairs) / len(pairs)
    ece = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        bucket = [
            (score, label)
            for score, label in pairs
            if low <= score < high or (index == bins - 1 and score == 1.0)
        ]
        if not bucket:
            continue
        mean_confidence = sum(score for score, _ in bucket) / len(bucket)
        accuracy = sum(float(label) for _, label in bucket) / len(bucket)
        ece += len(bucket) / len(pairs) * abs(mean_confidence - accuracy)
    return {"brier": brier, "ece": ece}


def write_gold_manifest(
    rows: list[dict[str, Any]],
    output: Path,
    strata: tuple[str, ...],
    seed: int,
) -> None:
    payload = {
        "schema": "embodied_gold_manifest_v1",
        "seed": seed,
        "strata": list(strata),
        "sample_count": len(rows),
        "review_policy": {
            "reviewers_required": 2,
            "adjudication_required_on_disagreement": True,
        },
        "items": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
