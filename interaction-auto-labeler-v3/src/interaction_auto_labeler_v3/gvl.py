from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

PHASES = ("approach", "contact", "manipulate", "release", "retreat")


@dataclass(frozen=True)
class ProgressProbe:
    episode_id: str
    frame_count: int
    presented_frames: tuple[int, ...]
    required_entity_ids: tuple[str, ...] = ()
    phases: tuple[str, ...] = PHASES
    schema: str = "grounded_video_language_progress_probe_v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_progress_probe(
    episode_id: str,
    frame_count: int,
    sample_count: int = 12,
    seed: int = 0,
    required_entity_ids: Iterable[str] = (),
) -> ProgressProbe:
    if frame_count < 2:
        raise ValueError("progress probing requires at least two frames")
    count = min(max(2, int(sample_count)), frame_count)
    frames = np.linspace(0, frame_count - 1, count).round().astype(int).tolist()
    frames = list(dict.fromkeys(frames))
    random.Random(f"{seed}:{episode_id}").shuffle(frames)
    return ProgressProbe(
        episode_id=episode_id,
        frame_count=frame_count,
        presented_frames=tuple(frames),
        required_entity_ids=tuple(required_entity_ids),
    )


def progress_prompt(probe: ProgressProbe) -> str:
    return (
        "These robot frames are intentionally shuffled. For every supplied frame, return its "
        "frame_index, normalized task progress in [0,1], one phase from "
        f"{list(probe.phases)}, visible_entity_ids, and a short grounded description. "
        "Do not infer an object that is not visible. Return JSON only using schema "
        "{frames:[{frame_index,progress,phase,visible_entity_ids,description}]}. "
        f"Required tracked entities: {list(probe.required_entity_ids)}."
    )


def _average_ranks(values: list[float]) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    cursor = 0
    while cursor < len(values):
        stop = cursor + 1
        while stop < len(values) and values[order[stop]] == values[order[cursor]]:
            stop += 1
        ranks[order[cursor:stop]] = (cursor + stop - 1) / 2.0
        cursor = stop
    return ranks


def _spearman(first: list[float], second: list[float]) -> float:
    if len(first) < 2:
        return 0.0
    a, b = _average_ranks(first), _average_ranks(second)
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _pairwise_order_accuracy(frames: list[int], progress: list[float]) -> float:
    correct = total = 0
    for left in range(len(frames)):
        for right in range(left + 1, len(frames)):
            if frames[left] == frames[right]:
                continue
            expected = frames[left] < frames[right]
            predicted = progress[left] < progress[right]
            correct += int(expected == predicted)
            total += 1
    return correct / total if total else 0.0


def evaluate_progress_response(probe: ProgressProbe, response: dict[str, Any]) -> dict[str, Any]:
    rows = response.get("frames", [])
    by_frame = {int(row["frame_index"]): row for row in rows if "frame_index" in row}
    missing = [frame for frame in probe.presented_frames if frame not in by_frame]
    evaluated = [by_frame[frame] for frame in probe.presented_frames if frame in by_frame]
    progress = [float(row.get("progress", math.nan)) for row in evaluated]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in progress):
        raise ValueError("all returned progress values must be finite and in [0, 1]")
    frames = [int(row["frame_index"]) for row in evaluated]
    expected = [frame / (probe.frame_count - 1) for frame in frames]
    rank_correlation = _spearman(expected, progress)
    order_accuracy = _pairwise_order_accuracy(frames, progress)
    progress_mae = float(np.mean(np.abs(np.asarray(expected) - progress))) if progress else 1.0
    required = set(probe.required_entity_ids)
    returned_entities = {
        str(value) for row in evaluated for value in row.get("visible_entity_ids", [])
    }
    grounding_recall = len(required & returned_entities) / len(required) if required else 1.0
    unknown_entities = returned_entities - required if required else set()
    entity_precision = (
        len(returned_entities - unknown_entities) / len(returned_entities)
        if returned_entities
        else 1.0
    )
    phases = [str(row.get("phase", "")) for row in evaluated]
    phase_valid_rate = (
        sum(phase in probe.phases for phase in phases) / len(phases) if phases else 0.0
    )
    ordered_rows = sorted(evaluated, key=lambda row: int(row["frame_index"]))
    phase_indices = [
        probe.phases.index(str(row.get("phase")))
        for row in ordered_rows
        if str(row.get("phase")) in probe.phases
    ]
    phase_pairs = list(pairwise(phase_indices))
    phase_order_accuracy = (
        sum(left <= right for left, right in phase_pairs) / len(phase_pairs)
        if phase_pairs
        else phase_valid_rate
    )
    completeness = len(evaluated) / len(probe.presented_frames)
    score = (
        0.30 * max(0.0, rank_correlation)
        + 0.25 * order_accuracy
        + 0.15 * max(0.0, 1.0 - progress_mae)
        + 0.10 * grounding_recall
        + 0.05 * entity_precision
        + 0.05 * phase_valid_rate
        + 0.05 * phase_order_accuracy
        + 0.05 * completeness
    )
    return {
        "schema": "grounded_video_language_progress_assessment_v1",
        "episode_id": probe.episode_id,
        "sample_count": len(evaluated),
        "missing_frames": missing,
        "rank_correlation": rank_correlation,
        "pairwise_order_accuracy": order_accuracy,
        "progress_mae": progress_mae,
        "grounding_recall": grounding_recall,
        "entity_precision": entity_precision,
        "unknown_entity_ids": sorted(unknown_entities),
        "phase_valid_rate": phase_valid_rate,
        "phase_order_accuracy": phase_order_accuracy,
        "completeness": completeness,
        "score": float(np.clip(score, 0.0, 1.0)),
        "needs_review": bool(score < 0.65 or completeness < 1.0),
    }


def load_progress_response(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
