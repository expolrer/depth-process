from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import atomic_write_json, read_jsonl

CORRECTION_TYPES = {"bbox", "mask", "instance", "boundary", "language", "phase", "pose"}


@dataclass(frozen=True)
class HumanCorrection:
    correction_id: str
    dataset_id: str
    dataset_version: str
    episode_id: str
    correction_type: str
    before: dict[str, Any]
    after: dict[str, Any]
    frame_index: int | None = None
    camera: str | None = None
    event_id: str | None = None
    reason: str = "human_review"
    reviewer_hash: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema: str = "embodied_human_correction_v1"

    def __post_init__(self) -> None:
        if self.correction_type not in CORRECTION_TYPES:
            raise ValueError(f"unsupported correction type: {self.correction_type}")
        if not self.correction_id or not self.dataset_id or not self.episode_id:
            raise ValueError("correction, dataset, and episode ids are required")
        if self.frame_index is not None and self.frame_index < 0:
            raise ValueError("correction frame_index must be non-negative")
        if self.before == self.after:
            raise ValueError("a correction must change the annotation")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def reviewer_hash(reviewer: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{reviewer}".encode()).hexdigest()[:16]


def append_correction(path: Path, correction: HumanCorrection) -> None:
    """Append one independently parseable record; writes stay below PIPE_BUF in normal use."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(correction.to_dict(), ensure_ascii=False) + "\n").encode("utf-8")
    if len(encoded) > 256 * 1024:
        raise ValueError("correction record is too large; store masks by path instead of inline")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, encoded)
    finally:
        os.close(descriptor)


def correction_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    counts: dict[str, int] = {}
    episodes: set[str] = set()
    for row in rows:
        kind = str(row["correction_type"])
        counts[kind] = counts.get(kind, 0) + 1
        episodes.add(str(row["episode_id"]))
    return {
        "schema": "embodied_correction_summary_v1",
        "correction_count": len(rows),
        "episode_count": len(episodes),
        "by_type": counts,
    }


def _image_id(episode_id: str, camera: str, frame_index: int) -> int:
    digest = hashlib.sha256(f"{episode_id}:{camera}:{frame_index}".encode()).digest()
    return int.from_bytes(digest[:7], "big")


def export_distillation_sets(correction_log: Path, output_root: Path) -> dict[str, Any]:
    rows = read_jsonl(correction_log)
    output_root.mkdir(parents=True, exist_ok=True)
    detection_images: dict[int, dict[str, Any]] = {}
    detection_annotations: list[dict[str, Any]] = []
    boundaries: list[dict[str, Any]] = []
    language: list[dict[str, Any]] = []
    tracking: list[dict[str, Any]] = []

    for row in rows:
        kind = str(row["correction_type"])
        after = dict(row.get("after", {}))
        common = {
            "correction_id": row["correction_id"],
            "dataset_id": row["dataset_id"],
            "dataset_version": row["dataset_version"],
            "episode_id": row["episode_id"],
        }
        if kind in {"bbox", "mask", "instance"}:
            frame = int(row["frame_index"])
            camera = str(row["camera"])
            image_id = _image_id(str(row["episode_id"]), camera, frame)
            width, height = [int(value) for value in after.get("image_size", [0, 0])]
            detection_images[image_id] = {
                "id": image_id,
                "file_name": after.get("image_path", ""),
                "width": width,
                "height": height,
                **common,
                "camera": camera,
                "frame_index": frame,
            }
            box = after.get("bbox_xyxy")
            if box:
                x1, y1, x2, y2 = [float(value) for value in box]
                detection_annotations.append(
                    {
                        "id": len(detection_annotations) + 1,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": (x2 - x1) * (y2 - y1),
                        "iscrowd": 0,
                        "segmentation_path": after.get("mask_path"),
                        "instance_id": after.get("instance_id"),
                        "source_correction_id": row["correction_id"],
                    }
                )
            tracking.append({**common, "camera": camera, "frame_index": frame, **after})
        elif kind in {"boundary", "phase"}:
            boundaries.append({**common, "event_id": row.get("event_id"), **after})
        elif kind == "language":
            language.append(
                {
                    **common,
                    "event_id": row.get("event_id"),
                    "rejected_text": row.get("before", {}).get("text", ""),
                    "accepted_text": after.get("text", ""),
                    "entity_ids": after.get("entity_ids", []),
                }
            )

    coco = {
        "info": {"description": "V3 human corrections for domain detector distillation"},
        "images": list(detection_images.values()),
        "annotations": detection_annotations,
        "categories": [{"id": 1, "name": "interaction target"}],
    }
    atomic_write_json(output_root / "detector_coco.json", coco)
    for filename, values in (
        ("tracker_prompts.jsonl", tracking),
        ("boundary_corrections.jsonl", boundaries),
        ("language_preferences.jsonl", language),
    ):
        (output_root / filename).write_text(
            "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
            encoding="utf-8",
        )
    manifest = {
        "schema": "embodied_distillation_dataset_v1",
        "source_log": str(correction_log.resolve()),
        "counts": {
            "corrections": len(rows),
            "detector_images": len(detection_images),
            "detector_annotations": len(detection_annotations),
            "tracker_prompts": len(tracking),
            "boundary_examples": len(boundaries),
            "language_preferences": len(language),
        },
        "privacy": {"raw_reviewer_identity_exported": False},
    }
    atomic_write_json(output_root / "manifest.json", manifest)
    return manifest
