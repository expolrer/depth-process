#!/usr/bin/env python3
"""Validate generated interaction candidates and SAM2 tracking artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    outputs = root / "outputs"
    ranked = read_jsonl(outputs / "interaction_candidates" / "ranked_index.jsonl")
    queue = read_jsonl(outputs / "interaction_candidates" / "ambiguity_queue.jsonl")
    vlm = read_jsonl(outputs / "interaction_candidates" / "vlm_resolutions.jsonl")
    tracks = read_jsonl(outputs / "target_tracks" / "track_index.jsonl")
    summary = json.loads((outputs / "target_tracks" / "summary.json").read_text(encoding="utf-8"))

    failures: list[str] = []
    empty_masks = 0
    unreadable_masks = 0
    missing_masks = 0
    for row in tracks:
        path = resolve(root, row["mask_path"])
        if not path.exists():
            missing_masks += 1
            continue
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            unreadable_masks += 1
        elif not image.any():
            empty_masks += 1

    event_counts = Counter(row["event_id"] for row in tracks)
    summary_ids = {row["event_id"] for row in summary["events_detail"]}
    track_ids = set(event_counts)
    if summary_ids != track_ids:
        failures.append("summary and track event IDs differ")
    if len(ranked) != len(summary_ids) * 3:
        failures.append("expected exactly three ranked camera rows per event")
    if len(tracks) != int(summary["tracked_frames"]):
        failures.append("track index row count differs from summary")
    if missing_masks or unreadable_masks or empty_masks:
        failures.append("mask files are missing, unreadable, or empty")

    report = {
        "schema": "interaction_annotation_validation_v1",
        "passed": not failures,
        "failures": failures,
        "events": len(summary_ids),
        "ranked_camera_frames": len(ranked),
        "grounding_candidates_before_nms": sum(int(row.get("candidates_before_nms", 0)) for row in ranked),
        "ambiguity_events": len(queue),
        "vlm_resolutions": len(vlm),
        "vlm_valid_choices": sum(bool(row.get("valid_choice")) for row in vlm),
        "tracked_frames": len(tracks),
        "selection_sources": dict(Counter(row["selection_source"] for row in summary["events_detail"])),
        "quality_flagged_frames": sum(bool(row.get("needs_review")) for row in tracks),
        "quality_flagged_events": sum(bool(row.get("needs_review_frames")) for row in summary["events_detail"]),
        "missing_masks": missing_masks,
        "unreadable_masks": unreadable_masks,
        "empty_masks": empty_masks,
        "per_event_frames": dict(event_counts),
    }
    report_dir = outputs / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "interaction_annotation_validation.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
