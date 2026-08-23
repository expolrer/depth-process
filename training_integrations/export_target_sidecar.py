#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
from PIL import Image

from training_integrations.target_annotations import normalize_box, phase_for_frame


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def view_size(view: dict[str, Any]) -> tuple[int, int]:
    if view["source_type"] == "video":
        capture = cv2.VideoCapture(view["video_path"])
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        return width, height
    first = read_jsonl(Path(view["manifest_path"]))[0]
    path = Path(first["rgb_path"])
    if not path.is_absolute():
        path = Path(view["manifest_path"]).parent / path
    with Image.open(path) as image:
        return image.size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    session = json.loads(args.session.read_text(encoding="utf-8"))
    events = {event["event_id"]: event for event in session["events"]}
    view_lookup = {
        (event["event_id"], view["camera"]): view
        for event in session["events"]
        for view in event["views"]
    }
    size_lookup = {key: view_size(view) for key, view in view_lookup.items()}
    records: dict[tuple[int, int, str], dict[str, Any]] = defaultdict(dict)
    for row in read_jsonl(args.tracks):
        event = events[row["event_id"]]
        frame = int(row["frame_index"])
        camera = row["camera"]
        width, height = size_lookup[(row["event_id"], camera)]
        bbox = row.get("bbox_xyxy")
        visible = bool(bbox) and float(row.get("area_ratio", 0.0)) > 0
        key = (int(event["episode_index"]), frame, row["event_id"])
        records[key][camera] = {
            "bbox_xyxy": bbox,
            "bbox_normalized": list(normalize_box(bbox, width, height)) if bbox else None,
            "visible": visible,
            "mask_path": row.get("mask_path"),
            "instance_id": f"{row['event_id']}:{camera}:target",
            "phase": phase_for_frame(
                frame,
                int(row["start_frame_index"]),
                int(row["anchor_frame_index"]),
                int(row["release_frame_index"]),
                int(row["end_frame_index"]),
            ),
            "confidence": row.get("confidence"),
            "needs_review": bool(row.get("needs_review")),
            "image_size": [width, height],
        }
    rows = [
        {
            "schema": "lerobot_target_annotation_sidecar_v1",
            "episode_index": episode,
            "frame_index": frame,
            "event_id": event_id,
            "targets": targets,
        }
        for (episode, frame, event_id), targets in sorted(records.items())
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "frames": len(rows),
                "camera_annotations": sum(len(row["targets"]) for row in rows),
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
