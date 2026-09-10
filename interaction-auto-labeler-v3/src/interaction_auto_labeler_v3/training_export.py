from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .event_graph import EpisodeEventGraph


def export_policy_sidecar(
    graphs: Iterable[EpisodeEventGraph], output: Path, include_noncausal_labels: bool = True
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for graph in graphs:
        events_by_frame: dict[int, list[Any]] = defaultdict(list)
        for event in graph.events:
            for frame in range(event.span.start_frame, event.span.end_frame + 1):
                events_by_frame[frame].append(event)
        observations: dict[int, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
        for item in graph.observations:
            observations[item.frame_index][item.camera].append(item)
        task_language = next((item.text for item in graph.language if item.level == "task"), "")
        for frame in sorted(set(events_by_frame) | set(observations)):
            targets: dict[str, list[dict[str, Any]]] = {}
            for camera, items in observations.get(frame, {}).items():
                targets[camera] = [
                    {
                        "entity_id": item.entity_id,
                        "bbox_xyxy": item.bbox_xyxy,
                        "mask_path": item.mask_path,
                        "centroid_camera_m": item.centroid_camera_m,
                        "centroid_world_m": item.centroid_world_m,
                        "visible": item.visible,
                        "confidence": item.confidence,
                    }
                    for item in items
                ]
            events = events_by_frame.get(frame, [])
            rows.append(
                {
                    "schema": "lerobot_v3_interaction_sidecar_v1",
                    "dataset_id": graph.dataset_id,
                    "dataset_version": graph.dataset_version,
                    "episode_id": graph.episode_id,
                    "frame_index": frame,
                    "task_language": task_language,
                    "event_ids": [event.event_id for event in events],
                    "active_arms": list(dict.fromkeys(event.active_arm for event in events)),
                    "targets": targets,
                    "training_contract": {
                        "auxiliary_supervision_allowed": include_noncausal_labels,
                        "policy_input_allowed": False,
                        "reason": "offline tracks can contain future interaction evidence",
                    },
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return {
        "schema": "policy_sidecar_export_report_v1",
        "row_count": len(rows),
        "output": str(output.resolve()),
    }
