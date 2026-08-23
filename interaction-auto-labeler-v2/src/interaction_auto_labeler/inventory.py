from __future__ import annotations

import math
from itertools import pairwise
from typing import Any

import numpy as np


def center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def box_iou(first: list[float], second: list[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _is_gripper(label: Any) -> bool:
    lowered = str(label or "").casefold()
    return any(token in lowered for token in ("gripper", "robot hand", "robotic hand", "claw"))


def _label_similarity(first: Any, second: Any) -> float:
    first_label = str(first or "").strip().casefold()
    second_label = str(second or "").strip().casefold()
    if first_label == second_label:
        return 1.0
    if (
        first_label
        and second_label
        and (first_label in second_label or second_label in first_label)
    ):
        return 0.8
    return 0.25


def build_scene_tracks(
    inventory_rows: list[dict[str, Any]],
    kind: str = "object",
    max_frame_gap: int = 45,
) -> list[dict[str, Any]]:
    """Greedily associate sparse inventory detections without pretending to be a mask tracker."""
    if kind not in {"object", "gripper"}:
        raise ValueError("kind must be object or gripper")
    tracks: list[dict[str, Any]] = []
    for row in sorted(inventory_rows, key=lambda item: int(item["frame_index"])):
        frame = int(row["frame_index"])
        width, height = row.get("image_size", [848, 480])
        diagonal = math.hypot(float(width), float(height))
        detections = [
            detection
            for detection in row.get("detections", [])
            if _is_gripper(detection.get("label")) == (kind == "gripper")
        ]
        proposals = []
        for track_index, track in enumerate(tracks):
            previous = track["observations"][-1]
            if frame - int(previous["frame_index"]) > max_frame_gap:
                continue
            for detection_index, detection in enumerate(detections):
                first_center = center(previous["box_xyxy"])
                second_center = center(detection["box_xyxy"])
                distance = math.hypot(
                    second_center[0] - first_center[0], second_center[1] - first_center[1]
                ) / max(diagonal, 1.0)
                association = (
                    0.45 * _label_similarity(previous.get("label"), detection.get("label"))
                    + 0.30 * box_iou(previous["box_xyxy"], detection["box_xyxy"])
                    + 0.25 * math.exp(-distance / 0.12)
                )
                if association >= 0.42:
                    proposals.append((association, track_index, detection_index))
        assigned_tracks: set[int] = set()
        assigned_detections: set[int] = set()
        for association, track_index, detection_index in sorted(proposals, reverse=True):
            if track_index in assigned_tracks or detection_index in assigned_detections:
                continue
            detection = detections[detection_index]
            tracks[track_index]["observations"].append(
                {
                    **detection,
                    "frame_index": frame,
                    "association_score": float(association),
                }
            )
            assigned_tracks.add(track_index)
            assigned_detections.add(detection_index)
        for detection_index, detection in enumerate(detections):
            if detection_index in assigned_detections:
                continue
            tracks.append(
                {
                    "scene_track_id": f"{kind}_{len(tracks):04d}",
                    "kind": kind,
                    "observations": [{**detection, "frame_index": frame, "association_score": 1.0}],
                }
            )
    return tracks


def match_scene_track(
    candidate: dict[str, Any],
    tracks: list[dict[str, Any]],
    frame_index: int,
    image_diagonal: float,
) -> tuple[dict[str, Any] | None, float | None]:
    best_track, best_score = None, None
    for track in tracks:
        observation = min(
            track["observations"],
            key=lambda row: abs(int(row["frame_index"]) - int(frame_index)),
        )
        first_center = center(candidate["box_xyxy"])
        second_center = center(observation["box_xyxy"])
        distance = math.hypot(
            second_center[0] - first_center[0], second_center[1] - first_center[1]
        ) / max(image_diagonal, 1.0)
        frame_penalty = math.exp(-abs(int(observation["frame_index"]) - frame_index) / 30.0)
        score = frame_penalty * (
            0.45 * _label_similarity(candidate.get("label"), observation.get("label"))
            + 0.35 * box_iou(candidate["box_xyxy"], observation["box_xyxy"])
            + 0.20 * math.exp(-distance / 0.12)
        )
        if best_score is None or score > best_score:
            best_track, best_score = track, float(score)
    return (
        (best_track, best_score)
        if best_score is not None and best_score >= 0.25
        else (None, best_score)
    )


def track_co_motion(
    object_track: dict[str, Any] | None,
    gripper_tracks: list[dict[str, Any]],
    start_frame: int,
    end_frame: int,
    image_diagonal: float,
) -> float | None:
    if object_track is None:
        return None
    object_rows = {
        int(row["frame_index"]): row
        for row in object_track["observations"]
        if start_frame <= int(row["frame_index"]) <= end_frame
    }
    best = None
    for gripper_track in gripper_tracks:
        gripper_rows = {
            int(row["frame_index"]): row
            for row in gripper_track["observations"]
            if start_frame <= int(row["frame_index"]) <= end_frame
        }
        common = sorted(set(object_rows).intersection(gripper_rows))
        if len(common) < 3:
            continue
        score = co_motion_score(
            [center(object_rows[frame]["box_xyxy"]) for frame in common],
            [center(gripper_rows[frame]["box_xyxy"]) for frame in common],
            image_diagonal,
        )
        if score is not None and (best is None or score > best):
            best = score
    return best


def summarize_track(observations: list[dict[str, Any]], image_diagonal: float) -> dict[str, Any]:
    visible = [row for row in observations if row.get("bbox_xyxy")]
    if not visible:
        return {
            "visibility_ratio": 0.0,
            "start_end_displacement": 0.0,
            "depth_displacement_m": None,
            "area_jump_ratio": None,
        }
    first, last = visible[0], visible[-1]
    first_center, last_center = center(first["bbox_xyxy"]), center(last["bbox_xyxy"])
    displacement = math.hypot(
        last_center[0] - first_center[0],
        last_center[1] - first_center[1],
    ) / max(image_diagonal, 1.0)
    depths = [row.get("median_depth_m") for row in visible if row.get("median_depth_m") is not None]
    areas = [
        max(
            1.0,
            (row["bbox_xyxy"][2] - row["bbox_xyxy"][0])
            * (row["bbox_xyxy"][3] - row["bbox_xyxy"][1]),
        )
        for row in visible
    ]
    jumps = [max(a, b) / min(a, b) for a, b in pairwise(areas)]
    return {
        "visibility_ratio": len(visible) / max(len(observations), 1),
        "start_end_displacement": min(1.0, displacement / 0.25),
        "depth_displacement_m": abs(float(depths[-1]) - float(depths[0]))
        if len(depths) >= 2
        else None,
        "area_jump_ratio": max(jumps) if jumps else 1.0,
        "first_frame": int(first["frame_index"]),
        "last_frame": int(last["frame_index"]),
    }


def inventory_schedule(view: dict[str, Any], stride: int) -> list[int]:
    forced = {
        int(view["start_frame"]),
        int(view["contact_frame"]),
        int(view["release_frame"]),
        int(view["end_frame"]),
    }
    sampled = set(range(int(view["start_frame"]), int(view["end_frame"]) + 1, max(1, stride)))
    return sorted(forced | sampled)


def co_motion_score(
    object_centers: list[tuple[float, float]],
    gripper_centers: list[tuple[float, float]],
    image_diagonal: float,
) -> float | None:
    if len(object_centers) != len(gripper_centers) or len(object_centers) < 3:
        return None
    relative = np.asarray(object_centers, dtype=np.float64) - np.asarray(
        gripper_centers, dtype=np.float64
    )
    residual = np.linalg.norm(relative - np.median(relative, axis=0), axis=1)
    normalized = float(np.median(residual)) / max(image_diagonal, 1.0)
    return float(math.exp(-normalized / 0.025))


def source_destination_score(
    start_center: tuple[float, float],
    end_center: tuple[float, float],
    source_box: list[float] | None,
    destination_box: list[float] | None,
) -> float | None:
    if source_box is None or destination_box is None:
        return None

    def inside(point: tuple[float, float], box: list[float]) -> bool:
        return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]

    return 1.0 if inside(start_center, source_box) and inside(end_center, destination_box) else 0.0
