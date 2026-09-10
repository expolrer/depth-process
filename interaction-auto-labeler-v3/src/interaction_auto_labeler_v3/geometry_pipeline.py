from __future__ import annotations

import itertools
import json
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .event_graph import EpisodeEventGraph
from .geometry import (
    CameraIntrinsics,
    backproject_depth,
    fuse_point_clouds,
    invert_transform,
    reproject_mask,
    reprojection_metrics,
    transform_points,
    validate_transform,
    write_ply,
)
from .pose_tracking import ObjectPose, depth_proxy_pose, trajectory_quality


def _resolve(path: str, base: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (base / value).resolve()


def _intrinsics(value: Any, base: Path) -> CameraIntrinsics:
    payload = (
        json.loads(_resolve(value, base).read_text(encoding="utf-8"))
        if isinstance(value, str)
        else value
    )
    return CameraIntrinsics(**payload)


def _transform(value: Any, base: Path) -> np.ndarray:
    payload = (
        json.loads(_resolve(value, base).read_text(encoding="utf-8"))
        if isinstance(value, str)
        else value
    )
    return validate_transform(payload, "world_from_camera")


def _load_view(row: dict[str, Any], base: Path) -> dict[str, Any]:
    depth_raw = np.asarray(Image.open(_resolve(str(row["depth_path"]), base)))
    depth = depth_raw.astype(np.float64) * float(row.get("depth_scale_m", 0.001))
    mask = np.asarray(Image.open(_resolve(str(row["mask_path"]), base))) > 0
    intrinsics = _intrinsics(row["intrinsics"], base)
    if depth.shape != mask.shape:
        raise ValueError("geometry manifest depth and mask shapes differ")
    points_camera, _ = backproject_depth(depth, intrinsics, mask)
    world_from_camera = _transform(row["world_from_camera"], base)
    points_world = transform_points(points_camera, world_from_camera)
    return {
        **row,
        "depth_m": depth,
        "mask": mask,
        "intrinsics_object": intrinsics,
        "world_from_camera_matrix": world_from_camera,
        "points_camera": points_camera,
        "points_world": points_world,
    }


def process_episode_geometry(
    graph: EpisodeEventGraph,
    rows: Iterable[dict[str, Any]],
    manifest_base: Path,
    output_root: Path,
    export_point_clouds: bool = False,
) -> tuple[EpisodeEventGraph, dict[str, Any]]:
    selected_rows = [row for row in rows if str(row["episode_id"]) == graph.episode_id]
    loaded = [_load_view(row, manifest_base) for row in selected_rows]
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for row in loaded:
        by_frame.setdefault(int(row["frame_index"]), []).append(row)

    pose_candidates: dict[int, list[ObjectPose]] = {}
    geometry_by_observation: dict[tuple[str, int], dict[str, Any]] = {}
    fused_outputs = []
    reprojections = []
    failures = []
    for frame_index, frame_rows in sorted(by_frame.items()):
        nonempty = [row for row in frame_rows if len(row["points_camera"])]
        for row in nonempty:
            camera = str(row["camera"])
            minimum_points = int(row.get("minimum_points", 10))
            if len(row["points_camera"]) < minimum_points:
                failures.append(
                    {
                        "frame_index": frame_index,
                        "camera": camera,
                        "reason": "insufficient_target_depth",
                        "valid_points": len(row["points_camera"]),
                        "minimum_points": minimum_points,
                    }
                )
                continue
            pose = depth_proxy_pose(
                frame_index,
                row["depth_m"],
                row["mask"],
                row["intrinsics_object"],
                row["world_from_camera_matrix"],
                minimum_points=minimum_points,
            )
            pose_candidates.setdefault(frame_index, []).append(pose)
            geometry_by_observation[(camera, frame_index)] = {
                "centroid_camera_m": tuple(
                    float(value) for value in np.median(row["points_camera"], axis=0)
                ),
                "centroid_world_m": tuple(
                    float(value) for value in np.median(row["points_world"], axis=0)
                ),
                "pose_world": pose.world_from_object,
            }
        if nonempty:
            fused, counts = fuse_point_clouds(
                [row["points_camera"] for row in nonempty],
                [row["world_from_camera_matrix"] for row in nonempty],
                float(nonempty[0].get("voxel_size_m", 0.005)),
            )
            fused_row = {
                "frame_index": frame_index,
                "camera_count": len(nonempty),
                "input_points": sum(len(row["points_camera"]) for row in nonempty),
                "fused_points": len(fused),
                "mean_observations_per_voxel": float(counts.mean()) if len(counts) else 0.0,
            }
            if export_point_clouds:
                ply_path = output_root / graph.episode_id / f"frame-{frame_index:06d}.ply"
                write_ply(ply_path, fused)
                fused_row["ply_path"] = str(ply_path.resolve())
            fused_outputs.append(fused_row)

        for source, target in itertools.permutations(frame_rows, 2):
            if source["camera"] == target["camera"]:
                continue
            target_from_source = (
                invert_transform(target["world_from_camera_matrix"])
                @ source["world_from_camera_matrix"]
            )
            projected, _ = reproject_mask(
                source["depth_m"],
                source["mask"],
                source["intrinsics_object"],
                target_from_source,
                target["intrinsics_object"],
            )
            reprojections.append(
                {
                    "frame_index": frame_index,
                    "source_camera": source["camera"],
                    "target_camera": target["camera"],
                    **reprojection_metrics(projected, target["mask"]),
                }
            )

    graph.observations = [
        replace(item, **geometry_by_observation[(item.camera, item.frame_index)])
        if (item.camera, item.frame_index) in geometry_by_observation
        else item
        for item in graph.observations
    ]
    best_poses = [
        max(items, key=lambda item: item.confidence) for items in pose_candidates.values()
    ]
    valid_ious = [float(row["iou"]) for row in reprojections]
    report = {
        "schema": "episode_calibrated_geometry_v1",
        "episode_id": graph.episode_id,
        "manifest_rows": len(selected_rows),
        "frames": len(by_frame),
        "observation_updates": sum(
            (item.camera, item.frame_index) in geometry_by_observation
            for item in graph.observations
        ),
        "fused_frames": fused_outputs,
        "reprojections": reprojections,
        "failures": failures,
        "mean_cross_view_iou": float(np.mean(valid_ious)) if valid_ious else None,
        "pose_trajectory": [
            item.to_dict() for item in sorted(best_poses, key=lambda item: item.frame_index)
        ],
        "trajectory_quality": trajectory_quality(best_poses, graph.fps),
        "pose_contract": {
            "metric_translation": True,
            "canonical_rotation": False,
            "reason": "RGB-D centroid/PCA proxy has no CAD or canonical object frame",
        },
    }
    graph.lineage["calibrated_geometry"] = {
        "manifest_rows": len(selected_rows),
        "mean_cross_view_iou": report["mean_cross_view_iou"],
    }
    graph.validate()
    return graph, report
