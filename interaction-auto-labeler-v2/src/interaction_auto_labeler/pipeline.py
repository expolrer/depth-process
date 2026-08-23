from __future__ import annotations

import io
import json
import math
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

from PIL import Image

from .backends import load_detector
from .cross_view import soft_cross_view_score
from .inventory import (
    build_scene_tracks,
    center,
    inventory_schedule,
    match_scene_track,
    source_destination_score,
    summarize_track,
    track_co_motion,
)
from .plan import build_plan
from .schema import EvidenceWeights, load_v2_task
from .scoring import rank_candidates


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def initialize_v2(workspace: Path, task: dict[str, Any]) -> dict[str, Any]:
    session_path = workspace / "session.json"
    session = _read_json(session_path)
    session["pipeline_version"] = "v2"
    session["v2"] = task["v2"]
    session["task"] = {**session.get("task", {}), **task}
    _write_json(session_path, session)
    _write_json(workspace / "v2" / "task_descriptor.json", task)
    plan = build_plan(session, task, workspace)
    _write_json(workspace / "v2" / "plan.json", plan)
    return plan


def write_action_phases(workspace: Path) -> list[dict[str, Any]]:
    session = _read_json(workspace / "session.json", {})
    rows = []
    for event in session.get("events", []):
        for view in event.get("views", []):
            rows.append(
                {
                    "event_id": event["event_id"],
                    "camera": view["camera"],
                    "side": event["side"],
                    "start_frame": int(view["start_frame"]),
                    "contact_frame": int(view["contact_frame"]),
                    "transport_frame": round(
                        (int(view["contact_frame"]) + int(view["release_frame"])) / 2
                    ),
                    "release_frame": int(view["release_frame"]),
                    "end_frame": int(view["end_frame"]),
                    "source": "adapter_or_reviewed_session_timing",
                }
            )
    _write_jsonl(workspace / "v2" / "action_phases.jsonl", rows)
    return rows


def run_head_inventory(
    workspace: Path,
    task: dict[str, Any],
    model_path: Path | None,
    device: str,
) -> list[dict[str, Any]]:
    from interaction_labeler.server import LabelerApplication

    session = _read_json(workspace / "session.json", {})
    backend_name = "sam3" if task["v2"]["detector_backend"] == "sam3" else "grounded-sam2"
    detector = load_detector(
        "sam3" if backend_name == "sam3" else "groundingdino", model_path, device
    )
    app = LabelerApplication(workspace)
    rows = []
    try:
        for event in session["events"]:
            head = next((view for view in event["views"] if view["role"] == "head"), None)
            if not head:
                continue
            for frame_index in inventory_schedule(head, int(task["v2"]["inventory_stride"])):
                with Image.open(
                    io.BytesIO(app.frame(event["event_id"], head["camera"], frame_index))
                ) as source:
                    image = source.convert("RGB")
                detections = detector.detect(image, list(task["detector_prompt"]))
                rows.append(
                    {
                        "event_id": event["event_id"],
                        "camera": head["camera"],
                        "frame_index": frame_index,
                        "image_size": [image.width, image.height],
                        "detections": detections,
                    }
                )
    finally:
        del detector
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    _write_jsonl(workspace / "v2" / "head_inventory.jsonl", rows)
    return rows


def _selected_candidate(camera_row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not camera_row:
        return None
    selected_id = camera_row.get("selected_candidate_id")
    return next(
        (
            candidate
            for candidate in camera_row.get("ranked_candidates", [])
            if int(candidate["candidate_id"]) == int(selected_id)
        ),
        None,
    )


def _label_similarity(first: Any, second: Any) -> float:
    first_label = str(first or "").strip().casefold()
    second_label = str(second or "").strip().casefold()
    if not first_label or not second_label:
        return 0.5
    if first_label == second_label:
        return 1.0
    if first_label in second_label or second_label in first_label:
        return 0.8
    return 0.35


def _build_v2_evidence(
    workspace: Path,
    task: dict[str, Any],
    include_policy_tracks: bool = True,
) -> None:
    session = _read_json(workspace / "session.json", {})
    event_metadata = {event["event_id"]: event for event in session.get("events", [])}
    ranked_rows = _read_jsonl(
        workspace / "outputs" / "interaction_candidates" / "ranked_index.jsonl"
    )
    track_rows = (
        _read_jsonl(workspace / "outputs" / "target_tracks_required" / "track_index.jsonl")
        if include_policy_tracks
        else []
    )
    tracks: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in track_rows:
        tracks.setdefault((row["event_id"], row["camera"]), []).append(row)
    by_event: dict[str, list[dict[str, Any]]] = {}
    for row in ranked_rows:
        by_event.setdefault(row["event_id"], []).append(row)
    inventory_by_event: dict[str, list[dict[str, Any]]] = {}
    for row in _read_jsonl(workspace / "v2" / "head_inventory.jsonl"):
        inventory_by_event.setdefault(row["event_id"], []).append(row)
    prior_evidence = (
        {row["event_id"]: row for row in _read_jsonl(workspace / "v2" / "instance_evidence.jsonl")}
        if include_policy_tracks
        else {}
    )
    weights_payload = task["v2"]["weights"]
    allowed_weights = {field.name for field in fields(EvidenceWeights)}
    weights = EvidenceWeights(
        **{key: value for key, value in weights_payload.items() if key in allowed_weights}
    )
    evidence_rows, links, review_queue, scene_track_rows = [], [], [], []
    review_config = task["v2"].get("review", {})
    for event_id, camera_rows in by_event.items():
        event = event_metadata.get(event_id, {})
        side = event.get("side", camera_rows[0].get("side", "right"))
        view_roles = {view["camera"]: view.get("role", "other") for view in event.get("views", [])}
        rows_by_role = {view_roles.get(row["camera"], "other"): row for row in camera_rows}
        head_row = rows_by_role.get("head")
        active_role = "left_wrist" if side == "left" else "right_wrist"
        active_wrist_row = rows_by_role.get(active_role)
        primary_row = head_row or active_wrist_row
        if primary_row is None:
            primary_row = camera_rows[0]

        selected_primary = _selected_candidate(primary_row)
        prior_instance_id = str(prior_evidence.get(event_id, {}).get("selected_instance_id", ""))
        prior_camera, _, prior_candidate_id = prior_instance_id.partition(":")
        if prior_camera == primary_row["camera"] and prior_candidate_id:
            selected_primary = next(
                (
                    candidate
                    for candidate in primary_row.get("ranked_candidates", [])
                    if int(candidate["candidate_id"]) == int(prior_candidate_id)
                ),
                selected_primary,
            )
        selected_wrist = _selected_candidate(active_wrist_row)
        primary_view = next(
            (view for view in event.get("views", []) if view["camera"] == primary_row["camera"]),
            {},
        )
        inventory_rows = inventory_by_event.get(event_id, []) if head_row else []
        inventory_gap = max(45, int(task["v2"].get("inventory_stride", 10)) * 3)
        scene_object_tracks = build_scene_tracks(
            inventory_rows, kind="object", max_frame_gap=inventory_gap
        )
        scene_gripper_tracks = build_scene_tracks(
            inventory_rows, kind="gripper", max_frame_gap=inventory_gap
        )
        for scene_track in scene_object_tracks + scene_gripper_tracks:
            scene_track_rows.append({"event_id": event_id, **scene_track})
        if inventory_rows:
            width, height = inventory_rows[0].get("image_size", [848, 480])
        else:
            width, height = 848, 480
        image_diagonal = math.hypot(float(width), float(height))
        contact_frame = int(primary_view.get("contact_frame", primary_row.get("frame_index", 0)))
        release_frame = int(primary_view.get("release_frame", contact_frame))
        selected_track = sorted(
            tracks.get((event_id, primary_row["camera"]), []),
            key=lambda row: int(row["frame_index"]),
        )
        selected_summary = summarize_track(selected_track, image_diagonal) if selected_track else {}

        geometry_applied = False
        geometric_result: dict[str, Any] = {}
        if primary_row.get("ranked_candidates") and selected_wrist and active_wrist_row:
            geometric_result = _read_json(
                workspace / "v2" / "calibrated_cross_view" / f"{event_id.replace(':', '_')}.json",
                {},
            )
            geometry_applied = bool(geometric_result.get("projected_box_xyxy"))

        candidates = []
        for candidate in primary_row.get("ranked_candidates", []):
            is_tracked_instance = bool(
                selected_primary
                and int(candidate["candidate_id"]) == int(selected_primary["candidate_id"])
            )
            cross_view = None
            if selected_wrist:
                cross_view = soft_cross_view_score(
                    {
                        "appearance_similarity": _label_similarity(
                            candidate.get("label"), selected_wrist.get("label")
                        ),
                        "depth_consistency": min(
                            float(candidate.get("depth_score", 0.5)),
                            float(selected_wrist.get("depth_score", 0.5)),
                        ),
                        "contact_sync": 1.0,
                    }
                )
            scene_track, inventory_association = match_scene_track(
                candidate,
                scene_object_tracks,
                contact_frame,
                image_diagonal,
            )
            inventory_summary = {}
            if scene_track:
                inventory_summary = summarize_track(
                    [
                        {
                            "frame_index": observation["frame_index"],
                            "bbox_xyxy": observation["box_xyxy"],
                        }
                        for observation in scene_track["observations"]
                    ],
                    image_diagonal,
                )
                association_scores = [
                    float(observation["association_score"])
                    for observation in scene_track["observations"][1:]
                    if observation.get("association_score") is not None
                ]
                inventory_summary["min_scene_association"] = (
                    min(association_scores) if association_scores else 1.0
                )
            quality_summary = inventory_summary
            if is_tracked_instance and selected_summary:
                quality_summary = {**inventory_summary, **selected_summary}
            co_motion = track_co_motion(
                scene_track,
                scene_gripper_tracks,
                contact_frame,
                release_frame,
                image_diagonal,
            )
            source_destination = None
            target_descriptor = task.get("target", {})
            source_box = target_descriptor.get("source_box_xyxy")
            destination_box = target_descriptor.get("destination_box_xyxy")
            if scene_track and scene_track["observations"]:
                source_destination = source_destination_score(
                    center(scene_track["observations"][0]["box_xyxy"]),
                    center(scene_track["observations"][-1]["box_xyxy"]),
                    source_box,
                    destination_box,
                )
            candidates.append(
                {
                    "instance_id": f"{primary_row['camera']}:{candidate['candidate_id']}",
                    "camera": primary_row["camera"],
                    "contact": 0.5 * float(candidate.get("proximity_score", 0.5))
                    + 0.5 * float(candidate.get("depth_score", 0.5)),
                    "co_motion": co_motion,
                    "start_end_displacement": inventory_summary.get("start_end_displacement"),
                    "source_destination": source_destination,
                    "descriptor_match": float(
                        candidate.get("detector_score", candidate.get("score", 0.0))
                    ),
                    "cross_view": cross_view,
                    "track_summary": quality_summary,
                    "scene_track_id": scene_track.get("scene_track_id") if scene_track else None,
                    "inventory_association": inventory_association,
                    "candidate_id": candidate["candidate_id"],
                    "label": candidate.get("label"),
                    "box_xyxy": candidate.get("box_xyxy"),
                }
            )
        result = rank_candidates(candidates, weights, float(task["v2"]["ambiguity_margin"]))
        if prior_instance_id and any(
            candidate["instance_id"] == prior_instance_id
            for candidate in result["ranked_candidates"]
        ):
            post_track_top = result["selected_instance_id"]
            selected_row = next(
                candidate
                for candidate in result["ranked_candidates"]
                if candidate["instance_id"] == prior_instance_id
            )
            competitors = [
                candidate["interaction_instance_score"]
                for candidate in result["ranked_candidates"]
                if candidate["instance_id"] != prior_instance_id
            ]
            result["post_track_top_instance_id"] = post_track_top
            result["selected_instance_id"] = prior_instance_id
            result["score"] = selected_row["interaction_instance_score"]
            result["margin"] = result["score"] - max(competitors, default=0.0)
            result["needs_review"] = post_track_top != prior_instance_id or result[
                "margin"
            ] < float(task["v2"]["ambiguity_margin"])
        result["event_id"] = event_id
        result["primary_camera"] = primary_row["camera"]
        result["active_wrist_camera"] = active_wrist_row["camera"] if active_wrist_row else None
        selected_result = next(
            (
                candidate
                for candidate in result["ranked_candidates"]
                if candidate["instance_id"] == result["selected_instance_id"]
            ),
            {},
        )
        if selected_result and selected_wrist and active_wrist_row:
            links.append(
                {
                    "event_id": event_id,
                    "source_camera": primary_row["camera"],
                    "target_camera": active_wrist_row["camera"],
                    "source_instance_id": result["selected_instance_id"],
                    "target_instance_id": (
                        f"{active_wrist_row['camera']}:{selected_wrist['candidate_id']}"
                    ),
                    "mode": task["v2"]["cross_view_mode"],
                    "score": selected_result.get("cross_view"),
                    "is_geometric_projection": geometry_applied,
                    "projected_box_xyxy": geometric_result.get("projected_box_xyxy"),
                }
            )
        evidence_rows.append(result)
        flags = []
        selected_summary_for_review = selected_result.get("track_summary", {})
        if selected_summary_for_review.get("visibility_ratio", 1.0) < float(
            review_config.get("min_track_visibility", 0.8)
        ):
            flags.append("low_track_visibility")
        area_jump = selected_summary_for_review.get("area_jump_ratio")
        if area_jump is not None and area_jump > float(review_config.get("max_area_jump", 2.5)):
            flags.append("mask_area_jump")
        if result.get("post_track_top_instance_id") not in {
            None,
            result["selected_instance_id"],
        }:
            flags.append("post_track_selection_disagreement")
        association = selected_result.get("inventory_association")
        if inventory_rows and (association is None or association < 0.45):
            flags.append("low_inventory_association")
        min_scene_association = selected_result.get("track_summary", {}).get(
            "min_scene_association"
        )
        if min_scene_association is not None and min_scene_association < 0.55:
            flags.append("head_inventory_track_drift")
        if result["needs_review"]:
            flags.append("low_selection_margin")
        if task["v2"]["cross_view_mode"] == "calibrated" and not geometry_applied:
            flags.append("calibrated_projection_missing")
        if flags:
            review_queue.append(
                {
                    "event_id": event_id,
                    "flags": sorted(set(flags)),
                    "score": result["score"],
                    "margin": result["margin"],
                }
            )
    _write_jsonl(workspace / "v2" / "instance_evidence.jsonl", evidence_rows)
    _write_jsonl(workspace / "v2" / "head_scene_tracks.jsonl", scene_track_rows)
    _write_jsonl(workspace / "v2" / "cross_view_links.jsonl", links)
    _write_jsonl(workspace / "v2" / "review_queue.jsonl", review_queue)
    _write_json(
        workspace / "v2" / "summary.json",
        {
            "schema": "robot_interaction_auto_label_summary_v2",
            "events": len(evidence_rows),
            "cross_view_links": len(links),
            "events_needing_review": len(review_queue),
        },
    )


def run_v2_pipeline(
    workspace: Path,
    v2_task_path: Path,
    run_inventory: bool = True,
    detector_backend: str | None = None,
    grounding_model: Path | None = None,
    device: str = "cuda:0",
    **v1_options: Any,
) -> None:
    from interaction_labeler.pipeline import (
        _set_status,
        materialize_lerobot,
        run_pipeline,
        run_required_tracking,
    )

    file_task = load_v2_task(v2_task_path)
    session_task = _read_json(workspace / "session.json", {}).get("task", {})
    task = {
        **file_task,
        **session_task,
        "target": {**file_task.get("target", {}), **session_task.get("target", {})},
        "v2": {**file_task["v2"], **session_task.get("v2", {})},
    }
    _write_json(workspace / "v2" / "task_descriptor.json", task)
    backend = detector_backend or (
        "sam3" if task["v2"]["detector_backend"] == "sam3" else "groundingdino"
    )
    skip_tracking = bool(v1_options.pop("skip_tracking", False))
    try:
        session = _read_json(workspace / "session.json", {})
        if session.get("input_format") == "lerobot":
            _set_status(workspace, "materialize_lerobot", "running")
            materialize_lerobot(session, workspace)
        _set_status(workspace, "v2_action_phases", "running")
        write_action_phases(workspace)
        if run_inventory:
            _set_status(workspace, "v2_head_scene_inventory", "running")
            run_head_inventory(workspace, task, grounding_model, device)
        run_pipeline(
            workspace,
            detector_backend=backend,
            grounding_model=grounding_model,
            device=device,
            skip_tracking=True,
            **v1_options,
        )
        _set_status(workspace, "v2_instance_evidence", "running")
        _build_v2_evidence(workspace, task, include_policy_tracks=False)
        if not skip_tracking:
            _set_status(workspace, "v2_bidirectional_tracking", "running")
            run_required_tracking(
                workspace,
                engine_root=v1_options.get("engine_root"),
                sam2_checkpoint=v1_options.get("sam2_checkpoint"),
                sam2_config=v1_options.get("sam2_config", "configs/sam2.1/sam2.1_hiera_l.yaml"),
                device=device,
            )
            _set_status(workspace, "v2_post_track_evidence", "running")
            _build_v2_evidence(workspace, task, include_policy_tracks=True)
        _set_status(workspace, "complete", "completed")
    except Exception as error:
        _set_status(
            workspace,
            "v2_pipeline",
            "failed",
            error=str(error),
            failed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        )
        raise
