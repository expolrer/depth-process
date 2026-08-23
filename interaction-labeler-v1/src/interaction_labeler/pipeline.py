from __future__ import annotations

import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .io import atomic_write_json, read_json, read_jsonl, write_jsonl
from .task import prompt_text

StatusCallback = Callable[[dict[str, Any]], None]
ROLE_CAMERA = {"head": "cam_h", "left_wrist": "cam_l", "right_wrist": "cam_r"}


def default_engine_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "sequence"


def _set_status(workspace: Path, stage: str, state: str, **extra: Any) -> dict[str, Any]:
    value = {
        "stage": stage,
        "state": state,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **extra,
    }
    atomic_write_json(workspace / "pipeline_status.json", value)
    return value


def _run(
    command: list[str],
    cwd: Path,
    workspace: Path,
    stage: str,
    callback: StatusCallback | None,
) -> None:
    status = _set_status(workspace, stage, "running", command=command)
    if callback:
        callback(status)
    log_path = workspace / "logs" / f"{stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
        exit_code = process.wait()
    if exit_code:
        status = _set_status(
            workspace,
            stage,
            "failed",
            exit_code=exit_code,
            log=str(log_path),
        )
        if callback:
            callback(status)
        raise RuntimeError(f"pipeline stage {stage} failed; see {log_path}")
    status = _set_status(workspace, stage, "completed", log=str(log_path))
    if callback:
        callback(status)


def _extract_video(video_path: Path, output_dir: Path) -> tuple[list[Path], float]:
    existing = sorted(output_dir.glob("*.jpg")) if output_dir.exists() else []
    if existing:
        capture = cv2.VideoCapture(str(video_path))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        capture.release()
        return existing, fps
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open LeRobot video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    paths = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        path = output_dir / f"{index:06d}.jpg"
        if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"failed to write extracted frame: {path}")
        paths.append(path)
        index += 1
    capture.release()
    if not paths:
        raise RuntimeError(f"video contains no decodable frames: {video_path}")
    return paths, fps


def materialize_lerobot(session: dict[str, Any], workspace: Path) -> dict[str, Any]:
    extracted_root = workspace / "outputs" / "extracted"
    signals_root = workspace / "outputs" / "robot_signals"
    summaries = []
    camera_map: dict[tuple[str, str], str] = {}
    for event in session["events"]:
        sequence = _slug(event["sequence"])
        event_rows = []
        for view in event["views"]:
            canonical_camera = ROLE_CAMERA.get(view["role"])
            if canonical_camera is None:
                continue
            camera_map[(event["event_id"], view["camera"])] = canonical_camera
            video_path = Path(view["video_path"])
            rgb_dir = extracted_root / sequence / canonical_camera / "rgb"
            frames, fps = _extract_video(video_path, rgb_dir)
            height, width = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR).shape[:2]
            zero_depth = extracted_root / sequence / canonical_camera / "zero_depth_mm.png"
            if not zero_depth.exists():
                Image.fromarray(np.zeros((height, width), dtype=np.uint16)).save(zero_depth)
            manifest_rows = []
            for frame_index, frame_path in enumerate(frames):
                manifest_rows.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_ns": int(frame_index / fps * 1e9),
                        "rgb_path": str(frame_path.resolve()),
                        "depth_raw_mm_path": str(zero_depth.resolve()),
                        "depth_aligned_rgb_mm_path": str(zero_depth.resolve()),
                    }
                )
            manifest_path = extracted_root / sequence / canonical_camera / "manifest.jsonl"
            write_jsonl(manifest_path, manifest_rows)
            contact = min(len(frames) - 1, int(view["contact_frame"]))
            event_timestamp = int(contact / fps * 1e9)
            robot_rows = [
                {
                    "frame_index": index,
                    "timestamp_ns": int(index / fps * 1e9),
                    "eef_pose": {},
                    "arm_joint_state": {"position": []},
                }
                for index in range(len(frames))
            ]
            camera_signal_root = signals_root / sequence / canonical_camera
            write_jsonl(camera_signal_root / "robot_signals.jsonl", robot_rows)
            camera_event = {
                "kind": "confirmed_grab",
                "side": event["side"],
                "frame_index": contact,
                "timestamp_ns": event_timestamp,
                "amplitude": 100.0,
            }
            write_jsonl(camera_signal_root / "events.jsonl", [camera_event])
            event_rows.append(camera_event)
            view.update(
                {
                    "source_camera": view["camera"],
                    "camera": canonical_camera,
                    "source_type": "manifest",
                    "manifest_path": str(manifest_path.resolve()),
                    "frame_count": len(frames),
                }
            )
        summaries.append(
            {
                "sequence": sequence,
                "events": event_rows[:1],
            }
        )
        event["sequence"] = sequence
        event["event_id"] = f"{sequence}:grasp_000_{event['side']}"
    atomic_write_json(signals_root / "summary.json", {"sequences": summaries})
    session.update(
        {
            "input_format": "lerobot_materialized",
            "manifest_root": str(extracted_root.resolve()),
            "signals_root": str(signals_root.resolve()),
        }
    )
    annotations_path = workspace / "annotations.json"
    annotations = read_json(annotations_path, {"revision": 0, "items": []})
    for item in annotations["items"]:
        mapped = camera_map.get((item["event_id"], item["camera"]))
        if mapped:
            item["camera"] = mapped
        for event in session["events"]:
            if item["event_id"].split(":", 1)[0] == event["sequence"]:
                item["event_id"] = event["event_id"]
                break
    atomic_write_json(annotations_path, annotations)
    atomic_write_json(workspace / "session.json", session)
    return session


def merge_manual_candidates(workspace: Path) -> None:
    candidate_path = workspace / "outputs" / "grounded_candidates" / "candidate_index.jsonl"
    rows = read_jsonl(candidate_path)
    annotations = read_json(workspace / "annotations.json", {"items": []})["items"]
    session = read_json(workspace / "session.json", {})
    contact_frames = {
        (event["event_id"], view["camera"]): int(view["contact_frame"])
        for event in session.get("events", [])
        for view in event.get("views", [])
    }
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for item in annotations:
        if item.get("deleted"):
            continue
        if int(item["frame_index"]) != contact_frames.get((item["event_id"], item["camera"])):
            continue
        latest[(item["event_id"], item["camera"])] = item
    overrides = []
    for row in rows:
        manual = latest.get((row["event_id"], row["camera"]))
        if not manual:
            continue
        candidate_id = max([int(item["candidate_id"]) for item in row["candidates"]] + [-1]) + 1
        row["candidates"].append(
            {
                "candidate_id": candidate_id,
                "label": manual.get("label", "manual target"),
                "role": "object_candidate",
                "score": 1.0,
                "box_xyxy": [float(value) for value in manual["box_xyxy"]],
                "source": "human_preannotation",
            }
        )
        overrides.append(
            {
                "event_id": row["event_id"],
                "camera": row["camera"],
                "candidate_id": candidate_id,
                "reason": manual.get("note", "Interactive manual box override"),
            }
        )
    write_jsonl(candidate_path, rows)
    write_jsonl(
        workspace / "outputs" / "interaction_candidates" / "human_overrides.jsonl", overrides
    )


def build_required_selections(workspace: Path) -> Path:
    outputs = workspace / "outputs"
    ranked = read_jsonl(outputs / "interaction_candidates" / "ranked_index.jsonl")
    manual = {
        (row["event_id"], row["camera"]): row
        for row in read_jsonl(outputs / "interaction_candidates" / "human_overrides.jsonl")
    }
    vlm = {
        row["event_id"]: row
        for row in read_jsonl(outputs / "interaction_candidates" / "vlm_resolutions.jsonl")
        if row.get("valid_choice")
    }
    v2 = {
        row["event_id"]: row
        for row in read_jsonl(workspace / "v2" / "instance_evidence.jsonl")
        if row.get("selected_instance_id")
    }
    session = read_json(workspace / "session.json", {})
    contact_frames = {
        (event["event_id"], view["camera"]): int(view["contact_frame"])
        for event in session.get("events", [])
        for view in event.get("views", [])
    }
    secondary_by_view: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in read_json(workspace / "annotations.json", {"items": []}).get("items", []):
        key = (item["event_id"], item["camera"])
        if int(item["frame_index"]) == contact_frames.get(key):
            continue
        secondary_by_view.setdefault(key, []).append(
            {
                "frame_index": int(item["frame_index"]),
                "box_xyxy": [float(value) for value in item["box_xyxy"]],
                "source": "human_post_correction",
            }
        )
    by_event: dict[str, list[dict[str, Any]]] = {}
    for row in ranked:
        by_event.setdefault(row["event_id"], []).append(row)
    selections = []
    for event_id, rows in by_event.items():
        side = rows[0]["side"]
        required = (("head", "cam_h"), ("wrist", "cam_l" if side == "left" else "cam_r"))
        for role, camera in required:
            row = next((item for item in rows if item["camera"] == camera), None)
            if row is None or not row["ranked_candidates"]:
                continue
            override = manual.get((event_id, camera))
            choice = vlm.get(event_id, {}).get("choice") or {}
            v2_choice = v2.get(event_id, {})
            v2_camera, _, v2_candidate = str(v2_choice.get("selected_instance_id", "")).partition(
                ":"
            )
            available_ids = {
                int(candidate["candidate_id"]) for candidate in row["ranked_candidates"]
            }
            if override:
                candidate_id = int(override["candidate_id"])
                source = "human_preannotation"
                evidence = override.get("reason")
            elif v2_camera == camera and v2_candidate and int(v2_candidate) in available_ids:
                candidate_id = int(v2_candidate)
                source = "v2_interaction_evidence"
                evidence = (
                    "Task descriptor, contact, head-scene motion and active-wrist cross-view evidence; "
                    f"score={float(v2_choice.get('score', 0.0)):.3f}, "
                    f"margin={float(v2_choice.get('margin', 0.0)):.3f}."
                )
            elif (
                choice.get("camera") == camera
                and int(choice.get("candidate_id", -1)) in available_ids
            ):
                candidate_id = int(choice["candidate_id"])
                source = "vlm_ambiguity_resolution"
                evidence = choice.get("reason_zh", choice.get("reason"))
            else:
                candidate_id = int(row["selected_candidate_id"])
                source = "automatic_interaction_ranking"
                evidence = "Grounding score, gripper proximity, RGB-D and camera-role ranking."
            candidate = next(
                item
                for item in row["ranked_candidates"]
                if int(item["candidate_id"]) == candidate_id
            )
            box = [float(value) for value in candidate["box_xyxy"]]
            selections.append(
                {
                    "event_id": event_id,
                    "sequence": row["sequence"],
                    "event_index": row["event_index"],
                    "side": side,
                    "role": role,
                    "camera": camera,
                    "candidate_id": candidate_id,
                    "candidate_label": candidate.get("label", "target"),
                    "box_xyxy": box,
                    "selection_source": source,
                    "confidence": candidate.get("interaction_score", candidate.get("score")),
                    "evidence": evidence,
                    "needs_review": bool(row.get("needs_vlm")),
                    "secondary_prompts": sorted(
                        secondary_by_view.get((event_id, camera), []),
                        key=lambda item: item["frame_index"],
                    ),
                    "visibility_exceptions": [],
                    "start_frame_override": None,
                    "end_frame_override": None,
                }
            )
    output = outputs / "interaction_review" / "required_view_selections.jsonl"
    write_jsonl(output, selections)
    return output


def run_required_tracking(
    workspace: Path,
    engine_root: Path | None = None,
    sam2_checkpoint: Path | None = None,
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    device: str = "cuda:0",
    callback: StatusCallback | None = None,
) -> None:
    workspace = workspace.resolve()
    engine_root = (engine_root or default_engine_root()).resolve()
    session = read_json(workspace / "session.json")
    if not session:
        raise FileNotFoundError(f"session.json is missing under {workspace}")
    outputs = workspace / "outputs"
    extracted_root = Path(session["manifest_root"])
    signals_root = (
        Path(session["signals_root"]) if session.get("signals_root") else outputs / "robot_signals"
    )
    sam2_checkpoint = (
        sam2_checkpoint or engine_root / "models" / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt"
    )
    selections = build_required_selections(workspace)
    for role in ("head", "wrist"):
        command = [
            sys.executable,
            str(engine_root / "scripts" / "track_required_views_sam2.py"),
            "--role",
            role,
            "--selections",
            str(selections),
            "--ranked-index",
            str(outputs / "interaction_candidates" / "ranked_index.jsonl"),
            "--signals-root",
            str(signals_root),
            "--extracted-root",
            str(extracted_root),
            "--output-root",
            str(outputs / "target_tracks_required"),
            "--checkpoint",
            str(sam2_checkpoint),
            "--config",
            sam2_config,
            "--device",
            device,
        ]
        _run(command, engine_root, workspace, f"sam2_{role}", callback)
    _merge_tracks(workspace)


def _merge_tracks(workspace: Path) -> None:
    root = workspace / "outputs" / "target_tracks_required"
    rows = read_jsonl(root / "head" / "track_index.jsonl") + read_jsonl(
        root / "wrist" / "track_index.jsonl"
    )
    details = []
    for role in ("head", "wrist"):
        summary = read_json(root / role / "summary.json", {})
        details.extend(summary.get("views_detail", []))
    write_jsonl(root / "track_index.jsonl", rows)
    atomic_write_json(
        root / "summary.json",
        {
            "schema": "required_dual_view_tracks_v1",
            "events": len({row["event_id"] for row in rows}),
            "views": len(details),
            "tracked_frames": len(rows),
            "quality_flagged_frames": sum(bool(row.get("needs_review")) for row in rows),
            "views_detail": details,
        },
    )


def run_pipeline(
    workspace: Path,
    engine_root: Path | None = None,
    grounding_model: Path | None = None,
    sam2_checkpoint: Path | None = None,
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    vlm_model: Path | None = None,
    detector_backend: str = "groundingdino",
    device: str = "cuda:0",
    skip_vlm: bool = False,
    skip_tracking: bool = False,
    callback: StatusCallback | None = None,
) -> None:
    workspace = workspace.resolve()
    engine_root = (engine_root or default_engine_root()).resolve()
    session = read_json(workspace / "session.json")
    if not session:
        raise FileNotFoundError(f"session.json is missing under {workspace}")
    try:
        if session["input_format"] == "lerobot":
            status = _set_status(workspace, "materialize_lerobot", "running")
            if callback:
                callback(status)
            session = materialize_lerobot(session, workspace)
        outputs = workspace / "outputs"
        extracted_root = Path(session["manifest_root"])
        signals_root = (
            Path(session["signals_root"])
            if session.get("signals_root")
            else outputs / "robot_signals"
        )
        grounding_model = grounding_model or engine_root / "models" / "grounding-dino-base"
        sam2_checkpoint = (
            sam2_checkpoint
            or engine_root / "models" / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt"
        )
        detect_command = [
            sys.executable,
            str(engine_root / "scripts" / "detect_grounded_candidates.py"),
            "--model",
            str(grounding_model),
            "--backend",
            detector_backend,
            "--signals-root",
            str(signals_root),
            "--extracted-root",
            str(extracted_root),
            "--output-root",
            str(outputs / "grounded_candidates"),
            "--device",
            device,
            "--prompt",
            prompt_text(session["task"]),
        ]
        _run(detect_command, engine_root, workspace, "grounding_candidates", callback)
        merge_manual_candidates(workspace)
        rank_command = [
            sys.executable,
            str(engine_root / "scripts" / "rank_interaction_candidates.py"),
            "--candidate-index",
            str(outputs / "grounded_candidates" / "candidate_index.jsonl"),
            "--signals-root",
            str(signals_root),
            "--output-root",
            str(outputs / "interaction_candidates"),
        ]
        _run(rank_command, engine_root, workspace, "interaction_ranking", callback)
        if not skip_vlm and vlm_model:
            vlm_command = [
                sys.executable,
                str(engine_root / "scripts" / "resolve_ambiguity_with_vlm.py"),
                "--queue",
                str(outputs / "interaction_candidates" / "ambiguity_queue.jsonl"),
                "--model",
                str(vlm_model),
                "--output",
                str(outputs / "interaction_candidates" / "vlm_resolutions.jsonl"),
                "--device",
                device,
                "--instruction",
                session["task"]["instruction"],
            ]
            _run(vlm_command, engine_root, workspace, "vlm_ambiguity", callback)
        if not skip_tracking:
            run_required_tracking(
                workspace,
                engine_root=engine_root,
                sam2_checkpoint=sam2_checkpoint,
                sam2_config=sam2_config,
                device=device,
                callback=callback,
            )
        else:
            build_required_selections(workspace)
        session["phase"] = "ready_for_post_review"
        atomic_write_json(workspace / "session.json", session)
        status = _set_status(workspace, "complete", "completed")
        if callback:
            callback(status)
    except Exception as error:
        status = _set_status(workspace, "pipeline", "failed", error=str(error))
        if callback:
            callback(status)
        raise
