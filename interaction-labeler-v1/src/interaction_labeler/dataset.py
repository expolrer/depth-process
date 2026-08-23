from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2

from .io import atomic_write_json, read_json, read_jsonl

CAMERA_ORDER = {"head": 0, "left_wrist": 1, "right_wrist": 2, "other": 3}


def detect_input_format(path: Path) -> str:
    path = path.resolve()
    if path.suffix.lower() == ".bag" or (path.is_dir() and next(path.rglob("*.bag"), None)):
        return "rosbag"
    if path.is_dir() and (path / "meta" / "info.json").exists():
        return "lerobot"
    if path.is_dir() and next(path.rglob("manifest.jsonl"), None):
        return "extracted"
    raise ValueError(f"cannot determine input format for {path}")


def camera_role(camera: str, task: dict[str, Any]) -> str:
    lowered = camera.lower()
    for role, aliases in task["camera_roles"].items():
        if any(str(alias).lower() in lowered for alias in aliases):
            return role
    return "other"


def _active_side(task: dict[str, Any], event: dict[str, Any] | None = None) -> str:
    configured = task.get("active_hand", "auto")
    if configured in {"left", "right"}:
        return configured
    if event and event.get("side") in {"left", "right"}:
        return str(event["side"])
    return "right"


def _manifest_root(path: Path) -> Path:
    candidates = [path, path / "outputs" / "extracted", path / "extracted"]
    for candidate in candidates:
        if candidate.is_dir() and next(candidate.rglob("manifest.jsonl"), None):
            return candidate.resolve()
    raise FileNotFoundError(f"no extracted manifest tree found under {path}")


def _signals_root(input_path: Path, manifest_root: Path, workspace: Path) -> Path | None:
    candidates = [
        workspace / "outputs" / "robot_signals",
        manifest_root.parent / "robot_signals",
        input_path / "outputs" / "robot_signals",
        input_path / "robot_signals",
    ]
    return next(
        (candidate.resolve() for candidate in candidates if (candidate / "summary.json").exists()),
        None,
    )


def _view_from_manifest(
    manifest_path: Path,
    role: str,
    anchor: int,
    start: int,
    release: int,
    end: int,
) -> dict[str, Any]:
    frame_count = sum(1 for line in manifest_path.open("r", encoding="utf-8") if line.strip())
    if frame_count <= 0:
        raise ValueError(f"empty manifest: {manifest_path}")
    clamp = lambda value: max(0, min(frame_count - 1, int(value)))
    return {
        "camera": manifest_path.parent.name,
        "role": role,
        "source_type": "manifest",
        "manifest_path": str(manifest_path.resolve()),
        "frame_count": frame_count,
        "start_frame": clamp(start),
        "contact_frame": clamp(anchor),
        "release_frame": clamp(release),
        "end_frame": clamp(end),
    }


def index_extracted(
    input_path: Path,
    workspace: Path,
    task: dict[str, Any],
    input_format: str = "extracted",
) -> dict[str, Any]:
    manifest_root = _manifest_root(input_path)
    signals_root = _signals_root(input_path, manifest_root, workspace)
    manifests: dict[str, dict[str, Path]] = {}
    for manifest in manifest_root.rglob("manifest.jsonl"):
        relative = manifest.relative_to(manifest_root)
        if len(relative.parts) < 3:
            continue
        sequence, camera = relative.parts[-3], relative.parts[-2]
        manifests.setdefault(sequence, {})[camera] = manifest

    event_rows: list[dict[str, Any]] = []
    summary = read_json(signals_root / "summary.json", {}) if signals_root else {}
    events_by_sequence = {
        row["sequence"]: [
            event
            for event in row.get("events", [])
            if event.get("kind") in {"grasp_command_end", "confirmed_grab"}
            and abs(float(event.get("amplitude", 100.0))) >= 50.0
        ]
        for row in summary.get("sequences", [])
    }
    timing = task["event_detection"]
    for sequence, camera_manifests in sorted(manifests.items()):
        source_events = events_by_sequence.get(sequence, [])
        if not source_events:
            first_manifest = next(iter(camera_manifests.values()))
            count = sum(1 for line in first_manifest.open("r", encoding="utf-8") if line.strip())
            source_events = [
                {
                    "kind": "manual_or_midpoint",
                    "side": _active_side(task),
                    "frame_index": max(0, count // 2),
                    "timestamp_ns": 0,
                }
            ]
        for event_index, event in enumerate(source_events):
            side = _active_side(task, event)
            views = []
            for camera, manifest_path in sorted(
                camera_manifests.items(), key=lambda item: CAMERA_ORDER[camera_role(item[0], task)]
            ):
                mapped_frame = int(event.get("frame_index", 0))
                if signals_root and int(event.get("timestamp_ns", 0)):
                    camera_events = read_jsonl(signals_root / sequence / camera / "events.jsonl")
                    matches = [
                        row
                        for row in camera_events
                        if row.get("side") == side and row.get("kind") == event.get("kind")
                    ]
                    if matches:
                        selected = min(
                            matches,
                            key=lambda row: abs(
                                int(row["timestamp_ns"]) - int(event["timestamp_ns"])
                            ),
                        )
                        mapped_frame = int(selected["frame_index"])
                release = mapped_frame + int(timing["post_release_frames"])
                views.append(
                    _view_from_manifest(
                        manifest_path,
                        camera_role(camera, task),
                        mapped_frame,
                        mapped_frame - int(timing["pre_frames"]),
                        release,
                        min(
                            mapped_frame + int(timing["max_span"]),
                            release + int(timing["post_release_frames"]),
                        ),
                    )
                )
            event_rows.append(
                {
                    "event_id": f"{sequence}:grasp_{event_index:03d}_{side}",
                    "sequence": sequence,
                    "episode_index": event_index,
                    "side": side,
                    "instruction": task["instruction"],
                    "views": views,
                }
            )
    return {
        "schema": "robot_interaction_label_session_v1",
        "pipeline_version": "v1",
        "input_format": input_format,
        "input_path": str(input_path.resolve()),
        "manifest_root": str(manifest_root),
        "signals_root": str(signals_root) if signals_root else None,
        "workspace": str(workspace.resolve()),
        "task": task,
        "events": event_rows,
        "phase": "ready_for_preannotation",
    }


def _episode_index(path: Path) -> int:
    match = re.search(r"episode[_-]?(\d+)", path.stem, flags=re.IGNORECASE)
    return int(match.group(1)) if match else 0


def _video_key(path: Path, videos_root: Path) -> str:
    relative = path.relative_to(videos_root)
    return ".".join(relative.parts[:-1]) or path.parent.name


def index_lerobot(input_path: Path, workspace: Path, task: dict[str, Any]) -> dict[str, Any]:
    videos_root = input_path / "videos"
    videos = sorted(videos_root.rglob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"no LeRobot MP4 files found under {videos_root}")
    episodes: dict[int, list[Path]] = {}
    for video in videos:
        if "depth" in video.as_posix().lower():
            continue
        episodes.setdefault(_episode_index(video), []).append(video)
    events = []
    timing = task["event_detection"]
    for episode, episode_videos in sorted(episodes.items()):
        side = _active_side(task)
        views = []
        for video in episode_videos:
            capture = cv2.VideoCapture(str(video))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
            capture.release()
            if frame_count <= 0:
                continue
            contact = frame_count // 2
            camera = _video_key(video, videos_root)
            views.append(
                {
                    "camera": camera,
                    "role": camera_role(camera, task),
                    "source_type": "video",
                    "video_path": str(video.resolve()),
                    "fps": fps,
                    "frame_count": frame_count,
                    "start_frame": max(0, contact - int(timing["pre_frames"])),
                    "contact_frame": contact,
                    "release_frame": min(
                        frame_count - 1, contact + int(timing["post_release_frames"])
                    ),
                    "end_frame": min(frame_count - 1, contact + int(timing["max_span"])),
                }
            )
        views.sort(key=lambda row: CAMERA_ORDER[row["role"]])
        events.append(
            {
                "event_id": f"episode_{episode:06d}:grasp_000_{side}",
                "sequence": f"episode_{episode:06d}",
                "episode_index": episode,
                "side": side,
                "instruction": task["instruction"],
                "views": views,
                "event_source": "episode_midpoint_fallback",
            }
        )
    return {
        "schema": "robot_interaction_label_session_v1",
        "pipeline_version": "v1",
        "input_format": "lerobot",
        "input_path": str(input_path.resolve()),
        "workspace": str(workspace.resolve()),
        "task": task,
        "events": events,
        "phase": "ready_for_preannotation",
        "warnings": [
            (
                "No explicit grasp event file was supplied; LeRobot episodes use midpoint anchors. "
                "Correct the contact frame in the review UI or provide exported event metadata."
            )
        ],
    }


def prepare_rosbag(
    input_path: Path,
    workspace: Path,
    task: dict[str, Any],
    engine_root: Path,
) -> dict[str, Any]:
    outputs = workspace / "outputs"
    extracted = outputs / "extracted"
    signals = outputs / "robot_signals"
    commands = [
        [
            sys.executable,
            str(engine_root / "scripts" / "extract_rgbd.py"),
            str(input_path),
            "--output-root",
            str(extracted),
        ],
        [
            sys.executable,
            str(engine_root / "scripts" / "align_depth_to_rgb.py"),
            "--input-root",
            str(extracted),
        ],
        [
            sys.executable,
            str(engine_root / "scripts" / "extract_robot_timeline.py"),
            str(input_path),
            "--extracted-root",
            str(extracted),
            "--output-root",
            str(signals),
        ],
    ]
    for command in commands:
        subprocess.run(command, cwd=engine_root, check=True)
    return index_extracted(outputs, workspace, task, input_format="rosbag")


def prepare_session(
    input_path: Path,
    workspace: Path,
    task: dict[str, Any],
    input_format: str = "auto",
    engine_root: Path | None = None,
) -> dict[str, Any]:
    workspace.mkdir(parents=True, exist_ok=True)
    resolved_format = detect_input_format(input_path) if input_format == "auto" else input_format
    if resolved_format == "rosbag":
        if engine_root is None:
            raise ValueError("--engine-root is required for raw rosbag extraction")
        session = prepare_rosbag(input_path, workspace, task, engine_root)
    elif resolved_format == "lerobot":
        session = index_lerobot(input_path.resolve(), workspace, task)
    elif resolved_format == "extracted":
        session = index_extracted(input_path.resolve(), workspace, task)
    else:
        raise ValueError(f"unsupported input format: {resolved_format}")
    atomic_write_json(workspace / "session.json", session)
    annotations = workspace / "annotations.json"
    if not annotations.exists():
        atomic_write_json(
            annotations,
            {"schema": "robot_interaction_manual_annotations_v1", "revision": 0, "items": []},
        )
    return session
