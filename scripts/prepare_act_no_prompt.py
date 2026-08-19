#!/usr/bin/env python3
"""Align RGB-D frames with common 14-DoF arm state/action labels for ACT."""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path
from typing import Any

import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore


CAMERAS = ("cam_h", "cam_l", "cam_r")
METHODS = (
    "raw_aligned",
    "rgb_guided",
    "temporal_rgb_guided",
    "lingbot_v05",
    "depth_anything_v2_fused",
    "lingbot_v05_sensor_fused",
    "ai_consensus_fused",
)


def register_bag_types(reader: Reader, typestore: Any) -> None:
    types: dict[str, Any] = {}
    for connection in reader.connections:
        definition = getattr(connection.msgdef, "data", connection.msgdef)
        types.update(get_types_from_msg(definition, connection.msgtype))
    typestore.register(types)


def read_arm_streams(bag_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    typestore = get_typestore(Stores.ROS1_NOETIC)
    state_times: list[int] = []
    states: list[np.ndarray] = []
    action_times: list[int] = []
    actions: list[np.ndarray] = []

    with Reader(bag_path) as reader:
        register_bag_types(reader, typestore)
        wanted = {"/sensors_data_raw", "/kuavo_arm_traj"}
        connections = [connection for connection in reader.connections if connection.topic in wanted]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            message = typestore.deserialize_ros1(rawdata, connection.msgtype)
            if connection.topic == "/sensors_data_raw":
                joint_q = np.asarray(message.joint_data.joint_q, dtype=np.float32)
                if joint_q.size >= 26:
                    state_times.append(int(timestamp))
                    states.append(joint_q[12:26].copy())
            else:
                position_deg = np.asarray(message.position, dtype=np.float32)
                if position_deg.size == 14:
                    action_times.append(int(timestamp))
                    actions.append(np.deg2rad(position_deg).astype(np.float32))

    if not states or not actions:
        raise RuntimeError(f"Missing common arm state/action streams in {bag_path}")
    return (
        np.asarray(state_times, dtype=np.int64),
        np.stack(states),
        np.asarray(action_times, dtype=np.int64),
        np.stack(actions),
    )


def read_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def nearest_index(times: list[int] | np.ndarray, timestamp: int) -> int:
    position = bisect.bisect_left(times, timestamp)
    candidates = [index for index in (position - 1, position) if 0 <= index < len(times)]
    return min(candidates, key=lambda index: abs(int(times[index]) - timestamp))


def depth_path(
    method: str,
    camera_row: dict[str, Any],
    sequence: str,
    camera: str,
    processed_root: Path,
) -> Path:
    if method == "raw_aligned":
        return Path(camera_row.get("depth_aligned_rgb_mm_path", camera_row["depth_raw_mm_path"]))
    frame_index = int(camera_row["frame_index"])
    return processed_root / sequence / camera / method / f"{frame_index:06d}.png"


def build_sequence(
    sequence: str,
    bag_path: Path,
    extracted_root: Path,
    processed_root: Path,
    max_delta_ms: float,
    eval_block_frames: int,
    eval_every_blocks: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifests = {
        camera: read_manifest(extracted_root / sequence / camera / "manifest.jsonl")
        for camera in CAMERAS
    }
    camera_times = {
        camera: [int(row["rgb_bag_timestamp_ns"]) for row in rows]
        for camera, rows in manifests.items()
    }
    state_times, states, action_times, actions = read_arm_streams(bag_path)
    max_delta_ns = int(max_delta_ms * 1e6)
    rows: list[dict[str, Any]] = []
    camera_deltas: list[float] = []
    state_deltas: list[float] = []
    action_deltas: list[float] = []
    missing_files = 0

    for head_row in manifests["cam_h"]:
        timestamp = int(head_row["rgb_bag_timestamp_ns"])
        camera_rows: dict[str, dict[str, Any]] = {"cam_h": head_row}
        valid = True
        for camera in CAMERAS[1:]:
            index = nearest_index(camera_times[camera], timestamp)
            delta = abs(camera_times[camera][index] - timestamp)
            if delta > max_delta_ns:
                valid = False
                break
            camera_rows[camera] = manifests[camera][index]
            camera_deltas.append(delta / 1e6)
        if not valid:
            continue

        state_index = nearest_index(state_times, timestamp)
        action_index = nearest_index(action_times, timestamp)
        state_delta = abs(int(state_times[state_index]) - timestamp)
        action_delta = abs(int(action_times[action_index]) - timestamp)
        if state_delta > max_delta_ns or action_delta > max_delta_ns:
            continue

        rgb_paths = {camera: str(Path(camera_rows[camera]["rgb_path"])) for camera in CAMERAS}
        depth_paths = {
            method: {
                camera: str(depth_path(method, camera_rows[camera], sequence, camera, processed_root))
                for camera in CAMERAS
            }
            for method in METHODS
        }
        paths = list(rgb_paths.values()) + [path for values in depth_paths.values() for path in values.values()]
        absent = [path for path in paths if not Path(path).is_file()]
        if absent:
            missing_files += len(absent)
            continue

        frame_position = len(rows)
        block_index = frame_position // eval_block_frames
        rows.append(
            {
                "sequence": sequence,
                "frame_position": frame_position,
                "timestamp_ns": timestamp,
                "is_eval": block_index % eval_every_blocks == eval_every_blocks - 1,
                "state": states[state_index].tolist(),
                "action": actions[action_index].tolist(),
                "rgb_paths": rgb_paths,
                "depth_paths": depth_paths,
            }
        )
        state_deltas.append(state_delta / 1e6)
        action_deltas.append(action_delta / 1e6)

    if not rows:
        raise RuntimeError(f"No aligned ACT frames for {sequence}")
    minimum_eval_frames = min(30, max(1, len(rows) // 10))
    if sum(row["is_eval"] for row in rows) < minimum_eval_frames:
        block_sizes: dict[int, int] = {}
        for row in rows:
            block = int(row["frame_position"]) // eval_block_frames
            block_sizes[block] = block_sizes.get(block, 0) + 1
        target_block = (max(block_sizes) * 3) / 4
        selected_block = max(
            block_sizes,
            key=lambda block: (block_sizes[block], -abs(block - target_block)),
        )
        for row in rows:
            row["is_eval"] = int(row["frame_position"]) // eval_block_frames == selected_block
    summary = {
        "sequence": sequence,
        "bag_path": str(bag_path),
        "frames": len(rows),
        "train_frames": sum(not row["is_eval"] for row in rows),
        "eval_frames": sum(row["is_eval"] for row in rows),
        "missing_file_references": missing_files,
        "camera_sync_abs_ms_p95": float(np.percentile(camera_deltas, 95)) if camera_deltas else 0.0,
        "state_sync_abs_ms_p95": float(np.percentile(state_deltas, 95)),
        "action_sync_abs_ms_p95": float(np.percentile(action_deltas, 95)),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag-root", type=Path, required=True)
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-delta-ms", type=float, default=50.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--eval-block-seconds", type=int, default=5)
    parser.add_argument("--eval-every-blocks", type=int, default=5)
    args = parser.parse_args()

    manifests = sorted(args.extracted_root.glob("*/cam_h/manifest.jsonl"))
    if not manifests:
        raise SystemExit(f"No head-camera manifests under {args.extracted_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for manifest in manifests:
        sequence = manifest.parent.parent.name
        bag_path = args.bag_root / f"{sequence}.bag"
        rows, summary = build_sequence(
            sequence,
            bag_path,
            args.extracted_root,
            args.processed_root,
            args.max_delta_ms,
            args.fps * args.eval_block_seconds,
            args.eval_every_blocks,
        )
        all_rows.extend(rows)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    index_path = args.output_dir / "index.jsonl"
    index_path.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in all_rows),
        encoding="utf-8",
    )
    summary = {
        "schema": "act_no_prompt_arm14_v1",
        "prompt_used": False,
        "state": "sensors_data_raw.joint_q[12:26] radians",
        "action": "kuavo_arm_traj.position converted degrees_to_radians",
        "cameras": list(CAMERAS),
        "methods": list(METHODS),
        "fps": args.fps,
        "split": {
            "eval_block_seconds": args.eval_block_seconds,
            "eval_every_blocks": args.eval_every_blocks,
        },
        "total_frames": len(all_rows),
        "train_frames": sum(not row["is_eval"] for row in all_rows),
        "eval_frames": sum(row["is_eval"] for row in all_rows),
        "sequences": summaries,
        "index": str(index_path),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {index_path} with {len(all_rows)} aligned frames")


if __name__ == "__main__":
    main()
