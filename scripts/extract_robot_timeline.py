#!/usr/bin/env python3
"""Align robot state streams and grasp candidates to extracted RGB-D frames."""

from __future__ import annotations

import argparse
import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore


TOPICS = {
    "/leju_claw_state",
    "/leju_claw_command",
    "/control_robot_hand_position",
    "/ik_fk_result/eef_pose",
    "/kuavo_arm_traj",
}
CAMERAS = ("cam_h", "cam_l", "cam_r")


@dataclass
class Stream:
    timestamps_ns: list[int]
    rows: list[dict[str, Any]]

    def nearest(self, timestamp_ns: int) -> tuple[dict[str, Any] | None, float | None]:
        if not self.timestamps_ns:
            return None, None
        position = bisect.bisect_left(self.timestamps_ns, timestamp_ns)
        candidates = [index for index in (position - 1, position) if 0 <= index < len(self.timestamps_ns)]
        index = min(candidates, key=lambda item: abs(self.timestamps_ns[item] - timestamp_ns))
        return self.rows[index], (self.timestamps_ns[index] - timestamp_ns) / 1e6


def values(value: Any) -> list[float]:
    if value is None:
        return []
    return np.asarray(value).reshape(-1).tolist()


def register_bag_types(reader: Reader, typestore: Any) -> None:
    types: dict[str, Any] = {}
    for connection in reader.connections:
        definition = getattr(connection.msgdef, "data", connection.msgdef)
        types.update(get_types_from_msg(definition, connection.msgtype))
    typestore.register(types)


def parse_message(topic: str, timestamp_ns: int, message: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"timestamp_ns": timestamp_ns}
    if topic == "/leju_claw_state":
        row.update(
            state=[int(item) for item in values(message.state)],
            position=values(message.data.position),
            velocity=values(message.data.velocity),
            effort=values(message.data.effort),
        )
    elif topic == "/leju_claw_command":
        row.update(
            position=values(message.data.position),
            velocity=values(message.data.velocity),
            effort=values(message.data.effort),
        )
    elif topic == "/control_robot_hand_position":
        row.update(
            left_position=[int(item) for item in values(message.left_hand_position)],
            right_position=[int(item) for item in values(message.right_hand_position)],
        )
    elif topic == "/ik_fk_result/eef_pose":
        row.update(
            frame_id=str(message.header.frame_id),
            left_position_m=values(message.left_pose.pos_xyz),
            left_quaternion_xyzw=values(message.left_pose.quat_xyzw),
            right_position_m=values(message.right_pose.pos_xyz),
            right_quaternion_xyzw=values(message.right_pose.quat_xyzw),
        )
    elif topic == "/kuavo_arm_traj":
        row.update(
            names=[str(item) for item in message.name],
            position=values(message.position),
            velocity=values(message.velocity),
            effort=values(message.effort),
        )
    return row


def read_streams(bag_path: Path) -> dict[str, Stream]:
    typestore = get_typestore(Stores.ROS1_NOETIC)
    data = {topic: Stream([], []) for topic in TOPICS}
    with Reader(bag_path) as reader:
        register_bag_types(reader, typestore)
        connections = [connection for connection in reader.connections if connection.topic in TOPICS]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            message = typestore.deserialize_ros1(rawdata, connection.msgtype)
            stream = data[connection.topic]
            stream.timestamps_ns.append(int(timestamp))
            stream.rows.append(parse_message(connection.topic, int(timestamp), message))
    return {topic: stream for topic, stream in data.items() if stream.rows}


def transition_events(stream: Stream, field: str, side: int, target: int, kind: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    previous: int | None = None
    for row in stream.rows:
        sequence = row.get(field, [])
        if side >= len(sequence):
            continue
        current = int(sequence[side])
        if current == target and previous != target:
            events.append(
                {
                    "timestamp_ns": row["timestamp_ns"],
                    "kind": kind,
                    "side": "left" if side == 0 else "right",
                    "confidence": 0.95,
                    "needs_review": False,
                }
            )
        previous = current
    return events


def motion_events(
    stream: Stream,
    field: str,
    side: int,
    min_amplitude: float,
    positive_kind: str,
    negative_kind: str,
) -> list[dict[str, Any]]:
    samples: list[tuple[int, float]] = []
    for row in stream.rows:
        sequence = row.get(field, [])
        if field in {"left_position", "right_position"}:
            if sequence:
                samples.append((row["timestamp_ns"], float(np.mean(sequence))))
        elif side < len(sequence):
            samples.append((row["timestamp_ns"], float(sequence[side])))
    if len(samples) < 3:
        return []

    times = np.asarray([item[0] for item in samples], dtype=np.int64)
    signal = np.asarray([item[1] for item in samples], dtype=np.float64)
    delta = np.diff(signal)
    active = np.flatnonzero(np.abs(delta) >= max(min_amplitude / 10.0, 1e-5))
    if not active.size:
        return []

    groups: list[list[int]] = [[int(active[0])]]
    for index in active[1:]:
        index = int(index)
        gap_seconds = (times[index] - times[groups[-1][-1]]) / 1e9
        if gap_seconds <= 0.25:
            groups[-1].append(index)
        else:
            groups.append([index])

    events: list[dict[str, Any]] = []
    for group in groups:
        start = group[0]
        end = min(group[-1] + 1, signal.size - 1)
        amplitude = float(signal[end] - signal[start])
        duration = float((times[end] - times[start]) / 1e9)
        if abs(amplitude) < min_amplitude:
            continue
        is_positive = amplitude > 0
        events.append(
            {
                "timestamp_ns": int(times[end]),
                "start_timestamp_ns": int(times[start]),
                "kind": positive_kind if is_positive else negative_kind,
                "side": "left" if side == 0 else "right",
                "amplitude": amplitude,
                "duration_seconds": duration,
                "confidence": 0.65 if is_positive else 0.55,
                "needs_review": True,
            }
        )
    return events


def detect_events(streams: dict[str, Stream]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    claw_state = streams.get("/leju_claw_state")
    if claw_state:
        for side in (0, 1):
            events.extend(transition_events(claw_state, "state", side, 3, "confirmed_grab"))
    claw_command = streams.get("/leju_claw_command")
    if claw_command:
        for side in (0, 1):
            events.extend(
                motion_events(
                    claw_command,
                    "position",
                    side,
                    5.0,
                    "grasp_command_end",
                    "release_command_end",
                )
            )
    dex_command = streams.get("/control_robot_hand_position")
    if dex_command:
        events.extend(
            motion_events(
                dex_command,
                "left_position",
                0,
                5.0,
                "grasp_command_end",
                "release_command_end",
            )
        )
        events.extend(
            motion_events(
                dex_command,
                "right_position",
                1,
                5.0,
                "grasp_command_end",
                "release_command_end",
            )
        )
    events.sort(key=lambda row: (row["timestamp_ns"], row["side"], row["kind"]))
    return events


def numeric_summary(stream: Stream, field: str) -> dict[str, Any] | None:
    arrays = [np.asarray(row[field], dtype=np.float64).reshape(-1) for row in stream.rows if row.get(field)]
    if not arrays:
        return None
    width = max(array.size for array in arrays)
    aligned = np.full((len(arrays), width), np.nan, dtype=np.float64)
    for index, array in enumerate(arrays):
        aligned[index, : array.size] = array
    return {
        "count": len(arrays),
        "min": np.nanmin(aligned, axis=0).tolist(),
        "max": np.nanmax(aligned, axis=0).tolist(),
        "median": np.nanmedian(aligned, axis=0).tolist(),
    }


def stream_summary(streams: dict[str, Stream]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    fields = {
        "/leju_claw_state": ("state", "position", "velocity", "effort"),
        "/leju_claw_command": ("position",),
        "/control_robot_hand_position": ("left_position", "right_position"),
        "/ik_fk_result/eef_pose": ("left_position_m", "right_position_m"),
    }
    for topic, stream in streams.items():
        row: dict[str, Any] = {"messages": len(stream.rows)}
        for field in fields.get(topic, ()):
            summary = numeric_summary(stream, field)
            if summary:
                row[field] = summary
        report[topic] = row
    return report


def read_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def align_frames(
    sequence: str,
    extracted_root: Path,
    output_root: Path,
    streams: dict[str, Stream],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    camera_report: dict[str, Any] = {}
    for camera in CAMERAS:
        manifest_path = extracted_root / sequence / camera / "manifest.jsonl"
        if not manifest_path.exists():
            continue
        frames = read_manifest(manifest_path)
        aligned_rows: list[dict[str, Any]] = []
        frame_times = [int(frame["rgb_bag_timestamp_ns"]) for frame in frames]
        for frame in frames:
            timestamp_ns = int(frame["rgb_bag_timestamp_ns"])
            row: dict[str, Any] = {
                "sequence": sequence,
                "camera": camera,
                "frame_index": int(frame["frame_index"]),
                "rgb_timestamp_ns": timestamp_ns,
            }
            for topic, key in (
                ("/leju_claw_state", "claw_state"),
                ("/leju_claw_command", "claw_command"),
                ("/control_robot_hand_position", "dex_hand_command"),
                ("/ik_fk_result/eef_pose", "eef_pose"),
                ("/kuavo_arm_traj", "arm_joint_state"),
            ):
                stream = streams.get(topic)
                if stream:
                    sample, delta_ms = stream.nearest(timestamp_ns)
                    row[key] = sample
                    row[f"{key}_delta_ms"] = delta_ms
            aligned_rows.append(row)
        output_path = output_root / sequence / camera / "robot_signals.jsonl"
        write_jsonl(output_path, aligned_rows)

        mapped_events: list[dict[str, Any]] = []
        for event in events:
            position = bisect.bisect_left(frame_times, int(event["timestamp_ns"]))
            candidates = [index for index in (position - 1, position) if 0 <= index < len(frame_times)]
            if not candidates:
                continue
            index = min(candidates, key=lambda item: abs(frame_times[item] - int(event["timestamp_ns"])))
            mapped_events.append(
                {
                    **event,
                    "camera": camera,
                    "frame_index": int(frames[index]["frame_index"]),
                    "frame_delta_ms": (frame_times[index] - int(event["timestamp_ns"])) / 1e6,
                }
            )
        write_jsonl(output_root / sequence / camera / "events.jsonl", mapped_events)
        camera_report[camera] = {
            "frames": len(aligned_rows),
            "events": len(mapped_events),
            "signals_path": str(output_path),
        }
    return camera_report


def resolve_bags(inputs: Iterable[Path]) -> list[Path]:
    bags: list[Path] = []
    for item in inputs:
        bags.extend(sorted(item.glob("*.bag")) if item.is_dir() else [item])
    return sorted({path.resolve() for path in bags})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--extracted-root", type=Path, default=Path("outputs/extracted"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/robot_signals"))
    args = parser.parse_args()
    reports: list[dict[str, Any]] = []
    for bag_path in resolve_bags(args.inputs):
        sequence = bag_path.stem
        print(f"[signals] reading {bag_path}", flush=True)
        streams = read_streams(bag_path)
        events = detect_events(streams)
        cameras = align_frames(sequence, args.extracted_root, args.output_root, streams, events)
        report = {
            "sequence": sequence,
            "bag_path": str(bag_path),
            "streams": stream_summary(streams),
            "events": events,
            "cameras": cameras,
        }
        report_path = args.output_root / sequence / "summary.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        reports.append(report)
        print(f"[signals] {sequence}: {len(events)} event candidates", flush=True)
    summary = {"schema": "rgbd_robot_timeline_v1", "sequences": reports}
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"sequences": len(reports), "events": sum(len(row["events"]) for row in reports)}, indent=2))


if __name__ == "__main__":
    main()
