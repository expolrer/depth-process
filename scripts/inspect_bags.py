#!/usr/bin/env python3
"""Inspect ROS1 bag topics and sample RGB-D message metadata without ROS."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore


IMAGE_TYPES = {
    "sensor_msgs/msg/Image",
    "sensor_msgs/msg/CompressedImage",
}
CAMERA_INFO_TYPE = "sensor_msgs/msg/CameraInfo"


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def stamp_ns(message: Any) -> int | None:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    sec = getattr(stamp, "sec", getattr(stamp, "secs", 0))
    nanosec = getattr(stamp, "nanosec", getattr(stamp, "nsecs", 0))
    return int(sec) * 1_000_000_000 + int(nanosec)


def sample_metadata(message: Any, msgtype: str) -> dict[str, Any]:
    header = getattr(message, "header", None)
    result: dict[str, Any] = {
        "header_stamp_ns": stamp_ns(message),
        "frame_id": str(getattr(header, "frame_id", "")),
    }
    if msgtype == "sensor_msgs/msg/Image":
        result.update(
            height=int(message.height),
            width=int(message.width),
            encoding=str(message.encoding),
            is_bigendian=int(message.is_bigendian),
            step=int(message.step),
            data_bytes=int(np.asarray(message.data).nbytes),
        )
    elif msgtype == "sensor_msgs/msg/CompressedImage":
        payload = np.asarray(message.data, dtype=np.uint8).tobytes()
        png_signature = b"\x89PNG\r\n\x1a\n"
        start = payload.find(png_signature)
        encoded = payload[start:] if start >= 0 else payload
        image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_UNCHANGED)
        result.update(format=str(message.format), data_bytes=len(payload))
        if image is not None:
            result.update(
                decoded_shape=list(image.shape),
                decoded_dtype=str(image.dtype),
                decoded_min=json_value(image.min()),
                decoded_max=json_value(image.max()),
            )
            if image.ndim == 2:
                result["zero_fraction"] = float(np.mean(image == 0))
    elif msgtype == CAMERA_INFO_TYPE:
        d = np.asarray(getattr(message, "d", getattr(message, "D", [])))
        k = np.asarray(getattr(message, "k", getattr(message, "K")))
        r = np.asarray(getattr(message, "r", getattr(message, "R")))
        p = np.asarray(getattr(message, "p", getattr(message, "P")))
        result.update(
            height=int(message.height),
            width=int(message.width),
            distortion_model=str(message.distortion_model),
            d=json_value(d),
            k=json_value(k.reshape(3, 3)),
            r=json_value(r.reshape(3, 3)),
            p=json_value(p.reshape(3, 4)),
        )
    return result


def inspect_bag(path: Path) -> dict[str, Any]:
    typestore = get_typestore(Stores.ROS1_NOETIC)
    samples: dict[int, dict[str, Any]] = {}
    errors: dict[int, str] = {}

    with Reader(path) as reader:
        interesting = [
            connection
            for connection in reader.connections
            if connection.msgtype in IMAGE_TYPES or connection.msgtype == CAMERA_INFO_TYPE
        ]
        pending = {connection.id for connection in interesting}
        if pending:
            for connection, timestamp, rawdata in reader.messages(connections=interesting):
                if connection.id not in pending:
                    continue
                try:
                    message = typestore.deserialize_ros1(rawdata, connection.msgtype)
                    samples[connection.id] = sample_metadata(message, connection.msgtype)
                    samples[connection.id]["bag_timestamp_ns"] = int(timestamp)
                except Exception as exc:  # Keep inspecting other topics.
                    errors[connection.id] = f"{type(exc).__name__}: {exc}"
                pending.remove(connection.id)
                if not pending:
                    break

        topic_rows = []
        by_type: dict[str, int] = defaultdict(int)
        for connection in reader.connections:
            by_type[connection.msgtype] += int(connection.msgcount)
            row = {
                "id": int(connection.id),
                "topic": connection.topic,
                "msgtype": connection.msgtype,
                "message_count": int(connection.msgcount),
            }
            if connection.id in samples:
                row["sample"] = samples[connection.id]
            if connection.id in errors:
                row["sample_error"] = errors[connection.id]
            topic_rows.append(row)

        return {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "start_time_ns": int(reader.start_time),
            "end_time_ns": int(reader.end_time),
            "duration_seconds": (reader.end_time - reader.start_time) / 1e9,
            "message_count": int(reader.message_count),
            "message_counts_by_type": dict(sorted(by_type.items())),
            "topics": sorted(topic_rows, key=lambda row: row["topic"]),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path, help="ROS1 bag files or directories")
    parser.add_argument("--output", type=Path, help="Write JSON report to this path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bag_paths: list[Path] = []
    for item in args.inputs:
        if item.is_dir():
            bag_paths.extend(sorted(item.glob("*.bag")))
        else:
            bag_paths.append(item)
    if not bag_paths:
        raise SystemExit("No .bag files found")

    report = {"bags": [inspect_bag(path.resolve()) for path in bag_paths]}
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
