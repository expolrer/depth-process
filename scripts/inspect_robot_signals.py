#!/usr/bin/env python3
"""Inspect custom robot signal schemas and representative values in ROS1 bags."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore


DEFAULT_TOPICS = (
    "/leju_claw_state",
    "/leju_claw_command",
    "/control_robot_hand_position",
    "/ik_fk_result/eef_pose",
    "/kuavo_arm_traj",
    "/joint_cmd",
)


def register_bag_types(reader: Reader, typestore: Any) -> None:
    types: dict[str, Any] = {}
    for connection in reader.connections:
        definition = getattr(connection.msgdef, "data", connection.msgdef)
        types.update(get_types_from_msg(definition, connection.msgtype))
    typestore.register(types)


def jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1)
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "values": flat[:32].tolist(),
            "truncated": flat.size > 32,
        }
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value[:32]]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def inspect(path: Path, topics: set[str], samples_per_topic: int) -> dict[str, Any]:
    typestore = get_typestore(Stores.ROS1_NOETIC)
    result: dict[str, Any] = {"bag": str(path), "topics": {}}
    with Reader(path) as reader:
        register_bag_types(reader, typestore)
        connections = [connection for connection in reader.connections if connection.topic in topics]
        for connection in connections:
            result["topics"][connection.topic] = {
                "msgtype": connection.msgtype,
                "message_count": int(connection.msgcount),
                "samples": [],
            }
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            row = result["topics"][connection.topic]
            if len(row["samples"]) >= samples_per_topic:
                continue
            message = typestore.deserialize_ros1(rawdata, connection.msgtype)
            row["samples"].append({"bag_timestamp_ns": int(timestamp), "message": jsonable(message)})
            if all(len(item["samples"]) >= samples_per_topic for item in result["topics"].values()):
                break
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bags", nargs="+", type=Path)
    parser.add_argument("--topic", action="append", dest="topics")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    topics = set(args.topics or DEFAULT_TOPICS)
    report = {"bags": [inspect(path, topics, args.samples) for path in args.bags]}
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
