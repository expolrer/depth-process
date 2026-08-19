#!/usr/bin/env python3
"""Print unique transforms stored in /tf_static and /tf for camera frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("--include-dynamic", action="store_true")
    args = parser.parse_args()

    typestore = get_typestore(Stores.ROS1_NOETIC)
    wanted = {"/tf_static"}
    if args.include_dynamic:
        wanted.add("/tf")

    rows: dict[tuple[str, str], dict[str, object]] = {}
    with Reader(args.bag) as reader:
        bag_types = {}
        for connection in reader.connections:
            definition = getattr(connection.msgdef, "data", connection.msgdef)
            bag_types.update(get_types_from_msg(definition, connection.msgtype))
        typestore.register(bag_types)
        connections = [connection for connection in reader.connections if connection.topic in wanted]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            message = typestore.deserialize_ros1(rawdata, connection.msgtype)
            for item in message.transforms:
                parent = str(item.header.frame_id)
                child = str(item.child_frame_id)
                if "cam_" not in parent and "cam_" not in child:
                    continue
                translation = item.transform.translation
                rotation = item.transform.rotation
                rows[(parent, child)] = {
                    "topic": connection.topic,
                    "bag_timestamp_ns": int(timestamp),
                    "parent": parent,
                    "child": child,
                    "translation_xyz_m": [translation.x, translation.y, translation.z],
                    "rotation_xyzw": [rotation.x, rotation.y, rotation.z, rotation.w],
                }
            if connection.topic == "/tf" and len(rows) > 100:
                break

    print(json.dumps(list(rows.values()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
