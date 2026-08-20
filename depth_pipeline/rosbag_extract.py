"""ROS1 RGB-D extraction without a ROS installation."""

from __future__ import annotations

import bisect
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
from rosbags.typesys import Stores, get_types_from_msg, get_typestore
from tqdm import tqdm

if TYPE_CHECKING:
    from rosbags.rosbag1 import Reader


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8"


@dataclass(frozen=True)
class CompressedFrame:
    bag_timestamp_ns: int
    header_timestamp_ns: int
    frame_id: str
    payload: bytes
    format: str


def message_stamp_ns(message: Any) -> int:
    stamp = message.header.stamp
    sec = getattr(stamp, "sec", getattr(stamp, "secs", 0))
    nanosec = getattr(stamp, "nanosec", getattr(stamp, "nsecs", 0))
    return int(sec) * 1_000_000_000 + int(nanosec)


def camera_info_dict(message: Any) -> dict[str, Any]:
    def field(lower: str, upper: str) -> np.ndarray:
        return np.asarray(getattr(message, lower, getattr(message, upper)), dtype=np.float64)

    return {
        "frame_id": str(message.header.frame_id),
        "header_timestamp_ns": message_stamp_ns(message),
        "height": int(message.height),
        "width": int(message.width),
        "distortion_model": str(message.distortion_model),
        "D": field("d", "D").tolist(),
        "K": field("k", "K").reshape(3, 3).tolist(),
        "R": field("r", "R").reshape(3, 3).tolist(),
        "P": field("p", "P").reshape(3, 4).tolist(),
    }


def transform_dict(item: Any, topic: str, bag_timestamp_ns: int) -> dict[str, Any]:
    translation = item.transform.translation
    rotation = item.transform.rotation
    return {
        "topic": topic,
        "bag_timestamp_ns": bag_timestamp_ns,
        "parent": str(item.header.frame_id),
        "child": str(item.child_frame_id),
        "translation_xyz_m": [translation.x, translation.y, translation.z],
        "rotation_xyzw": [rotation.x, rotation.y, rotation.z, rotation.w],
    }


def encoded_image(payload: bytes, signature: bytes) -> bytes:
    start = payload.find(signature)
    if start < 0:
        raise ValueError(f"Image signature {signature.hex()} not found in compressed payload")
    return payload[start:]


def nearest_pairs(
    colors: list[CompressedFrame],
    depths: list[CompressedFrame],
    max_delta_ns: int,
) -> tuple[list[tuple[CompressedFrame, CompressedFrame]], list[int]]:
    colors = sorted(colors, key=lambda frame: frame.bag_timestamp_ns)
    depths = sorted(depths, key=lambda frame: frame.bag_timestamp_ns)
    color_times = [frame.bag_timestamp_ns for frame in colors]
    pairs: list[tuple[CompressedFrame, CompressedFrame]] = []
    deltas: list[int] = []
    for depth in depths:
        position = bisect.bisect_left(color_times, depth.bag_timestamp_ns)
        candidates = [index for index in (position - 1, position) if 0 <= index < len(colors)]
        if not candidates:
            continue
        color_index = min(
            candidates,
            key=lambda index: abs(color_times[index] - depth.bag_timestamp_ns),
        )
        delta = color_times[color_index] - depth.bag_timestamp_ns
        if abs(delta) <= max_delta_ns:
            pairs.append((colors[color_index], depth))
            deltas.append(delta)
    return pairs, deltas


def write_pair(
    index: int,
    color: CompressedFrame,
    depth: CompressedFrame,
    rgb_dir: Path,
    depth_dir: Path,
) -> dict[str, Any]:
    stem = f"{index:06d}"
    rgb_path = rgb_dir / f"{stem}.jpg"
    depth_path = depth_dir / f"{stem}.png"
    rgb_path.write_bytes(encoded_image(color.payload, JPEG_SIGNATURE))
    depth_path.write_bytes(encoded_image(depth.payload, PNG_SIGNATURE))
    return {
        "frame_index": index,
        "rgb_path": str(rgb_path),
        "depth_raw_mm_path": str(depth_path),
        "rgb_bag_timestamp_ns": color.bag_timestamp_ns,
        "depth_bag_timestamp_ns": depth.bag_timestamp_ns,
        "rgb_header_timestamp_ns": color.header_timestamp_ns,
        "depth_header_timestamp_ns": depth.header_timestamp_ns,
        "sync_delta_ms": (color.bag_timestamp_ns - depth.bag_timestamp_ns) / 1e6,
        "rgb_frame_id": color.frame_id,
        "depth_frame_id": depth.frame_id,
        "rgb_format": color.format,
        "depth_format": depth.format,
    }


def register_bag_types(reader: "Reader", typestore: Any) -> None:
    types: dict[str, Any] = {}
    for connection in reader.connections:
        definition = getattr(connection.msgdef, "data", connection.msgdef)
        types.update(get_types_from_msg(definition, connection.msgtype))
    typestore.register(types)


def extract_camera(
    bag_path: Path,
    output_root: Path,
    camera: str,
    max_sync_delta_ms: float,
    workers: int,
) -> dict[str, Any]:
    from rosbags.rosbag1 import Reader

    color_topic = f"/{camera}/color/image_raw/compressed"
    depth_suffix = "image_raw/compressedDepth" if camera == "cam_h" else "image_rect_raw/compressedDepth"
    depth_topic = f"/{camera}/depth/{depth_suffix}"
    color_info_topic = f"/{camera}/color/camera_info"
    depth_info_topic = f"/{camera}/depth/camera_info"
    wanted_topics = {color_topic, depth_topic, color_info_topic, depth_info_topic, "/tf_static"}

    typestore = get_typestore(Stores.ROS1_NOETIC)
    colors: list[CompressedFrame] = []
    depths: list[CompressedFrame] = []
    color_info: dict[str, Any] | None = None
    depth_info: dict[str, Any] | None = None
    transforms: dict[tuple[str, str], dict[str, Any]] = {}

    with Reader(bag_path) as reader:
        register_bag_types(reader, typestore)
        connections = [connection for connection in reader.connections if connection.topic in wanted_topics]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            message = typestore.deserialize_ros1(rawdata, connection.msgtype)
            if connection.topic in {color_topic, depth_topic}:
                frame = CompressedFrame(
                    bag_timestamp_ns=int(timestamp),
                    header_timestamp_ns=message_stamp_ns(message),
                    frame_id=str(message.header.frame_id),
                    payload=np.asarray(message.data, dtype=np.uint8).tobytes(),
                    format=str(message.format),
                )
                (colors if connection.topic == color_topic else depths).append(frame)
            elif connection.topic == color_info_topic and color_info is None:
                color_info = camera_info_dict(message)
            elif connection.topic == depth_info_topic and depth_info is None:
                depth_info = camera_info_dict(message)
            elif connection.topic == "/tf_static":
                for item in message.transforms:
                    parent = str(item.header.frame_id)
                    child = str(item.child_frame_id)
                    if camera in parent or camera in child:
                        transforms[(parent, child)] = transform_dict(item, connection.topic, int(timestamp))

    if not colors or not depths:
        raise RuntimeError(f"Missing RGB or depth topic for {camera} in {bag_path}")
    if color_info is None or depth_info is None:
        raise RuntimeError(f"Missing CameraInfo for {camera} in {bag_path}")

    pairs, deltas_ns = nearest_pairs(colors, depths, int(max_sync_delta_ms * 1e6))
    camera_root = output_root / bag_path.stem / camera
    rgb_dir = camera_root / "rgb_raw"
    depth_dir = camera_root / "depth_raw_mm"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    tasks = ((index, color, depth, rgb_dir, depth_dir) for index, (color, depth) in enumerate(pairs))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(
            tqdm(
                executor.map(lambda args: write_pair(*args), tasks),
                total=len(pairs),
                desc=f"{bag_path.stem}/{camera}",
                unit="frame",
            )
        )

    manifest_path = camera_root / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    metadata = {
        "bag_path": str(bag_path),
        "camera": camera,
        "topics": {
            "rgb": color_topic,
            "depth": depth_topic,
            "rgb_camera_info": color_info_topic,
            "depth_camera_info": depth_info_topic,
        },
        "depth_unit": "millimeter",
        "depth_scale_to_meters": 0.001,
        "rgb_messages": len(colors),
        "depth_messages": len(depths),
        "paired_frames": len(pairs),
        "unpaired_depth_frames": len(depths) - len(pairs),
        "sync_basis": "bag_receive_timestamp",
        "max_sync_delta_ms": max_sync_delta_ms,
        "sync_delta_ms": {
            "min": min(deltas_ns) / 1e6 if deltas_ns else None,
            "median": float(np.median(deltas_ns)) / 1e6 if deltas_ns else None,
            "max": max(deltas_ns) / 1e6 if deltas_ns else None,
            "abs_p95": float(np.percentile(np.abs(deltas_ns), 95)) / 1e6 if deltas_ns else None,
        },
        "rgb_camera_info": color_info,
        "depth_camera_info": depth_info,
        "camera_transforms": list(transforms.values()),
        "manifest": str(manifest_path),
    }
    (camera_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def resolve_bags(inputs: Iterable[Path]) -> list[Path]:
    bags: list[Path] = []
    for item in inputs:
        if item.is_dir():
            bags.extend(sorted(item.glob("*.bag")))
        else:
            bags.append(item)
    return sorted({path.resolve() for path in bags})
