from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import validate_transform


@dataclass(frozen=True)
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


@dataclass(frozen=True)
class RobotKinematicTree:
    joints_by_child: dict[str, Joint]

    def chain(self, base_link: str, target_link: str) -> list[Joint]:
        chain: list[Joint] = []
        cursor = target_link
        visited: set[str] = set()
        while cursor != base_link:
            if cursor in visited:
                raise ValueError("cycle found in URDF kinematic tree")
            visited.add(cursor)
            joint = self.joints_by_child.get(cursor)
            if joint is None:
                raise KeyError(f"no kinematic chain from {base_link!r} to {target_link!r}")
            chain.append(joint)
            cursor = joint.parent
        return list(reversed(chain))


def _numbers(text: str | None, expected: int, default: tuple[float, ...]) -> np.ndarray:
    values = [float(value) for value in text.split()] if text else list(default)
    if len(values) != expected:
        raise ValueError(f"expected {expected} numeric values, got {values}")
    return np.asarray(values, dtype=np.float64)


def rpy_rotation(rpy: Any) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def origin_transform(xyz: Any, rpy: Any) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rpy_rotation(rpy)
    transform[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return transform


def axis_angle_transform(axis: Any, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        raise ValueError("joint axis cannot be zero")
    x, y, z = axis / norm
    skew = np.asarray([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    rotation = np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    return transform


def parse_urdf(source: Path | str) -> RobotKinematicTree:
    root = ET.parse(source).getroot() if isinstance(source, Path) else ET.fromstring(source)
    joints: dict[str, Joint] = {}
    for element in root.findall("joint"):
        parent = element.find("parent")
        child = element.find("child")
        if parent is None or child is None:
            raise ValueError("every URDF joint requires parent and child links")
        origin = element.find("origin")
        xyz = _numbers(origin.get("xyz") if origin is not None else None, 3, (0, 0, 0))
        rpy = _numbers(origin.get("rpy") if origin is not None else None, 3, (0, 0, 0))
        axis_element = element.find("axis")
        axis = _numbers(
            axis_element.get("xyz") if axis_element is not None else None,
            3,
            (1, 0, 0),
        )
        joint = Joint(
            name=str(element.get("name", "")),
            kind=str(element.get("type", "fixed")),
            parent=str(parent.get("link")),
            child=str(child.get("link")),
            origin=origin_transform(xyz, rpy),
            axis=axis,
        )
        if not joint.name or not joint.parent or not joint.child:
            raise ValueError("URDF joint names and links cannot be empty")
        joints[joint.child] = joint
    return RobotKinematicTree(joints)


def forward_kinematics(
    tree: RobotKinematicTree,
    joint_positions: Mapping[str, float],
    base_link: str,
    target_link: str,
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    for joint in tree.chain(base_link, target_link):
        transform = transform @ joint.origin
        value = float(joint_positions.get(joint.name, 0.0))
        if joint.kind in {"revolute", "continuous"}:
            transform = transform @ axis_angle_transform(joint.axis, value)
        elif joint.kind == "prismatic":
            motion = np.eye(4, dtype=np.float64)
            motion[:3, 3] = joint.axis / np.linalg.norm(joint.axis) * value
            transform = transform @ motion
        elif joint.kind != "fixed":
            raise ValueError(f"unsupported URDF joint type: {joint.kind}")
    return validate_transform(transform, "base_from_target")


def dynamic_camera_transform(
    tree: RobotKinematicTree,
    joint_positions: Mapping[str, float],
    base_link: str,
    end_effector_link: str,
    end_effector_from_camera: Any,
    world_from_base: Any | None = None,
) -> np.ndarray:
    world_from_base = np.eye(4) if world_from_base is None else validate_transform(world_from_base)
    base_from_eef = forward_kinematics(tree, joint_positions, base_link, end_effector_link)
    eef_from_camera = validate_transform(end_effector_from_camera, "end_effector_from_camera")
    return validate_transform(
        world_from_base @ base_from_eef @ eef_from_camera, "world_from_camera"
    )


def dynamic_camera_trajectory(
    tree: RobotKinematicTree,
    joint_rows: list[Mapping[str, float]],
    base_link: str,
    end_effector_link: str,
    end_effector_from_camera: Any,
    world_from_base: Any | None = None,
) -> np.ndarray:
    return np.stack(
        [
            dynamic_camera_transform(
                tree,
                row,
                base_link,
                end_effector_link,
                end_effector_from_camera,
                world_from_base,
            )
            for row in joint_rows
        ]
    )
