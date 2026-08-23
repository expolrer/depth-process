from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_TASK: dict[str, Any] = {
    "name": "robot_manipulation",
    "instruction": "Identify the object physically grasped by the active gripper.",
    "target": {
        "names": ["task object"],
        "attributes": [],
        "negative_descriptions": [],
    },
    "detector_prompt": [
        "task object",
        "robotic gripper",
        "robot hand",
        "box",
        "tabletop",
    ],
    "active_hand": "auto",
    "camera_roles": {
        "head": ["cam_h", "head", "top"],
        "left_wrist": ["cam_l", "left_wrist", "wrist_l"],
        "right_wrist": ["cam_r", "right_wrist", "wrist_r"],
    },
    "event_detection": {
        "pre_frames": 45,
        "post_release_frames": 20,
        "max_span": 240,
    },
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_task(path: Path | None) -> dict[str, Any]:
    if path is None:
        return DEFAULT_TASK
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError("task configuration must be a YAML mapping")
    task = _merge(DEFAULT_TASK, payload)
    if task["active_hand"] not in {"auto", "left", "right"}:
        raise ValueError("active_hand must be auto, left, or right")
    if not task["detector_prompt"]:
        raise ValueError("detector_prompt must contain at least one phrase")
    return task


def prompt_text(task: dict[str, Any]) -> str:
    return " . ".join(str(item).strip(" .") for item in task["detector_prompt"]) + " ."
