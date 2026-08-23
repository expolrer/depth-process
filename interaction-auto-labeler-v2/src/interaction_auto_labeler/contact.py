from __future__ import annotations

from typing import Any

import numpy as np


def _smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    if values.size < 3 or window <= 1:
        return values
    window = min(window, values.size if values.size % 2 else values.size - 1)
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same")


def detect_action_phases(
    gripper_position: list[float],
    eef_speed: list[float] | None = None,
    close_direction: str = "decrease",
    pre_frames: int = 45,
    post_frames: int = 20,
) -> dict[str, int | float]:
    values = np.asarray(gripper_position, dtype=np.float64)
    if values.size < 4:
        raise ValueError("at least four gripper samples are required")
    smoothed = _smooth(values)
    derivative = np.diff(smoothed, prepend=smoothed[0])
    closure_signal = -derivative if close_direction == "decrease" else derivative
    contact = int(np.argmax(closure_signal))
    opening_signal = derivative if close_direction == "decrease" else -derivative
    search_start = min(values.size - 1, contact + 2)
    release = search_start + int(np.argmax(opening_signal[search_start:]))
    if eef_speed is not None and len(eef_speed) == len(values):
        speed = _smooth(np.asarray(eef_speed, dtype=np.float64))
        transport_slice = speed[contact : max(contact + 1, release + 1)]
        transport = contact + int(np.argmax(transport_slice))
    else:
        transport = (contact + release) // 2
    return {
        "start_frame": max(0, contact - pre_frames),
        "contact_frame": contact,
        "transport_frame": transport,
        "release_frame": release,
        "end_frame": min(values.size - 1, release + post_frames),
        "closure_strength": float(max(0.0, closure_signal[contact])),
        "opening_strength": float(max(0.0, opening_signal[release])),
    }


def phase_from_signal_rows(
    rows: list[dict[str, Any]],
    side: str,
    close_direction: str = "decrease",
) -> dict[str, int | float]:
    keys = (f"{side}_gripper", f"gripper_{side}", "gripper_position")
    values = []
    speeds = []
    for row in rows:
        value = next((row.get(key) for key in keys if row.get(key) is not None), None)
        if value is None:
            value = row.get("gripper", {}).get(side)
        if value is None:
            raise ValueError(f"no {side} gripper signal found")
        values.append(float(value))
        speeds.append(float(row.get("eef_speed", {}).get(side, 0.0)))
    return detect_action_phases(values, speeds, close_direction)
