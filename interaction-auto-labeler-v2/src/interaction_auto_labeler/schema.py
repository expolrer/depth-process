from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EvidenceWeights:
    contact: float = 0.30
    co_motion: float = 0.25
    start_end_displacement: float = 0.20
    source_destination: float = 0.10
    descriptor_match: float = 0.10
    cross_view: float = 0.05

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(value < 0 for value in values.values()):
            raise ValueError("evidence weights must be non-negative")
        if sum(values.values()) <= 0:
            raise ValueError("at least one evidence weight must be positive")


def load_v2_task(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError("task configuration must be a YAML mapping")
    target = payload.get("target", {})
    if not target.get("names"):
        raise ValueError("target.names must contain at least one concept")
    v2 = payload.setdefault("v2", {})
    backend = v2.setdefault("detector_backend", "grounded-sam2")
    if backend not in {"grounded-sam2", "sam3"}:
        raise ValueError("v2.detector_backend must be grounded-sam2 or sam3")
    mode = v2.setdefault("cross_view_mode", "soft")
    if mode not in {"soft", "calibrated"}:
        raise ValueError("v2.cross_view_mode must be soft or calibrated")
    v2.setdefault("inventory_stride", 10)
    v2.setdefault("ambiguity_margin", 0.08)
    v2["weights"] = {**asdict(EvidenceWeights()), **v2.get("weights", {})}
    v2.setdefault("review", {})
    payload.setdefault(
        "detector_prompt",
        list(target["names"]) + ["robotic gripper", "robot hand", "tabletop"],
    )
    if mode == "calibrated" and not v2.get("calibration"):
        raise ValueError("calibrated cross-view mode requires v2.calibration")
    return payload
