from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_v3_task(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError("task configuration must be a YAML mapping")
    try:
        from interaction_auto_labeler.schema import load_v2_task
    except ImportError as error:
        raise RuntimeError("V3 run mode requires the sibling V2 package") from error
    payload = load_v2_task(path)
    v3 = payload.setdefault("v3", {})
    v3.setdefault("dataset_id", str(payload.get("name", "robot-dataset")))
    v3.setdefault("dataset_version", "v3.0.0")
    v3.setdefault("event_graph_schema", "embodied_event_graph_v1")
    temporal = v3.setdefault("temporal", {})
    temporal.setdefault("enabled", True)
    temporal.setdefault("modalities_npz_root", None)
    reliability = v3.setdefault("reliability", {})
    reliability.setdefault("calibrator", None)
    reliability.setdefault("accept_threshold", 0.8)
    gvl = v3.setdefault("gvl", {})
    gvl.setdefault("enabled", True)
    gvl.setdefault("responses_root", None)
    gvl.setdefault("sample_count", 12)
    quality = v3.setdefault("quality", {})
    quality.setdefault("reject_non_finite", True)
    quality.setdefault("minimum_evidence_coverage", 0.5)
    geometry = v3.setdefault("geometry", {})
    geometry.setdefault("mode", "disabled")
    geometry.setdefault("manifest", None)
    geometry.setdefault("export_point_clouds", False)
    if geometry["mode"] not in {"disabled", "calibrated"}:
        raise ValueError("v3.geometry.mode must be disabled or calibrated")
    return payload
