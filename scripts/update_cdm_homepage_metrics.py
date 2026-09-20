#!/usr/bin/env python3
"""Merge D405-only CDM processing and model-free evidence into homepage metrics."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from statistics import fmean
from typing import Any


METHODS = ("cdm_camera_specific", "cdm_sensor_fused")


def mean_frames(rows: list[dict[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows]
    if not values:
        raise ValueError(f"No values for {field}")
    return fmean(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cdm-summary", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--homepage-metrics", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.cdm_summary.read_text(encoding="utf-8"))
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    homepage = json.loads(args.homepage_metrics.read_text(encoding="utf-8"))
    frames = summary["frames"]
    if len(frames) != 9824:
        raise ValueError(f"Expected 9824 D405 frames, got {len(frames)}")
    if evidence["sampling"].get("cameras") != ["cam_l", "cam_r"]:
        raise ValueError("Evidence is not restricted to cam_l/cam_r")

    spatial = evidence["no_model_quality"]["spatial_by_method"]
    existing = {row["id"]: row for row in homepage["methods"]}
    pure_spatial = spatial["cdm_camera_specific"]
    fused_spatial = spatial["cdm_sensor_fused"]
    rows = {
        "cdm_camera_specific": {
            "id": "cdm_camera_specific",
            "label": "CDM D405 纯模型",
            "kind": "相机专属 RGB-D 模型",
            "comparison_scope": "仅双腕 D405",
            "sensor_behavior": "predicted",
            "valid_fraction": mean_frames(frames, "cdm_valid_fraction"),
            "filled_fraction": mean_frames(frames, "filled_fraction"),
            "sensor_mae_m": pure_spatial.get("sensor_overlap_mae_m"),
            "sensor_rmse_m": pure_spatial.get("sensor_preservation_rmse_m"),
            "act_chunk_mae_rad": None,
            "act_first_step_mae_rad": None,
            "roi_attention_lift": None,
        },
        "cdm_sensor_fused": {
            "id": "cdm_sensor_fused",
            "label": "CDM D405 传感器融合",
            "kind": "传感器保真融合",
            "comparison_scope": "仅双腕 D405",
            "sensor_behavior": "preserved",
            "valid_fraction": mean_frames(frames, "fused_valid_fraction"),
            "filled_fraction": mean_frames(frames, "filled_fraction"),
            "sensor_mae_m": 0.0,
            "sensor_rmse_m": 0.0,
            "act_chunk_mae_rad": None,
            "act_first_step_mae_rad": None,
            "roi_attention_lift": None,
        },
    }
    existing.update(rows)
    ordered = [row for row in homepage["methods"] if row["id"] not in METHODS]
    homepage["methods"] = ordered + [existing[method] for method in METHODS]
    homepage["updated"] = date.today().isoformat()
    homepage["scope"]["cdm_d405_camera_streams"] = 10
    homepage["scope"]["cdm_d405_frames"] = len(frames)
    homepage["sources"] = list(dict.fromkeys(homepage["sources"] + [
        "outputs/processed/cdm_summary.json",
        "outputs/cdm_quality_evidence/depth_quality_evidence.json",
    ]))
    homepage["cdm_d405_evidence"] = {
        "scope": "Five datasets, left/right D405 wrist cameras only",
        "head_camera": "Gemini-335L shown as a dimmed raw reference; no D435 CDM is applied",
        "no_model_quality": {
            method: {
                "valid_fraction": spatial[method].get("valid_fraction"),
                "flat_region_roughness_m": spatial[method].get("flat_region_roughness_m"),
                "rgb_depth_edge_f1": spatial[method].get("rgb_depth_edge_f1"),
            }
            for method in METHODS
        },
        "occlusion_recovery": {
            method: evidence["occlusion_recovery"]["by_method"][method] for method in METHODS
        },
        "multiview_geometry": {
            method: evidence["multiview_geometry"]["planarity_by_method"][method]
            for method in METHODS
        },
        "task_roi_geometry": {
            method: evidence["task_roi_geometry"]["by_method"][method] for method in METHODS
        },
    }
    args.homepage_metrics.write_text(
        json.dumps(homepage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
