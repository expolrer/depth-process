#!/usr/bin/env python3
"""Compute ROI attention mass/lift from saved ROI masks and raw heatmaps."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import cv2
import numpy as np


ATTENTION_DIRS = (
    "lingbot_cross_attention_raw",
    "lingbot_depth_token_attention_raw",
)


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def attention_metrics(mask: np.ndarray, attention: np.ndarray) -> dict[str, float]:
    roi = mask > 0
    att = attention.astype(np.float64)
    if attention.dtype == np.uint16:
        att /= 65535.0
    total = float(att.sum())
    area_ratio = float(roi.mean())
    roi_mass = float(att[roi].sum() / total) if total > 0 and roi.any() else 0.0
    background_mass = 1.0 - roi_mass
    lift = roi_mass / max(area_ratio, 1e-9)
    if total > 0:
        prob = att.ravel() / total
        entropy = float(-(prob[prob > 0] * np.log(prob[prob > 0])).sum() / np.log(prob.size))
    else:
        entropy = 0.0
    return {
        "roi_area_ratio": area_ratio,
        "roi_attention_mass": roi_mass,
        "roi_attention_lift": lift,
        "background_attention_mass": background_mass,
        "attention_entropy": entropy,
    }


def summarize(rows: list[dict[str, object]], keys: tuple[str, ...]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    out = []
    for group_key, items in sorted(groups.items()):
        row = {key: value for key, value in zip(keys, group_key, strict=True)}
        row["frames"] = len(items)
        for metric in (
            "roi_area_ratio",
            "roi_attention_mass",
            "roi_attention_lift",
            "background_attention_mass",
            "attention_entropy",
        ):
            row[f"mean_{metric}"] = mean(float(item[metric]) for item in items)
        out.append(row)
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi-index", type=Path, default=Path("outputs/roi_masks/roi_index.jsonl"))
    parser.add_argument("--attention-root", type=Path, default=Path("outputs/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports"))
    args = parser.parse_args()

    roi_rows = load_jsonl(args.roi_index)
    metric_rows: list[dict[str, object]] = []
    missing = 0
    for row in roi_rows:
        mask_path = Path(str(row["mask_path"]))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            missing += 1
            continue
        sequence = str(row["sequence"])
        camera = str(row["camera"])
        frame_index = int(row["frame_index"])
        for attention_name in ATTENTION_DIRS:
            att_path = args.attention_root / sequence / camera / attention_name / f"{frame_index:06d}.png"
            attention = cv2.imread(str(att_path), cv2.IMREAD_UNCHANGED)
            if attention is None:
                missing += 1
                continue
            if attention.shape != mask.shape:
                mask_eval = cv2.resize(mask, (attention.shape[1], attention.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                mask_eval = mask
            metric_rows.append(
                {
                    "sequence": sequence,
                    "camera": camera,
                    "frame_index": frame_index,
                    "attention": attention_name.removesuffix("_raw"),
                    "backend": row.get("backend", "unknown"),
                    **attention_metrics(mask_eval, attention),
                }
            )

    summary = {
        "schema": "roi_attention_metrics_v1",
        "roi_index": str(args.roi_index),
        "attention_root": str(args.attention_root),
        "frames_with_metrics": len(metric_rows),
        "missing_items": missing,
        "by_attention": summarize(metric_rows, ("attention",)),
        "by_sequence_attention": summarize(metric_rows, ("sequence", "attention")),
        "by_sequence_camera_attention": summarize(metric_rows, ("sequence", "camera", "attention")),
        "caution": (
            "ROI masks are pseudo-labels unless backend is grounded_sam2 and human review is completed. "
            "Use ROI mass together with lift, action MAE, and qualitative review."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "roi_attention_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(args.output_dir / "roi_attention_metrics_by_sequence.csv", summary["by_sequence_attention"])
    write_csv(
        args.output_dir / "roi_attention_metrics_by_sequence_camera.csv",
        summary["by_sequence_camera_attention"],
    )
    print(json.dumps(summary["by_attention"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
