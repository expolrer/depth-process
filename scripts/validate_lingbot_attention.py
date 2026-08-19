#!/usr/bin/env python3
"""Validate LingBot attention maps against every extracted RGB-D manifest."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


METHODS = (
    "lingbot_cross_attention",
    "lingbot_depth_token_attention",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def inspect_image(path: Path, expected_shape: tuple[int, int], raw: bool) -> dict[str, Any]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("cannot decode image")
    if image.shape[:2] != expected_shape:
        raise ValueError(f"shape={image.shape[:2]}, expected={expected_shape}")
    if raw:
        if image.dtype != np.uint16 or image.ndim != 2:
            raise ValueError(f"raw map must be uint16 single-channel, got {image.dtype} {image.shape}")
        unique_count = int(np.unique(image).size)
        if unique_count < 64 or int(image.max()) <= int(image.min()):
            raise ValueError(
                f"degenerate raw map: unique={unique_count}, min={int(image.min())}, max={int(image.max())}"
            )
        return {
            "dtype": str(image.dtype),
            "shape": list(image.shape),
            "min": int(image.min()),
            "max": int(image.max()),
            "mean": float(image.mean()),
            "unique": unique_count,
        }
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"overlay must be uint8 BGR, got {image.dtype} {image.shape}")
    if float(image.std()) < 5.0:
        raise ValueError(f"degenerate overlay: std={float(image.std()):.3f}")
    return {
        "dtype": str(image.dtype),
        "shape": list(image.shape),
        "mean": float(image.mean()),
        "std": float(image.std()),
    }


def main() -> int:
    args = parse_args()
    issues: list[dict[str, str]] = []
    streams: list[dict[str, Any]] = []
    sampled: list[dict[str, Any]] = []
    expected_total = 0

    manifests = sorted(args.input_root.glob("*/cam_*/manifest.jsonl"))
    if not manifests:
        issues.append({"asset": str(args.input_root), "error": "no manifests found"})
    for manifest_path in manifests:
        relative = manifest_path.parent.relative_to(args.input_root)
        rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
        expected_frames = len(rows)
        expected_total += expected_frames
        first_rgb = cv2.imread(rows[0]["rgb_path"], cv2.IMREAD_COLOR) if rows else None
        if first_rgb is None:
            issues.append({"asset": relative.as_posix(), "error": "cannot read source RGB"})
            continue
        expected_shape = first_rgb.shape[:2]
        stream_record: dict[str, Any] = {
            "stream": relative.as_posix(),
            "expectedFrames": expected_frames,
            "methods": {},
        }
        for method in METHODS:
            overlay_dir = args.output_root / relative / method
            raw_dir = args.output_root / relative / f"{method}_raw"
            overlay_paths = sorted(overlay_dir.glob("*.jpg"))
            raw_paths = sorted(raw_dir.glob("*.png"))
            stream_record["methods"][method] = {
                "overlayCount": len(overlay_paths),
                "rawCount": len(raw_paths),
            }
            if len(overlay_paths) != expected_frames:
                issues.append(
                    {
                        "asset": overlay_dir.as_posix(),
                        "error": f"overlay count={len(overlay_paths)}, expected={expected_frames}",
                    }
                )
            if len(raw_paths) != expected_frames:
                issues.append(
                    {
                        "asset": raw_dir.as_posix(),
                        "error": f"raw count={len(raw_paths)}, expected={expected_frames}",
                    }
                )
            if expected_frames:
                indices = sorted({0, expected_frames // 2, expected_frames - 1})
                for index in indices:
                    expected_name = f"{index:06d}"
                    for path, raw in (
                        (overlay_dir / f"{expected_name}.jpg", False),
                        (raw_dir / f"{expected_name}.png", True),
                    ):
                        try:
                            stats = inspect_image(path, expected_shape, raw)
                            sampled.append({"asset": path.as_posix(), **stats})
                        except Exception as exc:
                            issues.append({"asset": path.as_posix(), "error": str(exc)})
        streams.append(stream_record)

    summary: dict[str, Any] = {}
    if not args.summary.is_file():
        issues.append({"asset": str(args.summary), "error": "summary report missing"})
    else:
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
        if int(summary.get("selected_frames", -1)) != expected_total:
            issues.append(
                {
                    "asset": str(args.summary),
                    "error": f"selected_frames={summary.get('selected_frames')}, expected={expected_total}",
                }
            )
        accounted = int(summary.get("processed_frames", 0)) + int(summary.get("resumed_frames", 0))
        if accounted != expected_total:
            issues.append(
                {
                    "asset": str(args.summary),
                    "error": f"accounted frames={accounted}, expected={expected_total}",
                }
            )

    report = {
        "ok": not issues,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "streamCount": len(manifests),
        "expectedFrames": expected_total,
        "expectedFiles": expected_total * len(METHODS) * 2,
        "issues": issues,
        "attentionDefinition": summary.get("definition"),
        "streams": streams,
        "sampledImages": sampled,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: report[key] for key in ("ok", "streamCount", "expectedFrames", "expectedFiles", "issues")},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
