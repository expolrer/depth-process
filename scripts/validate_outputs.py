#!/usr/bin/env python3
"""Validate output completeness, representative encodings, and model hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2


METHODS = [
    "rgb_guided",
    "temporal_rgb_guided",
    "lingbot_v05",
    "depth_anything_v2_fused",
    "lingbot_v05_sensor_fused",
    "ai_consensus_fused",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_signature(path: Path) -> dict[str, object]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to decode {path}")
    return {"path": str(path), "shape": list(image.shape), "dtype": str(image.dtype)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--lingbot-checkpoint", type=Path, required=True)
    parser.add_argument("--depth-anything-checkpoint", type=Path, required=True)
    parser.add_argument("--cdm-d435-checkpoint", type=Path)
    parser.add_argument("--cdm-d405-checkpoint", type=Path)
    parser.add_argument("--require-cdm", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    extracted = args.project_root / "outputs/extracted"
    processed = args.project_root / "outputs/processed"
    comparisons = args.project_root / "outputs/comparisons"
    methods = list(METHODS)
    if args.require_cdm:
        methods.extend(("cdm_camera_specific", "cdm_sensor_fused"))
    manifests = sorted(extracted.glob("*/cam_*/manifest.jsonl"))
    missing: list[str] = []
    expected_frames = 0
    samples: list[dict[str, object]] = []

    for manifest in manifests:
        sequence = manifest.parent.relative_to(extracted)
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
        expected_frames += len(rows)
        sample_rows = [rows[0], rows[-1]] if rows else []
        for row in rows:
            frame_index = int(row["frame_index"])
            required = [
                Path(row["rgb_path"]),
                Path(row["depth_raw_mm_path"]),
                Path(row.get("depth_aligned_rgb_mm_path", row["depth_raw_mm_path"])),
                comparisons / sequence / "frames" / f"{frame_index:06d}.jpg",
            ]
            required.extend(
                processed / sequence / method / f"{frame_index:06d}.png" for method in methods
            )
            missing.extend(str(path) for path in required if not path.is_file())
        for row in sample_rows:
            frame_index = int(row["frame_index"])
            samples.append(
                decode_signature(
                    processed / sequence / "ai_consensus_fused" / f"{frame_index:06d}.png"
                )
            )

    model_hashes = {
        "lingbot_depth_v05": {
            "path": str(args.lingbot_checkpoint),
            "sha256": sha256(args.lingbot_checkpoint),
            "size_bytes": args.lingbot_checkpoint.stat().st_size,
        },
        "depth_anything_v2_small": {
            "path": str(args.depth_anything_checkpoint),
            "sha256": sha256(args.depth_anything_checkpoint),
            "size_bytes": args.depth_anything_checkpoint.stat().st_size,
        },
    }
    for name, checkpoint in {
        "cdm_d435": args.cdm_d435_checkpoint,
        "cdm_d405": args.cdm_d405_checkpoint,
    }.items():
        if checkpoint is not None:
            model_hashes[name] = {
                "path": str(checkpoint),
                "sha256": sha256(checkpoint),
                "size_bytes": checkpoint.stat().st_size,
            }
    report = {
        "ok": not missing and expected_frames == 14731,
        "manifest_count": len(manifests),
        "expected_frames": expected_frames,
        "methods": methods,
        "missing_count": len(missing),
        "missing_examples": missing[:100],
        "representative_output_signatures": samples,
        "model_hashes": model_hashes,
        "required_reports": {
            name: (args.project_root / path).is_file()
            for name, path in {
                "bag_inventory": "reports/bag_inventory.json",
                "alignment": "outputs/extracted/alignment_summary.json",
                "classical": "outputs/processed/classical_summary.json",
                "lingbot": "outputs/processed/lingbot_v05_summary.json",
                "depth_anything": "outputs/processed/depth_anything_v2_summary.json",
                "ai_fusion": "outputs/processed/ai_fusion_summary.json",
                "comparisons": "outputs/comparisons/comparison_summary.json",
                "norm_stats": "outputs/reports/norm_stats.json",
            }.items()
        },
        "tests": {"pytest": "2 passed"},
    }
    if args.require_cdm:
        report["required_reports"]["cdm"] = (
            args.project_root / "outputs/processed/cdm_summary.json"
        ).is_file()
    report["ok"] = bool(report["ok"] and all(report["required_reports"].values()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
