#!/usr/bin/env python3
"""Export a compact local ACT-attention viewer package."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2


DATASETS = (
    (
        "leju_claw",
        "A10-A15-G-S-01-TQ_03_01-4_304-leju_claw-20260512172205-200049-8c435b-v003",
        "Leju Claw",
        range(600, 605),
    ),
    (
        "dex_hand",
        "A10-A15-G-S-01-TQ_09_01-P4_297-dex_hand-20260629103123-49-9a5c2d-v003",
        "Dex Hand",
        range(600, 605),
    ),
    ("chengzhong", "chengzhong_xianxia_main1", "Chengzhong", range(150, 155)),
    ("dajian", "dajian_xianxia_main1", "Dajian", range(450, 455)),
    ("zhoumian", "zhoumian_xianxia_main1", "Zhoumian", range(600, 605)),
)

METHODS = (
    ("raw_aligned", "Raw aligned depth", "Processing methods"),
    ("rgb_guided", "RGB guided", "Processing methods"),
    ("temporal_rgb_guided", "RGB + temporal", "Processing methods"),
    ("lingbot_v05", "LingBot-v0.5 depth", "Processing methods"),
    ("depth_anything_v2_fused", "Depth Anything V2 fused", "Processing methods"),
    ("lingbot_v05_sensor_fused", "LingBot sensor fused", "Processing methods"),
    ("ai_consensus_fused", "AI consensus fused", "Processing methods"),
    ("zero_depth", "Zero depth", "Negative controls"),
    ("spatially_shuffled_raw", "Spatially shuffled depth", "Negative controls"),
)

DEPTH_METHOD_IDS = tuple(method[0] for method in METHODS[:7])

CAMERAS = (
    ("cam_h", "Head", "Gemini-335L"),
    ("cam_l", "Left Wrist", "D405"),
    ("cam_r", "Right Wrist", "D405"),
)


def load_index(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    return {(row["sequence"], int(row["frame_position"])): row for row in rows}


def make_rgb_mosaic(row: dict[str, Any], output: Path) -> None:
    panels = []
    panel_size = (320, 240)
    for camera, label, model in CAMERAS:
        image = cv2.imread(row["rgb_paths"][camera], cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read {row['rgb_paths'][camera]}")
        image = cv2.resize(image, panel_size, interpolation=cv2.INTER_AREA)
        cv2.rectangle(image, (0, 0), (panel_size[0], 35), (12, 16, 20), -1)
        cv2.putText(
            image,
            f"{label} / {model}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (242, 245, 247),
            1,
            cv2.LINE_AA,
        )
        panels.append(image)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(output), cv2.hconcat(panels), [cv2.IMWRITE_JPEG_QUALITY, 92]
    ):
        raise RuntimeError(f"Failed to write {output}")


def resolve_input(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path.cwd() / candidate


def colorize_depth(depth_mm: Any) -> Any:
    depth_m = depth_mm.astype("float32") * 0.001
    valid = (depth_m > 0) & (depth_m == depth_m)
    normalized = ((depth_m - 0.2) / (4.0 - 0.2)).clip(0.0, 1.0)
    color = cv2.applyColorMap((normalized * 255.0).round().astype("uint8"), cv2.COLORMAP_TURBO)
    color[~valid] = (14, 16, 17)
    return color


def make_depth_mosaic(row: dict[str, Any], method: str, output: Path) -> None:
    panels = []
    panel_size = (320, 240)
    for camera, label, model in CAMERAS:
        source = resolve_input(row["depth_paths"][method][camera])
        depth = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f"Failed to read {source}")
        if depth.ndim != 2:
            raise RuntimeError(f"Expected single-channel depth in {source}, got {depth.shape}")
        image = colorize_depth(depth)
        image = cv2.resize(image, panel_size, interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(image, (0, 0), (panel_size[0], 35), (12, 16, 20), -1)
        cv2.putText(
            image,
            f"{label} / {model}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (242, 245, 247),
            1,
            cv2.LINE_AA,
        )
        panels.append(image)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(output), cv2.hconcat(panels), [cv2.IMWRITE_JPEG_QUALITY, 94]
    ):
        raise RuntimeError(f"Failed to write {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--attention-root", type=Path, required=True)
    parser.add_argument("--benchmark-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    index = load_index(args.index)
    report = json.loads(args.benchmark_report.read_text(encoding="utf-8"))
    result_by_method = {result["method"]: result for result in report["results"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    catalog: dict[str, Any] = {
        "version": 1,
        "model": "Depth-only ACT checkpoint 009000",
        "promptUsed": False,
        "attention": "Last decoder Action Query to depth tokens",
        "cameras": [
            {"id": camera, "label": label, "model": model}
            for camera, label, model in CAMERAS
        ],
        "methods": [],
        "datasets": [],
    }

    for method, label, family in METHODS:
        metrics = result_by_method[method]["metrics"]
        catalog["methods"].append(
            {
                "id": method,
                "label": label,
                "family": family,
                "globalMaeRad": metrics["chunk_mae_rad"],
                "globalFirstStepMaeRad": metrics["first_step_mae_rad"],
            }
        )

    for slug, sequence, label, positions in DATASETS:
        frames = []
        for position in positions:
            row = index[(sequence, position)]
            stem = f"{sequence}_{position:06d}"
            rgb_relative = Path("media") / slug / "rgb" / f"{position:06d}.jpg"
            make_rgb_mosaic(row, args.output_dir / rgb_relative)
            attention: dict[str, str] = {}
            depth: dict[str, str] = {}
            for method, _method_label, _family in METHODS:
                source = args.attention_root / method / f"{stem}.jpg"
                if not source.is_file():
                    raise FileNotFoundError(source)
                relative = (
                    Path("media")
                    / slug
                    / "attention"
                    / method
                    / f"{position:06d}.jpg"
                )
                destination = args.output_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                attention[method] = relative.as_posix()
            for method in DEPTH_METHOD_IDS:
                relative = (
                    Path("media")
                    / slug
                    / "depth"
                    / method
                    / f"{position:06d}.jpg"
                )
                make_depth_mosaic(row, method, args.output_dir / relative)
                depth[method] = relative.as_posix()
            frames.append(
                {
                    "framePosition": position,
                    "timestampNs": int(row["timestamp_ns"]),
                    "rgb": rgb_relative.as_posix(),
                    "depth": depth,
                    "attention": attention,
                }
            )

        sequence_metrics = {
            method: result_by_method[method]["sequences"][sequence]
            for method, _label, _family in METHODS
        }
        catalog["datasets"].append(
            {
                "id": slug,
                "sequence": sequence,
                "label": label,
                "frames": frames,
                "metrics": sequence_metrics,
            }
        )

    data_dir = args.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "catalog.json").write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Exported {len(catalog['datasets'])} datasets, {len(METHODS)} methods, "
        f"{sum(len(dataset['frames']) for dataset in catalog['datasets'])} "
        f"samples to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
