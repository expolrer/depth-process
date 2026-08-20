#!/usr/bin/env python3
"""Build deterministic three-view RGB-D augmentation review assets."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from depth_pipeline.augmentation import RGBDAugmenter, TransformPlan


CAMERAS = ("cam_h", "cam_l", "cam_r")
TRANSFORM_LABELS = {
    "notransform": "不增强",
    "brightness": "亮度",
    "contrast": "对比度",
    "saturation": "饱和度",
    "hue": "色相",
    "sharpness": "锐度",
    "random_mask": "随机遮挡",
    "random_border_cutout": "边缘裁除",
    "gaussian_noise": "高斯噪声",
    "gamma_correction": "Gamma 校正",
}
TRANSFORM_FAMILIES = {
    "notransform": "identity",
    "brightness": "photometric",
    "contrast": "photometric",
    "saturation": "photometric",
    "hue": "photometric",
    "sharpness": "photometric",
    "random_mask": "spatial_occlusion",
    "random_border_cutout": "spatial_occlusion",
    "gaussian_noise": "photometric",
    "gamma_correction": "photometric",
}


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer-root", type=Path, default=workspace / "RGB-D-Depth-Lab-Video")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    return parser.parse_args()


def read_video_frame(path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Unable to decode frame {frame_index}: {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def split_views(frame: np.ndarray) -> list[np.ndarray]:
    if frame.shape[1] % 3:
        raise ValueError(f"Expected a three-view horizontal frame, got {frame.shape}")
    return list(np.split(frame, 3, axis=1))


def save_rgb(path: Path, rgb: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"Unable to write image: {path}")
    path.write_bytes(encoded.tobytes())


def visible_enough(plan: TransformPlan) -> bool:
    params = plan.params
    if "brightness" in params or "contrast" in params or "saturation" in params or "sharpness" in params:
        factor = float(next(iter(params.values())))
        return abs(factor - 1.0) >= 0.16
    if "hue" in params:
        return abs(float(params["hue"])) >= 0.025
    if "gamma" in params:
        return abs(float(params["gamma"]) - 1.0) >= 0.25
    return True


def demo_plan(
    augmenter: RGBDAugmenter,
    image_shape: tuple[int, ...],
    base_key: str,
    transform: str,
) -> tuple[TransformPlan, int]:
    for trial in range(100):
        plan = augmenter.sample_plan(
            image_shape,
            f"{base_key}:trial-{trial}",
            force_transform=transform,
        )
        if visible_enough(plan):
            return plan, trial
    raise RuntimeError(f"Unable to sample an illustrative plan for {transform}")


def mean_metric(audits: list[dict[str, Any]], key: str) -> float:
    values = [float(audit[key]) for audit in audits if audit.get(key) is not None]
    return float(np.mean(values)) if values else 0.0


def main() -> None:
    args = parse_args()
    viewer_root = args.viewer_root.resolve()
    output_root = (args.output_root or viewer_root / "augmentation").resolve()
    assets_root = output_root / "assets"
    config_root = Path(__file__).resolve().parents[1] / "config"
    rgb_only_path = config_root / "rgb_augmentation_only.yaml"
    paired_path = config_root / "rgbd_paired_augmentation.yaml"
    rgb_only = RGBDAugmenter.from_yaml(rgb_only_path, seed=args.seed)
    paired = RGBDAugmenter.from_yaml(paired_path, seed=args.seed)
    if rgb_only.names != paired.names or rgb_only.probabilities != paired.probabilities:
        raise ValueError("RGB-only and paired configs must share the same RGB policy")

    catalog = json.loads((viewer_root / "data" / "catalog.json").read_text(encoding="utf-8"))
    quality_report = json.loads(
        (viewer_root / "quality_evidence" / "depth_quality_evidence.json").read_text(encoding="utf-8")
    )
    representative_frames = {
        item["sequence"]: int(item["frame_index"]) for item in quality_report["representatives"]
    }

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "config").mkdir(parents=True, exist_ok=True)
    shutil.copy2(rgb_only_path, output_root / "config" / rgb_only_path.name)
    shutil.copy2(paired_path, output_root / "config" / paired_path.name)

    transform_rows = []
    for name in rgb_only.names:
        spec = rgb_only.transforms[name]
        family = TRANSFORM_FAMILIES[name]
        transform_rows.append(
            {
                "id": name,
                "label": TRANSFORM_LABELS[name],
                "type": spec["type"],
                "family": family,
                "weight": float(spec.get("weight", 1.0)),
                "probability": rgb_only.probabilities[name],
                "rgb_rule": "保持原图" if family == "identity" else "应用增强",
                "rgb_only_depth_rule": "深度保持原值",
                "paired_depth_rule": "同位置写入无效深度 0"
                if family == "spatial_occlusion"
                else "保持 metric depth 原值",
            }
        )

    dataset_rows = []
    for dataset in catalog["datasets"]:
        sequence = dataset["sequence"]
        if sequence not in representative_frames:
            continue
        frame_index = representative_frames[sequence]
        rgb_frame = read_video_frame(viewer_root / dataset["rgb"], frame_index)
        depth_frame = read_video_frame(
            viewer_root / dataset["depth"]["lingbot_v05_sensor_fused"], frame_index
        )
        rgb_views = split_views(rgb_frame)
        depth_views = split_views(depth_frame)
        dataset_asset_root = assets_root / dataset["id"]
        save_rgb(dataset_asset_root / "original_rgb.jpg", rgb_frame, args.jpeg_quality)
        save_rgb(dataset_asset_root / "original_depth.jpg", depth_frame, args.jpeg_quality)

        sample_rows: dict[str, Any] = {}
        for transform in rgb_only.names:
            augmented_views = []
            paired_depth_views = []
            rgb_only_audits = []
            paired_audits = []
            trials = []
            for camera, rgb_view, depth_view in zip(CAMERAS, rgb_views, depth_views):
                base_key = f"{sequence}:{frame_index}:{camera}:{transform}"
                plan, trial = demo_plan(paired, rgb_view.shape, base_key, transform)
                rgb_only_output, rgb_only_depth, rgb_audit = rgb_only.apply_plan(
                    rgb_view, depth_view, plan
                )
                paired_rgb, paired_depth, paired_audit = paired.apply_plan(
                    rgb_view, depth_view, plan
                )
                if not np.array_equal(rgb_only_output, paired_rgb):
                    raise AssertionError("Both versions must use identical RGB augmentation")
                if not np.array_equal(rgb_only_depth, depth_view):
                    raise AssertionError("RGB-only version must preserve depth")
                augmented_views.append(rgb_only_output)
                paired_depth_views.append(paired_depth)
                rgb_only_audits.append({"camera": camera, **rgb_audit})
                paired_audits.append({"camera": camera, **paired_audit})
                trials.append(trial)

            sample_root = dataset_asset_root / transform
            augmented_rgb = np.concatenate(augmented_views, axis=1)
            paired_depth_image = np.concatenate(paired_depth_views, axis=1)
            save_rgb(sample_root / "augmented_rgb.jpg", augmented_rgb, args.jpeg_quality)
            save_rgb(sample_root / "paired_depth.jpg", paired_depth_image, args.jpeg_quality)
            spatial = TRANSFORM_FAMILIES[transform] == "spatial_occlusion"
            sample_rows[transform] = {
                "augmented_rgb": str(
                    (Path("assets") / dataset["id"] / transform / "augmented_rgb.jpg").as_posix()
                ),
                "paired_depth": str(
                    (Path("assets") / dataset["id"] / transform / "paired_depth.jpg").as_posix()
                ),
                "rgb_changed_fraction": mean_metric(rgb_only_audits, "rgb_changed_fraction"),
                "rgb_only_depth_changed_fraction": mean_metric(
                    rgb_only_audits, "depth_changed_fraction"
                ),
                "paired_depth_changed_fraction": mean_metric(
                    paired_audits, "depth_changed_fraction"
                ),
                "intended_spatial_mask_fraction": mean_metric(
                    paired_audits, "spatial_mask_fraction"
                ),
                "paired_mask_iou": 1.0 if spatial else None,
                "rgb_only_cross_modal_mismatch": spatial,
                "trials": trials,
                "audits": paired_audits,
            }

        dataset_rows.append(
            {
                "id": dataset["id"],
                "label": dataset["label"],
                "sequence": sequence,
                "frame_index": frame_index,
                "original_rgb": str(
                    (Path("assets") / dataset["id"] / "original_rgb.jpg").as_posix()
                ),
                "original_depth": str(
                    (Path("assets") / dataset["id"] / "original_depth.jpg").as_posix()
                ),
                "depth_method": "lingbot_v05_sensor_fused",
                "samples": sample_rows,
            }
        )

    report = {
        "version": 1,
        "seed": args.seed,
        "sampling": {
            "datasets": len(dataset_rows),
            "camera_views": len(dataset_rows) * 3,
            "transforms": len(transform_rows),
            "review_samples": len(dataset_rows) * len(transform_rows) * 3,
            "max_num_transforms": rgb_only.max_num_transforms,
        },
        "source": {
            "rgb": "offline three-view RGB videos",
            "depth": "LingBot v0.5 + sensor fusion visualization",
            "frame_selection": "human-approved interaction representative frames",
        },
        "versions": [
            {
                "id": "rgb_only",
                "label": "版本一：仅 RGB 增强",
                "depth_policy": "所有变换均保持深度原值",
            },
            {
                "id": "rgbd_paired",
                "label": "版本二：RGB-D 物理一致增强",
                "depth_policy": "空间遮挡同步写入无效深度；光度变换保持 metric depth",
            },
        ],
        "transforms": transform_rows,
        "datasets": dataset_rows,
    }
    json_text = json.dumps(report, ensure_ascii=False, indent=2)
    (output_root / "augmentation_data.json").write_text(json_text + "\n", encoding="utf-8")
    (output_root / "augmentation_data.js").write_text(
        "window.RGBD_AUGMENTATION_REVIEW = " + json_text + ";\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output_root),
                "datasets": len(dataset_rows),
                "transforms": len(transform_rows),
                "samples": report["sampling"]["review_samples"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
