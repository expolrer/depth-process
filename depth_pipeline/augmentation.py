"""Deterministic RGB and paired RGB-D augmentation utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import yaml
from PIL import Image, ImageEnhance


@dataclass(frozen=True)
class TransformPlan:
    """A sampled transform that can be reused across aligned observations."""

    name: str
    transform_type: str
    params: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class RGBDAugmenter:
    """Apply one weighted RGB transform with optional paired depth occlusion."""

    def __init__(self, config: Mapping[str, Any], seed: int | None = None) -> None:
        self.config = dict(config)
        rgb_config = self.config.get("RGB_Augmenter", self.config)
        self.enabled = bool(rgb_config.get("enable", True))
        self.max_num_transforms = int(rgb_config.get("max_num_transforms", 1))
        if self.max_num_transforms != 1:
            raise ValueError("This augmenter currently requires max_num_transforms: 1")

        transforms = rgb_config.get("tfs", {})
        if not transforms:
            raise ValueError("RGB_Augmenter.tfs must contain at least one transform")
        self.transforms = dict(transforms)
        self.names = list(self.transforms)
        weights = np.asarray(
            [float(self.transforms[name].get("weight", 1.0)) for name in self.names],
            dtype=np.float64,
        )
        if np.any(weights < 0) or float(weights.sum()) <= 0:
            raise ValueError("Transform weights must be non-negative with a positive sum")
        self.probabilities = {
            name: float(weight / weights.sum()) for name, weight in zip(self.names, weights)
        }
        self._probability_vector = weights / weights.sum()
        self.seed = int(self.config.get("seed", 0) if seed is None else seed)

        depth_config = self.config.get("Depth_Augmenter", {})
        self.depth_mode = str(depth_config.get("mode", "unchanged"))
        self.depth_invalid_value = depth_config.get("invalid_value", 0)
        self.paired_spatial_transforms = set(
            depth_config.get(
                "paired_spatial_transforms",
                ["random_mask", "random_border_cutout"]
                if self.depth_mode == "paired_spatial"
                else [],
            )
        )

    @classmethod
    def from_yaml(cls, path: str | Path, seed: int | None = None) -> "RGBDAugmenter":
        with Path(path).open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        return cls(config, seed=seed)

    def _rng(self, sample_key: str) -> np.random.Generator:
        digest = sha256(f"{self.seed}:{sample_key}".encode("utf-8")).digest()
        return np.random.default_rng(int.from_bytes(digest[:8], "little", signed=False))

    @staticmethod
    def _sample_range(rng: np.random.Generator, value: Any) -> float:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return float(rng.uniform(float(value[0]), float(value[1])))
        return float(value)

    def sample_plan(
        self,
        image_shape: tuple[int, ...],
        sample_key: str,
        force_transform: str | None = None,
    ) -> TransformPlan:
        if len(image_shape) < 2:
            raise ValueError("image_shape must include height and width")
        height, width = int(image_shape[0]), int(image_shape[1])
        rng = self._rng(sample_key)
        name = force_transform
        if not self.enabled:
            name = "notransform"
        elif name is None:
            name = str(rng.choice(self.names, p=self._probability_vector))
        if name not in self.transforms:
            raise KeyError(f"Unknown transform: {name}")

        spec = self.transforms[name]
        transform_type = str(spec.get("type", "Identity"))
        kwargs = dict(spec.get("kwargs", {}))
        params: dict[str, Any] = {}

        if transform_type == "ColorJitter":
            fields = [key for key in ("brightness", "contrast", "saturation", "hue") if key in kwargs]
            if len(fields) != 1:
                raise ValueError(f"{name} must configure exactly one ColorJitter field")
            field = fields[0]
            params[field] = self._sample_range(rng, kwargs[field])
        elif transform_type == "SharpnessJitter":
            params["sharpness"] = self._sample_range(rng, kwargs["sharpness"])
        elif transform_type == "RandomMask":
            ratio_h, ratio_w = kwargs.get("mask_size", [0.1, 0.1])
            mask_h = max(1, min(height, int(round(height * float(ratio_h)))))
            mask_w = max(1, min(width, int(round(width * float(ratio_w)))))
            params.update(
                {
                    "top": int(rng.integers(0, height - mask_h + 1)),
                    "left": int(rng.integers(0, width - mask_w + 1)),
                    "height": mask_h,
                    "width": mask_w,
                }
            )
        elif transform_type == "RandomBorderCutout":
            cut_ratio = float(kwargs.get("cut_ratio", 0.15))
            side = str(rng.choice(["top", "right", "bottom", "left"]))
            extent = height if side in {"top", "bottom"} else width
            params.update({"side": side, "pixels": max(1, int(round(extent * cut_ratio)))})
        elif transform_type == "GaussianNoise":
            params.update(
                {
                    "mean": float(kwargs.get("mean", 0.0)),
                    "std": float(kwargs.get("std", 0.05)),
                    "noise_seed": int(rng.integers(0, np.iinfo(np.uint32).max)),
                }
            )
        elif transform_type == "GammaCorrection":
            params["gamma"] = self._sample_range(rng, kwargs.get("gamma", [0.5, 2.0]))
        elif transform_type != "Identity":
            raise ValueError(f"Unsupported transform type: {transform_type}")

        return TransformPlan(name=name, transform_type=transform_type, params=params)

    @staticmethod
    def _spatial_mask(shape: tuple[int, int], plan: TransformPlan) -> np.ndarray:
        height, width = shape
        mask = np.zeros((height, width), dtype=bool)
        if plan.transform_type == "RandomMask":
            top = int(plan.params["top"])
            left = int(plan.params["left"])
            mask[top : top + int(plan.params["height"]), left : left + int(plan.params["width"])] = True
        elif plan.transform_type == "RandomBorderCutout":
            pixels = int(plan.params["pixels"])
            side = plan.params["side"]
            if side == "top":
                mask[:pixels, :] = True
            elif side == "right":
                mask[:, width - pixels :] = True
            elif side == "bottom":
                mask[height - pixels :, :] = True
            else:
                mask[:, :pixels] = True
        return mask

    @staticmethod
    def _pil_enhance(rgb: np.ndarray, enhancer: type, factor: float) -> np.ndarray:
        image = Image.fromarray(rgb)
        return np.asarray(enhancer(image).enhance(factor), dtype=np.uint8)

    def apply_plan(
        self,
        rgb: np.ndarray,
        depth: np.ndarray | None,
        plan: TransformPlan,
        frame_index: int = 0,
    ) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("rgb must be an HxWx3 uint8 RGB array")
        if depth is not None and depth.shape[:2] != rgb.shape[:2]:
            raise ValueError("depth must be spatially aligned with rgb")

        output_rgb = rgb.copy()
        output_depth = None if depth is None else depth.copy()
        transform_type = plan.transform_type
        params = plan.params

        if transform_type == "ColorJitter":
            if "brightness" in params:
                output_rgb = self._pil_enhance(output_rgb, ImageEnhance.Brightness, params["brightness"])
            elif "contrast" in params:
                output_rgb = self._pil_enhance(output_rgb, ImageEnhance.Contrast, params["contrast"])
            elif "saturation" in params:
                output_rgb = self._pil_enhance(output_rgb, ImageEnhance.Color, params["saturation"])
            else:
                hsv = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2HSV)
                hue = hsv[..., 0].astype(np.int16)
                hsv[..., 0] = np.mod(hue + int(round(params["hue"] * 180.0)), 180).astype(np.uint8)
                output_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        elif transform_type == "SharpnessJitter":
            output_rgb = self._pil_enhance(output_rgb, ImageEnhance.Sharpness, params["sharpness"])
        elif transform_type == "GaussianNoise":
            noise_rng = np.random.default_rng(int(params["noise_seed"]) + int(frame_index))
            image = output_rgb.astype(np.float32) / 255.0
            image += noise_rng.normal(params["mean"], params["std"], image.shape).astype(np.float32)
            output_rgb = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
        elif transform_type == "GammaCorrection":
            image = output_rgb.astype(np.float32) / 255.0
            output_rgb = np.clip(np.rint(np.power(image, 1.0 / params["gamma"]) * 255.0), 0, 255).astype(np.uint8)

        spatial_mask = self._spatial_mask(rgb.shape[:2], plan)
        if spatial_mask.any():
            output_rgb[spatial_mask] = 0
            if output_depth is not None and plan.name in self.paired_spatial_transforms:
                output_depth[spatial_mask] = self.depth_invalid_value

        rgb_changed = np.any(output_rgb != rgb, axis=2)
        if depth is None:
            depth_changed_fraction = None
        else:
            equal = np.isclose(output_depth, depth, equal_nan=True)
            if equal.ndim == 3:
                equal = np.all(equal, axis=2)
            depth_changed_fraction = float(np.mean(~equal))
        audit = {
            **plan.as_dict(),
            "depth_mode": self.depth_mode,
            "depth_rule": "paired_invalid_mask"
            if plan.name in self.paired_spatial_transforms
            else "preserve_metric_depth",
            "rgb_changed_fraction": float(np.mean(rgb_changed)),
            "depth_changed_fraction": depth_changed_fraction,
            "spatial_mask_fraction": float(np.mean(spatial_mask)),
        }
        return output_rgb, output_depth, audit

    def apply(
        self,
        rgb: np.ndarray,
        depth: np.ndarray | None = None,
        sample_key: str = "sample",
        force_transform: str | None = None,
        frame_index: int = 0,
    ) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
        plan = self.sample_plan(rgb.shape, sample_key, force_transform=force_transform)
        return self.apply_plan(rgb, depth, plan, frame_index=frame_index)
