from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from .target_annotations import AnnotationIndex


def _crop_with_context(
    image: np.ndarray,
    box: tuple[float, float, float, float],
    output_size: tuple[int, int],
    context_ratio: float,
) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    center_x, center_y = (x1 + x2) * 0.5 * width, (y1 + y2) * 0.5 * height
    box_width = max(2.0, (x2 - x1) * width * (1.0 + context_ratio))
    box_height = max(2.0, (y2 - y1) * height * (1.0 + context_ratio))
    left = max(0, int(center_x - box_width * 0.5))
    right = min(width, max(left + 1, int(center_x + box_width * 0.5)))
    top = max(0, int(center_y - box_height * 0.5))
    bottom = min(height, max(top + 1, int(center_y + box_height * 0.5)))
    crop = Image.fromarray(image[top:bottom, left:right])
    return np.asarray(crop.resize((output_size[1], output_size[0]), Image.Resampling.BILINEAR))


@dataclass
class OpenPiTargetCropTransform:
    annotations: AnnotationIndex
    cameras: tuple[str, ...]
    output_size: tuple[int, int] = (224, 224)
    context_ratio: float = 0.20
    dropout_probability: float = 0.40
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        output = dict(sample)
        episode = int(sample["episode_index"])
        frame = int(sample["frame_index"])
        for camera in self.cameras:
            annotation = self.annotations.get(episode, frame, camera)
            image_key = f"observation.images.{camera}"
            crop_key = f"observation.images.{camera}_target_crop"
            bbox_key = f"observation.target_roi.{camera}.bbox"
            valid_key = f"observation.target_roi.{camera}.valid"
            use_roi = (
                annotation is not None
                and annotation.visible
                and self.rng.random() >= self.dropout_probability
            )
            if use_roi:
                output[crop_key] = _crop_with_context(
                    np.asarray(sample[image_key]),
                    annotation.bbox_normalized,
                    self.output_size,
                    self.context_ratio,
                )
                output[bbox_key] = np.asarray(annotation.bbox_normalized, dtype=np.float32)
                output[valid_key] = np.float32(1.0)
            else:
                output[crop_key] = np.asarray(
                    Image.fromarray(np.asarray(sample[image_key])).resize(
                        (self.output_size[1], self.output_size[0]),
                        Image.Resampling.BILINEAR,
                    )
                )
                output[bbox_key] = np.zeros(4, dtype=np.float32)
                output[valid_key] = np.float32(0.0)
        return output
