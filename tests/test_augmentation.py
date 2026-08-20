from pathlib import Path

import numpy as np

from depth_pipeline.augmentation import RGBDAugmenter


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def sample_rgbd() -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[:40, :60]
    rgb = np.stack((x * 4, y * 6, (x + y) * 2), axis=2).clip(0, 255).astype(np.uint8)
    depth = (0.8 + x * 0.01 + y * 0.005).astype(np.float32)
    return rgb, depth


def test_weights_match_requested_probabilities() -> None:
    augmenter = RGBDAugmenter.from_yaml(CONFIG_DIR / "rgb_augmentation_only.yaml")
    assert augmenter.probabilities["notransform"] == 0.25
    for name, probability in augmenter.probabilities.items():
        if name != "notransform":
            assert np.isclose(probability, 1.0 / 12.0)


def test_rgb_only_mask_keeps_depth_unchanged() -> None:
    rgb, depth = sample_rgbd()
    augmenter = RGBDAugmenter.from_yaml(CONFIG_DIR / "rgb_augmentation_only.yaml")
    output_rgb, output_depth, audit = augmenter.apply(
        rgb, depth, sample_key="rgb-only", force_transform="random_mask"
    )
    assert audit["spatial_mask_fraction"] == 0.01
    assert np.any(output_rgb != rgb)
    assert np.array_equal(output_depth, depth)


def test_paired_mask_uses_the_same_invalid_region() -> None:
    rgb, depth = sample_rgbd()
    augmenter = RGBDAugmenter.from_yaml(CONFIG_DIR / "rgbd_paired_augmentation.yaml")
    output_rgb, output_depth, audit = augmenter.apply(
        rgb, depth, sample_key="paired", force_transform="random_mask"
    )
    rgb_mask = np.all(output_rgb == 0, axis=2) & np.any(rgb != 0, axis=2)
    depth_mask = output_depth == 0
    assert audit["depth_rule"] == "paired_invalid_mask"
    assert np.array_equal(rgb_mask, depth_mask)


def test_photometric_transform_never_changes_metric_depth() -> None:
    rgb, depth = sample_rgbd()
    augmenter = RGBDAugmenter.from_yaml(CONFIG_DIR / "rgbd_paired_augmentation.yaml")
    output_rgb, output_depth, audit = augmenter.apply(
        rgb, depth, sample_key="gamma", force_transform="gamma_correction"
    )
    assert np.any(output_rgb != rgb)
    assert np.array_equal(output_depth, depth)
    assert audit["depth_rule"] == "preserve_metric_depth"


def test_sampling_is_reproducible() -> None:
    rgb, depth = sample_rgbd()
    augmenter = RGBDAugmenter.from_yaml(CONFIG_DIR / "rgbd_paired_augmentation.yaml")
    first = augmenter.apply(rgb, depth, sample_key="episode-3/frame-10/cam-l")
    second = augmenter.apply(rgb, depth, sample_key="episode-3/frame-10/cam-l")
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[2] == second[2]
