from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VariantSpec:
    name: str
    packed_channels: int
    geometry: str
    frontend: str
    requires_depth: bool
    requires_xyz: bool = False
    requires_lingbot: bool = False


VARIANTS = {
    "ACT0_RGB": VariantSpec(
        "ACT0_RGB", 3, "none", "official", False
    ),
    "ACT1_EARLY_RGBD": VariantSpec(
        "ACT1_EARLY_RGBD", 4, "metric_z", "early", True
    ),
    "ACT2_DEPTH_CNN": VariantSpec(
        "ACT2_DEPTH_CNN", 5, "metric_z_validity", "depth_cnn", True
    ),
    "ACT3_DEPTH_RESNET": VariantSpec(
        "ACT3_DEPTH_RESNET", 5, "metric_z_validity", "depth_resnet", True
    ),
    "ACT4_XYZMAP": VariantSpec(
        "ACT4_XYZMAP", 7, "camera_xyz_validity", "xyz_cnn", True, requires_xyz=True
    ),
    "ACT5_POINT_TOKENS": VariantSpec(
        "ACT5_POINT_TOKENS", 7, "unordered_xyz_validity", "point_tokens", True, requires_xyz=True
    ),
    "ACT6_LINGBOT_DEPTH": VariantSpec(
        "ACT6_LINGBOT_DEPTH", 5, "lingbot_depth_v0.5", "lingbot", True, requires_lingbot=True
    ),
    "ACT7_DEPTH_TRANSFORMER": VariantSpec(
        "ACT7_DEPTH_TRANSFORMER", 5, "metric_z_validity", "depth_transformer", True
    ),
}

SUPPORTED_VARIANTS = tuple(VARIANTS)


def get_variant(name: str) -> VariantSpec:
    try:
        return VARIANTS[name]
    except KeyError as error:
        choices = ", ".join(SUPPORTED_VARIANTS)
        raise ValueError(f"Unsupported ACT RGB-D variant {name!r}; choose from {choices}") from error

