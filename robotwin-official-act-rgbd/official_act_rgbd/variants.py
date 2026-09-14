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
    "ACT2_DUAL_SHARED": VariantSpec(
        "ACT2_DUAL_SHARED", 4, "metric_z", "depth_resnet", True
    ),
    "ACT3_DUAL_PER_VIEW": VariantSpec(
        "ACT3_DUAL_PER_VIEW", 7, "metric_z_camera_onehot", "per_view_depth_resnet", True
    ),
    "ACT4_XYZMAP": VariantSpec(
        "ACT4_XYZMAP", 6, "camera_xyz", "xyz_cnn", True, requires_xyz=True
    ),
    "ACT5_POINT_TOKENS": VariantSpec(
        "ACT5_POINT_TOKENS", 6, "unordered_xyz", "point_tokens", True, requires_xyz=True
    ),
    "ACT6_LINGBOT_DEPTH": VariantSpec(
        "ACT6_LINGBOT_DEPTH", 4, "lingbot_depth_v0.5", "lingbot", True, requires_lingbot=True
    ),
    "ACT7_DEPTH_TRANSFORMER": VariantSpec(
        "ACT7_DEPTH_TRANSFORMER", 4, "metric_z", "depth_transformer", True
    ),
}

SUPPORTED_VARIANTS = tuple(VARIANTS)


def get_variant(name: str) -> VariantSpec:
    try:
        return VARIANTS[name]
    except KeyError as error:
        choices = ", ".join(SUPPORTED_VARIANTS)
        raise ValueError(f"Unsupported ACT RGB-D variant {name!r}; choose from {choices}") from error
