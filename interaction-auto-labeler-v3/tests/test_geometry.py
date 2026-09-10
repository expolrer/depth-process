import math

import numpy as np

from interaction_auto_labeler_v3.geometry import (
    CameraIntrinsics,
    backproject_depth,
    fuse_point_clouds,
    mask_iou,
    reproject_mask,
)
from interaction_auto_labeler_v3.kinematics import dynamic_camera_transform, parse_urdf
from interaction_auto_labeler_v3.pose_tracking import depth_proxy_pose


def test_identity_reprojection_and_point_cloud_fusion() -> None:
    intrinsics = CameraIntrinsics(100, 100, 2, 2, 5, 5)
    depth = np.ones((5, 5), dtype=float)
    mask = np.zeros_like(depth, dtype=bool)
    mask[1:4, 1:4] = True
    projected, _ = reproject_mask(depth, mask, intrinsics, np.eye(4), intrinsics)
    assert mask_iou(projected, mask) == 1.0
    points, _ = backproject_depth(depth, intrinsics, mask)
    fused, counts = fuse_point_clouds([points, points], [np.eye(4), np.eye(4)], 0.001)
    assert len(fused) == len(points)
    assert set(counts) == {2}


def test_dynamic_wrist_extrinsic_uses_fk() -> None:
    tree = parse_urdf(
        """
        <robot name="demo">
          <joint name="wrist_yaw" type="revolute">
            <parent link="base"/><child link="wrist"/>
            <origin xyz="1 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
          </joint>
        </robot>
        """
    )
    eef_from_camera = np.eye(4)
    eef_from_camera[0, 3] = 0.1
    pose = dynamic_camera_transform(
        tree, {"wrist_yaw": math.pi / 2}, "base", "wrist", eef_from_camera
    )
    assert np.allclose(pose[:3, 3], [1.0, 0.1, 0.0], atol=1e-8)


def test_depth_pose_is_marked_as_noncanonical_proxy() -> None:
    intrinsics = CameraIntrinsics(100, 100, 5, 5, 11, 11)
    depth = np.ones((11, 11), dtype=float)
    mask = np.ones_like(depth, dtype=bool)
    pose = depth_proxy_pose(4, depth, mask, intrinsics, np.eye(4), minimum_points=20)
    assert pose.metric_translation is True
    assert pose.canonical_rotation is False
    assert pose.source == "rgbd_centroid_pca_proxy"
