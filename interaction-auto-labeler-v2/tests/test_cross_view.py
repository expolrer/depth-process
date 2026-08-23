import numpy as np

from interaction_auto_labeler.cross_view import (
    backproject_depth,
    project_points,
    soft_cross_view_score,
)
from interaction_auto_labeler.inventory import (
    build_scene_tracks,
    co_motion_score,
    match_scene_track,
    source_destination_score,
)


def test_calibrated_projection_round_trip() -> None:
    depth = np.zeros((8, 8), dtype=np.uint16)
    depth[2:6, 2:6] = 1000
    intrinsics = np.array([[100.0, 0.0, 4.0], [0.0, 100.0, 4.0], [0.0, 0.0, 1.0]])
    points = backproject_depth(depth, intrinsics, [2, 2, 6, 6], stride=1)
    pixels, box = project_points(points, intrinsics, (8, 8))
    assert len(pixels) == 16
    assert box is not None
    assert box[0] == 2.0


def test_soft_match_uses_available_features_only() -> None:
    score = soft_cross_view_score({"appearance_similarity": 1.0, "contact_sync": 0.5})
    assert 0.70 < score < 0.80


def test_motion_and_region_evidence() -> None:
    objects = [(10, 10), (20, 10), (30, 10)]
    gripper = [(8, 9), (18, 9), (28, 9)]
    assert co_motion_score(objects, gripper, 100.0) > 0.99
    assert source_destination_score((5, 5), (95, 5), [0, 0, 10, 10], [90, 0, 100, 10]) == 1.0


def test_head_inventory_keeps_identical_objects_as_distinct_tracks() -> None:
    rows = [
        {
            "frame_index": frame,
            "image_size": [100, 100],
            "detections": [
                {"label": "toy", "box_xyxy": [10 + frame, 10, 20 + frame, 20]},
                {"label": "toy", "box_xyxy": [70, 10 + frame, 80, 20 + frame]},
            ],
        }
        for frame in (0, 5, 10)
    ]
    tracks = build_scene_tracks(rows)
    assert len(tracks) == 2
    assert all(len(track["observations"]) == 3 for track in tracks)
    matched, score = match_scene_track(
        {"label": "toy", "box_xyxy": [20, 10, 30, 20]},
        tracks,
        frame_index=10,
        image_diagonal=141.4,
    )
    assert matched is not None
    assert score is not None and score > 0.8
