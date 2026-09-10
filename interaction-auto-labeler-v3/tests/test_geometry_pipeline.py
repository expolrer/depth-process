import numpy as np
from PIL import Image

from interaction_auto_labeler_v3.event_graph import (
    Entity,
    EpisodeEventGraph,
    TrackObservation,
)
from interaction_auto_labeler_v3.geometry_pipeline import process_episode_geometry


def test_calibrated_geometry_updates_graph_and_cross_view_metrics(tmp_path) -> None:
    depth = np.full((7, 7), 1000, dtype=np.uint16)
    mask = np.zeros((7, 7), dtype=np.uint8)
    mask[1:6, 1:6] = 255
    Image.fromarray(depth).save(tmp_path / "depth.png")
    Image.fromarray(mask).save(tmp_path / "mask.png")
    intrinsics = {"fx": 100, "fy": 100, "cx": 3, "cy": 3, "width": 7, "height": 7}
    rows = [
        {
            "episode_id": "0",
            "frame_index": 2,
            "camera": camera,
            "depth_path": "depth.png",
            "mask_path": "mask.png",
            "intrinsics": intrinsics,
            "world_from_camera": np.eye(4).tolist(),
        }
        for camera in ("head", "right_wrist")
    ]
    graph = EpisodeEventGraph(
        dataset_id="demo",
        dataset_version="v3",
        episode_id="0",
        frame_count=5,
        fps=30,
        entities=[Entity("target", "object")],
        observations=[
            TrackObservation("head-2", "target", "head", 2, (1, 1, 6, 6)),
            TrackObservation("wrist-2", "target", "right_wrist", 2, (1, 1, 6, 6)),
        ],
    )
    result, report = process_episode_geometry(
        graph, rows, tmp_path, tmp_path / "output", export_point_clouds=False
    )
    assert report["mean_cross_view_iou"] == 1.0
    assert report["observation_updates"] == 2
    assert all(item.centroid_world_m is not None for item in result.observations)
    assert report["pose_contract"]["canonical_rotation"] is False
