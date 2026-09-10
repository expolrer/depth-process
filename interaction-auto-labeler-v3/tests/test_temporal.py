import numpy as np

from interaction_auto_labeler_v3.temporal import (
    BoundaryConfig,
    propose_hierarchical_boundaries,
    segments_from_boundaries,
)


def test_multimodal_boundary_proposals_find_interaction_change() -> None:
    frames = 120
    gripper = np.zeros(frames)
    gripper[40:] = 1.0
    object_position = np.zeros((frames, 2))
    object_position[40:, 0] = np.linspace(0, 20, frames - 40)
    config = BoundaryConfig(
        threshold_quantiles={"task": 0.9, "subtask": 0.8, "event": 0.7},
        min_gap_frames={"task": 20, "subtask": 15, "event": 10},
        smoothing_frames=3,
    )
    proposals, scores = propose_hierarchical_boundaries(
        {"gripper": gripper, "object": object_position}, config
    )
    event_frames = [item.frame_index for item in proposals if item.level == "event"]
    assert any(abs(frame - 40) <= 3 for frame in event_frames)
    assert scores["event"].shape == (frames,)


def test_hierarchical_segments_are_nested_when_cuts_align() -> None:
    from interaction_auto_labeler_v3.temporal import BoundaryProposal

    proposals = [
        BoundaryProposal("task", 50, 0.9, 2, {"scene": 0.9}),
        BoundaryProposal("subtask", 25, 0.8, 2, {"visual": 0.8}),
        BoundaryProposal("subtask", 50, 0.8, 2, {"visual": 0.8}),
        BoundaryProposal("subtask", 75, 0.8, 2, {"visual": 0.8}),
        BoundaryProposal("event", 25, 0.8, 2, {"gripper": 0.8}),
        BoundaryProposal("event", 50, 0.8, 2, {"gripper": 0.8}),
        BoundaryProposal("event", 75, 0.8, 2, {"gripper": 0.8}),
    ]
    segments = segments_from_boundaries(100, proposals)
    assert sum(item.level == "task" for item in segments) == 2
    assert all(item.parent_segment_id for item in segments if item.level != "task")
