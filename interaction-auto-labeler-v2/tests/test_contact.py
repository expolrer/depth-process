from interaction_auto_labeler.contact import detect_action_phases


def test_detect_close_and_release() -> None:
    gripper = [1.0] * 8 + [0.9, 0.6, 0.2, 0.1] + [0.1] * 8 + [0.3, 0.7, 1.0] + [1.0] * 4
    phases = detect_action_phases(gripper, close_direction="decrease", pre_frames=3, post_frames=2)
    assert phases["contact_frame"] < phases["release_frame"]
    assert phases["start_frame"] <= phases["contact_frame"]
    assert phases["end_frame"] >= phases["release_frame"]
