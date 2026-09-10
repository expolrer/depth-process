from interaction_auto_labeler_v3.gvl import build_progress_probe, evaluate_progress_response


def test_shuffled_progress_probe_scores_temporal_understanding() -> None:
    probe = build_progress_probe("episode-1", 101, sample_count=6, required_entity_ids=["toy"])
    response = {
        "frames": [
            {
                "frame_index": frame,
                "progress": frame / 100,
                "phase": "manipulate",
                "visible_entity_ids": ["toy"],
                "description": "The robot manipulates the toy.",
            }
            for frame in probe.presented_frames
        ]
    }
    report = evaluate_progress_response(probe, response)
    assert report["rank_correlation"] > 0.99
    assert report["pairwise_order_accuracy"] == 1.0
    assert report["needs_review"] is False
