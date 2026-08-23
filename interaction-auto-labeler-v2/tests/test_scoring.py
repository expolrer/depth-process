from interaction_auto_labeler.scoring import rank_candidates, score_candidate


def test_missing_evidence_is_renormalized() -> None:
    row = score_candidate({"instance_id": "a", "contact": 1.0, "descriptor_match": 0.0})
    assert 0.70 < row["interaction_instance_score"] < 0.80
    assert row["evidence"]["co_motion"] is None


def test_low_margin_enters_review() -> None:
    result = rank_candidates(
        [
            {"instance_id": "a", "contact": 0.8},
            {"instance_id": "b", "contact": 0.76},
        ],
        ambiguity_margin=0.08,
    )
    assert result["selected_instance_id"] == "a"
    assert result["needs_review"] is True
