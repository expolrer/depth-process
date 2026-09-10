from interaction_auto_labeler_v3.reliability import (
    aggregate_reliability,
    choose_threshold,
    fit_isotonic_calibrator,
    selective_risk_curve,
)


def test_reliability_tracks_missing_evidence_and_calibrates_monotonically() -> None:
    raw = aggregate_reliability({"detection": 0.9, "contact": 0.8})
    assert raw["raw_score"] > 0.8
    assert raw["evidence_coverage"] < 1.0
    assert "track_stability" in raw["missing_required"]

    model = fit_isotonic_calibrator(
        [0.1, 0.2, 0.3, 0.6, 0.8, 0.9],
        [False, True, False, True, True, True],
    )
    predictions = model.predict([0.0, 0.25, 0.5, 1.0])
    assert predictions == sorted(predictions)


def test_selective_threshold_prefers_highest_coverage() -> None:
    curve = selective_risk_curve([0.2, 0.7, 0.8, 0.95], [False, False, True, True])
    selected = choose_threshold(curve, minimum_precision=0.9)
    assert selected is not None
    assert selected["precision"] >= 0.9
