from interaction_auto_labeler_v3.gold import (
    bbox_iou,
    boundary_metrics,
    calibration_metrics,
    stratified_gold_sample,
)


def test_stratified_sample_is_deterministic_and_covers_groups() -> None:
    rows = [
        {"episode_id": index, "task": "a" if index < 8 else "b", "scene": "lab"}
        for index in range(10)
    ]
    first = stratified_gold_sample(rows, 4, ("task", "scene"), seed=7)
    second = stratified_gold_sample(rows, 4, ("task", "scene"), seed=7)
    assert first == second
    assert {row["task"] for row in first} == {"a", "b"}


def test_gold_metrics() -> None:
    metrics = boundary_metrics([10, 31, 70], [12, 30, 90], tolerance_frames=2)
    assert metrics["true_positive"] == 2
    assert round(float(metrics["f1"]), 3) == 0.667
    assert bbox_iou([0, 0, 10, 10], [5, 0, 15, 10]) == 1 / 3
    calibration = calibration_metrics([0.9, 0.2], [True, False], bins=2)
    assert calibration["brier"] < 0.03
