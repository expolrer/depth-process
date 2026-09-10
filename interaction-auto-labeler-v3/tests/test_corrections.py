import json

from interaction_auto_labeler_v3.corrections import HumanCorrection, export_distillation_sets


def test_human_bbox_correction_exports_coco_and_tracker_prompt(tmp_path) -> None:
    correction = HumanCorrection(
        correction_id="fix-1",
        dataset_id="toys",
        dataset_version="v3",
        episode_id="4",
        correction_type="bbox",
        frame_index=20,
        camera="head",
        before={"bbox_xyxy": [0, 0, 5, 5]},
        after={
            "bbox_xyxy": [10, 20, 30, 50],
            "image_size": [100, 80],
            "image_path": "episode4/head/20.jpg",
            "instance_id": "toy-1",
        },
    )
    log = tmp_path / "corrections.jsonl"
    log.write_text(json.dumps(correction.to_dict()) + "\n", encoding="utf-8")
    report = export_distillation_sets(log, tmp_path / "distill")
    assert report["counts"]["detector_annotations"] == 1
    coco = json.loads((tmp_path / "distill" / "detector_coco.json").read_text())
    assert coco["annotations"][0]["bbox"] == [10.0, 20.0, 20.0, 30.0]


def test_correction_rejects_noop() -> None:
    try:
        HumanCorrection("x", "d", "v", "e", "language", {"text": "a"}, {"text": "a"})
    except ValueError:
        pass
    else:
        raise AssertionError("no-op correction should be rejected")
