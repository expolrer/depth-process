import json
from io import BytesIO

from PIL import Image

from interaction_auto_labeler_v3.server import V3LabelerApplication


def test_v3_server_records_interactive_bbox_correction(tmp_path) -> None:
    (tmp_path / "session.json").write_text(
        json.dumps(
            {
                "v3": {"dataset_id": "demo", "dataset_version": "v3"},
                "task": {"instruction": "pick"},
                "events": [{"event_id": "event-1", "episode_index": 3, "views": []}],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "annotations.json").write_text(
        json.dumps({"revision": 0, "items": []}), encoding="utf-8"
    )
    app = V3LabelerApplication(tmp_path)
    buffer = BytesIO()
    Image.new("RGB", (64, 48), "white").save(buffer, format="JPEG")
    app.frame = lambda _event, _camera, _frame: buffer.getvalue()  # type: ignore[method-assign]
    app.save_annotation(
        {
            "event_id": "event-1",
            "camera": "head",
            "frame_index": 9,
            "box_xyxy": [1, 2, 30, 40],
        }
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "v3" / "corrections.jsonl").read_text().splitlines()
    ]
    assert rows[0]["correction_type"] == "bbox"
    assert rows[0]["episode_id"] == "3"
    assert app.overview()["corrections"]["correction_count"] == 1


def test_v3_server_keeps_correction_when_frame_capture_fails(tmp_path) -> None:
    (tmp_path / "session.json").write_text(
        json.dumps(
            {
                "v3": {"dataset_id": "demo", "dataset_version": "v3"},
                "task": {"instruction": "pick"},
                "events": [{"event_id": "event-1", "episode_index": 3, "views": []}],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "annotations.json").write_text(
        json.dumps({"revision": 0, "items": []}), encoding="utf-8"
    )
    app = V3LabelerApplication(tmp_path)

    def fail_frame(_event, _camera, _frame):
        raise RuntimeError("decoder unavailable")

    app.frame = fail_frame  # type: ignore[method-assign]
    app.save_annotation(
        {
            "event_id": "event-1",
            "camera": "head",
            "frame_index": 9,
            "box_xyxy": [1, 2, 30, 40],
        }
    )
    row = json.loads(
        (tmp_path / "v3" / "corrections.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert row["after"]["bbox_xyxy"] == [1.0, 2.0, 30.0, 40.0]
    assert "decoder unavailable" in row["after"]["image_capture_error"]
