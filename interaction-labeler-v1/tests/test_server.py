import json
from pathlib import Path

import pytest

from interaction_labeler.server import LabelerApplication


def _workspace(tmp_path: Path) -> Path:
    (tmp_path / "session.json").write_text(
        json.dumps(
            {
                "task": {
                    "instruction": "old instruction",
                    "target": {"names": ["old target"]},
                    "detector_prompt": ["old target"],
                },
                "events": [{"event_id": "event:0", "instruction": "old instruction"}],
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_save_task_updates_session_and_event_instruction(tmp_path: Path) -> None:
    app = LabelerApplication(_workspace(tmp_path))
    task = app.save_task(
        {
            "instruction": "Pick up the red toy.",
            "target": {
                "names": ["red toy", "toy"],
                "attributes": ["red", "soft"],
                "source_region": "middle pile",
                "destination_region": "right side",
                "negative_descriptions": ["yellow toy"],
                "exemplar_images": ["/data/red-toy.png"],
            },
            "detector_prompt": ["red toy", "robotic gripper"],
        }
    )
    session = json.loads((tmp_path / "session.json").read_text(encoding="utf-8"))
    assert task["target"]["names"] == ["red toy", "toy"]
    assert task["target"]["source_region"] == "middle pile"
    assert session["events"][0]["instruction"] == "Pick up the red toy."


def test_save_task_rejects_empty_target_names(tmp_path: Path) -> None:
    app = LabelerApplication(_workspace(tmp_path))
    with pytest.raises(ValueError, match="target.names"):
        app.save_task(
            {
                "instruction": "Pick an object.",
                "target": {"names": []},
                "detector_prompt": ["object"],
            }
        )
