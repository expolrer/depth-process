from pathlib import Path

from interaction_labeler.task import load_task, prompt_text


def test_task_override_and_prompt(tmp_path: Path) -> None:
    path = tmp_path / "task.yaml"
    path.write_text("active_hand: left\ndetector_prompt: [red toy, gripper]\n", encoding="utf-8")
    task = load_task(path)
    assert task["active_hand"] == "left"
    assert task["camera_roles"]["head"]
    assert prompt_text(task) == "red toy . gripper ."
