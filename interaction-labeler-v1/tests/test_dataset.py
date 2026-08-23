import json
from pathlib import Path

from PIL import Image

from interaction_labeler.dataset import detect_input_format, index_extracted
from interaction_labeler.task import load_task


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_index_extracted_fallback_event(tmp_path: Path) -> None:
    root = tmp_path / "extracted"
    camera = root / "demo" / "cam_h"
    image = camera / "rgb" / "000000.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (32, 24), "white").save(image)
    write_jsonl(
        camera / "manifest.jsonl",
        [{"rgb_path": str(image), "depth_raw_mm_path": str(image)}],
    )
    task = load_task(None)
    session = index_extracted(root, tmp_path / "work", task)
    assert detect_input_format(root) == "extracted"
    assert len(session["events"]) == 1
    assert session["events"][0]["views"][0]["role"] == "head"
    assert session["events"][0]["views"][0]["contact_frame"] == 0
