#!/usr/bin/env python3
"""Build synchronized three-view temporal evidence for every grasp event."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


CAMERAS = (("cam_h", "头部相机"), ("cam_l", "左腕相机"), ("cam_r", "右腕相机"))
COLLAGE_CAMERA_NAMES = {"cam_h": "HEAD", "cam_l": "LEFT WRIST", "cam_r": "RIGHT WRIST"}
COLLAGE_STAGE_NAMES = {
    "approach": "APPROACH",
    "pre_contact": "PRE-CONTACT",
    "initialize": "CONTACT",
    "transport": "TRANSPORT",
    "release": "RELEASE",
    "post_release": "AFTER RELEASE",
}
STAGES = (
    ("approach", "接近", -30),
    ("pre_contact", "接触前", -10),
    ("initialize", "初始化/接触", 0),
    ("transport", "搬运中段", None),
    ("release", "释放", None),
    ("post_release", "释放后", None),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def font(size: int) -> ImageFont.ImageFont:
    for path in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def output_url(value: str) -> str:
    value = value.replace("\\", "/")
    marker = "/outputs/"
    if marker in value:
        value = value.split(marker, 1)[1]
    elif value.startswith("outputs/"):
        value = value[len("outputs/") :]
    return "../" + value.lstrip("/")


def stage_indices(anchor: int, release: int, length: int) -> list[tuple[str, str, int]]:
    transport = round((anchor + release) / 2)
    values = (anchor - 30, anchor - 10, anchor, transport, release, release + 15)
    return [
        (key, label, max(0, min(length - 1, value)))
        for (key, label, _), value in zip(STAGES, values)
    ]


def make_collage(cameras: list[dict[str, Any]], path: Path) -> None:
    tile_width, tile_height, caption_height = 320, 181, 28
    canvas = Image.new("RGB", (tile_width * 6, (tile_height + caption_height) * 3), (20, 24, 26))
    draw = ImageDraw.Draw(canvas)
    label_font = font(17)
    for row_index, camera in enumerate(cameras):
        for col_index, stage in enumerate(camera["stages"]):
            image = Image.open(stage["rgb_path"]).convert("RGB")
            image.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
            x = col_index * tile_width + (tile_width - image.width) // 2
            y = row_index * (tile_height + caption_height) + caption_height + (tile_height - image.height) // 2
            canvas.paste(image, (x, y))
            caption = f"{COLLAGE_CAMERA_NAMES[camera['camera']]} | {COLLAGE_STAGE_NAMES[stage['key']]} | F{stage['frame']}"
            draw.rectangle((col_index * tile_width, row_index * (tile_height + caption_height), (col_index + 1) * tile_width, row_index * (tile_height + caption_height) + caption_height), fill=(20, 24, 26))
            draw.text((col_index * tile_width + 7, row_index * (tile_height + caption_height) + 4), caption, fill=(245, 247, 247), font=label_font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    outputs = root / "outputs"
    summary = json.loads((outputs / "target_tracks" / "summary.json").read_text(encoding="utf-8"))
    ranked = read_jsonl(outputs / "interaction_candidates" / "ranked_index.jsonl")
    queue = read_jsonl(outputs / "interaction_candidates" / "ambiguity_queue.jsonl")
    ranked_by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ranked:
        ranked_by_event[row["event_id"]].append(row)
    queue_by_event = {row["event_id"]: row for row in queue}

    rows = []
    for event in summary["events_detail"]:
        event_id = event["event_id"]
        sequence = event_id.split(":grasp_", 1)[0]
        release_delta = int(event["release_frame"]) - int(event["anchor_frame"])
        camera_rows = {row["camera"]: row for row in ranked_by_event[event_id]}
        cameras = []
        for camera, camera_label in CAMERAS:
            ranked_row = camera_rows[camera]
            manifest = read_jsonl(outputs / "extracted" / sequence / camera / "manifest.jsonl")
            anchor = int(ranked_row["frame_index"])
            release = anchor + release_delta
            stages = []
            for key, label, frame_index in stage_indices(anchor, release, len(manifest)):
                frame = manifest[frame_index]
                rgb_path = str(frame["rgb_path"])
                stages.append(
                    {
                        "key": key,
                        "label": label,
                        "frame": frame_index,
                        "timestamp_ns": frame.get("timestamp_ns"),
                        "rgb_path": rgb_path,
                        "rgb_url": output_url(rgb_path),
                        "depth_path": frame.get("depth_aligned_rgb_mm_path", frame.get("depth_raw_mm_path")),
                    }
                )
            cameras.append({"camera": camera, "label": camera_label, "anchor": anchor, "release": release, "stages": stages})
        event_name = event_id.split(":", 1)[1]
        collage_path = outputs / "interaction_review" / "temporal_collages" / sequence / f"{event_name}.jpg"
        make_collage(cameras, collage_path)
        queue_row = queue_by_event[event_id]
        rows.append(
            {
                "event_id": event_id,
                "sequence": sequence,
                "event_index": queue_row["event_index"],
                "side": queue_row["side"],
                "temporal_collage_path": str(collage_path),
                "temporal_collage_url": output_url(str(collage_path)),
                "candidate_collage_path": queue_row["collage_path"],
                "candidate_collage_url": output_url(queue_row["collage_path"]),
                "cameras": cameras,
            }
        )

    output = outputs / "interaction_review" / "temporal_evidence.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"events": len(rows), "views": len(rows) * 3, "images": len(rows) * 18, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
