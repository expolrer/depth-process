#!/usr/bin/env python3
"""Render proposed human override boxes before expensive SAM2 reruns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    overrides = read_jsonl(root / "config" / "required_view_human_overrides.jsonl")
    overrides += read_jsonl(root / "config" / "required_view_human_overrides_v4.jsonl")
    ranked = read_jsonl(root / "outputs" / "interaction_candidates" / "ranked_index.jsonl")
    ranked_by_key = {(row["event_id"], row["camera"]): row for row in ranked}
    items = []
    for override in overrides:
        row = ranked_by_key[(override["event_id"], override["camera"])]
        candidate = next(item for item in row["ranked_candidates"] if int(item["candidate_id"]) == int(override["candidate_id"]))
        items.append({
            **override,
            "frame_index": int(row["frame_index"]),
            "rgb_path": row["rgb_path"],
            "box": override.get("box_xyxy", candidate["box_xyxy"]),
            "kind": "initial",
        })
        if override.get("secondary_prompts"):
            manifest = read_jsonl(root / "outputs" / "extracted" / row["sequence"] / override["camera"] / "manifest.jsonl")
            for secondary in override["secondary_prompts"]:
                frame_index = int(secondary.get("frame_index", row["frame_index"] + secondary.get("frame_offset", 0)))
                items.append({
                    **override,
                    "frame_index": frame_index,
                    "rgb_path": manifest[frame_index]["rgb_path"],
                    "box": secondary["box_xyxy"],
                    "kind": "secondary",
                })
    items.sort(key=lambda item: (item["event_id"], item["camera"], item["frame_index"]))
    out_dir = root / "outputs" / "interaction_review" / "human_override_preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
    outputs = []
    for page_start in range(0, len(items), 4):
        canvas = Image.new("RGB", (1696, 1080), (16, 20, 22))
        draw = ImageDraw.Draw(canvas)
        for offset, item in enumerate(items[page_start : page_start + 4]):
            column, row_index = offset % 2, offset // 2
            x, y = column * 848, row_index * 540
            image = Image.open(item["rgb_path"]).convert("RGB")
            image_draw = ImageDraw.Draw(image)
            box = tuple(round(float(value)) for value in item["box"])
            image_draw.rectangle(box, outline=(255, 212, 0), width=6)
            image_draw.text((max(4, box[0]), max(4, box[1] - 23)), f"{item['kind']} F{item['frame_index']}", fill=(255, 212, 0), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
            canvas.paste(image, (x, y + 60))
            title = f"{page_start + offset + 1:02d} {item['event_id']} | {item['camera']} | {item['kind']} F{item['frame_index']}"
            draw.text((x + 7, y + 9), title, fill=(255, 255, 255), font=font)
        output = out_dir / f"page_{page_start // 4 + 1:02d}.jpg"
        canvas.save(output, quality=94)
        outputs.append(str(output))
    (out_dir / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"boxes": len(items), "pages": len(outputs), "outputs": outputs}, ensure_ascii=False))


if __name__ == "__main__":
    main()
