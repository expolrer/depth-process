#!/usr/bin/env python3
"""Build six-stage contact sheets for required-view SAM2 tracks."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/target_tracks_required"))
    args = parser.parse_args()
    summary = json.loads((args.root / "summary.json").read_text(encoding="utf-8"))
    rows = read_jsonl(args.root / "track_index.jsonl")
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_key[(row["event_id"], row["role"])].append(row)
    labels = ("开始", "接触前", "初始化", "搬运", "释放", "结束")
    outputs = []
    for view in summary["views_detail"]:
        key = (view["event_id"], view["role"])
        event_rows = by_key[key]
        targets = (
            int(view["start_frame"]),
            max(int(view["start_frame"]), int(view["anchor_frame"]) - 10),
            int(view["anchor_frame"]),
            round((int(view["anchor_frame"]) + int(view["release_frame"])) / 2),
            int(view["release_frame"]),
            int(view["end_frame"]),
        )
        picked = [min(event_rows, key=lambda row: abs(int(row["frame_index"]) - target)) for target in targets]
        canvas = Image.new("RGB", (1272, 1164), (18, 22, 24))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        for index, (label, row) in enumerate(zip(labels, picked)):
            path = Path(row["overlay_path"])
            image = Image.open(path).convert("RGB")
            image.thumbnail((636, 360), Image.Resampling.LANCZOS)
            x = (index % 2) * 636
            y = (index // 2) * 388
            canvas.paste(image, (x, y + 28))
            draw.rectangle((x, y, x + 636, y + 28), fill=(18, 22, 24))
            draw.text((x + 7, y + 4), f"{label} / F{row['frame_index']} / area {100 * float(row['area_ratio']):.2f}%", fill=(255, 255, 255), font=font)
        event_name = view["event_id"].split(":", 1)[1]
        output = args.root / "review_sheets" / view["role"] / view["sequence"] / f"{event_name}_{view['camera']}.jpg"
        output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output, quality=92)
        outputs.append({"event_id": view["event_id"], "role": view["role"], "camera": view["camera"], "path": str(output)})
    index = args.root / "review_sheets.jsonl"
    index.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in outputs), encoding="utf-8")
    print(json.dumps({"sheets": len(outputs), "index": str(index)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
