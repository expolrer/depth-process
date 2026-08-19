#!/usr/bin/env python3
"""Build compact contact/transport atlases for visual QA of required views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/target_tracks_required"))
    parser.add_argument("--events-per-page", type=int, default=5)
    args = parser.parse_args()
    summary = json.loads((args.root / "summary.json").read_text(encoding="utf-8"))
    rows = read_jsonl(args.root / "track_index.jsonl")
    by_key = {}
    for row in rows:
        by_key.setdefault((row["event_id"], row["role"]), []).append(row)
    views = {(row["event_id"], row["role"]): row for row in summary["views_detail"]}
    event_ids = sorted({event_id for event_id, _ in views})
    output_dir = args.root / "audit_atlas"
    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 19)
    outputs = []
    for page_index in range(0, len(event_ids), args.events_per_page):
        page_events = event_ids[page_index : page_index + args.events_per_page]
        canvas = Image.new("RGB", (1696, len(page_events) * 570), (18, 22, 24))
        draw = ImageDraw.Draw(canvas)
        for event_offset, event_id in enumerate(page_events):
            y0 = event_offset * 570
            draw.text((8, y0 + 4), event_id, fill=(255, 255, 255), font=title_font)
            for role_index, role in enumerate(("head", "wrist")):
                detail = views[(event_id, role)]
                event_rows = by_key[(event_id, role)]
                anchor = int(detail["anchor_frame"])
                release = int(detail["release_frame"])
                targets = (anchor, round((anchor + release) / 2))
                for stage_index, target in enumerate(targets):
                    row = min(event_rows, key=lambda item: abs(int(item["frame_index"]) - target))
                    image = Image.open(row["overlay_path"]).convert("RGB").resize((424, 240), Image.Resampling.LANCZOS)
                    x = (role_index * 2 + stage_index) * 424
                    y = y0 + 58
                    canvas.paste(image, (x, y))
                    stage = "contact" if stage_index == 0 else "transport"
                    label = f"{role} {detail['camera']} #{detail['candidate_id']} {stage} F{row['frame_index']} area {100 * float(row['area_ratio']):.2f}%"
                    draw.rectangle((x, y + 240, x + 424, y + 270), fill=(18, 22, 24))
                    draw.text((x + 6, y + 245), label, fill=(255, 255, 255), font=font)
            draw.line((0, y0 + 568, 1696, y0 + 568), fill=(90, 100, 104), width=2)
        output = output_dir / f"page_{page_index // args.events_per_page + 1:02d}.jpg"
        canvas.save(output, quality=92)
        outputs.append(str(output))
    print(json.dumps({"pages": len(outputs), "outputs": outputs}, ensure_ascii=False))


if __name__ == "__main__":
    main()
