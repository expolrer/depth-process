#!/usr/bin/env python3
"""Build full-resolution candidate atlases for human-flagged required views."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--review", type=Path, default=Path("outputs/interaction_review/human_review_v3.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    review = json.loads(args.review.read_text(encoding="utf-8"))
    ranked = read_jsonl(root / "outputs" / "interaction_candidates" / "ranked_index.jsonl")
    ranked_by_key = {(row["event_id"], row["camera"]): row for row in ranked}
    issues = []
    for event_id, decision in review["decisions"].items():
        if decision.get("decision") != "correction":
            continue
        note = decision.get("note", "")
        side = "left" if event_id.endswith("_left") else "right"
        requested = []
        if "头部" in note:
            requested.append(("head", "cam_h"))
        if "执行腕" in note:
            requested.append(("wrist", "cam_l" if side == "left" else "cam_r"))
        for role, camera in requested:
            row = ranked_by_key[(event_id, camera)]
            issues.append({
                "event_id": event_id,
                "role": role,
                "camera": camera,
                "frame_index": row["frame_index"],
                "path": row["visualization_path"],
                "note": note,
            })
    issues.sort(key=lambda row: (row["event_id"], row["role"] != "head"))

    out_dir = root / "outputs" / "interaction_review" / "human_correction_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    note_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    outputs = []
    for page_start in range(0, len(issues), 4):
        canvas = Image.new("RGB", (1696, 1180), (16, 20, 22))
        draw = ImageDraw.Draw(canvas)
        for offset, issue in enumerate(issues[page_start : page_start + 4]):
            column, row_index = offset % 2, offset // 2
            x, y = column * 848, row_index * 590
            image = Image.open(root / issue["path"]).convert("RGB")
            canvas.paste(image, (x, y + 92))
            title = f"{page_start + offset + 1:02d} {issue['event_id']} | {issue['role']} {issue['camera']} | F{issue['frame_index']}"
            draw.text((x + 8, y + 7), title, fill=(255, 255, 255), font=title_font)
            relevant = f"See issues.json item {page_start + offset + 1:02d} for the full Chinese review note."
            wrapped = textwrap.wrap(relevant, width=70)[:2]
            draw.multiline_text((x + 8, y + 34), "\n".join(wrapped), fill=(255, 214, 92), font=note_font, spacing=3)
        output = out_dir / f"page_{page_start // 4 + 1:02d}.jpg"
        canvas.save(output, quality=94)
        outputs.append(str(output))
    (out_dir / "issues.json").write_text(json.dumps(issues, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"issues": len(issues), "pages": len(outputs), "outputs": outputs}, ensure_ascii=False))


if __name__ == "__main__":
    main()
