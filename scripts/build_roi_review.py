#!/usr/bin/env python3
"""Build a compact HTML page for human ROI mask spot checks."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def rel(path: str, output_dir: Path) -> str:
    p = Path(path)
    try:
        return p.relative_to(output_dir.parent).as_posix()
    except ValueError:
        return p.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi-root", type=Path, default=Path("outputs/roi_masks"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/roi_review"))
    parser.add_argument("--max-items", type=int, default=360)
    args = parser.parse_args()

    rows = load_jsonl(args.roi_root / "review_index.jsonl")[: args.max_items]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    for row in rows:
        overlay = rel(str(row["overlay_path"]), args.output_dir)
        title = f"{row['sequence']} / {row['camera']} / {int(row['frame_index']):06d}"
        cards.append(
            f"""
            <article class="card">
              <img src="../roi_masks/{html.escape(Path(row['sequence']).as_posix())}/{html.escape(str(row['camera']))}/overlay/{int(row['frame_index']):06d}.jpg" alt="{html.escape(title)}" loading="lazy">
              <div class="meta">
                <strong>{html.escape(title)}</strong>
                <span>area {float(row['area_ratio']):.3f} | conf {float(row['proposal_confidence']):.2f}</span>
              </div>
              <p>{html.escape(str(row['prompt']))}</p>
            </article>
            """
        )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ROI Mask Review</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, Arial, sans-serif; }}
    body {{ margin: 0; background: #f6f7f9; color: #18202a; }}
    header {{ position: sticky; top: 0; z-index: 2; background: #ffffff; border-bottom: 1px solid #d7dbe2; padding: 14px 18px; }}
    h1 {{ margin: 0; font-size: 18px; }}
    .sub {{ margin-top: 4px; color: #5d6978; font-size: 13px; }}
    main {{ padding: 18px; display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 14px; }}
    .card {{ background: #ffffff; border: 1px solid #dfe3ea; border-radius: 8px; overflow: hidden; }}
    img {{ display: block; width: 100%; aspect-ratio: 16 / 9; object-fit: contain; background: #0e1116; }}
    .meta {{ display: flex; justify-content: space-between; gap: 8px; padding: 10px 10px 0; font-size: 12px; }}
    .meta strong {{ font-size: 12px; overflow-wrap: anywhere; }}
    .meta span {{ color: #64748b; white-space: nowrap; }}
    p {{ margin: 8px 10px 12px; color: #425066; font-size: 12px; line-height: 1.35; }}
  </style>
</head>
<body>
  <header>
    <h1>ROI Mask Review</h1>
    <div class="sub">Yellow masks are pseudo-label proposals. Frames marked by this page should be spot-checked before using the numbers as final evidence.</div>
  </header>
  <main>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    (args.output_dir / "index.html").write_text(html_text, encoding="utf-8")
    print(f"wrote {args.output_dir / 'index.html'} with {len(rows)} items")


if __name__ == "__main__":
    main()
