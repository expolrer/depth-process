#!/usr/bin/env python3
"""Build multi-camera contact montages for grasp-event review."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import cv2
import numpy as np


CAMERAS = ("cam_h", "cam_l", "cam_r")
OFFSETS = (-15, -5, 0, 5, 15)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def fit(image: np.ndarray, width: int = 320, height: int = 180) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def label(image: np.ndarray, text: str, anchor: bool) -> np.ndarray:
    output = image.copy()
    color = (40, 220, 255) if anchor else (235, 235, 235)
    cv2.rectangle(output, (0, 0), (output.shape[1], 27), (0, 0, 0), -1)
    cv2.putText(output, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    if anchor:
        cv2.rectangle(output, (1, 1), (output.shape[1] - 2, output.shape[0] - 2), color, 3)
    return output


def build_montage(
    sequence: str,
    event: dict,
    extracted_root: Path,
    signals_root: Path,
) -> tuple[np.ndarray, dict[str, int]]:
    rows: list[np.ndarray] = []
    anchors: dict[str, int] = {}
    for camera in CAMERAS:
        events = read_jsonl(signals_root / sequence / camera / "events.jsonl")
        match = min(events, key=lambda row: abs(int(row["timestamp_ns"]) - int(event["timestamp_ns"])))
        anchor = int(match["frame_index"])
        anchors[camera] = anchor
        manifest = read_jsonl(extracted_root / sequence / camera / "manifest.jsonl")
        frames: list[np.ndarray] = []
        for offset in OFFSETS:
            index = min(max(anchor + offset, 0), len(manifest) - 1)
            image = cv2.imread(manifest[index]["rgb_path"], cv2.IMREAD_COLOR)
            if image is None:
                image = np.zeros((180, 320, 3), dtype=np.uint8)
            tile = label(fit(image), f"{camera}  frame {index}  {offset:+d}", offset == 0)
            frames.append(tile)
        rows.append(np.concatenate(frames, axis=1))
    return np.concatenate(rows, axis=0), anchors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals-root", type=Path, default=Path("outputs/robot_signals"))
    parser.add_argument("--extracted-root", type=Path, default=Path("outputs/extracted"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/contact_review"))
    args = parser.parse_args()

    summary = json.loads((args.signals_root / "summary.json").read_text(encoding="utf-8"))
    review_rows: list[dict] = []
    for sequence_row in summary["sequences"]:
        sequence = sequence_row["sequence"]
        grasp_events = [
            event
            for event in sequence_row["events"]
            if event["kind"] in {"grasp_command_end", "confirmed_grab"}
            and abs(float(event.get("amplitude", 100.0))) >= 50.0
        ]
        for event_index, event in enumerate(grasp_events):
            montage, anchors = build_montage(sequence, event, args.extracted_root, args.signals_root)
            filename = f"event_{event_index:03d}_{event['side']}.jpg"
            path = args.output_root / sequence / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), montage, [cv2.IMWRITE_JPEG_QUALITY, 91])
            review_rows.append(
                {
                    "sequence": sequence,
                    "event_index": event_index,
                    "timestamp_ns": int(event["timestamp_ns"]),
                    "side": event["side"],
                    "kind": event["kind"],
                    "confidence": float(event["confidence"]),
                    "camera_anchor_frames": anchors,
                    "montage_path": str(path),
                    "status": "pending",
                }
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    index_path = args.output_root / "review_index.jsonl"
    index_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in review_rows),
        encoding="utf-8",
    )
    cards = []
    for row in review_rows:
        relative = Path(row["montage_path"]).relative_to(args.output_root).as_posix()
        anchors = ", ".join(f"{key}: {value}" for key, value in row["camera_anchor_frames"].items())
        cards.append(
            f"<article><h2>{html.escape(row['sequence'])}</h2>"
            f"<p>Event {row['event_index']} | {row['side']} | {html.escape(anchors)}</p>"
            f"<img loading='lazy' src='{html.escape(relative)}'></article>"
        )
    page = """<!doctype html><html><head><meta charset='utf-8'>
<title>Robot Contact Review</title><style>
body{font:14px Arial,sans-serif;margin:0;background:#101214;color:#eceff1}header{position:sticky;top:0;padding:14px 20px;background:#191d20;border-bottom:1px solid #343a40;z-index:2}
main{max-width:1640px;margin:auto;padding:16px}article{margin:0 0 20px;border-bottom:1px solid #343a40;padding-bottom:18px}h1{font-size:20px;margin:0}h2{font-size:15px;margin:0 0 5px}p{color:#aeb7be;margin:0 0 9px}img{width:100%;height:auto;display:block;background:#000}
</style></head><body><header><h1>Grasp contact anchor review</h1></header><main>""" + "".join(cards) + "</main></body></html>"
    (args.output_root / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps({"review_events": len(review_rows), "index": str(args.output_root / "index.html")}, indent=2))


if __name__ == "__main__":
    main()
