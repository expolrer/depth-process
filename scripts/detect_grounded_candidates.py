#!/usr/bin/env python3
"""Run local GroundingDINO on robot grasp anchor frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


CAMERAS = ("cam_h", "cam_l", "cam_r")
TASK_PROMPTS = {
    "chengzhong": "metal sleeve . metal cylinder . weighing platform . box . robotic gripper . robot hand .",
    "dajian": "automotive sheet metal part . metal panel . support stand . robotic gripper . robot hand .",
    "zhoumian": "pin connector . seat belt . cable . empty box . robotic gripper . robot hand .",
    "leju_claw": "toy . task object . box . robotic claw . robotic gripper . robot hand . tabletop .",
    "dex_hand": "toy . task object . box . dexterous hand . robot hand . tabletop .",
}
GRIPPER_WORDS = ("gripper", "claw", "robot hand", "dexterous hand")
DESTINATION_WORDS = ("box", "platform", "stand", "tabletop")


def task_prompt(sequence: str) -> str:
    lowered = sequence.lower()
    for key, prompt in TASK_PROMPTS.items():
        if key in lowered:
            return prompt
    return "task object . robotic gripper . robot hand . box . tabletop ."


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def role_for(label: str) -> str:
    lowered = label.lower()
    if any(word in lowered for word in GRIPPER_WORDS):
        return "gripper"
    if any(word in lowered for word in DESTINATION_WORDS):
        return "destination"
    return "object_candidate"


def depth_stats(depth: np.ndarray, box: list[float]) -> dict[str, float | None]:
    height, width = depth.shape
    x1, y1, x2, y2 = box
    left = max(0, min(width - 1, int(np.floor(x1))))
    top = max(0, min(height - 1, int(np.floor(y1))))
    right = max(left + 1, min(width, int(np.ceil(x2))))
    bottom = max(top + 1, min(height, int(np.ceil(y2))))
    crop = depth[top:bottom, left:right]
    valid = crop[crop > 0]
    return {
        "valid_ratio": float(valid.size / max(crop.size, 1)),
        "median_mm": float(np.median(valid)) if valid.size else None,
        "p10_mm": float(np.percentile(valid, 10)) if valid.size else None,
        "p90_mm": float(np.percentile(valid, 90)) if valid.size else None,
    }


def postprocess(
    processor: Any,
    outputs: Any,
    input_ids: torch.Tensor,
    target_size: tuple[int, int],
    box_threshold: float,
    text_threshold: float,
) -> dict[str, Any]:
    kwargs = {
        "target_sizes": [target_size],
        "text_threshold": text_threshold,
    }
    try:
        return processor.post_process_grounded_object_detection(
            outputs,
            input_ids,
            box_threshold=box_threshold,
            **kwargs,
        )[0]
    except TypeError:
        return processor.post_process_grounded_object_detection(
            outputs,
            input_ids,
            threshold=box_threshold,
            **kwargs,
        )[0]


def text_labels(result: dict[str, Any]) -> list[str]:
    labels = result.get("text_labels", result.get("labels", []))
    output: list[str] = []
    for label in labels:
        if isinstance(label, str):
            output.append(label)
        elif hasattr(label, "item"):
            output.append(str(label.item()))
        else:
            output.append(str(label))
    return output


def draw_candidates(image: Image.Image, candidates: list[dict[str, Any]], title: str) -> Image.Image:
    output = image.copy().convert("RGB")
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    colors = {
        "object_candidate": (255, 220, 0),
        "gripper": (30, 110, 255),
        "destination": (90, 220, 90),
    }
    for candidate in candidates:
        x1, y1, x2, y2 = [int(round(item)) for item in candidate["box_xyxy"]]
        color = colors[candidate["role"]]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text = f"{candidate['candidate_id']} {candidate['label']} {candidate['score']:.2f}"
        text_box = draw.textbbox((x1, max(0, y1 - 15)), text, font=font)
        draw.rectangle(text_box, fill=color)
        draw.text((text_box[0], text_box[1]), text, fill=(0, 0, 0), font=font)
    draw.rectangle((0, 0, output.width, 20), fill=(0, 0, 0))
    draw.text((6, 4), title, fill=(255, 255, 255), font=font)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/grounding-dino-base"))
    parser.add_argument("--signals-root", type=Path, default=Path("outputs/robot_signals"))
    parser.add_argument("--extracted-root", type=Path, default=Path("outputs/extracted"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/grounded_candidates"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--box-threshold", type=float, default=0.20)
    parser.add_argument("--text-threshold", type=float, default=0.18)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--event-id")
    parser.add_argument("--frame-offset", type=int, default=0)
    parser.add_argument("--merge-existing", action="store_true")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.float32,
    ).to(args.device)
    model.eval()

    summary = json.loads((args.signals_root / "summary.json").read_text(encoding="utf-8"))
    index_path = args.output_root / "candidate_index.jsonl"
    if args.merge_existing and index_path.exists() and args.event_id:
        rows = [row for row in read_jsonl(index_path) if row["event_id"] != args.event_id]
    else:
        rows = []
    processed_events = 0
    for sequence_row in summary["sequences"]:
        sequence = sequence_row["sequence"]
        events = [
            event
            for event in sequence_row["events"]
            if event["kind"] in {"grasp_command_end", "confirmed_grab"}
            and abs(float(event.get("amplitude", 100.0))) >= 50.0
        ]
        for event_index, event in enumerate(events):
            if args.max_events is not None and processed_events >= args.max_events:
                break
            event_id = f"{sequence}:grasp_{event_index:03d}_{event['side']}"
            if args.event_id and event_id != args.event_id:
                continue
            for camera in CAMERAS:
                camera_events = read_jsonl(args.signals_root / sequence / camera / "events.jsonl")
                matches = [item for item in camera_events if item["kind"] == event["kind"] and item["side"] == event["side"]]
                if not matches:
                    continue
                mapped = min(matches, key=lambda item: abs(int(item["timestamp_ns"]) - int(event["timestamp_ns"])))
                manifest = read_jsonl(args.extracted_root / sequence / camera / "manifest.jsonl")
                frame_index = max(0, min(len(manifest) - 1, int(mapped["frame_index"]) + args.frame_offset))
                frame = manifest[frame_index]
                rgb_path = Path(frame["rgb_path"])
                depth_path = Path(frame.get("depth_aligned_rgb_mm_path", frame["depth_raw_mm_path"]))
                pil_image = Image.open(rgb_path).convert("RGB")
                prompt = task_prompt(sequence)
                inputs = processor(images=pil_image, text=prompt, return_tensors="pt")
                input_ids = inputs["input_ids"]
                inputs = {key: value.to(args.device) if hasattr(value, "to") else value for key, value in inputs.items()}
                with torch.inference_mode():
                    outputs = model(**inputs)
                result = postprocess(
                    processor,
                    outputs,
                    input_ids,
                    (pil_image.height, pil_image.width),
                    args.box_threshold,
                    args.text_threshold,
                )
                boxes = result["boxes"].detach().cpu().tolist()
                scores = result["scores"].detach().float().cpu().tolist()
                labels = text_labels(result)
                depth = np.asarray(Image.open(depth_path))
                candidates: list[dict[str, Any]] = []
                for candidate_index, (box, score, label_text) in enumerate(zip(boxes, scores, labels)):
                    candidate = {
                        "candidate_id": candidate_index,
                        "label": label_text,
                        "role": role_for(label_text),
                        "score": float(score),
                        "box_xyxy": [float(item) for item in box],
                    }
                    if depth.ndim == 2:
                        candidate["depth"] = depth_stats(depth, candidate["box_xyxy"])
                    candidates.append(candidate)
                relative = Path(sequence) / f"grasp_{event_index:03d}_{event['side']}" / camera
                visualization_path = args.output_root / relative / "candidates.jpg"
                visualization_path.parent.mkdir(parents=True, exist_ok=True)
                visualization = draw_candidates(
                    pil_image,
                    candidates,
                    f"{camera} frame {frame_index} {event['side']} grasp",
                )
                visualization.save(visualization_path, quality=92)
                rows.append(
                    {
                        "event_id": event_id,
                        "sequence": sequence,
                        "event_index": event_index,
                        "side": event["side"],
                        "camera": camera,
                        "frame_index": frame_index,
                        "timestamp_ns": int(event["timestamp_ns"]),
                        "rgb_path": str(rgb_path),
                        "depth_path": str(depth_path),
                        "prompt": prompt,
                        "candidates": candidates,
                        "visualization_path": str(visualization_path),
                    }
                )
                print(f"[grounding] {event_id} {camera}: {len(candidates)} candidates", flush=True)
            processed_events += 1
        if args.max_events is not None and processed_events >= args.max_events:
            break

    args.output_root.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = {
        "schema": "grounded_robot_candidates_v1",
        "model": str(args.model),
        "events": len({row["event_id"] for row in rows}),
        "camera_frames": len(rows),
        "candidates": sum(len(row["candidates"]) for row in rows),
        "index": str(index_path),
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
