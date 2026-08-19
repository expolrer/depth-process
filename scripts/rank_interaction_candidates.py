#!/usr/bin/env python3
"""Rank grounded candidates using robot-side, RGB-D, gripper, and motion cues."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def box_iou(first: list[float], second: list[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(first_area + second_area - intersection, 1e-9)


def nms(candidates: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda row: float(row["score"]), reverse=True):
        if any(box_iou(candidate["box_xyxy"], kept["box_xyxy"]) >= threshold for kept in selected):
            continue
        selected.append(candidate)
    return selected


def box_distance(first: list[float], second: list[float], diagonal: float) -> float:
    dx = max(second[0] - first[2], first[0] - second[2], 0.0)
    dy = max(second[1] - first[3], first[1] - second[3], 0.0)
    return math.hypot(dx, dy) / max(diagonal, 1.0)


def center_distance(box: list[float], point: tuple[float, float], diagonal: float) -> float:
    center = ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)
    return math.hypot(center[0] - point[0], center[1] - point[1]) / max(diagonal, 1.0)


def matching_wrist(camera: str, side: str) -> bool:
    return (camera == "cam_l" and side == "left") or (camera == "cam_r" and side == "right")


def motion_context(signals: list[dict[str, Any]], frame_index: int, side: str) -> dict[str, float | None]:
    if not signals:
        return {"eef_displacement_m": None, "joint_displacement_l2": None}
    before = signals[max(0, frame_index - 5)]
    after = signals[min(len(signals) - 1, frame_index + 5)]
    eef_key = f"{side}_position_m"
    eef_before = np.asarray(before.get("eef_pose", {}).get(eef_key, []), dtype=np.float64)
    eef_after = np.asarray(after.get("eef_pose", {}).get(eef_key, []), dtype=np.float64)
    eef_displacement = float(np.linalg.norm(eef_after - eef_before)) if eef_before.size == eef_after.size == 3 else None

    joint_before = np.asarray(before.get("arm_joint_state", {}).get("position", []), dtype=np.float64)
    joint_after = np.asarray(after.get("arm_joint_state", {}).get("position", []), dtype=np.float64)
    if joint_before.size >= 14 and joint_after.size >= 14:
        indices = slice(0, 7) if side == "left" else slice(7, 14)
        joint_displacement = float(np.linalg.norm(joint_after[indices] - joint_before[indices]))
    else:
        joint_displacement = None
    return {"eef_displacement_m": eef_displacement, "joint_displacement_l2": joint_displacement}


def select_grippers(grippers: list[dict[str, Any]], camera: str, side: str) -> list[dict[str, Any]]:
    if not grippers:
        return []
    if camera != "cam_h" or len(grippers) == 1:
        return grippers
    ordered = sorted(grippers, key=lambda row: (row["box_xyxy"][0] + row["box_xyxy"][2]) * 0.5)
    return [ordered[0] if side == "left" else ordered[-1]]


def rank_row(row: dict[str, Any], signals: list[dict[str, Any]], nms_threshold: float) -> dict[str, Any]:
    image = Image.open(row["rgb_path"])
    width, height = image.size
    diagonal = math.hypot(width, height)
    object_candidates = nms(
        [candidate for candidate in row["candidates"] if candidate["role"] == "object_candidate"],
        nms_threshold,
    )
    grippers = nms([candidate for candidate in row["candidates"] if candidate["role"] == "gripper"], 0.7)
    relevant_grippers = select_grippers(grippers, row["camera"], row["side"])
    camera_score = 1.0 if matching_wrist(row["camera"], row["side"]) else (0.72 if row["camera"] == "cam_h" else 0.18)

    ranked: list[dict[str, Any]] = []
    for candidate in object_candidates:
        box = candidate["box_xyxy"]
        area_ratio = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) / max(width * height, 1)
        if relevant_grippers:
            proximity_distance = min(box_distance(box, gripper["box_xyxy"], diagonal) for gripper in relevant_grippers)
            proximity_score = float(math.exp(-proximity_distance / 0.10))
            candidate_depth = candidate.get("depth", {}).get("median_mm")
            depth_differences = []
            for gripper in relevant_grippers:
                gripper_depth = gripper.get("depth", {}).get("median_mm")
                if candidate_depth is not None and gripper_depth is not None:
                    depth_differences.append(abs(float(candidate_depth) - float(gripper_depth)))
            depth_delta_mm = min(depth_differences) if depth_differences else None
            depth_score = float(math.exp(-depth_delta_mm / 250.0)) if depth_delta_mm is not None else 0.5
            anchor_source = "detected_gripper"
        else:
            if matching_wrist(row["camera"], row["side"]):
                anchor = (width * 0.5, height * 0.52)
            else:
                anchor = (width * (0.32 if row["side"] == "left" else 0.68), height * 0.68)
            proximity_distance = center_distance(box, anchor, diagonal)
            proximity_score = float(math.exp(-proximity_distance / 0.22))
            depth_delta_mm = None
            depth_score = 0.5
            anchor_source = "camera_side_prior"

        minimum_area = min(1.0, area_ratio / 0.004)
        if matching_wrist(row["camera"], row["side"]):
            large_area_penalty = math.exp(-max(0.0, area_ratio - 0.90) * 8.0)
        else:
            large_area_penalty = math.exp(-max(0.0, area_ratio - 0.42) * 5.0)
        area_score = float(minimum_area * large_area_penalty)
        detector_score = float(candidate["score"])
        interaction_score = (
            0.34 * detector_score
            + 0.34 * proximity_score
            + 0.11 * depth_score
            + 0.10 * camera_score
            + 0.11 * area_score
        )
        ranked.append(
            {
                **candidate,
                "area_ratio": area_ratio,
                "detector_score": detector_score,
                "proximity_distance_normalized": proximity_distance,
                "proximity_score": proximity_score,
                "depth_delta_to_gripper_mm": depth_delta_mm,
                "depth_score": depth_score,
                "camera_score": camera_score,
                "area_score": area_score,
                "anchor_source": anchor_source,
                "interaction_score": float(interaction_score),
            }
        )
    ranked.sort(key=lambda item: item["interaction_score"], reverse=True)
    top_score = float(ranked[0]["interaction_score"]) if ranked else 0.0
    second_score = float(ranked[1]["interaction_score"]) if len(ranked) > 1 else 0.0
    margin = top_score - second_score
    motion = motion_context(signals, int(row["frame_index"]), row["side"])
    return {
        **row,
        "candidates_before_nms": len(row["candidates"]),
        "object_candidates_after_nms": len(ranked),
        "ranked_candidates": ranked,
        "selected_candidate_id": ranked[0]["candidate_id"] if ranked else None,
        "top_interaction_score": top_score,
        "selection_margin": margin,
        "needs_vlm": not ranked or top_score < 0.48 or margin < 0.08,
        "robot_motion": motion,
    }


def draw_ranked(row: dict[str, Any], output_path: Path) -> None:
    image = Image.open(row["rgb_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for rank, candidate in enumerate(row["ranked_candidates"][:12], start=1):
        box = tuple(int(round(item)) for item in candidate["box_xyxy"])
        color = (255, 210, 0) if rank == 1 else (70, 210, 255)
        draw.rectangle(box, outline=color, width=4 if rank == 1 else 2)
        label = f"R{rank}/ID{candidate['candidate_id']} {candidate['label']} S={candidate['interaction_score']:.2f}"
        text_box = draw.textbbox((box[0], max(0, box[1] - 14)), label, font=font)
        draw.rectangle(text_box, fill=color)
        draw.text((text_box[0], text_box[1]), label, fill=(0, 0, 0), font=font)
    title = f"{row['camera']} {row['side']} margin={row['selection_margin']:.3f} VLM={row['needs_vlm']}"
    draw.rectangle((0, 0, image.width, 20), fill=(0, 0, 0))
    draw.text((6, 4), title, fill=(255, 255, 255), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92)


def build_event_collage(rows: list[dict[str, Any]], output_path: Path) -> None:
    tiles: list[Image.Image] = []
    for row in sorted(rows, key=lambda item: item["camera"]):
        image = Image.open(row["ranked_visualization_path"]).convert("RGB")
        image.thumbnail((520, 360))
        canvas = Image.new("RGB", (520, 380), (0, 0, 0))
        canvas.paste(image, ((520 - image.width) // 2, 20))
        tiles.append(canvas)
    collage = Image.new("RGB", (520 * len(tiles), 380), (0, 0, 0))
    for index, tile in enumerate(tiles):
        collage.paste(tile, (index * 520, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    collage.save(output_path, quality=91)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-index", type=Path, default=Path("outputs/grounded_candidates/candidate_index.jsonl"))
    parser.add_argument("--signals-root", type=Path, default=Path("outputs/robot_signals"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/interaction_candidates"))
    parser.add_argument("--nms-threshold", type=float, default=0.65)
    args = parser.parse_args()

    source_rows = read_jsonl(args.candidate_index)
    signal_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    ranked_rows: list[dict[str, Any]] = []
    for source in source_rows:
        key = (source["sequence"], source["camera"])
        if key not in signal_cache:
            signal_cache[key] = read_jsonl(args.signals_root / key[0] / key[1] / "robot_signals.jsonl")
        ranked = rank_row(source, signal_cache[key], args.nms_threshold)
        visualization = (
            args.output_root
            / source["sequence"]
            / f"grasp_{source['event_index']:03d}_{source['side']}"
            / source["camera"]
            / "ranked.jpg"
        )
        draw_ranked(ranked, visualization)
        ranked["ranked_visualization_path"] = str(visualization)
        ranked_rows.append(ranked)

    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ranked_rows:
        by_event[row["event_id"]].append(row)
    ambiguity_rows: list[dict[str, Any]] = []
    for event_id, event_rows in by_event.items():
        side = event_rows[0]["side"]
        preferred_camera = "cam_l" if side == "left" else "cam_r"
        preferred = next((row for row in event_rows if row["camera"] == preferred_camera), None)
        if preferred is None or not preferred["ranked_candidates"]:
            preferred = next((row for row in event_rows if row["camera"] == "cam_h"), event_rows[0])
        preferred_top = preferred["ranked_candidates"][0] if preferred["ranked_candidates"] else None
        oversized_top = bool(preferred_top and float(preferred_top.get("area_ratio", 0.0)) > 0.55)
        event_needs_vlm = bool(preferred["needs_vlm"] or oversized_top)
        collage_path = args.output_root / preferred["sequence"] / f"grasp_{preferred['event_index']:03d}_{side}" / "vlm_candidates.jpg"
        build_event_collage(event_rows, collage_path)
        ambiguity_rows.append(
            {
                "event_id": event_id,
                "sequence": preferred["sequence"],
                "event_index": preferred["event_index"],
                "side": side,
                "prompt": preferred["prompt"],
                "preferred_camera": preferred["camera"],
                "automatic_candidate_id": preferred["selected_candidate_id"],
                "automatic_score": preferred["top_interaction_score"],
                "automatic_margin": preferred["selection_margin"],
                "needs_vlm": event_needs_vlm,
                "collage_path": str(collage_path),
                "views": [
                    {
                        "camera": row["camera"],
                        "selected_candidate_id": row["selected_candidate_id"],
                        "score": row["top_interaction_score"],
                        "margin": row["selection_margin"],
                        "ranked_candidates": row["ranked_candidates"][:12],
                    }
                    for row in event_rows
                ],
            }
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    ranked_index = args.output_root / "ranked_index.jsonl"
    ranked_index.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ranked_rows), encoding="utf-8")
    ambiguity_index = args.output_root / "ambiguity_queue.jsonl"
    ambiguity_index.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ambiguity_rows), encoding="utf-8")
    summary = {
        "schema": "robot_interaction_candidate_ranking_v1",
        "camera_rows": len(ranked_rows),
        "events": len(ambiguity_rows),
        "events_needing_vlm": sum(bool(row["needs_vlm"]) for row in ambiguity_rows),
        "ranked_index": str(ranked_index),
        "ambiguity_queue": str(ambiguity_index),
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
