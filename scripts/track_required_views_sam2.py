#!/usr/bin/env python3
"""Run SAM2 tracking for either the head or active-wrist view of every grasp."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from sam2.build_sam import build_sam2_video_predictor

from track_targets_sam2 import propagate, quality, read_jsonl, release_frame, save_mask


def yellow_overlay(rgb_path: Path, mask: np.ndarray, bbox: list[int] | None, path: Path) -> None:
    image = Image.open(rgb_path).convert("RGB")
    yellow = Image.new("RGB", image.size, (255, 220, 0))
    alpha = Image.fromarray(mask.astype(np.uint8) * 55, mode="L")
    image = Image.composite(yellow, image, alpha)
    if bbox:
        ImageDraw.Draw(image).rectangle(tuple(bbox), outline=(255, 220, 0), width=5)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("head", "wrist"), required=True)
    parser.add_argument("--selections", type=Path, default=Path("outputs/interaction_review/required_view_selections.jsonl"))
    parser.add_argument("--ranked-index", type=Path, default=Path("outputs/interaction_candidates/ranked_index.jsonl"))
    parser.add_argument("--signals-root", type=Path, default=Path("outputs/robot_signals"))
    parser.add_argument("--extracted-root", type=Path, default=Path("outputs/extracted"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/target_tracks_required"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/sam2/checkpoints/sam2.1_hiera_large.pt"))
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pre-frames", type=int, default=45)
    parser.add_argument("--post-release-frames", type=int, default=20)
    parser.add_argument("--max-span", type=int, default=240)
    args = parser.parse_args()

    selections = [row for row in read_jsonl(args.selections) if row["role"] == args.role]
    ranked = {(row["event_id"], row["camera"]): row for row in read_jsonl(args.ranked_index)}
    args.output_root.mkdir(parents=True, exist_ok=True)
    predictor = build_sam2_video_predictor(args.config, str(args.checkpoint), device=args.device)
    predictor.eval()
    all_rows, reports = [], []
    role_root = args.output_root / args.role

    for number, selection in enumerate(selections):
        event_id, camera = selection["event_id"], selection["camera"]
        ranked_row = ranked[(event_id, camera)]
        anchor = int(ranked_row["frame_index"])
        manifest = read_jsonl(args.extracted_root / selection["sequence"] / camera / "manifest.jsonl")
        release = release_frame(args.signals_root, selection["sequence"], camera, selection["side"], anchor, min(len(manifest) - 1, anchor + args.max_span))
        start = int(selection["start_frame_override"]) if selection.get("start_frame_override") is not None else max(0, anchor - args.pre_frames)
        end = int(selection["end_frame_override"]) if selection.get("end_frame_override") is not None else min(len(manifest) - 1, release + args.post_release_frames, anchor + args.max_span)
        start = max(0, min(start, anchor))
        end = min(len(manifest) - 1, max(end, anchor))
        anchor_local = anchor - start
        frames = manifest[start:end + 1]
        event_name = event_id.split(":", 1)[1]
        event_root = role_root / selection["sequence"] / event_name / camera

        with tempfile.TemporaryDirectory(prefix=f"sam2_{args.role}_", dir=args.output_root) as temporary:
            video_dir = Path(temporary)
            for local_index, frame in enumerate(frames):
                (video_dir / f"{local_index:06d}.jpg").symlink_to(Path(frame["rgb_path"]))
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = predictor.init_state(video_path=str(video_dir), offload_video_to_cpu=True, offload_state_to_cpu=False, async_loading_frames=False)
                box = np.asarray(selection["box_xyxy"], dtype=np.float32)
                _, object_ids, logits = predictor.add_new_points_or_box(inference_state=state, frame_idx=anchor_local, obj_id=1, box=box)
                position = list(object_ids).index(1)
                masks = {anchor_local: (logits[position] > 0.0).detach().cpu().numpy().squeeze().astype(bool)}
                for secondary in selection.get("secondary_prompts", []):
                    if "frame_index" in secondary:
                        secondary_local = max(0, min(len(frames) - 1, int(secondary["frame_index"]) - start))
                    else:
                        secondary_local = max(0, min(len(frames) - 1, anchor_local + int(secondary["frame_offset"])))
                    secondary_box = np.asarray(secondary["box_xyxy"], dtype=np.float32)
                    _, secondary_ids, secondary_logits = predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=secondary_local,
                        obj_id=1,
                        box=secondary_box,
                    )
                    secondary_position = list(secondary_ids).index(1)
                    masks[secondary_local] = (
                        secondary_logits[secondary_position] > 0.0
                    ).detach().cpu().numpy().squeeze().astype(bool)
                masks.update(propagate(predictor, state, anchor_local, False, end - anchor + 1))
                masks.update(propagate(predictor, state, anchor_local, True, anchor - start + 1))

        stage_frames = {start, max(start, anchor - 10), anchor, round((anchor + release) / 2), release, end}
        previous_area = None
        event_rows = []
        for local_index in sorted(masks):
            frame_index = start + local_index
            metrics = quality(masks[local_index], previous_area)
            previous_area = metrics["area_ratio"]
            mask_path = event_root / "mask" / f"{frame_index:06d}.png"
            save_mask(mask_path, masks[local_index])
            overlay_path = None
            if frame_index in stage_frames or metrics["needs_review"]:
                overlay_path = event_root / "overlay" / f"{frame_index:06d}.jpg"
                yellow_overlay(Path(manifest[frame_index]["rgb_path"]), masks[local_index], metrics["bbox_xyxy"], overlay_path)
            event_rows.append({
                **selection,
                "frame_index": frame_index,
                "anchor_frame_index": anchor,
                "release_frame_index": release,
                "start_frame_index": start,
                "end_frame_index": end,
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path) if overlay_path else None,
                **metrics,
            })
        all_rows.extend(event_rows)
        reports.append({
            **selection,
            "anchor_frame": anchor,
            "release_frame": release,
            "start_frame": start,
            "end_frame": end,
            "tracked_frames": len(event_rows),
            "empty_frames": sum(float(row["area_ratio"]) == 0 for row in event_rows),
            "quality_flagged_frames": sum(bool(row["needs_review"]) for row in event_rows),
        })
        print(f"[sam2-{args.role}] {number + 1}/{len(selections)} {event_id} {camera}: {len(event_rows)} frames", flush=True)

    role_root.mkdir(parents=True, exist_ok=True)
    (role_root / "track_index.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows), encoding="utf-8")
    (role_root / "summary.json").write_text(json.dumps({"role": args.role, "views": len(reports), "tracked_frames": len(all_rows), "views_detail": reports}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"role": args.role, "views": len(reports), "tracked_frames": len(all_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
