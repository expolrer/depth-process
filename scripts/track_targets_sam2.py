#!/usr/bin/env python3
"""Track selected robot interaction targets bidirectionally with SAM2.1."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from sam2.build_sam import build_sam2_video_predictor


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_vlm(path: Path | None, minimum_confidence: float) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    return {
        row["event_id"]: row
        for row in read_jsonl(path)
        if row.get("valid_choice")
        and float((row.get("choice") or {}).get("confidence", 0.0)) >= minimum_confidence
    }


def select_view(
    row: dict[str, Any],
    vlm: dict[str, dict[str, Any]],
    overrides: dict[str, dict[str, Any]],
) -> tuple[str, int, str]:
    override = overrides.get(row["event_id"])
    if override:
        return str(override["camera"]), int(override["candidate_id"]), "review_override"
    resolution = vlm.get(row["event_id"])
    if resolution:
        choice = resolution["choice"]
        return str(choice["camera"]), int(choice["candidate_id"]), "vlm"
    return row["preferred_camera"], int(row["automatic_candidate_id"]), "automatic"


def selected_candidate(row: dict[str, Any], camera: str, candidate_id: int) -> dict[str, Any]:
    for view in row["views"]:
        if view["camera"] != camera:
            continue
        for candidate in view["ranked_candidates"]:
            if int(candidate["candidate_id"]) == candidate_id:
                return candidate
    raise KeyError(f"candidate {camera}/{candidate_id} missing for {row['event_id']}")


def release_frame(
    signals_root: Path,
    sequence: str,
    camera: str,
    side: str,
    anchor_frame: int,
    fallback: int,
) -> int:
    events = read_jsonl(signals_root / sequence / camera / "events.jsonl")
    releases = [
        int(event["frame_index"])
        for event in events
        if event["side"] == side
        and event["kind"] == "release_command_end"
        and int(event["frame_index"]) > anchor_frame
    ]
    return min(releases) if releases else fallback


def event_anchor_frame(row: dict[str, Any], camera: str) -> int:
    for view in row["views"]:
        if view["camera"] == camera:
            for source in ("frame_index",):
                if source in view:
                    return int(view[source])
    # The queue view omits frame_index; ranked source is indexed separately by caller.
    raise KeyError(camera)


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if not xs.size:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def quality(mask: np.ndarray, previous_area: float | None) -> dict[str, Any]:
    height, width = mask.shape
    area = float(mask.mean())
    bbox = mask_bbox(mask)
    touches_border = False
    if bbox:
        touches_border = bbox[0] <= 1 or bbox[1] <= 1 or bbox[2] >= width - 1 or bbox[3] >= height - 1
    area_ratio_change = None
    if previous_area is not None and min(previous_area, area) > 0:
        area_ratio_change = max(previous_area, area) / min(previous_area, area)
    needs_review = area < 0.0005 or area > 0.72 or (area_ratio_change is not None and area_ratio_change > 2.5)
    return {
        "area_ratio": area,
        "bbox_xyxy": bbox,
        "touches_border": touches_border,
        "area_ratio_change": area_ratio_change,
        "needs_review": needs_review,
    }


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def save_overlay(rgb_path: Path, mask: np.ndarray, bbox: list[int] | None, path: Path) -> None:
    image = Image.open(rgb_path).convert("RGB")
    color = Image.new("RGB", image.size, (255, 210, 0))
    alpha = Image.fromarray(mask.astype(np.uint8) * 100, mode="L")
    image = Image.composite(color, image, alpha)
    if bbox:
        ImageDraw.Draw(image).rectangle(tuple(bbox), outline=(0, 255, 255), width=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=90)


def propagate(
    predictor: Any,
    state: dict[str, Any],
    anchor: int,
    reverse: bool,
    frames_to_track: int,
) -> dict[int, np.ndarray]:
    masks: dict[int, np.ndarray] = {}
    iterator = predictor.propagate_in_video(
        state,
        start_frame_idx=anchor,
        max_frame_num_to_track=frames_to_track,
        reverse=reverse,
    )
    for frame_index, object_ids, logits in iterator:
        object_position = list(object_ids).index(1)
        mask = (logits[object_position] > 0.0).detach().cpu().numpy().squeeze()
        masks[int(frame_index)] = mask.astype(bool)
    return masks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, default=Path("outputs/interaction_candidates/ambiguity_queue.jsonl"))
    parser.add_argument("--ranked-index", type=Path, default=Path("outputs/interaction_candidates/ranked_index.jsonl"))
    parser.add_argument("--vlm-resolutions", type=Path)
    parser.add_argument(
        "--overrides",
        type=Path,
        default=Path("outputs/interaction_candidates/human_overrides.jsonl"),
    )
    parser.add_argument("--signals-root", type=Path, default=Path("outputs/robot_signals"))
    parser.add_argument("--extracted-root", type=Path, default=Path("outputs/extracted"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/target_tracks"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/sam2/checkpoints/sam2.1_hiera_large.pt"))
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pre-frames", type=int, default=45)
    parser.add_argument("--post-release-frames", type=int, default=20)
    parser.add_argument("--max-span", type=int, default=240)
    parser.add_argument("--review-stride", type=int, default=10)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--min-vlm-confidence", type=float, default=0.55)
    parser.add_argument("--event-id")
    parser.add_argument("--merge-existing", action="store_true")
    args = parser.parse_args()

    queue_rows = read_jsonl(args.queue)
    if args.event_id:
        queue_rows = [row for row in queue_rows if row["event_id"] == args.event_id]
    if args.max_events is not None:
        queue_rows = queue_rows[: args.max_events]
    ranked_rows = read_jsonl(args.ranked_index)
    ranked_lookup = {(row["event_id"], row["camera"]): row for row in ranked_rows}
    vlm = load_vlm(args.vlm_resolutions, args.min_vlm_confidence)
    overrides = {row["event_id"]: row for row in read_jsonl(args.overrides)}

    predictor = build_sam2_video_predictor(args.config, str(args.checkpoint), device=args.device)
    predictor.eval()
    replace_ids = {row["event_id"] for row in queue_rows}
    index_path = args.output_root / "track_index.jsonl"
    summary_path = args.output_root / "summary.json"
    if args.merge_existing and index_path.exists():
        all_rows = [row for row in read_jsonl(index_path) if row["event_id"] not in replace_ids]
    else:
        all_rows = []
    if args.merge_existing and summary_path.exists():
        old_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        event_reports = [row for row in old_summary.get("events_detail", []) if row["event_id"] not in replace_ids]
    else:
        event_reports = []
    args.output_root.mkdir(parents=True, exist_ok=True)

    for event_number, queue_row in enumerate(queue_rows):
        camera, candidate_id, selection_source = select_view(queue_row, vlm, overrides)
        ranked = ranked_lookup[(queue_row["event_id"], camera)]
        candidate = selected_candidate(queue_row, camera, candidate_id)
        anchor_global = int(ranked["frame_index"])
        manifest = read_jsonl(args.extracted_root / queue_row["sequence"] / camera / "manifest.jsonl")
        release_global = release_frame(
            args.signals_root,
            queue_row["sequence"],
            camera,
            queue_row["side"],
            anchor_global,
            min(len(manifest) - 1, anchor_global + args.max_span),
        )
        start_global = max(0, anchor_global - args.pre_frames)
        end_global = min(
            len(manifest) - 1,
            release_global + args.post_release_frames,
            anchor_global + args.max_span,
        )
        anchor_local = anchor_global - start_global
        frame_rows = manifest[start_global : end_global + 1]
        event_name = f"grasp_{queue_row['event_index']:03d}_{queue_row['side']}"
        event_root = args.output_root / queue_row["sequence"] / event_name / camera

        with tempfile.TemporaryDirectory(prefix="sam2_frames_", dir=args.output_root) as temporary:
            video_dir = Path(temporary)
            for local_index, frame in enumerate(frame_rows):
                source = Path(frame["rgb_path"])
                (video_dir / f"{local_index:06d}.jpg").symlink_to(source)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = predictor.init_state(
                    video_path=str(video_dir),
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=False,
                    async_loading_frames=False,
                )
                box = np.asarray(candidate["box_xyxy"], dtype=np.float32)
                _, object_ids, logits = predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=anchor_local,
                    obj_id=1,
                    box=box,
                )
                anchor_position = list(object_ids).index(1)
                anchor_mask = (logits[anchor_position] > 0.0).detach().cpu().numpy().squeeze().astype(bool)
                masks = {anchor_local: anchor_mask}
                masks.update(propagate(predictor, state, anchor_local, False, end_global - anchor_global + 1))
                masks.update(propagate(predictor, state, anchor_local, True, anchor_global - start_global + 1))

        previous_area: float | None = None
        review_count = 0
        for local_index in sorted(masks):
            global_index = start_global + local_index
            metrics = quality(masks[local_index], previous_area)
            previous_area = metrics["area_ratio"]
            mask_path = event_root / "mask" / f"{global_index:06d}.png"
            save_mask(mask_path, masks[local_index])
            overlay_path = None
            if (
                local_index == anchor_local
                or local_index % max(1, args.review_stride) == 0
                or metrics["needs_review"]
            ):
                overlay_path = event_root / "overlay" / f"{global_index:06d}.jpg"
                save_overlay(Path(manifest[global_index]["rgb_path"]), masks[local_index], metrics["bbox_xyxy"], overlay_path)
                review_count += 1
            all_rows.append(
                {
                    "event_id": queue_row["event_id"],
                    "sequence": queue_row["sequence"],
                    "event_index": queue_row["event_index"],
                    "side": queue_row["side"],
                    "camera": camera,
                    "frame_index": global_index,
                    "anchor_frame_index": anchor_global,
                    "release_frame_index": release_global,
                    "selection_source": selection_source,
                    "candidate_id": candidate_id,
                    "mask_path": str(mask_path),
                    "overlay_path": str(overlay_path) if overlay_path else None,
                    **metrics,
                }
            )
        report = {
            "event_id": queue_row["event_id"],
            "camera": camera,
            "candidate_id": candidate_id,
            "selection_source": selection_source,
            "anchor_frame": anchor_global,
            "release_frame": release_global,
            "start_frame": start_global,
            "end_frame": end_global,
            "tracked_frames": len(masks),
            "review_frames": review_count,
            "needs_review_frames": sum(
                1 for row in all_rows if row["event_id"] == queue_row["event_id"] and row["needs_review"]
            ),
        }
        event_reports.append(report)
        print(f"[sam2] {event_number + 1}/{len(queue_rows)} {queue_row['event_id']}: {len(masks)} frames", flush=True)

    index_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows), encoding="utf-8")
    summary = {
        "schema": "robot_target_track_v1",
        "model": "SAM2.1 Hiera Large",
        "checkpoint": str(args.checkpoint),
        "events": len(event_reports),
        "tracked_frames": len(all_rows),
        "events_detail": event_reports,
        "index": str(index_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"events": len(event_reports), "tracked_frames": len(all_rows), "index": str(index_path)}, indent=2))


if __name__ == "__main__":
    main()
