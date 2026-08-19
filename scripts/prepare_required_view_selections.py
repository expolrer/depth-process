#!/usr/bin/env python3
"""Select traceable target prompts for head and active-wrist tracking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def candidate_exists(row: dict[str, Any], candidate_id: int) -> bool:
    return any(int(item["candidate_id"]) == candidate_id for item in row.get("ranked_candidates", []))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    outputs = root / "outputs"
    summary = json.loads((outputs / "target_tracks" / "summary.json").read_text(encoding="utf-8"))
    ranked = read_jsonl(outputs / "interaction_candidates" / "ranked_index.jsonl")
    required = read_jsonl(outputs / "interaction_review" / "qwen_required_views.jsonl")
    temporal = read_jsonl(outputs / "interaction_review" / "qwen_temporal_annotations.jsonl")
    overrides = read_jsonl(outputs / "interaction_candidates" / "human_overrides.jsonl")
    required_overrides = read_jsonl(root / "config" / "required_view_overrides.jsonl")
    required_overrides += read_jsonl(root / "config" / "required_view_human_overrides.jsonl")
    required_overrides += read_jsonl(root / "config" / "required_view_human_overrides_v4.jsonl")
    ranked_by_key = {(row["event_id"], row["camera"]): row for row in ranked}
    required_by_event = {row["event_id"]: row for row in required}
    temporal_by_event = {row["event_id"]: row for row in temporal}
    override_by_event = {row["event_id"]: row for row in overrides}
    required_override_by_key = {(row["event_id"], row["camera"]): row for row in required_overrides}
    summary_by_event = {row["event_id"]: row for row in summary["events_detail"]}

    selections = []
    for event_id, existing in summary_by_event.items():
        side = "left" if event_id.endswith("_left") else "right"
        required_cameras = (("head", "cam_h"), ("wrist", "cam_l" if side == "left" else "cam_r"))
        dual = required_by_event[event_id]
        single = temporal_by_event[event_id]
        override = override_by_event.get(event_id)
        for role, camera in required_cameras:
            ranked_row = ranked_by_key[(event_id, camera)]
            candidate_id = None
            source = None
            confidence = None
            evidence = None
            needs_review = False
            required_override = required_override_by_key.get((event_id, camera))
            if required_override and candidate_exists(ranked_row, int(required_override["candidate_id"])):
                candidate_id = int(required_override["candidate_id"])
                source = "codex_dual_view_override"
                evidence = required_override.get("reason")
            elif override and override.get("camera") == camera and candidate_exists(ranked_row, int(override["candidate_id"])):
                candidate_id = int(override["candidate_id"])
                source = "codex_temporal_override"
                evidence = override.get("reason")
            elif existing["camera"] == camera and candidate_exists(ranked_row, int(existing["candidate_id"])):
                candidate_id = int(existing["candidate_id"])
                source = "existing_verified_track"
                evidence = "沿用首轮已完成连续性检查的 SAM2 初始化候选。"
            elif dual.get(f"{role}_valid"):
                item = dual["annotation"][role]
                candidate_id = int(item["candidate_id"])
                source = "qwen_dual_view"
                confidence = item.get("confidence")
                evidence = item.get("evidence")
            else:
                item = (single.get("annotation") or {})
                if single.get("valid_choice") and item.get("camera") == camera and candidate_exists(ranked_row, int(item["candidate_id"])):
                    candidate_id = int(item["candidate_id"])
                    source = "qwen_single_view"
                    confidence = item.get("confidence")
                    evidence = item.get("temporal_evidence")
                    needs_review = True
                else:
                    candidate_id = int(ranked_row["selected_candidate_id"])
                    source = "automatic_rank_fallback"
                    confidence = ranked_row.get("top_interaction_score")
                    evidence = "Qwen 未给出该视角的合法编号，暂用融合排序第一候选。"
                    needs_review = True
            candidate = next(item for item in ranked_row["ranked_candidates"] if int(item["candidate_id"]) == candidate_id)
            box = required_override.get("box_xyxy", candidate["box_xyxy"]) if required_override else candidate["box_xyxy"]
            area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) / (848 * 480)
            if area > 0.55 and source not in {"codex_dual_view_override"}:
                needs_review = True
                evidence = f"{evidence} 初始化框占画面 {100 * area:.1f}%，属于大框风险。"
            selections.append({
                "event_id": event_id,
                "sequence": event_id.split(":grasp_", 1)[0],
                "event_index": int(event_id.split(":grasp_", 1)[1].split("_", 1)[0]),
                "side": side,
                "role": role,
                "camera": camera,
                "candidate_id": candidate_id,
                "candidate_label": candidate.get("label", "target"),
                "box_xyxy": box,
                "box_area_ratio": area,
                "selection_source": source,
                "confidence": confidence,
                "evidence": evidence,
                "needs_review": needs_review,
                "secondary_prompts": required_override.get("secondary_prompts", []) if required_override else [],
                "visibility_exceptions": required_override.get("visibility_exceptions", []) if required_override else [],
                "start_frame_override": required_override.get("start_frame_override") if required_override else None,
                "end_frame_override": required_override.get("end_frame_override") if required_override else None,
            })

    output = outputs / "interaction_review" / "required_view_selections.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selections), encoding="utf-8")
    counts: dict[str, int] = {}
    for row in selections:
        counts[row["selection_source"]] = counts.get(row["selection_source"], 0) + 1
    print(json.dumps({"views": len(selections), "needs_review": sum(row["needs_review"] for row in selections), "sources": counts, "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
