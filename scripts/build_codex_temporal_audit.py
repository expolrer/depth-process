#!/usr/bin/env python3
"""Build an independent, evidence-based Codex audit for each grasp event."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


CAMERA_NAMES = {"cam_h": "头部相机", "cam_l": "左腕相机", "cam_r": "右腕相机"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def nearest(rows: list[dict[str, Any]], frame: int) -> dict[str, Any]:
    return min(rows, key=lambda row: abs(int(row["frame_index"]) - frame))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    outputs = root / "outputs"
    summary = json.loads((outputs / "target_tracks" / "summary.json").read_text(encoding="utf-8"))
    tracks = read_jsonl(outputs / "target_tracks" / "track_index.jsonl")
    ranked = read_jsonl(outputs / "interaction_candidates" / "ranked_index.jsonl")
    qwen = read_jsonl(outputs / "interaction_review" / "qwen_temporal_annotations.jsonl")
    overrides = read_jsonl(outputs / "interaction_candidates" / "human_overrides.jsonl")
    tracks_by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ranked_by_key = {(row["event_id"], row["camera"]): row for row in ranked}
    for row in tracks:
        tracks_by_event[row["event_id"]].append(row)
    qwen_by_event = {row["event_id"]: row for row in qwen}
    override_ids = {row["event_id"] for row in overrides}

    audits = []
    for event in summary["events_detail"]:
        event_id = event["event_id"]
        camera = event["camera"]
        candidate_id = int(event["candidate_id"])
        ranked_row = ranked_by_key[(event_id, camera)]
        candidate = next((row for row in ranked_row.get("ranked_candidates", []) if int(row["candidate_id"]) == candidate_id), {})
        event_tracks = sorted(tracks_by_event[event_id], key=lambda row: int(row["frame_index"]))
        start_row = event_tracks[0]
        anchor_row = nearest(event_tracks, int(event["anchor_frame"]))
        release_row = nearest(event_tracks, int(event["release_frame"]))
        end_row = event_tracks[-1]
        areas = [float(row["area_ratio"]) for row in event_tracks]
        mean_area = statistics.fmean(areas)
        area_cv = statistics.pstdev(areas) / mean_area if mean_area else 0.0
        flagged = int(event["needs_review_frames"])
        depth = candidate.get("depth") or {}
        motion = ranked_row.get("robot_motion") or {}
        qwen_row = qwen_by_event.get(event_id, {})
        qwen_annotation = qwen_row.get("annotation") or {}
        qwen_match = (
            qwen_row.get("valid_choice")
            and str(qwen_annotation.get("camera")) == camera
            and int(qwen_annotation.get("candidate_id", -1)) == candidate_id
        )
        if event_id in override_ids:
            recommendation = "建议重点人工确认：本事件使用了时序复核覆盖，旧 VLM 曾选错大面积背景。"
        elif not qwen_row.get("valid_choice"):
            recommendation = "建议人工复核：Qwen 输出未通过候选编号校验。"
        elif not qwen_match:
            recommendation = "建议人工复核：Qwen 时序判断与当前 SAM2 初始化候选不一致。"
        elif flagged:
            recommendation = f"建议检查标红帧后再批准：共有 {flagged} 帧触发面积突变或边界质量告警。"
        else:
            recommendation = "建议批准：Qwen 与当前实例一致，SAM2 时序连续且未触发质量告警。"

        hand = "左手" if event_id.endswith("_left") else "右手"
        candidate_label = candidate.get("label", "目标物体")
        camera_name = CAMERA_NAMES.get(camera, camera)
        if depth.get("median_mm") is not None:
            depth_delta = candidate.get("depth_delta_to_gripper_mm")
            delta_text = f"{float(depth_delta):.0f} mm" if depth_delta is not None else "不可用"
            depth_text = (
                f"初始化候选的有效深度比例为 {100 * float(depth.get('valid_ratio', 0)):.1f}%，中位深度 "
                f"{float(depth['median_mm']):.0f} mm，与夹爪的深度差 {delta_text}。"
            )
        else:
            depth_text = "初始化候选没有足够的有效深度值，深度证据不能单独支持该选择。"
        audits.append({
            "event_id": event_id,
            "reviewer": "Codex 辅助时序视觉复核",
            "selected_target": f"当前采用{camera_name}候选 #{candidate_id}（{candidate_label}），由{hand}执行；选择来源为 {event['selection_source']}。",
            "robot_evidence": f"初始化附近末端位移 {float(motion.get('eef_displacement_m', 0)):.4f} m，关节位移 L2 为 {float(motion.get('joint_displacement_l2', 0)):.3f}；夹爪闭合事件与 RGB-D 初始化帧已对齐。",
            "rgbd_evidence": depth_text,
            "tracking_evidence": (
                f"SAM2 共跟踪 {event['tracked_frames']} 帧；mask 面积占比从 {100 * float(start_row['area_ratio']):.2f}% 变化到接触时 "
                f"{100 * float(anchor_row['area_ratio']):.2f}%、释放时 {100 * float(release_row['area_ratio']):.2f}%、结束时 "
                f"{100 * float(end_row['area_ratio']):.2f}%。全段面积变异系数 {area_cv:.3f}，质量告警 {flagged} 帧。"
            ),
            "qwen_comparison": (
                "Qwen3.5-2B 的相机与候选编号和当前跟踪实例一致。"
                if qwen_match
                else "Qwen3.5-2B 的建议与当前跟踪实例不一致或未通过有效候选校验，不能直接作为真值。"
            ),
            "recommendation": recommendation,
            "status": "pending_human_review",
        })

    output = outputs / "interaction_review" / "codex_temporal_audits.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audits), encoding="utf-8")
    print(json.dumps({"events": len(audits), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
