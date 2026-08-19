#!/usr/bin/env python3
"""Build a traceable Codex audit for head and active-wrist target tracks."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def view_audit(selection: dict[str, Any], rows: list[dict[str, Any]], qwen: dict[str, Any]) -> dict[str, Any]:
    nonempty = [row for row in rows if float(row.get("area_ratio", 0)) > 0]
    failures = [row for row in rows if row.get("visibility_status") == "tracking_failure"]
    unobservable = [row for row in rows if row.get("visibility_status") == "unobservable"]
    post_release = [row for row in rows if row.get("visibility_status") == "post_release_out_of_view"]
    flagged = [row for row in rows if row.get("needs_review")]
    qwen_role = (qwen.get("annotation") or {}).get(selection["role"], {})
    qwen_match = bool(
        qwen_role
        and qwen_role.get("camera") == selection["camera"]
        and int(qwen_role.get("candidate_id", -1)) == int(selection["candidate_id"])
    )
    status = "通过"
    if failures:
        status = "需要修正"
    elif selection.get("needs_review") or flagged:
        status = "建议人工抽检"
    return {
        "role": selection["role"],
        "camera": selection["camera"],
        "candidate_id": selection["candidate_id"],
        "candidate_label": selection.get("candidate_label"),
        "selection_source": selection["selection_source"],
        "selection_evidence": selection.get("evidence"),
        "qwen_candidate_id": qwen_role.get("candidate_id"),
        "qwen_matches_final": qwen_match,
        "tracked_frames": len(rows),
        "visible_box_frames": len(nonempty),
        "tracking_failure_frames": len(failures),
        "unobservable_frames": len(unobservable),
        "post_release_out_of_view_frames": len(post_release),
        "quality_flagged_frames": len(flagged),
        "visibility_exceptions": selection.get("visibility_exceptions", []),
        "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    outputs = args.root.resolve() / "outputs"
    selections = read_jsonl(outputs / "interaction_review" / "required_view_selections.jsonl")
    tracks = read_jsonl(outputs / "target_tracks_required" / "track_index.jsonl")
    qwen_rows = read_jsonl(outputs / "interaction_review" / "qwen_required_views.jsonl")
    qwen_by_event = {row["event_id"]: row for row in qwen_rows}
    tracks_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in tracks:
        tracks_by_key[(row["event_id"], row["role"])].append(row)

    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for selection in selections:
        by_event[selection["event_id"]].append(selection)

    audits = []
    for event_id, event_selections in by_event.items():
        qwen = qwen_by_event.get(event_id, {})
        views = [
            view_audit(selection, tracks_by_key[(event_id, selection["role"])], qwen)
            for selection in sorted(event_selections, key=lambda row: (row["role"] != "head", row["role"]))
        ]
        failures = sum(view["tracking_failure_frames"] for view in views)
        unobservable = sum(view["unobservable_frames"] for view in views)
        review_views = sum(view["status"] != "通过" for view in views)
        if failures:
            recommendation = f"发现 {failures} 帧可见目标漏跟踪，需要修正后再批准。"
        elif review_views:
            recommendation = "两路必要视角已生成标注；存在自动候选或面积变化告警，建议人工抽检黄色框是否始终覆盖同一目标实例。"
        else:
            recommendation = "头部与执行腕视角的可见目标均保持连续黄色标注，可以批准。"
        if unobservable:
            recommendation += f" 另有 {unobservable} 帧因高速模糊、遮挡或离开视野而不画框，已单独标明，不计为漏跟踪。"
        audits.append({
            "event_id": event_id,
            "reviewer": "Codex 双视角时序视觉复核",
            "views": views,
            "same_instance_check": "仅比较头部相机与执行手腕相机；未执行动作的另一腕部相机不进入最终标注。",
            "recommendation": recommendation,
            "status": "needs_correction" if failures else ("manual_review" if review_views else "approved"),
        })

    output = outputs / "interaction_review" / "codex_required_view_audits.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audits), encoding="utf-8")
    print(json.dumps({"events": len(audits), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
