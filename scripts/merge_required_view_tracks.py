#!/usr/bin/env python3
"""Merge and validate head/active-wrist SAM2 tracking outputs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/target_tracks_required"))
    parser.add_argument("--selections", type=Path, default=Path("outputs/interaction_review/required_view_selections.jsonl"))
    args = parser.parse_args()
    rows = read_jsonl(args.root / "head" / "track_index.jsonl") + read_jsonl(args.root / "wrist" / "track_index.jsonl")
    selections = {(row["event_id"], row["role"]): row for row in read_jsonl(args.selections)}
    details = []
    for role in ("head", "wrist"):
        summary = json.loads((args.root / role / "summary.json").read_text(encoding="utf-8"))
        details.extend(summary["views_detail"])
    missing = unreadable = empty = critical_empty = unobservable = 0
    by_event: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_event[row["event_id"]].add(row["role"])
        row["visibility_exceptions"] = selections.get((row["event_id"], row["role"]), {}).get("visibility_exceptions", [])
        path = Path(row["mask_path"])
        if not path.exists():
            missing += 1
            continue
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            unreadable += 1
        elif not image.any():
            empty += 1
            frame_index = int(row["frame_index"])
            visibility_exception = next(
                (
                    item
                    for item in row.get("visibility_exceptions", [])
                    if int(item["start_frame"]) <= frame_index <= int(item["end_frame"])
                ),
                None,
            )
            if visibility_exception:
                row["visibility_status"] = "unobservable"
                row["visibility_reason"] = visibility_exception["reason"]
                unobservable += 1
            elif frame_index <= int(row["release_frame_index"]):
                row["visibility_status"] = "tracking_failure"
                critical_empty += 1
            else:
                row["visibility_status"] = "post_release_out_of_view"
    bad_role_sets = {event_id: sorted(roles) for event_id, roles in by_event.items() if roles != {"head", "wrist"}}
    report = {
        "schema": "required_dual_view_tracks_v1",
        "passed": not (missing or unreadable or critical_empty or bad_role_sets or len(details) != 38),
        "events": len(by_event),
        "views": len(details),
        "tracked_frames": len(rows),
        "missing_masks": missing,
        "unreadable_masks": unreadable,
        "empty_masks": empty,
        "empty_masks_through_release": critical_empty,
        "unobservable_frames": unobservable,
        "empty_masks_after_release": empty - critical_empty - unobservable,
        "quality_flagged_frames": sum(bool(row["needs_review"]) for row in rows),
        "views_needing_review": sum(bool(row["needs_review"] or row["quality_flagged_frames"]) for row in details),
        "bad_role_sets": bad_role_sets,
        "views_detail": details,
    }
    (args.root / "track_index.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (args.root / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "views_detail"}, ensure_ascii=False))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
