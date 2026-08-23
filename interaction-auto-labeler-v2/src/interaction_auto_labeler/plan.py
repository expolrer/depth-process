from __future__ import annotations

from pathlib import Path
from typing import Any

from .inventory import inventory_schedule


def build_plan(session: dict[str, Any], task: dict[str, Any], workspace: Path) -> dict[str, Any]:
    v2 = task["v2"]
    inventory = []
    for event in session["events"]:
        head = next((view for view in event["views"] if view["role"] == "head"), None)
        if head:
            inventory.append(
                {
                    "event_id": event["event_id"],
                    "camera": head["camera"],
                    "frames": inventory_schedule(head, int(v2["inventory_stride"])),
                }
            )
    return {
        "schema": "robot_interaction_auto_label_plan_v2",
        "workspace": str(workspace.resolve()),
        "backend": v2["detector_backend"],
        "cross_view_mode": v2["cross_view_mode"],
        "target": task["target"],
        "inventory_schedule": inventory,
        "stages": [
            {"id": "target_descriptor", "output": "v2/task_descriptor.json"},
            {"id": "head_scene_inventory", "output": "v2/head_inventory.jsonl"},
            {"id": "contact_phase_detection", "output": "v2/action_phases.jsonl"},
            {
                "id": "open_vocab_candidates",
                "output": "outputs/grounded_candidates/candidate_index.jsonl",
            },
            {"id": "interaction_evidence_scoring", "output": "v2/instance_evidence.jsonl"},
            {"id": "single_click_correction", "output": "annotations.json"},
            {
                "id": "bidirectional_tracking",
                "output": "outputs/target_tracks_required/track_index.jsonl",
            },
            {"id": "cross_view_association", "output": "v2/cross_view_links.jsonl"},
            {"id": "risk_only_review", "output": "v2/review_queue.jsonl"},
        ],
    }
