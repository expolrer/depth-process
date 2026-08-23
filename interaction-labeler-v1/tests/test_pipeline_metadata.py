import json
from pathlib import Path

from interaction_labeler.io import read_jsonl
from interaction_labeler.pipeline import build_required_selections, merge_manual_candidates


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_manual_box_becomes_selected_tracking_prompt(tmp_path: Path) -> None:
    workspace = tmp_path
    candidates = workspace / "outputs" / "grounded_candidates" / "candidate_index.jsonl"
    base = {
        "event_id": "demo:grasp_000_right",
        "sequence": "demo",
        "event_index": 0,
        "side": "right",
        "frame_index": 10,
        "prompt": "toy .",
        "candidates": [
            {
                "candidate_id": 0,
                "label": "toy",
                "role": "object_candidate",
                "score": 0.5,
                "box_xyxy": [1, 1, 4, 4],
            }
        ],
    }
    write_jsonl(candidates, [{**base, "camera": "cam_h"}, {**base, "camera": "cam_r"}])
    (workspace / "annotations.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "event_id": "demo:grasp_000_right",
                        "camera": "cam_h",
                        "frame_index": 10,
                        "box_xyxy": [5, 6, 20, 22],
                        "label": "target toy",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (workspace / "session.json").write_text(
        json.dumps(
            {
                "events": [
                    {
                        "event_id": "demo:grasp_000_right",
                        "views": [
                            {"camera": "cam_h", "contact_frame": 10},
                            {"camera": "cam_r", "contact_frame": 10},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    merge_manual_candidates(workspace)
    merged = read_jsonl(candidates)
    assert merged[0]["candidates"][-1]["source"] == "human_preannotation"

    ranked = []
    for row in merged:
        ranked_candidates = [
            {**candidate, "interaction_score": 0.9 if candidate.get("source") else 0.5}
            for candidate in row["candidates"]
        ]
        ranked.append(
            {
                **row,
                "ranked_candidates": ranked_candidates,
                "selected_candidate_id": ranked_candidates[0]["candidate_id"],
                "needs_vlm": False,
            }
        )
    write_jsonl(workspace / "outputs" / "interaction_candidates" / "ranked_index.jsonl", ranked)
    output = build_required_selections(workspace)
    selections = read_jsonl(output)
    head = next(row for row in selections if row["role"] == "head")
    assert head["selection_source"] == "human_preannotation"
    assert head["box_xyxy"] == [5.0, 6.0, 20.0, 22.0]


def test_non_contact_manual_box_becomes_secondary_prompt(tmp_path: Path) -> None:
    workspace = tmp_path
    ranked = []
    for camera in ("cam_h", "cam_r"):
        ranked.append(
            {
                "event_id": "demo:grasp_000_right",
                "sequence": "demo",
                "event_index": 0,
                "side": "right",
                "camera": camera,
                "ranked_candidates": [
                    {
                        "candidate_id": 0,
                        "label": "toy",
                        "box_xyxy": [1, 1, 4, 4],
                        "interaction_score": 0.8,
                    }
                ],
                "selected_candidate_id": 0,
                "needs_vlm": False,
            }
        )
    write_jsonl(workspace / "outputs" / "interaction_candidates" / "ranked_index.jsonl", ranked)
    (workspace / "session.json").write_text(
        json.dumps(
            {
                "events": [
                    {
                        "event_id": "demo:grasp_000_right",
                        "views": [
                            {"camera": "cam_h", "contact_frame": 10},
                            {"camera": "cam_r", "contact_frame": 10},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (workspace / "annotations.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "event_id": "demo:grasp_000_right",
                        "camera": "cam_h",
                        "frame_index": 20,
                        "box_xyxy": [8, 8, 16, 16],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    selections = read_jsonl(build_required_selections(workspace))
    head = next(row for row in selections if row["role"] == "head")
    assert head["secondary_prompts"] == [
        {"frame_index": 20, "box_xyxy": [8.0, 8.0, 16.0, 16.0], "source": "human_post_correction"}
    ]


def test_v2_evidence_selects_head_instance_before_tracking(tmp_path: Path) -> None:
    event_id = "demo:grasp_000_right"
    candidates = [
        {"candidate_id": 0, "label": "toy", "box_xyxy": [1, 1, 4, 4], "score": 0.8},
        {"candidate_id": 1, "label": "toy", "box_xyxy": [8, 8, 14, 14], "score": 0.7},
    ]
    ranked = [
        {
            "event_id": event_id,
            "sequence": "demo",
            "event_index": 0,
            "side": "right",
            "camera": camera,
            "ranked_candidates": candidates,
            "selected_candidate_id": 0,
            "needs_vlm": True,
        }
        for camera in ("cam_h", "cam_r")
    ]
    write_jsonl(tmp_path / "outputs" / "interaction_candidates" / "ranked_index.jsonl", ranked)
    write_jsonl(
        tmp_path / "v2" / "instance_evidence.jsonl",
        [
            {
                "event_id": event_id,
                "selected_instance_id": "cam_h:1",
                "score": 0.91,
                "margin": 0.22,
            }
        ],
    )
    (tmp_path / "session.json").write_text(
        json.dumps(
            {
                "events": [
                    {
                        "event_id": event_id,
                        "views": [
                            {"camera": "cam_h", "contact_frame": 10},
                            {"camera": "cam_r", "contact_frame": 10},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    selections = read_jsonl(build_required_selections(tmp_path))
    head = next(row for row in selections if row["role"] == "head")
    assert head["candidate_id"] == 1
    assert head["selection_source"] == "v2_interaction_evidence"
