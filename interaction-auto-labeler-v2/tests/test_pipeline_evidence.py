import json
from dataclasses import asdict
from pathlib import Path

from interaction_auto_labeler.pipeline import _build_v2_evidence
from interaction_auto_labeler.schema import EvidenceWeights


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _camera_row(camera: str, candidates: list[dict], selected: int) -> dict:
    return {
        "event_id": "demo:grasp_000_right",
        "camera": camera,
        "side": "right",
        "ranked_candidates": candidates,
        "selected_candidate_id": selected,
    }


def test_v2_ranks_head_instances_and_links_only_active_wrist(tmp_path: Path) -> None:
    event_id = "demo:grasp_000_right"
    (tmp_path / "session.json").write_text(
        json.dumps(
            {
                "events": [
                    {
                        "event_id": event_id,
                        "side": "right",
                        "views": [
                            {"camera": "cam_h", "role": "head"},
                            {"camera": "cam_l", "role": "left_wrist"},
                            {"camera": "cam_r", "role": "right_wrist"},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    head_candidates = [
        {
            "candidate_id": 0,
            "label": "red toy",
            "box_xyxy": [10, 10, 30, 30],
            "proximity_score": 0.95,
            "depth_score": 0.9,
            "detector_score": 0.9,
        },
        {
            "candidate_id": 1,
            "label": "yellow toy",
            "box_xyxy": [50, 10, 70, 30],
            "proximity_score": 0.2,
            "depth_score": 0.4,
            "detector_score": 0.6,
        },
    ]
    passive = [{**head_candidates[0], "candidate_id": 7, "detector_score": 1.0}]
    active = [{**head_candidates[0], "candidate_id": 4}]
    _write_jsonl(
        tmp_path / "outputs" / "interaction_candidates" / "ranked_index.jsonl",
        [
            _camera_row("cam_h", head_candidates, 0),
            _camera_row("cam_l", passive, 7),
            _camera_row("cam_r", active, 4),
        ],
    )
    _write_jsonl(
        tmp_path / "outputs" / "target_tracks_required" / "track_index.jsonl",
        [
            {
                "event_id": event_id,
                "camera": "cam_h",
                "frame_index": 0,
                "bbox_xyxy": [10, 10, 30, 30],
            },
            {
                "event_id": event_id,
                "camera": "cam_h",
                "frame_index": 5,
                "bbox_xyxy": [20, 10, 40, 30],
            },
        ],
    )
    task = {
        "v2": {
            "weights": asdict(EvidenceWeights()),
            "ambiguity_margin": 0.08,
            "cross_view_mode": "soft",
            "review": {},
        }
    }
    _build_v2_evidence(tmp_path, task, include_policy_tracks=False)
    pre_track = json.loads(
        (tmp_path / "v2" / "instance_evidence.jsonl").read_text(encoding="utf-8")
    )
    _build_v2_evidence(tmp_path, task, include_policy_tracks=True)
    evidence = json.loads((tmp_path / "v2" / "instance_evidence.jsonl").read_text(encoding="utf-8"))
    link = json.loads((tmp_path / "v2" / "cross_view_links.jsonl").read_text(encoding="utf-8"))

    assert evidence["primary_camera"] == "cam_h"
    assert evidence["active_wrist_camera"] == "cam_r"
    assert evidence["selected_instance_id"] == pre_track["selected_instance_id"]
    assert {row["instance_id"] for row in evidence["ranked_candidates"]} == {"cam_h:0", "cam_h:1"}
    assert link["source_camera"] == "cam_h"
    assert link["target_camera"] == "cam_r"
    assert link["source_instance_id"] == evidence["selected_instance_id"]
