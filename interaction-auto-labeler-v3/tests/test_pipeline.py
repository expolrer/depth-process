import json

from interaction_auto_labeler_v3.event_graph import EpisodeEventGraph
from interaction_auto_labeler_v3.pipeline import build_v3_workspace


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_workspace(root) -> None:
    _write_json(
        root / "session.json",
        {
            "task": {"instruction": "Pick up the red toy and place it on the right."},
            "events": [
                {
                    "event_id": "episode_000000:grasp_000_right",
                    "episode_index": 0,
                    "side": "right",
                    "views": [
                        {
                            "camera": "head",
                            "role": "head",
                            "start_frame": 0,
                            "contact_frame": 10,
                            "release_frame": 30,
                            "end_frame": 39,
                            "frame_count": 40,
                        }
                    ],
                }
            ],
        },
    )
    evidence = {
        "event_id": "episode_000000:grasp_000_right",
        "selected_instance_id": "head:0",
        "score": 0.9,
        "needs_review": False,
        "ranked_candidates": [
            {
                "instance_id": "head:0",
                "label": "red toy",
                "evidence": {"contact": 0.95, "co_motion": 0.9, "cross_view": 0.85},
            }
        ],
    }
    evidence_path = root / "v2" / "instance_evidence.jsonl"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence) + "\n", encoding="utf-8")
    track_path = root / "outputs" / "target_tracks_required" / "track_index.jsonl"
    track_path.parent.mkdir(parents=True, exist_ok=True)
    track_path.write_text(
        "".join(
            json.dumps(
                {
                    "event_id": evidence["event_id"],
                    "camera": "head",
                    "frame_index": frame,
                    "bbox_xyxy": [10, 10, 30, 30],
                    "confidence": 0.9,
                }
            )
            + "\n"
            for frame in (5, 10, 20, 30)
        ),
        encoding="utf-8",
    )


def _task(path) -> None:
    path.write_text(
        """
name: toy_demo
instruction: Pick up the red toy and place it on the right.
target:
  names: [red toy]
detector_prompt: [red toy, robot gripper]
active_hand: right
camera_roles:
  head: [head]
  left_wrist: [left_wrist]
  right_wrist: [right_wrist]
event_detection:
  pre_frames: 10
  post_release_frames: 10
  max_span: 40
v2:
  detector_backend: grounded-sam2
v3:
  dataset_id: toy_demo
  dataset_version: v3.0.0
""".strip(),
        encoding="utf-8",
    )


def test_build_v3_workspace_produces_graph_language_and_quality(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    _write_workspace(workspace)
    task = tmp_path / "task.yaml"
    _task(task)
    summary = build_v3_workspace(workspace, task)
    assert summary["event_graphs"] == 1
    assert summary["quality"]["review"] == 1
    graph = EpisodeEventGraph.read(
        workspace / "v3" / "event_graphs" / "episode-000000.event_graph.json"
    )
    assert {item.level for item in graph.segments} == {"task", "subtask", "event"}
    assert {item.level for item in graph.language} == {"task", "subtask", "event"}
    quality = json.loads((workspace / "v3" / "quality" / "episode-000000.json").read_text())
    assert "gvl_not_run" in quality["quality_gate"]["reasons"]
