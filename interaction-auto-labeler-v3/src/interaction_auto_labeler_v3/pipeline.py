from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .anomaly import detect_anomalies, quality_gate
from .event_graph import EpisodeEventGraph, FrameSpan, Segment, migrate_v2_workspace
from .geometry_pipeline import process_episode_geometry
from .gvl import build_progress_probe, evaluate_progress_response
from .io import atomic_write_json, read_jsonl
from .language import generate_grounded_language
from .reliability import aggregate_reliability, load_calibrator
from .schema import load_v3_task
from .temporal import load_modalities_npz, propose_hierarchical_boundaries, segments_from_boundaries


def _read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _config_path(value: str | None, base: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _set_status(workspace: Path, stage: str, state: str, **details: Any) -> None:
    atomic_write_json(
        workspace / "pipeline_status.json",
        {
            "stage": stage,
            "state": state,
            "pipeline_version": "v3",
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **details,
        },
    )


def initialize_v3(workspace: Path, task_path: Path) -> dict[str, Any]:
    task = load_v3_task(task_path)
    try:
        from interaction_auto_labeler.pipeline import initialize_v2
    except ImportError as error:
        raise RuntimeError("V3 run mode requires the sibling V2 package") from error
    session = _read_json(workspace / "session.json", {})
    initialize_v2(workspace, {**session.get("task", {}), **task})
    session = _read_json(workspace / "session.json", {})
    session["pipeline_version"] = "v3"
    session["v3"] = task["v3"]
    session["task"] = {**session.get("task", {}), **task}
    atomic_write_json(workspace / "session.json", session)
    atomic_write_json(workspace / "v3" / "task_descriptor.json", task)
    atomic_write_json(
        workspace / "v3" / "summary.json",
        {
            "schema": "robot_interaction_auto_label_summary_v3",
            "state": "ready",
            "event_graphs": 0,
            "quality": {"accept": 0, "review": 0, "reject": 0},
        },
    )
    return task


def _fallback_hierarchy(graph: EpisodeEventGraph) -> list[Segment]:
    task = Segment("task:0000", "task", FrameSpan(0, graph.frame_count - 1), "robot task")
    segments: list[Segment] = [task]
    for index, event in enumerate(sorted(graph.events, key=lambda item: item.span.start_frame)):
        subtask_id = f"subtask:{index:04d}"
        segments.append(
            Segment(
                subtask_id,
                "subtask",
                event.span,
                event.event_type.replace("_", " "),
                task.segment_id,
                event.confidence,
            )
        )
        segments.append(
            Segment(
                f"event-segment:{index:04d}",
                "event",
                event.span,
                event.event_type.replace("_", " "),
                subtask_id,
                event.confidence,
            )
        )
    return segments


def _feature_path(root: str | None, graph: EpisodeEventGraph) -> Path | None:
    if not root:
        return None
    base = Path(root)
    candidates = [
        base / f"episode-{int(graph.episode_id):06d}.npz",
        base / f"{graph.episode_id}.npz",
    ]
    return next((path for path in candidates if path.exists()), None)


def _reliability_features(
    graph: EpisodeEventGraph, geometry_report: dict[str, Any] | None = None
) -> dict[str, float | None]:
    evidence = [item for event in graph.events for item in event.evidence]
    by_kind: dict[str, list[float]] = {}
    for item in evidence:
        by_kind.setdefault(item.kind, []).append(item.score)

    def mean(*names: str) -> float | None:
        values = [value for name in names for value in by_kind.get(name, [])]
        return sum(values) / len(values) if values else None

    observation_confidence = [item.confidence for item in graph.observations if item.visible]
    return {
        "detection": mean("descriptor_match", "detection"),
        "track_stability": (
            sum(observation_confidence) / len(observation_confidence)
            if observation_confidence
            else None
        ),
        "contact": mean("contact"),
        "co_motion": mean("co_motion"),
        "depth_consistency": mean("depth_consistency"),
        "cross_view": (
            geometry_report.get("mean_cross_view_iou")
            if geometry_report and geometry_report.get("mean_cross_view_iou") is not None
            else mean("cross_view")
        ),
        "boundary": sum(event.confidence for event in graph.events) / len(graph.events)
        if graph.events
        else None,
        "language_grounding": 1.0
        if any(item.review_state == "approved" for item in graph.language)
        else None,
    }


def build_v3_workspace(workspace: Path, task_path: Path) -> dict[str, Any]:
    task = load_v3_task(task_path)
    config = task["v3"]
    config_base = task_path.resolve().parent
    graph_root = workspace / "v3" / "event_graphs"
    paths = migrate_v2_workspace(
        workspace,
        graph_root,
        str(config["dataset_id"]),
        str(config["dataset_version"]),
    )
    calibrator_path = _config_path(config["reliability"].get("calibrator"), config_base)
    calibrator = load_calibrator(calibrator_path) if calibrator_path else None
    geometry_config = config["geometry"]
    geometry_manifest_path = _config_path(
        geometry_config.get("manifest") if geometry_config["mode"] == "calibrated" else None,
        config_base,
    )
    geometry_rows = read_jsonl(geometry_manifest_path) if geometry_manifest_path else []
    geometry_episodes = 0
    counts = {"accept": 0, "review": 0, "reject": 0}
    for path in paths:
        graph = EpisodeEventGraph.read(path)
        modality_root = _config_path(config["temporal"].get("modalities_npz_root"), config_base)
        modality_path = _feature_path(str(modality_root) if modality_root else None, graph)
        loaded_modalities = load_modalities_npz(str(modality_path)) if modality_path else None
        timestamps = loaded_modalities.get("timestamps") if loaded_modalities else None
        modalities = (
            {name: value for name, value in loaded_modalities.items() if name != "timestamps"}
            if loaded_modalities
            else None
        )
        if modalities and config["temporal"].get("enabled", True):
            proposals, _ = propose_hierarchical_boundaries(modalities)
            graph.segments = segments_from_boundaries(graph.frame_count, proposals)
        else:
            graph.segments = _fallback_hierarchy(graph)
        graph = generate_grounded_language(graph)

        episode_geometry_rows = [
            row for row in geometry_rows if str(row.get("episode_id")) == graph.episode_id
        ]
        geometry_report: dict[str, Any] = {"state": "disabled"}
        if geometry_config["mode"] == "calibrated":
            if episode_geometry_rows and geometry_manifest_path:
                graph, geometry_report = process_episode_geometry(
                    graph,
                    episode_geometry_rows,
                    geometry_manifest_path.parent,
                    workspace / "v3" / "geometry" / "point_clouds",
                    bool(geometry_config.get("export_point_clouds", False)),
                )
                geometry_report["state"] = "completed"
                geometry_episodes += 1
                atomic_write_json(
                    workspace / "v3" / "geometry" / f"episode-{int(graph.episode_id):06d}.json",
                    geometry_report,
                )
            else:
                geometry_report = {
                    "state": "not_run",
                    "reason": "no calibrated geometry manifest rows for this episode",
                }

        reliability = aggregate_reliability(_reliability_features(graph, geometry_report))
        calibrated = (
            calibrator.predict_one(float(reliability["raw_score"]))
            if calibrator
            else float(reliability["raw_score"])
        )
        response_root = _config_path(config["gvl"].get("responses_root"), config_base)
        response_path = (
            response_root / f"episode-{int(graph.episode_id):06d}.json" if response_root else None
        )
        if response_path and response_path.exists():
            probe = build_progress_probe(
                graph.episode_id,
                graph.frame_count,
                int(config["gvl"].get("sample_count", 12)),
                required_entity_ids=[item.entity_id for item in graph.entities],
            )
            progress = evaluate_progress_response(probe, _read_json(response_path, {}))
            progress_score = float(progress["score"])
        else:
            progress = {"state": "not_run", "score": None, "needs_review": True}
            progress_score = 0.0
        anomalies = (
            detect_anomalies(modalities or {}, timestamps)
            if modalities or timestamps is not None
            else []
        )
        gate = quality_gate(
            calibrated,
            float(reliability["evidence_coverage"]),
            progress_score,
            anomalies,
            float(config["reliability"].get("accept_threshold", 0.8)),
            float(config["quality"].get("minimum_evidence_coverage", 0.5)),
            bool(config["quality"].get("reject_non_finite", True)),
        )
        if progress.get("state") == "not_run":
            gate["decision"] = "review"
            gate["reasons"] = list(dict.fromkeys([*gate["reasons"], "gvl_not_run"]))
        if calibrator is None:
            gate["decision"] = "review"
            gate["reasons"] = list(dict.fromkeys([*gate["reasons"], "reliability_uncalibrated"]))
        if loaded_modalities is None:
            gate["decision"] = "review"
            gate["reasons"] = list(
                dict.fromkeys([*gate["reasons"], "multimodal_anomaly_check_not_run"])
            )
        if geometry_config["mode"] == "calibrated" and geometry_report["state"] != "completed":
            gate["decision"] = "review"
            gate["reasons"] = list(dict.fromkeys([*gate["reasons"], "calibrated_geometry_not_run"]))
        counts[gate["decision"]] += 1
        report = {
            "schema": "episode_annotation_quality_v3",
            "episode_id": graph.episode_id,
            "event_graph": str(path.resolve()),
            "reliability": {
                **reliability,
                "calibrated_score": calibrated,
                "calibration_state": "gold_calibrated" if calibrator else "uncalibrated",
            },
            "gvl_progress": progress,
            "anomaly_state": (
                "evaluated" if loaded_modalities is not None else "modalities_unavailable"
            ),
            "geometry": geometry_report,
            "quality_gate": gate,
        }
        graph.lineage["v3_quality_report"] = f"../quality/episode-{int(graph.episode_id):06d}.json"
        graph.write(path)
        atomic_write_json(
            workspace / "v3" / "quality" / f"episode-{int(graph.episode_id):06d}.json",
            report,
        )
    summary = {
        "schema": "robot_interaction_auto_label_summary_v3",
        "state": "completed",
        "dataset_id": config["dataset_id"],
        "dataset_version": config["dataset_version"],
        "event_graphs": len(paths),
        "quality": counts,
        "gold_calibration_active": calibrator is not None,
        "geometry_mode": config["geometry"]["mode"],
        "geometry_episodes": geometry_episodes,
        "limitations": [
            "Episodes without GVL responses remain in review.",
            "Episodes without modality NPZ files do not receive anomaly or learned-boundary evidence.",
            "Calibrated 3D outputs require per-frame FK and camera calibration inputs.",
        ],
    }
    atomic_write_json(workspace / "v3" / "summary.json", summary)
    return summary


def run_v3_pipeline(workspace: Path, v3_task_path: Path, **v2_options: Any) -> None:
    try:
        from interaction_auto_labeler.pipeline import run_v2_pipeline
    except ImportError as error:
        raise RuntimeError("V3 run mode requires the sibling V2 package") from error
    try:
        _set_status(workspace, "v3_v2_interaction_labeling", "running")
        run_v2_pipeline(workspace, v2_task_path=v3_task_path, **v2_options)
        _set_status(workspace, "v3_event_graph_and_quality", "running")
        summary = build_v3_workspace(workspace, v3_task_path)
        _set_status(workspace, "complete", "completed", summary=summary)
    except Exception as error:
        _set_status(workspace, "v3_pipeline", "failed", error=str(error))
        raise
