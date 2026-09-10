from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "embodied_event_graph_v1"
SEGMENT_LEVELS = {"task", "subtask", "event", "phase"}
REVIEW_STATES = {"auto", "review", "approved", "rejected"}


def _score(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


@dataclass(frozen=True)
class FrameSpan:
    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        if self.start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        if self.end_frame < self.start_frame:
            raise ValueError("end_frame must be greater than or equal to start_frame")

    def contains(self, other: FrameSpan) -> bool:
        return self.start_frame <= other.start_frame and self.end_frame >= other.end_frame


@dataclass(frozen=True)
class Evidence:
    kind: str
    source: str
    score: float
    frame_index: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind or not self.source:
            raise ValueError("evidence kind and source are required")
        _score(self.score, "evidence score")
        if self.frame_index is not None and self.frame_index < 0:
            raise ValueError("evidence frame_index must be non-negative")


@dataclass(frozen=True)
class Entity:
    entity_id: str
    category: str
    names: tuple[str, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.entity_id or not self.category:
            raise ValueError("entity_id and category are required")


@dataclass(frozen=True)
class TrackObservation:
    observation_id: str
    entity_id: str
    camera: str
    frame_index: int
    bbox_xyxy: tuple[float, float, float, float] | None = None
    mask_path: str | None = None
    centroid_camera_m: tuple[float, float, float] | None = None
    centroid_world_m: tuple[float, float, float] | None = None
    pose_world: tuple[tuple[float, float, float, float], ...] | None = None
    visible: bool = True
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.observation_id or not self.entity_id or not self.camera:
            raise ValueError("observation id, entity id, and camera are required")
        if self.frame_index < 0:
            raise ValueError("observation frame_index must be non-negative")
        _score(self.confidence, "observation confidence")
        if self.bbox_xyxy is not None:
            x1, y1, x2, y2 = self.bbox_xyxy
            if not all(math.isfinite(float(value)) for value in self.bbox_xyxy):
                raise ValueError("bbox coordinates must be finite")
            if x2 <= x1 or y2 <= y1:
                raise ValueError("bbox must have positive area")
        if self.pose_world is not None and (
            len(self.pose_world) != 4 or any(len(row) != 4 for row in self.pose_world)
        ):
            raise ValueError("pose_world must be a 4x4 matrix")


@dataclass(frozen=True)
class Segment:
    segment_id: str
    level: str
    span: FrameSpan
    label: str
    parent_segment_id: str | None = None
    confidence: float = 1.0
    boundary_uncertainty_frames: int = 0

    def __post_init__(self) -> None:
        if self.level not in SEGMENT_LEVELS:
            raise ValueError(f"unsupported segment level: {self.level}")
        if not self.segment_id or not self.label:
            raise ValueError("segment id and label are required")
        _score(self.confidence, "segment confidence")
        if self.boundary_uncertainty_frames < 0:
            raise ValueError("boundary uncertainty must be non-negative")


@dataclass(frozen=True)
class InteractionEvent:
    event_id: str
    event_type: str
    span: FrameSpan
    active_arm: str = "unknown"
    target_entity_id: str | None = None
    source_entity_id: str | None = None
    destination_entity_id: str | None = None
    phase: str | None = None
    confidence: float = 1.0
    review_state: str = "auto"
    evidence: tuple[Evidence, ...] = ()
    observation_ids: tuple[str, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id or not self.event_type:
            raise ValueError("event id and type are required")
        _score(self.confidence, "event confidence")
        if self.review_state not in REVIEW_STATES:
            raise ValueError(f"unsupported review state: {self.review_state}")


@dataclass(frozen=True)
class LanguageAnnotation:
    language_id: str
    level: str
    text: str
    span: FrameSpan
    event_ids: tuple[str, ...] = ()
    entity_ids: tuple[str, ...] = ()
    source: str = "template"
    confidence: float = 1.0
    review_state: str = "auto"

    def __post_init__(self) -> None:
        if self.level not in {"task", "subtask", "event"}:
            raise ValueError(f"unsupported language level: {self.level}")
        if not self.language_id or not self.text.strip():
            raise ValueError("language id and text are required")
        _score(self.confidence, "language confidence")
        if self.review_state not in REVIEW_STATES:
            raise ValueError(f"unsupported review state: {self.review_state}")


@dataclass
class EpisodeEventGraph:
    dataset_id: str
    dataset_version: str
    episode_id: str
    frame_count: int
    fps: float
    modalities: dict[str, Any] = field(default_factory=dict)
    entities: list[Entity] = field(default_factory=list)
    observations: list[TrackObservation] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    events: list[InteractionEvent] = field(default_factory=list)
    language: list[LanguageAnnotation] = field(default_factory=list)
    lineage: dict[str, Any] = field(default_factory=dict)
    schema: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema != SCHEMA_VERSION:
            raise ValueError(f"unsupported event graph schema: {self.schema}")
        if not self.dataset_id or not self.dataset_version or not self.episode_id:
            raise ValueError("dataset id, dataset version, and episode id are required")
        if self.frame_count <= 0 or self.fps <= 0:
            raise ValueError("frame_count and fps must be positive")

        entity_ids = _unique((item.entity_id for item in self.entities), "entity")
        observation_ids = _unique(
            (item.observation_id for item in self.observations), "observation"
        )
        _unique((item.segment_id for item in self.segments), "segment")
        event_ids = _unique((item.event_id for item in self.events), "event")
        _unique((item.language_id for item in self.language), "language")

        episode_span = FrameSpan(0, self.frame_count - 1)
        segments = {item.segment_id: item for item in self.segments}
        for item in self.observations:
            if item.entity_id not in entity_ids:
                raise ValueError(f"observation references unknown entity: {item.entity_id}")
            if item.frame_index >= self.frame_count:
                raise ValueError(f"observation frame outside episode: {item.observation_id}")
        for item in self.segments:
            if not episode_span.contains(item.span):
                raise ValueError(f"segment outside episode: {item.segment_id}")
            if item.parent_segment_id:
                parent = segments.get(item.parent_segment_id)
                if parent is None:
                    raise ValueError(f"segment references unknown parent: {item.segment_id}")
                if not parent.span.contains(item.span):
                    raise ValueError(f"parent does not contain segment: {item.segment_id}")
        for item in self.events:
            if not episode_span.contains(item.span):
                raise ValueError(f"event outside episode: {item.event_id}")
            for entity_id in (
                item.target_entity_id,
                item.source_entity_id,
                item.destination_entity_id,
            ):
                if entity_id and entity_id not in entity_ids:
                    raise ValueError(f"event references unknown entity: {entity_id}")
            unknown_observations = set(item.observation_ids) - observation_ids
            if unknown_observations:
                raise ValueError(
                    f"event references unknown observations: {sorted(unknown_observations)}"
                )
        for item in self.language:
            if not episode_span.contains(item.span):
                raise ValueError(f"language span outside episode: {item.language_id}")
            if set(item.event_ids) - event_ids:
                raise ValueError(f"language references unknown event: {item.language_id}")
            if set(item.entity_ids) - entity_ids:
                raise ValueError(f"language references unknown entity: {item.language_id}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def write(self, path: Path) -> None:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EpisodeEventGraph:
        graph = cls(
            schema=payload.get("schema", SCHEMA_VERSION),
            dataset_id=str(payload["dataset_id"]),
            dataset_version=str(payload["dataset_version"]),
            episode_id=str(payload["episode_id"]),
            frame_count=int(payload["frame_count"]),
            fps=float(payload["fps"]),
            modalities=dict(payload.get("modalities", {})),
            entities=[
                Entity(**_tuple_fields(row, {"names"})) for row in payload.get("entities", [])
            ],
            observations=[
                TrackObservation(
                    **_tuple_fields(
                        row,
                        {"bbox_xyxy", "centroid_camera_m", "centroid_world_m", "pose_world"},
                    )
                )
                for row in payload.get("observations", [])
            ],
            segments=[
                Segment(
                    **{
                        **row,
                        "span": FrameSpan(**row["span"]),
                    }
                )
                for row in payload.get("segments", [])
            ],
            events=[
                InteractionEvent(
                    **{
                        **_tuple_fields(row, {"observation_ids"}),
                        "span": FrameSpan(**row["span"]),
                        "evidence": tuple(Evidence(**item) for item in row.get("evidence", [])),
                    }
                )
                for row in payload.get("events", [])
            ],
            language=[
                LanguageAnnotation(
                    **{
                        **_tuple_fields(row, {"event_ids", "entity_ids"}),
                        "span": FrameSpan(**row["span"]),
                    }
                )
                for row in payload.get("language", [])
            ],
            lineage=dict(payload.get("lineage", {})),
        )
        graph.validate()
        return graph

    @classmethod
    def read(cls, path: Path) -> EpisodeEventGraph:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _unique(values: Any, kind: str) -> set[str]:
    result: set[str] = set()
    for value in values:
        if value in result:
            raise ValueError(f"duplicate {kind} id: {value}")
        result.add(value)
    return result


def _tuple_fields(row: dict[str, Any], names: set[str]) -> dict[str, Any]:
    output = dict(row)
    for name in names:
        value = output.get(name)
        if value is not None:
            output[name] = tuple(tuple(item) if isinstance(item, list) else item for item in value)
    return output


def migrate_v2_workspace(
    workspace: Path,
    output: Path,
    dataset_id: str,
    dataset_version: str,
) -> list[Path]:
    """Convert reviewed V2 workspaces into one event graph per episode."""
    session = json.loads((workspace / "session.json").read_text(encoding="utf-8"))
    evidence = _read_jsonl(workspace / "v2" / "instance_evidence.jsonl")
    tracks = _read_jsonl(workspace / "outputs" / "target_tracks_required" / "track_index.jsonl")
    evidence_by_event = {row["event_id"]: row for row in evidence}
    tracks_by_event: dict[str, list[dict[str, Any]]] = {}
    for row in tracks:
        tracks_by_event.setdefault(str(row["event_id"]), []).append(row)
    events_by_episode: dict[int, list[dict[str, Any]]] = {}
    for event in session.get("events", []):
        events_by_episode.setdefault(int(event.get("episode_index", 0)), []).append(event)

    written: list[Path] = []
    output.mkdir(parents=True, exist_ok=True)
    for episode_index, source_events in sorted(events_by_episode.items()):
        entities: dict[str, Entity] = {}
        observations: list[TrackObservation] = []
        graph_events: list[InteractionEvent] = []
        segments: list[Segment] = []
        max_frame = 0
        for event in source_events:
            event_id = str(event["event_id"])
            views = event.get("views", [])
            starts = [int(view.get("start_frame", 0)) for view in views] or [0]
            ends = [int(view.get("end_frame", max(starts))) for view in views] or starts
            span = FrameSpan(min(starts), max(ends))
            max_frame = max(max_frame, span.end_frame)
            row = evidence_by_event.get(event_id, {})
            selected_instance = str(row.get("selected_instance_id", f"{event_id}:target"))
            entity_id = f"target:{selected_instance}"
            label = "target object"
            selected = next(
                (
                    item
                    for item in row.get("ranked_candidates", [])
                    if item.get("instance_id") == row.get("selected_instance_id")
                ),
                {},
            )
            label = str(selected.get("label") or label)
            entities.setdefault(entity_id, Entity(entity_id, "manipulated_object", (label,)))
            event_observations = []
            for index, track in enumerate(tracks_by_event.get(event_id, [])):
                box = track.get("bbox_xyxy")
                if not box:
                    continue
                observation_id = f"{event_id}:obs:{index:06d}"
                observations.append(
                    TrackObservation(
                        observation_id=observation_id,
                        entity_id=entity_id,
                        camera=str(track["camera"]),
                        frame_index=int(track["frame_index"]),
                        bbox_xyxy=tuple(float(value) for value in box),
                        mask_path=track.get("mask_path"),
                        visible=not bool(track.get("visibility_exception", False)),
                        confidence=float(track.get("confidence", row.get("score", 0.5)) or 0.5),
                    )
                )
                event_observations.append(observation_id)
            evidence_items = []
            for name, value in selected.get("evidence", {}).items():
                if value is not None:
                    evidence_items.append(
                        Evidence(str(name), "v2_interaction_evidence", float(value))
                    )
            review_state = "review" if row.get("needs_review") else "auto"
            graph_events.append(
                InteractionEvent(
                    event_id=event_id,
                    event_type="pick_and_place",
                    span=span,
                    active_arm=str(event.get("side", "unknown")),
                    target_entity_id=entity_id,
                    confidence=float(row.get("score", 0.5) or 0.5),
                    review_state=review_state,
                    evidence=tuple(evidence_items),
                    observation_ids=tuple(event_observations),
                )
            )
            segments.append(
                Segment(
                    segment_id=f"segment:{event_id}",
                    level="event",
                    span=span,
                    label="pick and place target",
                    confidence=float(row.get("score", 0.5) or 0.5),
                )
            )

        frame_count = max(
            max_frame + 1,
            max(
                (
                    int(view.get("frame_count", 0))
                    for event in source_events
                    for view in event.get("views", [])
                ),
                default=0,
            ),
        )
        instruction = str(session.get("task", {}).get("instruction", "")).strip()
        language = []
        if instruction:
            language.append(
                LanguageAnnotation(
                    language_id=f"episode:{episode_index}:task_language",
                    level="task",
                    text=instruction,
                    span=FrameSpan(0, frame_count - 1),
                    event_ids=tuple(event.event_id for event in graph_events),
                    entity_ids=tuple(entities),
                    source="reviewed_v2_task",
                    review_state="approved",
                )
            )
        graph = EpisodeEventGraph(
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            episode_id=str(episode_index),
            frame_count=frame_count,
            fps=float(session.get("fps", 30.0)),
            modalities={"cameras": sorted({obs.camera for obs in observations})},
            entities=list(entities.values()),
            observations=observations,
            segments=segments,
            events=graph_events,
            language=language,
            lineage={
                "source_schema": "robot_interaction_auto_label_v2",
                "source_workspace": str(workspace.resolve()),
            },
        )
        path = output / f"episode-{episode_index:06d}.event_graph.json"
        graph.write(path)
        written.append(path)
    return written


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
