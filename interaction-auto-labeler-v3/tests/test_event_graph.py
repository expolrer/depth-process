import json
from pathlib import Path

import pytest

from interaction_auto_labeler_v3.event_graph import (
    Entity,
    EpisodeEventGraph,
    FrameSpan,
    InteractionEvent,
    LanguageAnnotation,
    Segment,
    TrackObservation,
)


def test_event_graph_round_trip(tmp_path: Path) -> None:
    graph = EpisodeEventGraph(
        dataset_id="demo",
        dataset_version="v1",
        episode_id="0",
        frame_count=100,
        fps=30.0,
        entities=[Entity("object:1", "toy", ("red toy",))],
        observations=[TrackObservation("obs:1", "object:1", "cam_h", 20, (1.0, 2.0, 4.0, 6.0))],
        segments=[Segment("task:0", "task", FrameSpan(0, 99), "move toy")],
        events=[
            InteractionEvent(
                "event:0",
                "pick",
                FrameSpan(10, 50),
                target_entity_id="object:1",
                observation_ids=("obs:1",),
            )
        ],
        language=[
            LanguageAnnotation(
                "language:0",
                "task",
                "Move the red toy.",
                FrameSpan(0, 99),
                ("event:0",),
                ("object:1",),
            )
        ],
    )
    path = tmp_path / "graph.json"
    graph.write(path)
    loaded = EpisodeEventGraph.read(path)
    assert loaded.to_dict() == graph.to_dict()
    assert json.loads(path.read_text(encoding="utf-8"))["schema"] == "embodied_event_graph_v1"


def test_event_graph_rejects_dangling_reference() -> None:
    graph = EpisodeEventGraph(
        dataset_id="demo",
        dataset_version="v1",
        episode_id="0",
        frame_count=10,
        fps=30,
        events=[InteractionEvent("event:0", "pick", FrameSpan(0, 5), target_entity_id="missing")],
    )
    with pytest.raises(ValueError, match="unknown entity"):
        graph.validate()
