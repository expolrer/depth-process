from interaction_auto_labeler_v3.event_graph import (
    Entity,
    EpisodeEventGraph,
    FrameSpan,
    InteractionEvent,
    Segment,
)
from interaction_auto_labeler_v3.language import generate_grounded_language


def test_language_is_grounded_at_three_levels() -> None:
    graph = EpisodeEventGraph(
        dataset_id="demo",
        dataset_version="v1",
        episode_id="0",
        frame_count=60,
        fps=30,
        entities=[
            Entity("toy", "toy", ("red toy",)),
            Entity("pile", "region", ("center pile",)),
            Entity("left", "region", ("left side",)),
        ],
        segments=[
            Segment("task:0", "task", FrameSpan(0, 59), "move toy"),
            Segment("subtask:0", "subtask", FrameSpan(0, 59), "pick and place", "task:0"),
        ],
        events=[
            InteractionEvent(
                "event:0",
                "pick_and_place",
                FrameSpan(5, 50),
                active_arm="left",
                target_entity_id="toy",
                source_entity_id="pile",
                destination_entity_id="left",
            )
        ],
    )
    result = generate_grounded_language(graph)
    assert {item.level for item in result.language} == {"task", "subtask", "event"}
    assert "red toy" in next(item.text for item in result.language if item.level == "event")
