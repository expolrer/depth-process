from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.request
from pathlib import Path
from typing import Any, Protocol

from .event_graph import EpisodeEventGraph, FrameSpan, InteractionEvent, LanguageAnnotation


class LanguageBackend(Protocol):
    def generate(
        self, context: dict[str, Any], image_paths: tuple[Path, ...] = ()
    ) -> dict[str, Any]: ...


def _entity_name(graph: EpisodeEventGraph, entity_id: str | None, fallback: str) -> str:
    entity = next((item for item in graph.entities if item.entity_id == entity_id), None)
    if entity is None:
        return fallback
    return entity.names[0] if entity.names else entity.category.replace("_", " ")


def event_context(graph: EpisodeEventGraph, event: InteractionEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "event_type": event.event_type,
        "active_arm": event.active_arm,
        "target": {
            "entity_id": event.target_entity_id,
            "name": _entity_name(graph, event.target_entity_id, "target object"),
        },
        "source": {
            "entity_id": event.source_entity_id,
            "name": _entity_name(graph, event.source_entity_id, "source location"),
        },
        "destination": {
            "entity_id": event.destination_entity_id,
            "name": _entity_name(graph, event.destination_entity_id, "target location"),
        },
        "start_frame": event.span.start_frame,
        "end_frame": event.span.end_frame,
        "phase": event.phase,
        "attributes": event.attributes,
    }


class TemplateLanguageBackend:
    def generate(
        self, context: dict[str, Any], image_paths: tuple[Path, ...] = ()
    ) -> dict[str, Any]:
        arm = context["active_arm"]
        hand = f"{arm} hand" if arm in {"left", "right"} else "robot hand"
        target = context["target"]["name"]
        source = context["source"]["name"]
        destination = context["destination"]["name"]
        event_type = context["event_type"].replace("-", "_")
        source_clause = f" from the {source}" if context["source"].get("entity_id") else ""
        destination_clause = (
            f" at the {destination}" if context["destination"].get("entity_id") else ""
        )
        templates = {
            "pick": f"Use the {hand} to pick up the {target}{source_clause}.",
            "place": f"Use the {hand} to place the {target}{destination_clause}.",
            "pick_and_place": (
                f"Use the {hand} to pick up the {target}{source_clause}"
                + (f" and place it{destination_clause}." if destination_clause else ".")
            ),
            "insert": f"Use the {hand} to insert the {target} into the {destination}.",
            "open": f"Use the {hand} to open the {target}.",
            "close": f"Use the {hand} to close the {target}.",
            "press": f"Use the {hand} to press the {target}.",
            "rotate": f"Use the {hand} to rotate the {target}.",
            "pour": f"Use the {hand} to pour from the {target} into the {destination}.",
            "wipe": f"Use the {hand} and the {target} to wipe the {destination}.",
            "handover": f"Transfer the {target} between the robot hands.",
        }
        text = templates.get(
            event_type,
            f"Use the {hand} to perform {event_type.replace('_', ' ')} on the {target}.",
        )
        return {
            "text": text,
            "entity_ids": [
                value["entity_id"]
                for key, value in context.items()
                if key in {"target", "source", "destination"} and value.get("entity_id")
            ],
            "confidence": 1.0,
            "source": "grounded_template",
        }


class OpenAICompatibleLanguageBackend:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _image(path: Path) -> dict[str, Any]:
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}

    def generate(
        self, context: dict[str, Any], image_paths: tuple[Path, ...] = ()
    ) -> dict[str, Any]:
        prompt = (
            "Generate one concise VLA instruction for the grounded robot event below. "
            "Do not introduce objects, arms, actions, or locations absent from the JSON. "
            "Return JSON with text, entity_ids, confidence.\n"
            + json.dumps(context, ensure_ascii=False)
        )
        content = [{"type": "text", "text": prompt}] + [self._image(path) for path in image_paths]
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": content}],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
        content_text = result["choices"][0]["message"]["content"]
        generated = json.loads(content_text)
        generated.setdefault("source", f"openai_compatible:{self.model}")
        return generated


def generate_grounded_language(
    graph: EpisodeEventGraph,
    backend: LanguageBackend | None = None,
    images_by_event: dict[str, tuple[Path, ...]] | None = None,
) -> EpisodeEventGraph:
    backend = backend or TemplateLanguageBackend()
    images_by_event = images_by_event or {}
    known_entities = {item.entity_id for item in graph.entities}
    approved = [item for item in graph.language if item.review_state == "approved"]
    approved_event_ids = {
        event_id for item in approved if item.level == "event" for event_id in item.event_ids
    }
    generated: list[LanguageAnnotation] = []
    for index, event in enumerate(sorted(graph.events, key=lambda item: item.span.start_frame)):
        if event.event_id in approved_event_ids:
            continue
        payload = backend.generate(
            event_context(graph, event), images_by_event.get(event.event_id, ())
        )
        entity_ids = tuple(str(item) for item in payload.get("entity_ids", []))
        if set(entity_ids) - known_entities:
            raise ValueError(f"language backend introduced unknown entities for {event.event_id}")
        generated.append(
            LanguageAnnotation(
                language_id=f"language:event:{index:04d}",
                level="event",
                text=str(payload["text"]).strip(),
                span=event.span,
                event_ids=(event.event_id,),
                entity_ids=entity_ids,
                source=str(payload.get("source", "language_backend")),
                confidence=float(payload.get("confidence", 0.5)),
                review_state="auto",
            )
        )

    subtask_segments = [item for item in graph.segments if item.level == "subtask"]
    for index, segment in enumerate(subtask_segments):
        if any(item.level == "subtask" and item.span == segment.span for item in approved):
            continue
        children = [item for item in generated if segment.span.contains(item.span)]
        if not children:
            continue
        generated.append(
            LanguageAnnotation(
                language_id=f"language:subtask:{index:04d}",
                level="subtask",
                text=" Then ".join(item.text.rstrip(".") for item in children) + ".",
                span=segment.span,
                event_ids=tuple(event_id for item in children for event_id in item.event_ids),
                entity_ids=tuple(
                    dict.fromkeys(entity_id for item in children for entity_id in item.entity_ids)
                ),
                source="hierarchical_composition",
                confidence=min(item.confidence for item in children),
            )
        )
    event_language = [item for item in generated if item.level == "event"]
    if event_language and not any(item.level == "task" for item in approved):
        generated.append(
            LanguageAnnotation(
                language_id="language:task:0000",
                level="task",
                text=" Then ".join(item.text.rstrip(".") for item in event_language) + ".",
                span=FrameSpan(0, graph.frame_count - 1),
                event_ids=tuple(event.event_id for event in graph.events),
                entity_ids=tuple(
                    dict.fromkeys(
                        entity_id for item in event_language for entity_id in item.entity_ids
                    )
                ),
                source="hierarchical_composition",
                confidence=min(item.confidence for item in event_language),
            )
        )
    graph.language = approved + generated
    graph.validate()
    return graph


def backend_from_environment(base_url: str, model: str, api_key_env: str) -> LanguageBackend:
    return OpenAICompatibleLanguageBackend(base_url, model, os.environ.get(api_key_env))
