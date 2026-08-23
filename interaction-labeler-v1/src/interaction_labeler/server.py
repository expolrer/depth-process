from __future__ import annotations

import io
import json
import mimetypes
import threading
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any

import cv2
from PIL import Image

from .io import atomic_write_json, read_json, read_jsonl
from .pipeline import run_pipeline


class LabelerApplication:
    def __init__(
        self,
        workspace: Path,
        pipeline_options: dict[str, Any] | None = None,
        pipeline_runner: Any = run_pipeline,
    ) -> None:
        self.workspace = workspace.resolve()
        self.pipeline_options = pipeline_options or {}
        self.pipeline_runner = pipeline_runner
        self.lock = threading.Lock()
        self.worker: threading.Thread | None = None
        self.manifest_cache: dict[Path, list[dict[str, Any]]] = {}

    def session(self) -> dict[str, Any]:
        session = read_json(self.workspace / "session.json", {})
        annotations = read_json(self.workspace / "annotations.json", {"revision": 0, "items": []})
        ranked_rows = read_jsonl(
            self.workspace / "outputs" / "interaction_candidates" / "ranked_index.jsonl"
        )
        tracks = read_jsonl(
            self.workspace / "outputs" / "target_tracks_required" / "track_index.jsonl"
        )
        ranked = {
            f"{row['event_id']}::{row['camera']}": {
                "selected_candidate_id": row.get("selected_candidate_id"),
                "needs_vlm": bool(row.get("needs_vlm")),
                "selection_margin": row.get("selection_margin"),
                "candidates": row.get("ranked_candidates", []),
            }
            for row in ranked_rows
        }
        track_map: dict[str, list[dict[str, Any]]] = {}
        for row in tracks:
            key = f"{row['event_id']}::{row['camera']}"
            track_map.setdefault(key, []).append(
                {
                    "frame_index": int(row["frame_index"]),
                    "bbox_xyxy": row.get("bbox_xyxy"),
                    "area_ratio": row.get("area_ratio"),
                    "needs_review": bool(row.get("needs_review")),
                    "visibility_status": row.get("visibility_status"),
                }
            )
        v2_evidence_rows = read_jsonl(self.workspace / "v2" / "instance_evidence.jsonl")
        v2_evidence = {row["event_id"]: row for row in v2_evidence_rows}
        v2_review_queue = read_jsonl(self.workspace / "v2" / "review_queue.jsonl")
        vlm_rows = read_jsonl(
            self.workspace / "outputs" / "interaction_candidates" / "vlm_resolutions.jsonl"
        )
        return {
            **session,
            "annotations": annotations,
            "ranked": ranked,
            "tracks": track_map,
            "vlm_resolutions": {row["event_id"]: row for row in vlm_rows},
            "v2_evidence": v2_evidence,
            "v2_review_queue": v2_review_queue,
            "v2_summary": read_json(self.workspace / "v2" / "summary.json", None),
            "pipeline_status": read_json(
                self.workspace / "pipeline_status.json",
                {"stage": "idle", "state": "idle"},
            ),
        }

    @staticmethod
    def _string_list(value: Any, field: str) -> list[str]:
        if not isinstance(value, list):
            raise TypeError(f"{field} must be a list")
        if any(not isinstance(item, str) for item in value):
            raise ValueError(f"{field} entries must be strings")
        return [item.strip() for item in value if item.strip()]

    def save_task(self, item: dict[str, Any]) -> dict[str, Any]:
        instruction = str(item.get("instruction", "")).strip()
        target_input = item.get("target", {})
        if not instruction:
            raise ValueError("instruction is required")
        if not isinstance(target_input, dict):
            raise TypeError("target must be an object")
        names = self._string_list(target_input.get("names", []), "target.names")
        detector_prompt = self._string_list(item.get("detector_prompt", []), "detector_prompt")
        if not names:
            raise ValueError("target.names must contain at least one concept")
        if not detector_prompt:
            raise ValueError("detector_prompt must contain at least one phrase")
        with self.lock:
            session = read_json(self.workspace / "session.json", {})
            old_task = session.get("task", {})
            old_target = old_task.get("target", {})
            target = {
                **old_target,
                "names": names,
                "attributes": self._string_list(
                    target_input.get("attributes", []), "target.attributes"
                ),
                "negative_descriptions": self._string_list(
                    target_input.get("negative_descriptions", []),
                    "target.negative_descriptions",
                ),
                "exemplar_images": self._string_list(
                    target_input.get("exemplar_images", []), "target.exemplar_images"
                ),
                "source_region": str(target_input.get("source_region", "")).strip(),
                "destination_region": str(target_input.get("destination_region", "")).strip(),
            }
            session["task"] = {
                **old_task,
                "instruction": instruction,
                "target": target,
                "detector_prompt": detector_prompt,
            }
            for event in session.get("events", []):
                event["instruction"] = instruction
            atomic_write_json(self.workspace / "session.json", session)
        return session["task"]

    def _find_view(self, event_id: str, camera: str) -> dict[str, Any]:
        session = read_json(self.workspace / "session.json", {})
        for event in session.get("events", []):
            if event["event_id"] != event_id:
                continue
            for view in event["views"]:
                if view["camera"] == camera:
                    return view
        raise KeyError(f"unknown event/camera: {event_id}/{camera}")

    def frame(self, event_id: str, camera: str, frame_index: int) -> bytes:
        view = self._find_view(event_id, camera)
        frame_index = max(0, min(int(view["frame_count"]) - 1, frame_index))
        if view["source_type"] == "video":
            capture = cv2.VideoCapture(view["video_path"])
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            capture.release()
            if not ok:
                raise RuntimeError("video frame decode failed")
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            return encoded.tobytes()
        manifest_path = Path(view["manifest_path"])
        if manifest_path not in self.manifest_cache:
            self.manifest_cache[manifest_path] = read_jsonl(manifest_path)
        row = self.manifest_cache[manifest_path][frame_index]
        image_path = Path(row["rgb_path"])
        if not image_path.is_absolute():
            image_path = (manifest_path.parent / image_path).resolve()
        image = Image.open(image_path).convert("RGB")
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=92)
        return output.getvalue()

    def save_annotation(self, item: dict[str, Any]) -> dict[str, Any]:
        required = {"event_id", "camera", "frame_index", "box_xyxy"}
        if not required.issubset(item):
            raise ValueError(f"annotation requires {sorted(required)}")
        box = [float(value) for value in item["box_xyxy"]]
        if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("box_xyxy must be a non-empty [x1, y1, x2, y2] box")
        with self.lock:
            payload = read_json(
                self.workspace / "annotations.json",
                {"schema": "robot_interaction_manual_annotations_v1", "revision": 0, "items": []},
            )
            key = (item["event_id"], item["camera"], int(item["frame_index"]))
            retained = [
                row
                for row in payload["items"]
                if (row["event_id"], row["camera"], int(row["frame_index"])) != key
            ]
            retained.append(
                {
                    **item,
                    "frame_index": int(item["frame_index"]),
                    "box_xyxy": box,
                    "source": "human_interactive_override",
                }
            )
            payload["revision"] = int(payload.get("revision", 0)) + 1
            payload["items"] = retained
            atomic_write_json(self.workspace / "annotations.json", payload)
        return payload

    def delete_annotation(self, item: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            payload = read_json(
                self.workspace / "annotations.json",
                {"schema": "robot_interaction_manual_annotations_v1", "revision": 0, "items": []},
            )
            key = (item["event_id"], item["camera"], int(item["frame_index"]))
            payload["items"] = [
                row
                for row in payload["items"]
                if (row["event_id"], row["camera"], int(row["frame_index"])) != key
            ]
            payload["revision"] = int(payload.get("revision", 0)) + 1
            atomic_write_json(self.workspace / "annotations.json", payload)
        return payload

    def save_event_timing(self, item: dict[str, Any]) -> dict[str, Any]:
        allowed = {"start_frame", "contact_frame", "release_frame", "end_frame"}
        with self.lock:
            session = read_json(self.workspace / "session.json", {})
            for event in session.get("events", []):
                if event["event_id"] != item["event_id"]:
                    continue
                for view in event["views"]:
                    if view["camera"] != item["camera"]:
                        continue
                    for key in allowed.intersection(item):
                        view[key] = max(0, min(int(view["frame_count"]) - 1, int(item[key])))
                    if not (
                        view["start_frame"]
                        <= view["contact_frame"]
                        <= view["release_frame"]
                        <= view["end_frame"]
                    ):
                        raise ValueError(
                            "frame stages must be ordered start <= contact <= release <= end"
                        )
                    atomic_write_json(self.workspace / "session.json", session)
                    return session
        raise KeyError("event view not found")

    def start_pipeline(self) -> bool:
        with self.lock:
            if self.worker and self.worker.is_alive():
                return False

            def target() -> None:
                self.pipeline_runner(self.workspace, **self.pipeline_options)

            self.worker = threading.Thread(target=target, name="label-pipeline", daemon=True)
            self.worker.start()
            return True


class LabelerHandler(BaseHTTPRequestHandler):
    server_version = "InteractionLabeler/0.1"

    @property
    def app(self) -> LabelerApplication:
        return self.server.app  # type: ignore[attr-defined]

    def _json(self, value: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        try:
            if parsed.path == "/api/session":
                self._json(self.app.session())
                return
            if parsed.path == "/api/status":
                self._json(
                    read_json(
                        self.app.workspace / "pipeline_status.json",
                        {"stage": "idle", "state": "idle"},
                    )
                )
                return
            if parsed.path == "/api/frame":
                query = urllib.parse.parse_qs(parsed.query)
                body = self.app.frame(
                    query["event_id"][0],
                    query["camera"][0],
                    int(query["frame"][0]),
                )
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "private, max-age=300")
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return
            self._static(parsed.path)
        except Exception as error:  # noqa: BLE001 - HTTP boundary returns structured errors.
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def do_POST(self) -> None:
        try:
            if self.path == "/api/annotations":
                self._json(self.app.save_annotation(self._body()))
            elif self.path == "/api/annotations/delete":
                self._json(self.app.delete_annotation(self._body()))
            elif self.path == "/api/event-timing":
                self._json(self.app.save_event_timing(self._body()))
            elif self.path == "/api/task":
                self._json(self.app.save_task(self._body()))
            elif self.path == "/api/run":
                started = self.app.start_pipeline()
                self._json(
                    {"started": started}, HTTPStatus.ACCEPTED if started else HTTPStatus.CONFLICT
                )
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except Exception as error:  # noqa: BLE001 - HTTP boundary returns structured errors.
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def _static(self, request_path: str) -> None:
        relative = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
        if ".." in Path(relative).parts:
            raise ValueError("invalid static path")
        resource = files("interaction_labeler").joinpath("static", relative)
        if not resource.is_file():
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        body = resource.read_bytes()
        content_type = mimetypes.guess_type(relative)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type",
            f"{content_type}; charset=utf-8" if content_type.startswith("text/") else content_type,
        )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[labeler] {self.address_string()} {format % args}")


def serve(
    workspace: Path,
    host: str = "127.0.0.1",
    port: int = 8769,
    open_browser: bool = True,
    auto_start: bool = False,
    pipeline_options: dict[str, Any] | None = None,
    pipeline_runner: Any = run_pipeline,
) -> None:
    app = LabelerApplication(workspace, pipeline_options, pipeline_runner)
    server = ThreadingHTTPServer((host, port), LabelerHandler)
    server.app = app  # type: ignore[attr-defined]
    url = f"http://{host}:{port}/"
    print(f"Interaction labeler: {url}", flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    if auto_start:
        app.start_pipeline()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
