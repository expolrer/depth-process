from __future__ import annotations

import mimetypes
import threading
import uuid
import webbrowser
from hashlib import sha256
from http.server import ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any

from .corrections import HumanCorrection, append_correction, correction_summary
from .io import read_jsonl

try:
    from interaction_labeler.io import read_json
    from interaction_labeler.server import LabelerApplication, LabelerHandler
except ImportError as error:  # pragma: no cover - installation error has a direct message.
    raise ImportError("V3 server requires the sibling interaction-labeler-v1 package") from error


class V3LabelerApplication(LabelerApplication):
    def _identity(self) -> tuple[str, str]:
        session = read_json(self.workspace / "session.json", {})
        config = session.get("v3", {})
        return str(config.get("dataset_id", "unknown")), str(
            config.get("dataset_version", "unknown")
        )

    def _record(
        self,
        correction_type: str,
        before: dict[str, Any],
        after: dict[str, Any],
        *,
        event_id: str | None = None,
        frame_index: int | None = None,
        camera: str | None = None,
    ) -> None:
        if before == after:
            return
        dataset_id, dataset_version = self._identity()
        episode_id = "unknown"
        if event_id:
            session = read_json(self.workspace / "session.json", {})
            event = next(
                (item for item in session.get("events", []) if item["event_id"] == event_id), {}
            )
            episode_id = str(event.get("episode_index", event.get("sequence", "unknown")))
        append_correction(
            self.workspace / "v3" / "corrections.jsonl",
            HumanCorrection(
                correction_id=f"correction-{uuid.uuid4().hex}",
                dataset_id=dataset_id,
                dataset_version=dataset_version,
                episode_id=episode_id,
                correction_type=correction_type,
                before=before,
                after=after,
                event_id=event_id,
                frame_index=frame_index,
                camera=camera,
                reason="interactive_review",
            ),
        )

    def overview(self) -> dict[str, Any]:
        quality = []
        quality_root = self.workspace / "v3" / "quality"
        if quality_root.exists():
            quality = [read_json(path, {}) for path in sorted(quality_root.glob("*.json"))]
        corrections = read_jsonl(self.workspace / "v3" / "corrections.jsonl")
        return {
            "schema": "auto_labeler_v3_ui_overview_v1",
            "summary": read_json(self.workspace / "v3" / "summary.json", {}),
            "quality": quality,
            "corrections": correction_summary(corrections),
        }

    def session(self) -> dict[str, Any]:
        return {**super().session(), "v3_overview": self.overview()}

    def save_annotation(self, item: dict[str, Any]) -> dict[str, Any]:
        payload = read_json(self.workspace / "annotations.json", {"items": []})
        key = (item.get("event_id"), item.get("camera"), int(item.get("frame_index", -1)))
        previous = next(
            (
                row
                for row in payload.get("items", [])
                if (row.get("event_id"), row.get("camera"), int(row.get("frame_index", -2))) == key
            ),
            {},
        )
        result = super().save_annotation(item)
        current = next(
            row
            for row in result["items"]
            if (row["event_id"], row["camera"], int(row["frame_index"])) == key
        )
        correction_after = {
            "bbox_xyxy": current["box_xyxy"],
            "source": current["source"],
        }
        try:
            frame_name = sha256(
                f"{item['event_id']}:{item['camera']}:{int(item['frame_index'])}".encode()
            ).hexdigest()[:20]
            frame_path = self.workspace / "v3" / "correction_frames" / f"{frame_name}.jpg"
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            frame_path.write_bytes(
                self.frame(
                    str(item["event_id"]),
                    str(item["camera"]),
                    int(item["frame_index"]),
                )
            )
            from PIL import Image

            with Image.open(frame_path) as image:
                correction_after.update(
                    image_path=str(frame_path.resolve()), image_size=list(image.size)
                )
        except (OSError, RuntimeError, ValueError) as error:
            correction_after["image_capture_error"] = str(error)
        self._record(
            "bbox",
            {"bbox_xyxy": previous.get("box_xyxy")},
            correction_after,
            event_id=str(item["event_id"]),
            frame_index=int(item["frame_index"]),
            camera=str(item["camera"]),
        )
        return result

    def delete_annotation(self, item: dict[str, Any]) -> dict[str, Any]:
        payload = read_json(self.workspace / "annotations.json", {"items": []})
        key = (item.get("event_id"), item.get("camera"), int(item.get("frame_index", -1)))
        previous = next(
            (
                row
                for row in payload.get("items", [])
                if (row.get("event_id"), row.get("camera"), int(row.get("frame_index", -2))) == key
            ),
            {},
        )
        result = super().delete_annotation(item)
        if previous:
            self._record(
                "bbox",
                {"bbox_xyxy": previous.get("box_xyxy")},
                {"bbox_xyxy": None, "visible": False},
                event_id=str(item["event_id"]),
                frame_index=int(item["frame_index"]),
                camera=str(item["camera"]),
            )
        return result

    def save_event_timing(self, item: dict[str, Any]) -> dict[str, Any]:
        session = read_json(self.workspace / "session.json", {})
        event = next(row for row in session["events"] if row["event_id"] == item["event_id"])
        view = next(row for row in event["views"] if row["camera"] == item["camera"])
        keys = ("start_frame", "contact_frame", "release_frame", "end_frame")
        before = {key: view.get(key) for key in keys}
        result = super().save_event_timing(item)
        event = next(row for row in result["events"] if row["event_id"] == item["event_id"])
        view = next(row for row in event["views"] if row["camera"] == item["camera"])
        after = {key: view.get(key) for key in keys}
        self._record(
            "boundary",
            before,
            after,
            event_id=str(item["event_id"]),
            camera=str(item["camera"]),
        )
        return result

    def save_task(self, item: dict[str, Any]) -> dict[str, Any]:
        session = read_json(self.workspace / "session.json", {})
        before = {"text": session.get("task", {}).get("instruction", "")}
        result = super().save_task(item)
        after = {"text": result.get("instruction", "")}
        self._record("language", before, after)
        return result


class V3LabelerHandler(LabelerHandler):
    @property
    def app(self) -> V3LabelerApplication:
        return self.server.app  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] == "/api/v3/overview":
            try:
                self._json(self.app.overview())
            except Exception as error:  # noqa: BLE001 - HTTP boundary returns structured errors.
                self._json({"error": str(error)}, 400)
            return
        super().do_GET()

    def _static(self, request_path: str) -> None:
        relative = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
        if ".." in Path(relative).parts:
            raise ValueError("invalid static path")
        resource = files("interaction_auto_labeler_v3").joinpath("static", relative)
        if not resource.is_file():
            super()._static(request_path)
            return
        body = resource.read_bytes()
        content_type = mimetypes.guess_type(relative)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header(
            "Content-Type",
            f"{content_type}; charset=utf-8" if content_type.startswith("text/") else content_type,
        )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(
    workspace: Path,
    host: str = "127.0.0.1",
    port: int = 8773,
    open_browser: bool = True,
    auto_start: bool = False,
    pipeline_options: dict[str, Any] | None = None,
    pipeline_runner: Any = None,
) -> None:
    from .pipeline import run_v3_pipeline

    app = V3LabelerApplication(
        workspace,
        pipeline_options,
        pipeline_runner or run_v3_pipeline,
    )
    server = ThreadingHTTPServer((host, port), V3LabelerHandler)
    server.app = app  # type: ignore[attr-defined]
    url = f"http://{host}:{port}/"
    print(f"Interaction auto labeler V3: {url}", flush=True)
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
