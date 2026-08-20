#!/usr/bin/env python3
"""Validate the static viewer catalog and every referenced video asset."""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer-root", type=Path, default=Path(__file__).resolve().parents[1] / "viewer")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def probe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,nb_frames:format=duration,size",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"ffprobe failed with {result.returncode}")
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise ValueError("No video stream")
    return {"stream": streams[0], "format": payload.get("format", {})}


def main() -> int:
    args = parse_args()
    viewer_root = args.viewer_root.resolve()
    catalog_path = viewer_root / "data" / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    expected_fps = float(catalog["fps"])
    issues: list[dict[str, str]] = []
    assets: list[tuple[Path, int, str]] = []

    for static_name in ("index.html", "styles.css", "app.js", "serve_viewer.py", "vendor/lucide.min.js"):
        static_path = viewer_root / static_name
        if not static_path.is_file() or static_path.stat().st_size == 0:
            issues.append({"asset": static_name, "error": "Missing or empty static asset"})

    seen: set[str] = set()
    def add_references(references: list[tuple[str, str]], expected_frames: int) -> None:
        for stream_id, relative_path in references:
            if relative_path in seen:
                issues.append({"asset": relative_path, "error": "Duplicate catalog reference"})
                continue
            seen.add(relative_path)
            assets.append((viewer_root / relative_path, expected_frames, stream_id))

    for dataset in catalog["datasets"]:
        if "media" not in dataset:
            # Version 1 catalogs store each three-camera mosaic as one video.
            expected_frames = int(dataset["frameCount"])
            references = [("rgb", dataset["rgb"])]
            references.extend((f"depth:{key}", value) for key, value in dataset["depth"].items())
            references.extend(
                (f"attention:{key}", value) for key, value in dataset.get("attention", {}).items()
            )
            add_references(references, expected_frames)
            continue

        for camera in catalog["cameras"]:
            camera_id = camera["id"]
            expected_frames = int(dataset["frameCounts"][camera_id])
            media = dataset["media"][camera_id]
            add_references([("rgb", media["rgb"]), *media["methods"].items()], expected_frames)

        mosaic = dataset.get("mosaic")
        if not mosaic:
            issues.append({"asset": dataset["id"], "error": "Missing mosaic media catalog"})
            continue
        mosaic_references = [("mosaic_rgb", mosaic["rgb"]), *mosaic["methods"].items()]
        add_references(mosaic_references, int(dataset["frameCounts"]["cam_h"]))

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        pending = {executor.submit(probe, path): (path, frames, stream_id) for path, frames, stream_id in assets}
        for future in as_completed(pending):
            path, expected_frames, stream_id = pending[future]
            relative = path.relative_to(viewer_root).as_posix()
            try:
                payload = future.result()
                stream = payload["stream"]
                actual_frames = int(stream.get("nb_frames") or 0)
                actual_fps = float(Fraction(stream.get("r_frame_rate", "0/1")))
                width = int(stream.get("width") or 0)
                height = int(stream.get("height") or 0)
                errors: list[str] = []
                if stream.get("codec_name") != "h264":
                    errors.append(f"codec={stream.get('codec_name')}")
                if actual_frames != expected_frames:
                    errors.append(f"frames={actual_frames}, expected={expected_frames}")
                if abs(actual_fps - expected_fps) > 0.01:
                    errors.append(f"fps={actual_fps}, expected={expected_fps}")
                if width <= 0 or height <= 0 or width % 2 or height % 2:
                    errors.append(f"invalid dimensions={width}x{height}")
                if errors:
                    issues.append({"asset": relative, "error": "; ".join(errors)})
                results.append(
                    {
                        "asset": relative,
                        "stream": stream_id,
                        "frames": actual_frames,
                        "fps": actual_fps,
                        "width": width,
                        "height": height,
                        "bytes": int(payload["format"].get("size") or 0),
                    }
                )
            except Exception as exc:
                issues.append({"asset": relative, "error": str(exc)})

    results.sort(key=lambda item: item["asset"])
    report = {
        "ok": not issues,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "datasetCount": len(catalog["datasets"]),
        "cameraCount": len(catalog["cameras"]),
        "methodCount": len(catalog["methods"]),
        "videoCount": len(assets),
        "totalBytes": sum(item["bytes"] for item in results),
        "issues": issues,
        "videos": results,
    }
    output = args.output or viewer_root / "data" / "validation_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: report[key] for key in ("ok", "datasetCount", "cameraCount", "methodCount", "videoCount", "totalBytes", "issues")},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
