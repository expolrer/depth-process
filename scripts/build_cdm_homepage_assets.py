#!/usr/bin/env python3
"""Build wrist-only CDM comparison videos and register them in the homepage catalog."""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np


METHODS = (
    ("cdm_camera_specific", "CDM D405 camera-specific", "Processing methods"),
    ("cdm_sensor_fused", "CDM D405 sensor fused", "Processing methods"),
)
SEQUENCE_SLUGS = {
    "A10-A15-G-S-01-TQ_03_01-4_304-leju_claw-20260512172205-200049-8c435b-v003": "leju_claw",
    "A10-A15-G-S-01-TQ_09_01-P4_297-dex_hand-20260629103123-49-9a5c2d-v003": "dex_hand",
    "chengzhong_xianxia_main1": "chengzhong",
    "dajian_xianxia_main1": "dajian",
    "zhoumian_xianxia_main1": "zhoumian",
}


def read_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def colorize(path: Path, minimum: float, maximum: float) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(path)
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth_m = depth.astype(np.float32) * 0.001
    valid = np.isfinite(depth_m) & (depth_m > 0)
    normalized = np.clip((depth_m - minimum) / max(maximum - minimum, 1e-6), 0.0, 1.0)
    image = cv2.applyColorMap(np.rint(normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    image[~valid] = (14, 16, 17)
    return cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (320, 30), (14, 16, 18), -1)
    cv2.putText(result, label, (9, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def unsupported_head(path: Path, minimum: float, maximum: float) -> np.ndarray:
    panel = colorize(path, minimum, maximum)
    panel = np.rint(panel.astype(np.float32) * 0.28).astype(np.uint8)
    cv2.rectangle(panel, (20, 78), (300, 174), (14, 16, 18), -1)
    cv2.putText(panel, "HEAD / Gemini-335L", (50, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (230, 235, 238), 1, cv2.LINE_AA)
    cv2.putText(panel, "CDM checkpoint unavailable", (37, 137), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, "dimmed raw reference", (62, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (170, 176, 180), 1, cv2.LINE_AA)
    return panel


def probe_frames(path: Path) -> int | None:
    if not path.is_file():
        return None
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1", str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode or not result.stdout.strip().isdigit():
        return None
    return int(result.stdout.strip())


def encode_sequence(
    project_root: Path,
    output_root: Path,
    sequence: str,
    method: str,
    fps: float,
    minimum: float,
    maximum: float,
    overwrite: bool,
) -> dict[str, Any]:
    manifests = {
        camera: read_manifest(project_root / "outputs/extracted" / sequence / camera / "manifest.jsonl")
        for camera in ("cam_h", "cam_l", "cam_r")
    }
    frame_count = min(len(rows) for rows in manifests.values())
    slug = SEQUENCE_SLUGS[sequence]
    output = output_root / "media" / slug / "depth" / f"{method}.mp4"
    if not overwrite and probe_frames(output) == frame_count:
        return {"sequence": sequence, "method": method, "frames": frame_count, "status": "reused"}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".part.mp4")
    temporary.unlink(missing_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s:v", "960x240", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-frames:v", str(frame_count), "-y", str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in range(frame_count):
            head_path = Path(
                manifests["cam_h"][frame].get(
                    "depth_aligned_rgb_mm_path", manifests["cam_h"][frame]["depth_raw_mm_path"]
                )
            )
            panels = [unsupported_head(head_path, minimum, maximum)]
            for camera, label in (("cam_l", "LEFT WRIST / D405"), ("cam_r", "RIGHT WRIST / D405")):
                path = project_root / "outputs/processed" / sequence / camera / method / f"{frame:06d}.png"
                panels.append(add_label(colorize(path, minimum, maximum), label))
            process.stdin.write(cv2.hconcat(panels).tobytes())
        process.stdin.close()
        error = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        status = process.wait()
        if status:
            raise RuntimeError(error.strip() or f"ffmpeg exited {status}")
        temporary.replace(output)
    except BaseException:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        process.kill()
        process.wait()
        temporary.unlink(missing_ok=True)
        raise
    actual = probe_frames(output)
    if actual != frame_count:
        raise RuntimeError(f"Invalid frame count for {output}: {actual} != {frame_count}")
    return {"sequence": sequence, "method": method, "frames": frame_count, "status": "encoded"}


def update_catalog(output_root: Path) -> None:
    json_path = output_root / "data/catalog.json"
    catalog = json.loads(json_path.read_text(encoding="utf-8"))
    existing = {row["id"]: row for row in catalog["methods"]}
    for method, label, family in METHODS:
        existing[method] = {
            "id": method,
            "label": label,
            "family": family,
            "globalMaeRad": None,
            "globalFirstStepMaeRad": None,
            "evaluationScope": "D405 wrists only; ACT evaluation not run",
        }
    ordered = [row for row in catalog["methods"] if row["id"] not in {item[0] for item in METHODS}]
    insertion = next((index for index, row in enumerate(ordered) if row["family"] == "Negative controls"), len(ordered))
    catalog["methods"] = ordered[:insertion] + [existing[item[0]] for item in METHODS] + ordered[insertion:]
    for dataset in catalog["datasets"]:
        for method, _label, _family in METHODS:
            dataset["depth"][method] = f"media/{dataset['id']}/depth/{method}.mp4"
            dataset["metrics"][method] = {
                "frames": 0,
                "chunk_mae_rad": None,
                "chunk_rmse_rad": None,
                "first_step_mae_rad": None,
                "first_step_rmse_rad": None,
                "depth_attention_mass": None,
                "depth_attention_entropy": None,
            }
    payload = json.dumps(catalog, ensure_ascii=False, indent=2) + "\n"
    json_path.write_text(payload, encoding="utf-8")
    (output_root / "data/catalog.js").write_text(
        "window.RGBD_CATALOG = " + json.dumps(catalog, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--depth-min-m", type=float, default=0.2)
    parser.add_argument("--depth-max-m", type=float, default=4.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    tasks = [
        (args.project_root, args.output_root, sequence, method, args.fps, args.depth_min_m, args.depth_max_m, args.overwrite)
        for sequence in SEQUENCE_SLUGS
        for method, _label, _family in METHODS
    ]
    records = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(encode_sequence, *task): (task[2], task[3]) for task in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            record = future.result()
            records.append(record)
            print(f"[{index}/{len(tasks)}] {record['status']} {record['sequence']} {record['method']}", flush=True)
    update_catalog(args.output_root)
    report = {"methods": [item[0] for item in METHODS], "records": records}
    (args.output_root / "data/cdm_video_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
