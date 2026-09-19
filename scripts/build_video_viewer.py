#!/usr/bin/env python3
"""Build browser-friendly RGB and colorized metric-depth videos."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CAMERAS = (
    ("cam_h", "头部", "Gemini-335L"),
    ("cam_l", "左腕", "D405"),
    ("cam_r", "右腕", "D405"),
)

METHODS = (
    ("raw_aligned", "原始对齐深度", "传感器"),
    ("rgb_guided", "RGB 引导", "经典方法"),
    ("temporal_rgb_guided", "RGB + 时序", "经典方法"),
    ("lingbot_v05", "LingBot v0.5", "模型预测"),
    ("depth_anything_v2_fused", "Depth Anything V2 融合", "传感器融合"),
    ("lingbot_v05_sensor_fused", "LingBot 传感器融合", "传感器融合"),
    ("ai_consensus_fused", "AI 共识融合", "传感器融合"),
    ("lingbot_cross_attention", "LingBot 跨模态注意力", "模型解释"),
    ("lingbot_depth_token_attention", "LingBot 深度 Token 注意力", "模型解释"),
)

CDM_METHODS = (
    ("cdm_camera_specific", "CDM 相机专属输出", "模型预测"),
    ("cdm_sensor_fused", "CDM 传感器融合", "传感器融合"),
)

ATTENTION_METHODS = {
    "lingbot_cross_attention",
    "lingbot_depth_token_attention",
}

DATASET_HINTS = (
    ("leju_claw", "leju_claw", "夹爪采集", "Leju Claw"),
    ("dex_hand", "dex_hand", "灵巧手采集", "Dex Hand"),
    ("chengzhong", "chengzhong", "称重", "线下赛"),
    ("dajian", "dajian", "大件搬运", "线下赛"),
    ("zhoumian", "zhoumian", "桌面整理", "线下赛"),
)


@dataclass(frozen=True)
class VideoTask:
    dataset_key: str
    dataset_slug: str
    camera_id: str
    stream_id: str
    kind: str
    frame_paths: tuple[Path, ...]
    output_path: Path
    frame_count: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--viewer-root", type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--depth-min-m", type=float, default=0.2)
    parser.add_argument("--depth-max-m", type=float, default=4.0)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--ffmpeg-threads", type=int, default=2)
    parser.add_argument("--crf", type=int, default=22)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--dataset", action="append", help="Only build matching dataset key/slug")
    parser.add_argument("--max-frames", type=int, help="Limit frames per stream for a smoke test")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include-cdm",
        action="store_true",
        help="Include CDM videos after process_cdm.py has produced every selected camera stream",
    )
    return parser.parse_args()


def dataset_identity(key: str) -> tuple[str, str, str, int]:
    for order, (needle, slug, label, group) in enumerate(DATASET_HINTS):
        if needle in key.lower():
            return slug, label, group, order
    safe = "".join(char if char.isalnum() else "-" for char in key.lower()).strip("-")
    return safe or "dataset", key, "RGB-D", len(DATASET_HINTS)


def read_manifest(path: Path, max_frames: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if max_frames is not None and len(rows) >= max_frames:
                    break
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def processed_paths(
    processed_root: Path,
    dataset_key: str,
    camera_id: str,
    method_id: str,
    frame_count: int,
    extension: str = ".png",
) -> tuple[Path, ...]:
    directory = processed_root / dataset_key / camera_id / method_id
    paths = tuple(directory / f"{index:06d}{extension}" for index in range(frame_count))
    missing = next((path for path in paths if not path.is_file()), None)
    if missing:
        raise FileNotFoundError(f"Missing processed frame: {missing}")
    return paths


def collect_tasks(
    args: argparse.Namespace,
    methods: tuple[tuple[str, str, str], ...],
) -> tuple[list[VideoTask], list[dict[str, Any]]]:
    project_root = args.project_root.resolve()
    extracted_root = project_root / "outputs" / "extracted"
    processed_root = project_root / "outputs" / "processed"
    viewer_root = (args.viewer_root or project_root / "viewer").resolve()
    selected = {value.lower() for value in args.dataset or []}

    datasets: list[dict[str, Any]] = []
    tasks: list[VideoTask] = []
    dataset_dirs = [path for path in extracted_root.iterdir() if path.is_dir()]
    dataset_dirs.sort(key=lambda path: dataset_identity(path.name)[3])

    for dataset_dir in dataset_dirs:
        slug, label, group, _ = dataset_identity(dataset_dir.name)
        if selected and not any(value in {slug.lower(), dataset_dir.name.lower()} for value in selected):
            continue

        dataset_entry: dict[str, Any] = {
            "id": slug,
            "key": dataset_dir.name,
            "label": label,
            "group": group,
            "frameCounts": {},
            "media": {},
        }
        for camera_id, camera_label, camera_model in CAMERAS:
            manifest_path = dataset_dir / camera_id / "manifest.jsonl"
            rows = read_manifest(manifest_path, args.max_frames)
            frame_count = len(rows)
            dataset_entry["frameCounts"][camera_id] = frame_count
            media_dir = viewer_root / "media" / slug / camera_id
            media_dir.mkdir(parents=True, exist_ok=True)

            rgb_paths = tuple(Path(row["rgb_path"]) for row in rows)
            raw_paths = tuple(Path(row["depth_aligned_rgb_mm_path"]) for row in rows)
            for path in (*rgb_paths, *raw_paths):
                if not path.is_file():
                    raise FileNotFoundError(path)

            rgb_relative = f"media/{slug}/{camera_id}/rgb.mp4"
            method_media: dict[str, str] = {}
            tasks.append(
                VideoTask(
                    dataset_dir.name,
                    slug,
                    camera_id,
                    "rgb",
                    "rgb",
                    rgb_paths,
                    viewer_root / rgb_relative,
                )
            )

            for method_id, _method_label, _family in methods:
                if method_id == "raw_aligned":
                    paths = raw_paths
                    task_kind = "depth"
                else:
                    task_kind = "precolored" if method_id in ATTENTION_METHODS else "depth"
                    paths = processed_paths(
                        processed_root,
                        dataset_dir.name,
                        camera_id,
                        method_id,
                        frame_count,
                        extension=".jpg" if method_id in ATTENTION_METHODS else ".png",
                    )
                relative = f"media/{slug}/{camera_id}/{method_id}.mp4"
                method_media[method_id] = relative
                tasks.append(
                    VideoTask(
                        dataset_dir.name,
                        slug,
                        camera_id,
                        method_id,
                        task_kind,
                        paths,
                        viewer_root / relative,
                    )
                )

            dataset_entry["media"][camera_id] = {
                "label": camera_label,
                "model": camera_model,
                "rgb": rgb_relative,
                "methods": method_media,
            }

        mosaic_rgb = f"media/{slug}/mosaic/rgb.mp4"
        mosaic_methods: dict[str, str] = {}
        mosaic_rgb_inputs = tuple(
            viewer_root / dataset_entry["media"][camera_id]["rgb"]
            for camera_id, _label, _model in CAMERAS
        )
        tasks.append(
            VideoTask(
                dataset_dir.name,
                slug,
                "mosaic",
                "rgb",
                "mosaic",
                mosaic_rgb_inputs,
                viewer_root / mosaic_rgb,
                dataset_entry["frameCounts"]["cam_h"],
            )
        )
        for method_id, _method_label, _family in METHODS:
            relative = f"media/{slug}/mosaic/{method_id}.mp4"
            mosaic_methods[method_id] = relative
            inputs = tuple(
                viewer_root / dataset_entry["media"][camera_id]["methods"][method_id]
                for camera_id, _label, _model in CAMERAS
            )
            tasks.append(
                VideoTask(
                    dataset_dir.name,
                    slug,
                    "mosaic",
                    method_id,
                    "mosaic",
                    inputs,
                    viewer_root / relative,
                    dataset_entry["frameCounts"]["cam_h"],
                )
            )
        dataset_entry["mosaic"] = {"rgb": mosaic_rgb, "methods": mosaic_methods}

        master_count = dataset_entry["frameCounts"]["cam_h"]
        dataset_entry["durationSeconds"] = master_count / args.fps
        datasets.append(dataset_entry)

    if not datasets:
        raise ValueError("No datasets matched the requested selection")
    return tasks, datasets


def ffmpeg_output_args(args: argparse.Namespace, output_path: Path) -> list[str]:
    return [
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        args.preset,
        "-crf",
        str(args.crf),
        "-threads",
        str(args.ffmpeg_threads),
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-g",
        str(max(1, round(args.fps))),
        "-keyint_min",
        str(max(1, round(args.fps))),
        "-sc_threshold",
        "0",
        "-movflags",
        "+faststart",
        "-y",
        str(output_path),
    ]


def encode_rgb(task: VideoTask, args: argparse.Namespace, temporary_path: Path) -> None:
    first_path = task.frame_paths[0]
    contiguous_pattern = first_path.parent / "%06d.jpg"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(args.fps),
        "-start_number",
        "0",
        "-i",
        str(contiguous_pattern),
        "-frames:v",
        str(len(task.frame_paths)),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        *ffmpeg_output_args(args, temporary_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"ffmpeg exited {result.returncode}")


def colorize_depth(depth_mm: np.ndarray, depth_min_m: float, depth_max_m: float) -> np.ndarray:
    if depth_mm.ndim != 2:
        raise ValueError(f"Expected single-channel depth, got shape {depth_mm.shape}")
    depth_m = depth_mm.astype(np.float32) * 0.001
    valid = np.isfinite(depth_m) & (depth_m > 0)
    scale = max(depth_max_m - depth_min_m, 1e-6)
    normalized = np.clip((depth_m - depth_min_m) / scale, 0.0, 1.0)
    color_index = np.round(normalized * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(color_index, cv2.COLORMAP_TURBO)
    color[~valid] = (14, 16, 17)
    return color


def encode_depth(task: VideoTask, args: argparse.Namespace, temporary_path: Path) -> None:
    first = cv2.imread(str(task.frame_paths[0]), cv2.IMREAD_UNCHANGED)
    if first is None:
        raise ValueError(f"Cannot read {task.frame_paths[0]}")
    height, width = first.shape[:2]
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(args.fps),
        "-i",
        "-",
        *ffmpeg_output_args(args, temporary_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for path in task.frame_paths:
            depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise ValueError(f"Cannot read {path}")
            if depth.shape[:2] != (height, width):
                raise ValueError(f"Unexpected size {depth.shape[:2]} in {path}")
            frame = colorize_depth(depth, args.depth_min_m, args.depth_max_m)
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
        if return_code:
            raise RuntimeError(stderr.strip() or f"ffmpeg exited {return_code}")
    except BaseException:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        process.kill()
        process.wait()
        raise


def encode_mosaic(task: VideoTask, args: argparse.Namespace, temporary_path: Path) -> None:
    if len(task.frame_paths) != 3 or task.frame_count is None:
        raise ValueError("A mosaic task requires three camera videos and a frame count")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    for path in task.frame_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        command.extend(["-i", str(path)])
    command.extend(
        [
            "-filter_complex",
            (
                "[0:v]setpts=PTS-STARTPTS[v0];"
                "[1:v]setpts=PTS-STARTPTS[v1];"
                "[2:v]setpts=PTS-STARTPTS[v2];"
                "[v0][v1][v2]hstack=inputs=3[v]"
            ),
            "-map",
            "[v]",
            "-frames:v",
            str(task.frame_count),
            *ffmpeg_output_args(args, temporary_path),
        ]
    )
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"ffmpeg exited {result.returncode}")


def probe_video(path: Path) -> dict[str, Any]:
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
        raise RuntimeError(result.stderr.strip() or f"ffprobe failed for {path}")
    payload = json.loads(result.stdout)
    stream = payload.get("streams", [{}])[0]
    if stream.get("codec_name") != "h264":
        raise ValueError(f"Unexpected codec in {path}: {stream.get('codec_name')}")
    return {"stream": stream, "format": payload.get("format", {})}


def encode_task(task: VideoTask, args: argparse.Namespace) -> dict[str, Any]:
    task.output_path.parent.mkdir(parents=True, exist_ok=True)
    if task.output_path.is_file() and not args.overwrite:
        try:
            return {"status": "reused", "probe": probe_video(task.output_path)}
        except Exception:
            task.output_path.unlink(missing_ok=True)

    temporary_path = task.output_path.with_name(task.output_path.stem + ".part.mp4")
    temporary_path.unlink(missing_ok=True)
    try:
        if task.kind in {"rgb", "precolored"}:
            encode_rgb(task, args, temporary_path)
        elif task.kind == "mosaic":
            encode_mosaic(task, args, temporary_path)
        else:
            encode_depth(task, args, temporary_path)
        probe = probe_video(temporary_path)
        temporary_path.replace(task.output_path)
        return {"status": "encoded", "probe": probe}
    finally:
        temporary_path.unlink(missing_ok=True)


def write_catalog(
    viewer_root: Path,
    datasets: list[dict[str, Any]],
    args: argparse.Namespace,
    methods: tuple[tuple[str, str, str], ...],
) -> None:
    catalog = {
        "version": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "fps": args.fps,
        "depthRangeM": [args.depth_min_m, args.depth_max_m],
        "cameras": [
            {"id": camera_id, "label": label, "model": model}
            for camera_id, label, model in CAMERAS
        ],
        "methods": [
            {
                "id": method_id,
                "label": label,
                "family": family,
                "visualization": "attention" if method_id in ATTENTION_METHODS else "depth",
            }
            for method_id, label, family in methods
        ],
        "datasets": datasets,
    }
    data_dir = viewer_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_task_phase(
    tasks: list[VideoTask],
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    failures: list[dict[str, str]],
    completed_offset: int,
    total_tasks: int,
) -> None:
    if not tasks:
        return
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        pending = {executor.submit(encode_task, task, args): task for task in tasks}
        for phase_completed, future in enumerate(as_completed(pending), start=1):
            completed = completed_offset + phase_completed
            task = pending[future]
            identity = f"{task.dataset_slug}/{task.camera_id}/{task.stream_id}"
            try:
                result = future.result()
                records.append({"video": identity, **result})
                print(
                    f"[{completed:03d}/{total_tasks:03d}] "
                    f"{result['status']:7s} {identity}",
                    flush=True,
                )
            except Exception as exc:
                failures.append({"video": identity, "error": str(exc)})
                print(
                    f"[{completed:03d}/{total_tasks:03d}] FAILED  {identity}: {exc}",
                    file=sys.stderr,
                )


def main() -> int:
    args = parse_args()
    if args.depth_max_m <= args.depth_min_m:
        raise ValueError("--depth-max-m must be greater than --depth-min-m")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg and ffprobe are required")

    args.project_root = args.project_root.resolve()
    viewer_root = (args.viewer_root or args.project_root / "viewer").resolve()
    args.viewer_root = viewer_root
    methods = METHODS + CDM_METHODS if args.include_cdm else METHODS
    tasks, datasets = collect_tasks(args, methods)
    print(f"Building {len(tasks)} videos for {len(datasets)} datasets with {args.workers} workers")

    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    component_tasks = [task for task in tasks if task.kind != "mosaic"]
    mosaic_tasks = [task for task in tasks if task.kind == "mosaic"]
    run_task_phase(component_tasks, args, records, failures, 0, len(tasks))
    if failures:
        print("Skipping mosaic phase because component videos failed", file=sys.stderr)
    else:
        run_task_phase(
            mosaic_tasks,
            args,
            records,
            failures,
            len(component_tasks),
            len(tasks),
        )

    write_catalog(viewer_root, datasets, args, methods)
    report = {
        "ok": not failures,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "videoCount": len(tasks),
        "encodedOrReused": len(records),
        "failed": failures,
        "videos": records,
    }
    (viewer_root / "data" / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if failures:
        print(f"Build completed with {len(failures)} failure(s)", file=sys.stderr)
        return 1
    print(f"Viewer media ready at {viewer_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
