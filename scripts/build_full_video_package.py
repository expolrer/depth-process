#!/usr/bin/env python3
"""Build compact full-sequence RGB/depth mosaics and a local viewer catalog."""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


METHODS = (
    ("raw_aligned", "Raw aligned depth", "Processing methods"),
    ("rgb_guided", "RGB guided", "Processing methods"),
    ("temporal_rgb_guided", "RGB + temporal", "Processing methods"),
    ("lingbot_v05", "LingBot-v0.5 depth", "Processing methods"),
    ("depth_anything_v2_fused", "Depth Anything V2 fused", "Processing methods"),
    ("lingbot_v05_sensor_fused", "LingBot sensor fused", "Processing methods"),
    ("ai_consensus_fused", "AI consensus fused", "Processing methods"),
    ("zero_depth", "Zero depth", "Negative controls"),
    ("spatially_shuffled_raw", "Spatially shuffled depth", "Negative controls"),
)
DEPTH_METHOD_IDS = tuple(method[0] for method in METHODS[:7])
CAMERAS = ("cam_h", "cam_l", "cam_r")


def probe(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,nb_read_frames,duration,sample_aspect_ratio",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        return None
    stream = json.loads(result.stdout)["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": int(stream.get("nb_read_frames") or 0),
        "duration": float(stream.get("duration") or 0.0),
        "sampleAspectRatio": stream.get("sample_aspect_ratio") or "N/A",
    }


def encode_mosaic(sources: list[Path], output: Path, expected_frames: int, crf: int) -> dict[str, Any]:
    existing = probe(output)
    if (
        existing
        and existing["frames"] == expected_frames
        and existing["width"] == 960
        and existing["sampleAspectRatio"] == "1:1"
    ):
        return {"path": str(output), "status": "reused", **existing}
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".part.mp4")
    temporary.unlink(missing_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    for source in sources:
        command.extend(["-i", str(source)])
    command.extend(
        [
            "-filter_complex",
            "[0:v]scale=320:240:flags=lanczos,setsar=1[a];"
            "[1:v]scale=320:240:flags=lanczos,setsar=1[b];"
            "[2:v]scale=320:240:flags=lanczos,setsar=1[c];"
            "[a][b][c]hstack=inputs=3[v]",
            "-map",
            "[v]",
            "-frames:v",
            str(expected_frames),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-y",
            str(temporary),
        ]
    )
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(result.stderr.strip() or f"ffmpeg exited {result.returncode}")
    temporary.replace(output)
    details = probe(output)
    if not details or details["frames"] != expected_frames:
        raise RuntimeError(f"Invalid output {output}: {details}")
    return {"path": str(output), "status": "encoded", **details}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-viewer", type=Path, required=True)
    parser.add_argument("--benchmark-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--crf", type=int, default=20)
    args = parser.parse_args()

    source_catalog = json.loads((args.source_viewer / "data" / "catalog.json").read_text())
    benchmark = json.loads(args.benchmark_report.read_text())
    result_by_method = {result["method"]: result for result in benchmark["results"]}
    tasks = []
    catalog: dict[str, Any] = {
        "version": 1,
        "fps": float(source_catalog["fps"]),
        "depthRangeM": source_catalog["depthRangeM"],
        "model": "Depth-only ACT checkpoint 009000",
        "promptUsed": False,
        "attention": "Last decoder Action Query to depth tokens",
        "cameras": source_catalog["cameras"],
        "methods": [],
        "datasets": [],
    }
    for method, label, family in METHODS:
        metrics = result_by_method[method]["metrics"]
        catalog["methods"].append(
            {
                "id": method,
                "label": label,
                "family": family,
                "globalMaeRad": metrics["chunk_mae_rad"],
                "globalFirstStepMaeRad": metrics["first_step_mae_rad"],
            }
        )

    for dataset in source_catalog["datasets"]:
        slug = dataset["id"]
        frame_count = int(dataset["frameCounts"]["cam_h"])
        rgb_sources = [args.source_viewer / dataset["media"][camera]["rgb"] for camera in CAMERAS]
        rgb_output = args.output_dir / "media" / slug / "rgb.mp4"
        tasks.append((rgb_sources, rgb_output, frame_count, args.crf))
        depth: dict[str, str] = {}
        attention: dict[str, str] = {}
        for method in DEPTH_METHOD_IDS:
            sources = [
                args.source_viewer / dataset["media"][camera]["methods"][method]
                for camera in CAMERAS
            ]
            output = args.output_dir / "media" / slug / "depth" / f"{method}.mp4"
            tasks.append((sources, output, frame_count, args.crf))
            depth[method] = output.relative_to(args.output_dir).as_posix()
        for method, _label, _family in METHODS:
            output = args.output_dir / "media" / slug / "attention" / f"{method}.mp4"
            attention[method] = output.relative_to(args.output_dir).as_posix()
        sequence = dataset["key"]
        catalog["datasets"].append(
            {
                "id": slug,
                "sequence": sequence,
                "label": dataset["label"],
                "frameCount": frame_count,
                "durationSeconds": frame_count / float(source_catalog["fps"]),
                "rgb": rgb_output.relative_to(args.output_dir).as_posix(),
                "depth": depth,
                "attention": attention,
                "metrics": {
                    method: result_by_method[method]["sequences"][sequence]
                    for method, _label, _family in METHODS
                },
            }
        )

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(encode_mosaic, *task): task[1] for task in tasks}
        for index, future in enumerate(as_completed(futures), 1):
            output = futures[future]
            try:
                result = future.result()
                print(f"[{index}/{len(tasks)}] {result['status']} {output}", flush=True)
            except Exception as error:
                failures.append(f"{output}: {error}")
                print(f"[{index}/{len(tasks)}] FAILED {output}: {error}", flush=True)

    data_dir = args.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "catalog.json").write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if failures:
        raise RuntimeError("\n".join(failures))
    print(f"Built {len(tasks)} RGB/depth mosaics in {args.output_dir}")


if __name__ == "__main__":
    main()
