#!/usr/bin/env python3
"""Validate all full-video catalog references and write a SHA256 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def probe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,nb_read_frames,avg_frame_rate,duration,sample_aspect_ratio",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"ffprobe failed for {path}")
    stream = json.loads(result.stdout)["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": int(stream["nb_read_frames"]),
        "fps": stream["avg_frame_rate"],
        "duration": float(stream["duration"]),
        "sampleAspectRatio": stream.get("sample_aspect_ratio") or "N/A",
    }


def digest(path: Path) -> tuple[Path, str]:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return path, value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    catalog_path = args.root / "data" / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    records = []
    failures = []
    paths: list[Path] = []
    for dataset in catalog["datasets"]:
        expected = int(dataset["frameCount"])
        relatives = [
            dataset["rgb"],
            *dataset["depth"].values(),
            *dataset["attention"].values(),
        ]
        for relative in relatives:
            path = args.root / relative
            try:
                details = probe(path)
                if details["width"] != 960 or details["height"] != 240:
                    raise RuntimeError(f"unexpected dimensions {details['width']}x{details['height']}")
                if details["frames"] != expected:
                    raise RuntimeError(f"expected {expected} frames, got {details['frames']}")
                if details["sampleAspectRatio"] not in {"1:1", "N/A"}:
                    raise RuntimeError(
                        f"expected square pixels, got SAR {details['sampleAspectRatio']}"
                    )
                records.append({"path": relative, **details, "bytes": path.stat().st_size})
                paths.append(path)
            except Exception as error:
                failures.append(f"{relative}: {error}")

    report = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "ok": not failures,
        "datasets": len(catalog["datasets"]),
        "videos": len(records),
        "bytes": sum(record["bytes"] for record in records),
        "failures": failures,
        "records": records,
    }
    report_path = args.root / "data" / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError("\n".join(failures))

    manifest_paths = [catalog_path, report_path, *paths]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        digests = list(executor.map(digest, manifest_paths))
    manifest = args.root / "manifest.sha256"
    manifest.write_text(
        "".join(
            f"{value}  {path.relative_to(args.root).as_posix()}\n"
            for path, value in sorted(digests, key=lambda item: item[0].as_posix())
        ),
        encoding="ascii",
    )
    print(
        f"Validated {len(records)} videos, {report['bytes'] / 1024**2:.1f} MiB; "
        f"wrote {manifest}"
    )


if __name__ == "__main__":
    main()
