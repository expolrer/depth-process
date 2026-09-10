from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATASET_MANIFEST_SCHEMA = "embodied_dataset_version_v1"


@dataclass(frozen=True)
class DatasetVersionManifest:
    dataset_id: str
    version: str
    root: str
    layout: str
    created_at: str
    content_id: str
    parent_version: str | None = None
    sources: tuple[str, ...] = ()
    annotation_versions: dict[str, str] = field(default_factory=dict)
    files: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    schema: str = DATASET_MANIFEST_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_lerobot_layout(root: Path) -> str:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return "unknown"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    codebase_version = str(info.get("codebase_version", info.get("version", "")))
    if (root / "meta" / "episodes").is_dir() and any((root / "data").rglob("*.parquet")):
        return "lerobot_v3"
    if "v3" in codebase_version.casefold() or codebase_version.startswith("3"):
        return "lerobot_v3"
    if any((root / "data").rglob("episode_*.parquet")) or any(
        (root / "data").rglob("episode-*.parquet")
    ):
        return "lerobot_v2"
    return "lerobot"


def _sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_fingerprint(path: Path, root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": path.relative_to(root).as_posix(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def inventory_files(
    root: Path,
    hash_mode: str = "metadata",
    suffixes: Iterable[str] = (".json", ".jsonl", ".parquet", ".mp4", ".png"),
) -> list[dict[str, Any]]:
    if hash_mode not in {"metadata", "full"}:
        raise ValueError("hash_mode must be metadata or full")
    allowed = {suffix.casefold() for suffix in suffixes}
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if allowed and path.suffix.casefold() not in allowed:
            continue
        row = _metadata_fingerprint(path, root)
        if hash_mode == "full":
            row["sha256"] = _sha256(path)
        rows.append(row)
    return rows


def build_dataset_version(
    root: Path,
    dataset_id: str,
    version: str,
    output: Path,
    parent_version: str | None = None,
    sources: Iterable[str] = (),
    annotation_versions: dict[str, str] | None = None,
    hash_mode: str = "metadata",
) -> DatasetVersionManifest:
    root = root.resolve()
    files = inventory_files(root, hash_mode=hash_mode)
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    content_id = hashlib.sha256(canonical).hexdigest()
    manifest = DatasetVersionManifest(
        dataset_id=dataset_id,
        version=version,
        root=str(root),
        layout=detect_lerobot_layout(root),
        created_at=datetime.now(timezone.utc).isoformat(),
        content_id=content_id,
        parent_version=parent_version,
        sources=tuple(sources),
        annotation_versions=annotation_versions or {},
        files=tuple(files),
        metadata={"hash_mode": hash_mode, "file_count": len(files)},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return manifest


def validate_lerobot_v3(root: Path) -> dict[str, Any]:
    root = root.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    required = [root / "meta" / "info.json", root / "data"]
    for path in required:
        if not path.exists():
            errors.append(f"missing required path: {path}")
    info: dict[str, Any] = {}
    if (root / "meta" / "info.json").exists():
        try:
            info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            errors.append(f"invalid meta/info.json: {error}")
    parquet_files = sorted((root / "data").rglob("*.parquet")) if (root / "data").exists() else []
    video_files = sorted((root / "videos").rglob("*.mp4")) if (root / "videos").exists() else []
    episode_meta = (
        sorted((root / "meta" / "episodes").rglob("*.parquet"))
        if (root / "meta" / "episodes").exists()
        else []
    )
    if not parquet_files:
        errors.append("no data parquet shards found")
    if not episode_meta:
        warnings.append("no chunked meta/episodes parquet files found")
    features = info.get("features", {})
    video_features = [name for name, value in features.items() if value.get("dtype") == "video"]
    if video_features and not video_files:
        errors.append("video features are declared but no MP4 shards were found")
    if detect_lerobot_layout(root) != "lerobot_v3":
        errors.append("dataset does not look like LeRobot v3")
    return {
        "schema": "lerobot_v3_validation_v1",
        "root": str(root),
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "data_shards": len(parquet_files),
        "video_shards": len(video_files),
        "episode_metadata_shards": len(episode_meta),
        "declared_video_features": video_features,
    }


def official_v3_conversion_command(
    repo_id: str,
    python: Path | None = None,
    module: str = "lerobot.scripts.convert_dataset_v21_to_v30",
    extra_args: Iterable[str] = (),
) -> list[str]:
    if not repo_id.strip():
        raise ValueError("repo_id is required")
    return [
        str(python or Path(sys.executable)),
        "-m",
        module,
        "--repo-id",
        repo_id,
        *list(extra_args),
    ]


def run_official_v3_conversion(
    repo_id: str,
    python: Path | None = None,
    module: str = "lerobot.scripts.convert_dataset_v21_to_v30",
    extra_args: Iterable[str] = (),
    dry_run: bool = False,
) -> dict[str, Any]:
    command = official_v3_conversion_command(repo_id, python, module, extra_args)
    if dry_run:
        return {"command": command, "returncode": None, "dry_run": True}
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    result = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "dry_run": False,
    }
    if completed.returncode:
        raise RuntimeError(json.dumps(result, ensure_ascii=False))
    return result
