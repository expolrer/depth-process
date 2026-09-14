#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REQUIRED = ("policy_best.ckpt", "policy_last.ckpt", "dataset_stats.pkl", "config.json", "metrics.jsonl")
MANIFEST = "artifact_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        marker = repo / ".source_commit"
        if marker.is_file():
            value = marker.read_text(encoding="utf-8").strip()
            if len(value) == 40 and all(character in "0123456789abcdef" for character in value.lower()):
                return value
        raise SystemExit(
            f"cannot determine source commit for {repo}; deploy from Git or create {marker} with the deployed commit"
        )


def create(directory: Path, repo: Path) -> None:
    missing = [name for name in REQUIRED if not (directory / name).is_file()]
    if missing:
        raise SystemExit(f"missing required artifacts: {missing}")
    files = {
        name: {"bytes": (directory / name).stat().st_size, "sha256": sha256(directory / name)}
        for name in REQUIRED
    }
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit(repo),
        "files": files,
    }
    (directory / MANIFEST).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"created {directory / MANIFEST}")


def verify(directory: Path) -> None:
    path = directory / MANIFEST
    if not path.is_file():
        raise SystemExit(f"missing manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    for name in REQUIRED:
        file_path = directory / name
        expected = payload.get("files", {}).get(name)
        if not file_path.is_file() or expected is None:
            errors.append(f"missing {name}")
            continue
        actual_size = file_path.stat().st_size
        actual_hash = sha256(file_path)
        if actual_size != expected["bytes"]:
            errors.append(f"{name}: size {actual_size} != {expected['bytes']}")
        if actual_hash != expected["sha256"]:
            errors.append(f"{name}: sha256 mismatch")
    if errors:
        raise SystemExit("ARTIFACT VERIFY FAILED: " + "; ".join(errors))
    print(f"ARTIFACT VERIFY PASSED source_commit={payload.get('source_commit', 'unknown')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("create", "verify"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if args.command == "create":
        create(args.directory, args.repo)
    else:
        verify(args.directory)


if __name__ == "__main__":
    main()
