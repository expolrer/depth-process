#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/ssd/hhw/depth-model").resolve()
OLD_EXPERIMENTS = ("RGBDACTv2", "A0", "A1", "A2", "A4", "A5", "smoke", "aux_10h")
WEIGHT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}


def validated(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == ROOT or ROOT not in resolved.parents:
        raise SystemExit(f"refusing path outside project root: {resolved}")
    return resolved


def main() -> None:
    targets = [validated(ROOT / "experiments" / name) for name in OLD_EXPERIMENTS]
    official = validated(ROOT / "experiments/OfficialACTRGBD")
    targets.extend(validated(path) for path in official.glob("*/stack_blocks_two/clean500_seed0"))
    smoke = validated(official / "_smoke")
    if smoke.exists():
        targets.append(smoke)

    target_strings = tuple(str(target) for target in targets if target.exists())
    active = []
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            command = cmdline.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(target in command for target in target_strings):
            active.append({"pid": int(cmdline.parent.name), "command": command})
    if active:
        raise SystemExit(f"refusing to delete weights referenced by active processes: {active}")

    removed_files = 0
    removed_bytes = 0
    by_root = {}
    for target in targets:
        if not target.exists():
            continue
        count = 0
        size = 0
        for directory, _, names in os.walk(target):
            for name in names:
                path = Path(directory) / name
                if path.suffix.lower() not in WEIGHT_SUFFIXES:
                    continue
                bytes_value = path.stat().st_size
                path.unlink()
                count += 1
                size += bytes_value
        by_root[str(target)] = {"files": count, "bytes": size}
        removed_files += count
        removed_bytes += size

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": "delete_deprecated_weight_files_keep_logs_metrics_and_configs",
        "removed_files": removed_files,
        "removed_bytes": removed_bytes,
        "roots": by_root,
    }
    report_dir = ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "deprecated_weight_cleanup_official6000.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
