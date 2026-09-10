from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .io import atomic_write_json


def shard_for_episode(episode_id: str, shard_count: int) -> int:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    digest = hashlib.blake2b(str(episode_id).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % shard_count


class _HandlePool:
    def __init__(self, root: Path, maximum_open: int = 64) -> None:
        self.root = root
        self.maximum_open = maximum_open
        self.handles: OrderedDict[int, Any] = OrderedDict()

    def write(self, shard: int, text: str) -> None:
        handle = self.handles.pop(shard, None)
        if handle is None:
            path = self.root / f"part-{shard:05d}.jsonl"
            handle = path.open("a", encoding="utf-8")
        self.handles[shard] = handle
        handle.write(text)
        if len(self.handles) > self.maximum_open:
            _, oldest = self.handles.popitem(last=False)
            oldest.close()

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


def write_episode_shards(
    rows: Iterable[dict[str, Any]],
    output_root: Path,
    shard_count: int = 4096,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    existing = list(output_root.glob("part-*.jsonl"))
    if existing:
        raise FileExistsError("output_root already contains shard files")
    counts = [0] * shard_count
    pool = _HandlePool(output_root)
    total = 0
    try:
        for row in rows:
            episode_id = str(row.get("episode_id", row.get("episode_index", "")))
            if not episode_id:
                raise ValueError("every shard row requires episode_id or episode_index")
            shard = shard_for_episode(episode_id, shard_count)
            pool.write(shard, json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts[shard] += 1
            total += 1
    finally:
        pool.close()
    manifest = {
        "schema": "embodied_episode_shards_v1",
        "shard_count": shard_count,
        "record_count": total,
        "non_empty_shards": sum(value > 0 for value in counts),
        "counts": counts,
        "assignment": "blake2b_64(episode_id) modulo shard_count",
    }
    atomic_write_json(output_root / "manifest.json", manifest)
    return manifest


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSONL at line {line_number}: {error}") from error
