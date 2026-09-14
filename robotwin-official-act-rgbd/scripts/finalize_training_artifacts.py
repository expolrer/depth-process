#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def last_metric(path: Path) -> dict[str, object]:
    last = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                last = json.loads(line)
    if last is None:
        raise SystemExit(f"no metrics found in {path}")
    return last


def main() -> None:
    parser = argparse.ArgumentParser(description="Keep only the best evaluation checkpoint after training")
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-epochs", type=int, required=True)
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=Path("/ssd/hhw/depth-model/experiments/OfficialACTRGBD"),
    )
    args = parser.parse_args()

    output = args.output.resolve()
    experiments_root = args.experiments_root.resolve()
    if output == experiments_root or experiments_root not in output.parents:
        raise SystemExit(f"refusing cleanup outside {experiments_root}: {output}")

    required = ("policy_best.ckpt", "dataset_stats.pkl", "config.json", "metrics.jsonl")
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise SystemExit(f"cannot finalize incomplete artifacts; missing {missing}")
    metric = last_metric(output / "metrics.jsonl")
    epoch = int(metric.get("epoch", 0))
    if epoch < args.expected_epochs:
        raise SystemExit(f"training is incomplete: epoch {epoch} < {args.expected_epochs}")

    removed = []
    candidates = list(output.glob("policy_epoch_*.ckpt"))
    candidates.extend(output / name for name in ("policy_last.ckpt", "training_last.pt"))
    for path in candidates:
        if path.is_file():
            removed.append({"name": path.name, "bytes": path.stat().st_size})
            path.unlink()

    completion = {
        "status": "complete",
        "epoch": epoch,
        "evaluation_checkpoint": "policy_best.ckpt",
        "checkpoint_policy": "best_only_after_completion",
        "removed_files": len(removed),
        "removed_bytes": sum(item["bytes"] for item in removed),
    }
    (output / "training_complete.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(completion))


if __name__ == "__main__":
    main()
