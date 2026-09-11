#!/usr/bin/env python3
"""Export a compact, reproducible snapshot of the RoboTwin RGB-D benchmark."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path


TASKS = (
    "pick_dual_bottles",
    "stack_blocks_two",
    "handover_mic",
    "place_a2b_left",
    "place_a2b_right",
    "pick_diverse_bottles",
)

VARIANTS = {
    "A0": "RGB + joint ACT baseline",
    "A1": "RGB-D four-channel early concatenation",
    "A2": "RGB ResNet + depth CNN + joint, token-level late fusion",
    "A3": "RGB encoder + LingBot-Depth/ViT depth tokens",
    "A4": "Fused point cloud + DP3",
    "A5": "RGB tokens + depth tokens + joint-conditioned action DiT",
}

SCORE_RE = re.compile(r"success_rate=([^\s]+)")


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def classify(status: str, checkpoint_exists: bool) -> str:
    if " COMPLETE " in f" {status} ":
        return "evaluated"
    if "EVALUATING" in status:
        return "evaluating"
    if "TRAINING" in status:
        return "training"
    if checkpoint_exists or "TRAINED" in status:
        return "trained_eval_pending"
    if "FAILED" in status:
        return "failed_retry_pending"
    return "pending"


def parse_score(status: str) -> float | None:
    match = SCORE_RE.search(status)
    if not match:
        return None
    value = match.group(1).rstrip("%")
    try:
        score = float(value)
    except ValueError:
        return None
    return score / 100.0 if score > 1 else score


def build_snapshot(root: Path) -> dict:
    rows = []
    for variant in VARIANTS:
        for task in TASKS:
            run = root / "experiments" / variant / task / "seed0"
            status = read_text(run / "status.txt")
            checkpoint = run / "policy_step_030000.pt"
            if variant == "A4":
                checkpoint_exists = any(run.glob("**/*.ckpt")) or "COMPLETE" in status
            else:
                checkpoint_exists = checkpoint.is_file() and checkpoint.stat().st_size > 0
            rows.append(
                {
                    "variant": variant,
                    "task": task,
                    "stage": classify(status, checkpoint_exists),
                    "success_rate": parse_score(status),
                    "checkpoint_available": checkpoint_exists,
                    "status": status or "not_started",
                }
            )

    counts = {}
    for row in rows:
        counts[row["stage"]] = counts.get(row["stage"], 0) + 1

    scheduler = root / "scheduler" / "continuous_v2"
    workers = {
        f"gpu{gpu}": read_text(scheduler / f"gpu{gpu}.status")
        for gpu in (4, 5, 6)
    }
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "root": str(root),
        "resource_policy": {
            "project_gpus": [4, 5, 6],
            "excluded_gpus": [0, 1, 2, 3, 7],
            "sapien_gpu_isolation": True,
        },
        "variants": VARIANTS,
        "tasks": list(TASKS),
        "summary": {"total": len(rows), **counts},
        "workers": workers,
        "results": rows,
    }


def markdown(snapshot: dict) -> str:
    result_by_key = {
        (row["variant"], row["task"]): row for row in snapshot["results"]
    }
    summary = snapshot["summary"]
    lines = [
        "# RoboTwin RGB-D Benchmark Status",
        "",
        f"Snapshot: `{snapshot['generated_at']}`",
        "",
        "This is a live experiment snapshot. Success rates are published only after a full 100-rollout evaluation completes.",
        "",
        "## Progress",
        "",
        f"- Fully evaluated: **{summary.get('evaluated', 0)}/{summary['total']}**",
        f"- Trained, evaluation pending: **{summary.get('trained_eval_pending', 0)}**",
        f"- Training now: **{summary.get('training', 0)}**",
        f"- Evaluation now: **{summary.get('evaluating', 0)}**",
        f"- Pending: **{summary.get('pending', 0)}**",
        "- Project scheduler GPUs: **4, 5, 6 only**",
        "",
        "## Success Rate",
        "",
        "| Variant | " + " | ".join(TASKS) + " |",
        "|---|" + "---:|" * len(TASKS),
    ]
    for variant in VARIANTS:
        values = []
        for task in TASKS:
            row = result_by_key[(variant, task)]
            score = row["success_rate"]
            values.append(f"{score:.0%}" if score is not None else "-")
        lines.append(f"| {variant} | " + " | ".join(values) + " |")

    lines.extend(["", "## Architectures", ""])
    for variant, description in VARIANTS.items():
        lines.append(f"- **{variant}**: {description}")

    lines.extend(["", "## Active Workers", ""])
    for gpu, status in snapshot["workers"].items():
        lines.append(f"- **{gpu.upper()}**: `{status or 'status unavailable'}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A0 is the completed RGB-only control. A4 currently has the strongest completed depth-enabled results, but cross-architecture conclusions must wait until A1-A5 use the same task seeds and all 100-rollout evaluations finish.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/ssd/depth-model"))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    snapshot = build_snapshot(args.root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(markdown(snapshot), encoding="utf-8")


if __name__ == "__main__":
    main()
