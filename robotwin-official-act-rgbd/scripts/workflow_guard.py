#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Reject jobs outside the committed experiment stage")
    parser.add_argument("action", choices=("train", "eval"))
    parser.add_argument("variant")
    parser.add_argument("task")
    parser.add_argument("--plan", type=Path, default=Path(__file__).resolve().parents[1] / "execution_plan.json")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    current_id = plan["current_stage"]
    current = next(stage for stage in plan["stages"] if stage["id"] == current_id)
    errors = []
    if current["status"] != "active":
        errors.append(f"current stage {current_id} is not active")
    if args.action not in current["actions"]:
        errors.append(f"action {args.action} is not allowed in {current_id}")
    if current["variants"] and args.variant not in current["variants"]:
        errors.append(f"variant {args.variant} is not allowed in {current_id}: {current['variants']}")
    if current["tasks"] and args.task not in current["tasks"]:
        errors.append(f"task {args.task} is not allowed in {current_id}: {current['tasks']}")
    if errors:
        raise SystemExit("WORKFLOW GUARD REJECTED: " + "; ".join(errors))
    print(f"WORKFLOW GUARD PASSED stage={current_id} action={args.action} variant={args.variant} task={args.task}")


if __name__ == "__main__":
    main()

