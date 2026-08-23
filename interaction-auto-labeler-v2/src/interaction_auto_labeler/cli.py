from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import initialize_v2, run_v2_pipeline
from .plan import build_plan
from .schema import EvidenceWeights, load_v2_task
from .scoring import rank_candidates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="auto-labeler-v2")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run", help="Prepare a dataset and launch the V2 interactive labeler"
    )
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--format", choices=("auto", "rosbag", "lerobot", "extracted"), default="auto")
    run.add_argument("--workspace", type=Path, required=True)
    run.add_argument("--task", type=Path, required=True)
    run.add_argument("--engine-root", type=Path)
    run.add_argument(
        "--concept-model", type=Path, help="GroundingDINO directory or SAM3 checkpoint"
    )
    run.add_argument("--sam2-checkpoint", type=Path)
    run.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    run.add_argument("--vlm-model", type=Path)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8771)
    run.add_argument("--no-open", action="store_true")
    run.add_argument("--auto-start", action="store_true")
    run.add_argument("--skip-vlm", action="store_true")
    run.add_argument("--skip-tracking", action="store_true")
    run.add_argument("--skip-inventory", action="store_true")

    plan = subparsers.add_parser("plan", help="Build the deterministic V2 stage plan")
    plan.add_argument("--workspace", type=Path, required=True)
    plan.add_argument("--task", type=Path, required=True)

    score = subparsers.add_parser("score", help="Rank precomputed candidate evidence")
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--task", type=Path, required=True)

    project = subparsers.add_parser(
        "project", help="Project one RGB-D target box into another calibrated camera"
    )
    project.add_argument("--depth", type=Path, required=True)
    project.add_argument("--box", required=True, help="x1,y1,x2,y2 in source RGB pixels")
    project.add_argument("--source-intrinsics", type=Path, required=True)
    project.add_argument("--target-from-source", type=Path, required=True)
    project.add_argument("--target-intrinsics", type=Path, required=True)
    project.add_argument("--target-height", type=int, required=True)
    project.add_argument("--target-width", type=int, required=True)
    project.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "project":
        import numpy as np
        from PIL import Image

        from .cross_view import backproject_depth, project_points, transform_points

        def matrix(path: Path) -> np.ndarray:
            return np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=np.float64)

        depth = np.asarray(Image.open(args.depth))
        box = [float(value) for value in args.box.split(",")]
        if len(box) != 4:
            raise SystemExit("--box requires four comma-separated values")
        points = backproject_depth(depth, matrix(args.source_intrinsics), box)
        transformed = transform_points(points, matrix(args.target_from_source))
        pixels, projected_box = project_points(
            transformed,
            matrix(args.target_intrinsics),
            (args.target_height, args.target_width),
        )
        payload = {
            "schema": "calibrated_cross_view_projection_v1",
            "source_box_xyxy": box,
            "source_points": len(points),
            "projected_points": len(pixels),
            "projected_box_xyxy": projected_box,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(args.output.resolve())
        return
    task = load_v2_task(args.task)
    if args.command == "score":
        candidates = json.loads(args.input.read_text(encoding="utf-8"))
        weights = EvidenceWeights(**task["v2"]["weights"])
        result = rank_candidates(candidates, weights, float(task["v2"]["ambiguity_margin"]))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(args.output.resolve())
        return
    if args.command == "plan":
        session = json.loads((args.workspace / "session.json").read_text(encoding="utf-8"))
        plan = build_plan(session, task, args.workspace)
        output = args.workspace / "v2" / "plan.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(output.resolve())
        return

    try:
        from interaction_labeler.dataset import prepare_session
        from interaction_labeler.pipeline import default_engine_root
        from interaction_labeler.server import serve
        from interaction_labeler.task import load_task
    except ImportError as error:
        raise SystemExit(
            "V2 requires the sibling interaction-labeler-v1 package. Install both projects with uv pip install -e."
        ) from error

    engine_root = (args.engine_root or default_engine_root()).resolve()
    v1_task = load_task(args.task)
    v1_task["v2"] = task["v2"]
    session = prepare_session(
        args.data,
        args.workspace,
        v1_task,
        input_format=args.format,
        engine_root=engine_root,
    )
    initialize_v2(args.workspace, {**session["task"], "v2": task["v2"]})
    detector_backend = "sam3" if task["v2"]["detector_backend"] == "sam3" else "groundingdino"
    pipeline_options = {
        "v2_task_path": args.task.resolve(),
        "run_inventory": not args.skip_inventory,
        "detector_backend": detector_backend,
        "grounding_model": args.concept_model,
        "sam2_checkpoint": args.sam2_checkpoint,
        "sam2_config": args.sam2_config,
        "vlm_model": args.vlm_model,
        "device": args.device,
        "engine_root": engine_root,
        "skip_vlm": args.skip_vlm,
        "skip_tracking": args.skip_tracking,
    }
    print(
        json.dumps(
            {
                "events": len(session["events"]),
                "backend": task["v2"]["detector_backend"],
                "workspace": str(args.workspace.resolve()),
            },
            ensure_ascii=False,
        )
    )
    serve(
        args.workspace,
        host=args.host,
        port=args.port,
        open_browser=not args.no_open,
        auto_start=args.auto_start,
        pipeline_options=pipeline_options,
        pipeline_runner=run_v2_pipeline,
    )


if __name__ == "__main__":
    main()
