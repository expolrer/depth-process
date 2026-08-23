from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import prepare_session
from .io import read_json, read_jsonl
from .pipeline import default_engine_root, run_pipeline
from .server import serve
from .task import load_task


def _common_prepare(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="ROS bag, bag directory, LeRobot root, or extracted RGB-D root",
    )
    parser.add_argument(
        "--format", choices=("auto", "rosbag", "lerobot", "extracted"), default="auto"
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--task", type=Path)
    parser.add_argument("--engine-root", type=Path, default=default_engine_root())


def _pipeline_options(args: argparse.Namespace) -> dict:
    return {
        "engine_root": args.engine_root,
        "grounding_model": args.grounding_model,
        "sam2_checkpoint": args.sam2_checkpoint,
        "sam2_config": args.sam2_config,
        "vlm_model": args.vlm_model,
        "detector_backend": args.detector_backend,
        "device": args.device,
        "skip_vlm": args.skip_vlm,
        "skip_tracking": args.skip_tracking,
    }


def _model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine-root", type=Path, default=default_engine_root())
    parser.add_argument("--grounding-model", type=Path)
    parser.add_argument("--sam2-checkpoint", type=Path)
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--vlm-model", type=Path)
    parser.add_argument(
        "--detector-backend", choices=("groundingdino", "sam3"), default="groundingdino"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-vlm", action="store_true")
    parser.add_argument("--skip-tracking", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="interaction-labeler")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="Index a dataset and create an annotation session"
    )
    _common_prepare(prepare)

    pipeline = subparsers.add_parser("pipeline", help="Run detection, ranking, VLM and SAM2")
    pipeline.add_argument("--workspace", type=Path, required=True)
    _model_arguments(pipeline)

    serve_parser = subparsers.add_parser("serve", help="Open an existing session in the review UI")
    serve_parser.add_argument("--workspace", type=Path, required=True)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8769)
    serve_parser.add_argument("--no-open", action="store_true")
    serve_parser.add_argument("--auto-start", action="store_true")
    _model_arguments(serve_parser)

    run = subparsers.add_parser("run", help="Prepare data and launch the interactive UI")
    _common_prepare(run)
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=8769)
    run.add_argument("--no-open", action="store_true")
    run.add_argument("--auto-start", action="store_true")
    run.add_argument("--grounding-model", type=Path)
    run.add_argument("--sam2-checkpoint", type=Path)
    run.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    run.add_argument("--vlm-model", type=Path)
    run.add_argument(
        "--detector-backend", choices=("groundingdino", "sam3"), default="groundingdino"
    )
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--skip-vlm", action="store_true")
    run.add_argument("--skip-tracking", action="store_true")

    export = subparsers.add_parser(
        "export", help="Export session, detections, tracks and overrides"
    )
    export.add_argument("--workspace", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command in {"prepare", "run"}:
        task = load_task(args.task)
        session = prepare_session(
            args.data,
            args.workspace,
            task,
            input_format=args.format,
            engine_root=args.engine_root,
        )
        print(
            json.dumps(
                {"events": len(session["events"]), "workspace": str(args.workspace)},
                ensure_ascii=False,
            )
        )
        if args.command == "prepare":
            return
    if args.command == "pipeline":
        run_pipeline(args.workspace, **_pipeline_options(args))
    elif args.command in {"serve", "run"}:
        serve(
            args.workspace,
            host=args.host,
            port=args.port,
            open_browser=not args.no_open,
            auto_start=args.auto_start,
            pipeline_options=_pipeline_options(args),
        )
    elif args.command == "export":
        workspace = args.workspace.resolve()
        payload = {
            "schema": "robot_interaction_annotation_export_v1",
            "session": read_json(workspace / "session.json", {}),
            "manual_annotations": read_json(workspace / "annotations.json", {}),
            "ranked_candidates": read_jsonl(
                workspace / "outputs" / "interaction_candidates" / "ranked_index.jsonl"
            ),
            "target_tracks": read_jsonl(
                workspace / "outputs" / "target_tracks_required" / "track_index.jsonl"
            ),
            "v2_instance_evidence": read_jsonl(workspace / "v2" / "instance_evidence.jsonl"),
            "v2_review_queue": read_jsonl(workspace / "v2" / "review_queue.jsonl"),
            "pipeline_status": read_json(workspace / "pipeline_status.json", {}),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(args.output.resolve())


if __name__ == "__main__":
    main()
