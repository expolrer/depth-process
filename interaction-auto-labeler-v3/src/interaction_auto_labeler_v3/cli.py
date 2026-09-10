from __future__ import annotations

import argparse
import json
import os
import shlex
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .dataset_lifecycle import (
    build_dataset_version,
    run_official_v3_conversion,
    validate_lerobot_v3,
)
from .event_graph import EpisodeEventGraph, migrate_v2_workspace
from .gold import stratified_gold_sample, write_gold_manifest
from .io import atomic_write_json, read_jsonl


def _print(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _add_run_arguments(parser: argparse.ArgumentParser, include_data: bool = True) -> None:
    if include_data:
        parser.add_argument("--data", type=Path, required=True)
        parser.add_argument(
            "--format", choices=("auto", "rosbag", "lerobot", "extracted"), default="auto"
        )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=include_data)
    parser.add_argument("--engine-root", type=Path)
    parser.add_argument("--concept-model", type=Path)
    parser.add_argument("--sam2-checkpoint", type=Path)
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--vlm-model", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8773)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument("--skip-vlm", action="store_true")
    parser.add_argument("--skip-tracking", action="store_true")
    parser.add_argument("--skip-inventory", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auto-labeler-v3",
        description="Scalable multimodal robot-data annotation and quality pipeline",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Prepare data and launch the V3 interactive system")
    _add_run_arguments(run)
    serve = commands.add_parser("serve", help="Reopen an existing V3 review workspace")
    _add_run_arguments(serve, include_data=False)

    process = commands.add_parser(
        "process-v3", help="Build V3 outputs from a completed V2 workspace"
    )
    process.add_argument("--workspace", type=Path, required=True)
    process.add_argument("--task", type=Path, required=True)

    migrate = commands.add_parser("migrate-v2", help="Convert a reviewed V2 workspace")
    migrate.add_argument("--workspace", type=Path, required=True)
    migrate.add_argument("--output", type=Path, required=True)
    migrate.add_argument("--dataset-id", required=True)
    migrate.add_argument("--dataset-version", required=True)

    version = commands.add_parser("version-dataset", help="Create a dataset version manifest")
    version.add_argument("--root", type=Path, required=True)
    version.add_argument("--dataset-id", required=True)
    version.add_argument("--version", required=True)
    version.add_argument("--output", type=Path, required=True)
    version.add_argument("--parent-version")
    version.add_argument("--source", action="append", default=[])
    version.add_argument("--hash-mode", choices=("metadata", "full"), default="metadata")

    validate = commands.add_parser("validate-v3", help="Validate a LeRobot v3 dataset")
    validate.add_argument("--root", type=Path, required=True)
    validate.add_argument("--output", type=Path)

    convert = commands.add_parser("convert-lerobot-v3", help="Run the official LeRobot converter")
    convert.add_argument("--repo-id", required=True)
    convert.add_argument("--python", type=Path)
    convert.add_argument("--module", default="lerobot.scripts.convert_dataset_v21_to_v30")
    convert.add_argument("--extra-arg", action="append", default=[])
    convert.add_argument("--dry-run", action="store_true")

    gold = commands.add_parser("sample-gold", help="Build a deterministic stratified gold set")
    gold.add_argument("--episodes", type=Path, required=True)
    gold.add_argument("--output", type=Path, required=True)
    gold.add_argument("--size", type=int, required=True)
    gold.add_argument("--strata", default="task,scene,active_arm")
    gold.add_argument("--seed", type=int, default=0)

    segment = commands.add_parser("segment", help="Propose task/subtask/event boundaries")
    segment.add_argument("--graph", type=Path, required=True)
    segment.add_argument("--modalities", type=Path, required=True)
    segment.add_argument("--output", type=Path, required=True)
    segment.add_argument("--proposal-output", type=Path)

    language = commands.add_parser("language", help="Generate grounded three-level language")
    language.add_argument("--graph", type=Path, required=True)
    language.add_argument("--output", type=Path, required=True)
    language.add_argument("--base-url")
    language.add_argument("--model")
    language.add_argument("--api-key-env", default="VLM_API_KEY")

    calibrate = commands.add_parser(
        "fit-calibrator", help="Fit reliability calibration on gold data"
    )
    calibrate.add_argument("--gold", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--score-field", default="raw_score")
    calibrate.add_argument("--correct-field", default="correct")
    calibrate.add_argument("--minimum-precision", type=float, default=0.95)

    probe = commands.add_parser("gvl-probe", help="Create a shuffled visual progress probe")
    probe.add_argument("--graph", type=Path, required=True)
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--samples", type=int, default=12)
    probe.add_argument("--seed", type=int, default=0)

    evaluate = commands.add_parser("gvl-evaluate", help="Evaluate a VLM progress response")
    evaluate.add_argument("--probe", type=Path, required=True)
    evaluate.add_argument("--response", type=Path, required=True)
    evaluate.add_argument("--output", type=Path)

    anomaly = commands.add_parser("detect-anomalies", help="Audit synchronized modality arrays")
    anomaly.add_argument("--modalities", type=Path, required=True)
    anomaly.add_argument("--timestamp-key", default="timestamps")
    anomaly.add_argument("--output", type=Path, required=True)

    reproject = commands.add_parser(
        "reproject-mask", help="Reproject a metric RGB-D target mask between calibrated cameras"
    )
    reproject.add_argument("--source-depth", type=Path, required=True)
    reproject.add_argument("--source-mask", type=Path, required=True)
    reproject.add_argument("--source-intrinsics", type=Path, required=True)
    reproject.add_argument("--target-from-source", type=Path, required=True)
    reproject.add_argument("--target-intrinsics", type=Path, required=True)
    reproject.add_argument("--depth-scale", type=float, default=0.001)
    reproject.add_argument("--target-mask", type=Path)
    reproject.add_argument("--output-mask", type=Path, required=True)
    reproject.add_argument("--output-report", type=Path, required=True)

    fk = commands.add_parser(
        "fk-camera-trajectory", help="Compute per-frame dynamic wrist camera extrinsics"
    )
    fk.add_argument("--urdf", type=Path, required=True)
    fk.add_argument("--joints", type=Path, required=True)
    fk.add_argument("--base-link", required=True)
    fk.add_argument("--eef-link", required=True)
    fk.add_argument("--eef-from-camera", type=Path, required=True)
    fk.add_argument("--world-from-base", type=Path)
    fk.add_argument("--output", type=Path, required=True)

    shard = commands.add_parser("shard-episodes", help="Stream JSONL metadata into stable shards")
    shard.add_argument("--input", type=Path, required=True)
    shard.add_argument("--output", type=Path, required=True)
    shard.add_argument("--shards", type=int, default=4096)

    queue_add = commands.add_parser("queue-enqueue", help="Idempotently enqueue JSONL jobs")
    queue_add.add_argument("--database", type=Path, required=True)
    queue_add.add_argument("--jobs", type=Path, required=True)
    queue_status = commands.add_parser("queue-status", help="Show queue counts")
    queue_status.add_argument("--database", type=Path, required=True)
    queue_lease = commands.add_parser("queue-lease", help="Lease jobs to a worker")
    queue_lease.add_argument("--database", type=Path, required=True)
    queue_lease.add_argument("--worker", required=True)
    queue_lease.add_argument("--count", type=int, default=1)
    queue_lease.add_argument("--lease-seconds", type=float, default=900)

    distill = commands.add_parser(
        "distill-corrections", help="Export human corrections for training"
    )
    distill.add_argument("--corrections", type=Path, required=True)
    distill.add_argument("--output", type=Path, required=True)

    sidecar = commands.add_parser(
        "export-sidecar", help="Export V3 graph annotations for VLA training"
    )
    sidecar.add_argument("--graph-root", type=Path, required=True)
    sidecar.add_argument("--output", type=Path, required=True)

    ablation = commands.add_parser("create-ablation", help="Create paired ACT or pi0.5 runs")
    ablation.add_argument("--framework", choices=("act", "openpi_pi05"), required=True)
    ablation.add_argument("--experiment-id", required=True)
    ablation.add_argument("--dataset", required=True)
    ablation.add_argument("--output-root", required=True)
    ablation.add_argument("--base-config", required=True)
    ablation.add_argument("--split-manifest", required=True)
    ablation.add_argument(
        "--train-command", required=True, help="Use {config} and optionally {output}"
    )
    ablation.add_argument("--steps", type=int, default=5000)
    ablation.add_argument("--seed", type=int, default=0)
    ablation.add_argument("--gpus", default="")
    ablation.add_argument("--include-input-variants", action="store_true")
    ablation.add_argument("--output", type=Path, required=True)
    ablation_run = commands.add_parser(
        "run-ablation", help="Inspect or explicitly execute a bundle"
    )
    ablation_run.add_argument("--bundle", type=Path, required=True)
    ablation_run.add_argument("--execute", action="store_true")
    return parser


def _run_interactive(args: argparse.Namespace) -> None:
    from interaction_auto_labeler.schema import load_v2_task
    from interaction_labeler.dataset import prepare_session
    from interaction_labeler.pipeline import default_engine_root
    from interaction_labeler.task import load_task

    from .pipeline import initialize_v3
    from .schema import load_v3_task
    from .server import serve

    if args.command == "serve":
        session = json.loads((args.workspace / "session.json").read_text(encoding="utf-8"))
        stored_task_path = session.get("v3_task_path")
        task_path = args.task or (Path(stored_task_path) if stored_task_path else None)
        if task_path is None or not task_path.is_file():
            raise SystemExit("serve requires --task unless session.json contains v3_task_path")
    else:
        task_path = args.task.resolve()
        task = load_v3_task(task_path)
        engine_root = (args.engine_root or default_engine_root()).resolve()
        v1_task = load_task(task_path)
        v1_task["v2"] = load_v2_task(task_path)["v2"]
        v1_task["v3"] = task["v3"]
        session = prepare_session(
            args.data,
            args.workspace,
            v1_task,
            input_format=args.format,
            engine_root=engine_root,
        )
        session["v3_task_path"] = str(task_path)
        atomic_write_json(args.workspace / "session.json", session)
        initialize_v3(args.workspace, task_path)
    task = load_v3_task(task_path)
    detector_backend = "sam3" if task["v2"]["detector_backend"] == "sam3" else "groundingdino"
    engine_root = (args.engine_root or default_engine_root()).resolve()
    options = {
        "v3_task_path": task_path,
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
    _print(
        {
            "pipeline": "v3",
            "events": len(session.get("events", [])),
            "workspace": str(args.workspace.resolve()),
            "url": f"http://{args.host}:{args.port}/",
        }
    )
    serve(
        args.workspace,
        host=args.host,
        port=args.port,
        open_browser=not args.no_open,
        auto_start=args.auto_start,
        pipeline_options=options,
    )


def _graphs(root: Path) -> Iterable[EpisodeEventGraph]:
    for path in sorted(root.glob("*.event_graph.json")):
        yield EpisodeEventGraph.read(path)


def main() -> None:
    args = build_parser().parse_args()
    if args.command in {"run", "serve"}:
        _run_interactive(args)
        return
    if args.command == "process-v3":
        from .pipeline import build_v3_workspace

        _print(build_v3_workspace(args.workspace, args.task))
        return
    if args.command == "migrate-v2":
        paths = migrate_v2_workspace(
            args.workspace, args.output, args.dataset_id, args.dataset_version
        )
        _print(
            {
                "schema": "v2_to_v3_migration_v1",
                "episodes": len(paths),
                "files": [str(path) for path in paths],
            }
        )
        return
    if args.command == "version-dataset":
        manifest = build_dataset_version(
            args.root,
            args.dataset_id,
            args.version,
            args.output,
            parent_version=args.parent_version,
            sources=args.source,
            hash_mode=args.hash_mode,
        )
        _print(manifest.to_dict())
        return
    if args.command == "validate-v3":
        report = validate_lerobot_v3(args.root)
        if args.output:
            atomic_write_json(args.output, report)
        _print(report)
        if not report["valid"]:
            raise SystemExit(2)
        return
    if args.command == "convert-lerobot-v3":
        _print(
            run_official_v3_conversion(
                args.repo_id, args.python, args.module, args.extra_arg, args.dry_run
            )
        )
        return
    if args.command == "sample-gold":
        strata = tuple(item.strip() for item in args.strata.split(",") if item.strip())
        sampled = stratified_gold_sample(read_jsonl(args.episodes), args.size, strata, args.seed)
        write_gold_manifest(sampled, args.output, strata, args.seed)
        _print(
            {
                "schema": "embodied_gold_manifest_v1",
                "sample_count": len(sampled),
                "output": str(args.output.resolve()),
            }
        )
        return
    if args.command == "segment":
        from .temporal import (
            load_modalities_npz,
            propose_hierarchical_boundaries,
            segments_from_boundaries,
        )

        graph = EpisodeEventGraph.read(args.graph)
        proposals, _ = propose_hierarchical_boundaries(load_modalities_npz(str(args.modalities)))
        graph.segments = segments_from_boundaries(graph.frame_count, proposals)
        graph.write(args.output)
        if args.proposal_output:
            atomic_write_json(
                args.proposal_output, {"proposals": [item.to_dict() for item in proposals]}
            )
        _print(
            {
                "segments": len(graph.segments),
                "proposals": len(proposals),
                "output": str(args.output.resolve()),
            }
        )
        return
    if args.command == "language":
        from .language import OpenAICompatibleLanguageBackend, generate_grounded_language

        backend = None
        if args.base_url or args.model:
            if not args.base_url or not args.model:
                raise SystemExit("--base-url and --model must be supplied together")
            backend = OpenAICompatibleLanguageBackend(
                args.base_url, args.model, os.environ.get(args.api_key_env)
            )
        graph = generate_grounded_language(EpisodeEventGraph.read(args.graph), backend)
        graph.write(args.output)
        _print({"annotations": len(graph.language), "output": str(args.output.resolve())})
        return
    if args.command == "fit-calibrator":
        from .reliability import (
            choose_threshold,
            fit_isotonic_calibrator,
            save_calibrator,
            selective_risk_curve,
        )

        rows = read_jsonl(args.gold)
        scores = [float(row[args.score_field]) for row in rows]
        correct = [bool(row[args.correct_field]) for row in rows]
        calibrator = fit_isotonic_calibrator(scores, correct)
        save_calibrator(calibrator, args.output)
        calibrated = calibrator.predict(scores)
        curve = selective_risk_curve(calibrated, correct)
        _print(
            {
                "samples": len(rows),
                "threshold": choose_threshold(curve, args.minimum_precision),
                "output": str(args.output.resolve()),
            }
        )
        return
    if args.command == "gvl-probe":
        from .gvl import build_progress_probe, progress_prompt

        graph = EpisodeEventGraph.read(args.graph)
        probe = build_progress_probe(
            graph.episode_id,
            graph.frame_count,
            args.samples,
            args.seed,
            [item.entity_id for item in graph.entities],
        )
        payload = {**probe.to_dict(), "prompt": progress_prompt(probe)}
        atomic_write_json(args.output, payload)
        _print(payload)
        return
    if args.command == "gvl-evaluate":
        from .gvl import ProgressProbe, evaluate_progress_response

        payload = json.loads(args.probe.read_text(encoding="utf-8"))
        probe = ProgressProbe(
            episode_id=payload["episode_id"],
            frame_count=int(payload["frame_count"]),
            presented_frames=tuple(payload["presented_frames"]),
            required_entity_ids=tuple(payload.get("required_entity_ids", [])),
            phases=tuple(payload.get("phases", [])),
        )
        report = evaluate_progress_response(
            probe, json.loads(args.response.read_text(encoding="utf-8"))
        )
        if args.output:
            atomic_write_json(args.output, report)
        _print(report)
        return
    if args.command == "detect-anomalies":
        import numpy as np

        from .anomaly import detect_anomalies

        with np.load(args.modalities, allow_pickle=False) as payload:
            timestamps = (
                payload[args.timestamp_key] if args.timestamp_key in payload.files else None
            )
            modalities = {
                name: payload[name] for name in payload.files if name != args.timestamp_key
            }
        anomalies = detect_anomalies(modalities, timestamps)
        report = {
            "schema": "embodied_anomaly_report_v1",
            "anomalies": [item.to_dict() for item in anomalies],
        }
        atomic_write_json(args.output, report)
        _print({"anomaly_count": len(anomalies), "output": str(args.output.resolve())})
        return
    if args.command == "reproject-mask":
        import numpy as np
        from PIL import Image

        from .geometry import CameraIntrinsics, reproject_mask, reprojection_metrics

        depth = np.asarray(Image.open(args.source_depth), dtype=np.float64) * args.depth_scale
        source_mask = np.asarray(Image.open(args.source_mask)) > 0
        source_intrinsics = CameraIntrinsics(
            **json.loads(args.source_intrinsics.read_text(encoding="utf-8"))
        )
        target_intrinsics = CameraIntrinsics(
            **json.loads(args.target_intrinsics.read_text(encoding="utf-8"))
        )
        transform = json.loads(args.target_from_source.read_text(encoding="utf-8"))
        projected, _ = reproject_mask(
            depth, source_mask, source_intrinsics, transform, target_intrinsics
        )
        args.output_mask.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((projected * 255).astype(np.uint8)).save(args.output_mask)
        report: dict[str, Any] = {
            "schema": "calibrated_mask_reprojection_v1",
            "projected_pixels": int(projected.sum()),
            "metric": False,
        }
        if args.target_mask:
            target_mask = np.asarray(Image.open(args.target_mask)) > 0
            report.update(reprojection_metrics(projected, target_mask))
            report["metric"] = True
        atomic_write_json(args.output_report, report)
        _print(report)
        return
    if args.command == "fk-camera-trajectory":
        import numpy as np

        from .kinematics import dynamic_camera_trajectory, parse_urdf

        joint_rows = read_jsonl(args.joints)
        positions = [dict(row.get("positions", row.get("joints", {}))) for row in joint_rows]
        eef_from_camera = json.loads(args.eef_from_camera.read_text(encoding="utf-8"))
        world_from_base = (
            json.loads(args.world_from_base.read_text(encoding="utf-8"))
            if args.world_from_base
            else None
        )
        trajectory = dynamic_camera_trajectory(
            parse_urdf(args.urdf),
            positions,
            args.base_link,
            args.eef_link,
            eef_from_camera,
            world_from_base,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            frame_index=np.asarray(
                [row.get("frame_index", index) for index, row in enumerate(joint_rows)]
            ),
            world_from_camera=trajectory,
        )
        _print({"frames": len(trajectory), "output": str(args.output.resolve())})
        return
    if args.command == "shard-episodes":
        from .sharding import iter_jsonl, write_episode_shards

        _print(write_episode_shards(iter_jsonl(args.input), args.output, args.shards))
        return
    if args.command.startswith("queue-"):
        from .jobstore import Job, JobStore

        store = JobStore(args.database)
        if args.command == "queue-enqueue":
            _print(store.enqueue(Job.from_dict(row) for row in read_jsonl(args.jobs)))
        elif args.command == "queue-lease":
            _print(store.lease(args.worker, args.count, args.lease_seconds))
        else:
            _print(store.summary())
        return
    if args.command == "distill-corrections":
        from .corrections import export_distillation_sets

        _print(export_distillation_sets(args.corrections, args.output))
        return
    if args.command == "export-sidecar":
        from .training_export import export_policy_sidecar

        _print(export_policy_sidecar(_graphs(args.graph_root), args.output))
        return
    if args.command == "create-ablation":
        from .ablation import DEFAULT_VARIANTS, AblationSpec, write_ablation_bundle

        variants = DEFAULT_VARIANTS if args.include_input_variants else DEFAULT_VARIANTS[:2]
        spec = AblationSpec(
            experiment_id=args.experiment_id,
            framework=args.framework,
            dataset=args.dataset,
            output_root=args.output_root,
            base_config=args.base_config,
            train_entrypoint=tuple(shlex.split(args.train_command, posix=os.name != "nt")),
            seed=args.seed,
            steps=args.steps,
            gpu_ids=tuple(int(value) for value in args.gpus.split(",") if value.strip()),
            variants=variants,
            split_manifest=args.split_manifest,
        )
        _print(write_ablation_bundle(spec, args.output))
        return
    if args.command == "run-ablation":
        from .ablation import execute_bundle

        _print(execute_bundle(args.bundle, dry_run=not args.execute))
        return
    raise AssertionError(args.command)
