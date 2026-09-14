#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify RGB, qpos and action tensors remain official-ACT aligned")
    parser.add_argument("--project-root", type=Path, default=Path("/ssd/hhw/depth-model"))
    parser.add_argument(
        "--official-act-root",
        type=Path,
        default=Path("/ssd/hhw/depth-model/repos/RoboTwin/policy/ACT"),
    )
    parser.add_argument("--task", default="stack_blocks_two")
    parser.add_argument("--task-config", default="depth_master_clean")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = Path(__file__).resolve().parent
    sys.path.insert(0, str(project))
    sys.path.insert(0, str(args.official_act_root))

    from official_act_rgbd.data import (
        CAMERA_ORDER,
        PROCESSED_CAMERA,
        OfficialAlignedRGBDDataset,
        exact_official_stats,
    )
    from utils import get_norm_stats

    processed_dir = (
        args.project_root
        / "datasets/act_processed"
        / f"sim-{args.task}"
        / f"{args.task_config}-{args.episodes}"
    )
    master_dir = args.project_root / "datasets/master" / args.task / args.task_config / "data"
    ours, max_action_len = exact_official_stats(processed_dir, list(range(args.episodes)))
    upstream, upstream_max_action_len = get_norm_stats(str(processed_dir), args.episodes)
    stats_delta = {
        key: float(np.max(np.abs(np.asarray(ours[key]) - np.asarray(upstream[key]))))
        for key in ("qpos_mean", "qpos_std", "action_mean", "action_std")
    }

    dataset = OfficialAlignedRGBDDataset(
        processed_dir,
        master_dir,
        [0],
        ours,
        max_action_len,
        sample_mode="frame_grid",
        frames_per_episode=1,
    )
    sample = dataset[0]
    frame = int(sample["frame"])
    with h5py.File(processed_dir / "episode_0.hdf5", "r") as source:
        expected_image = np.stack(
            [
                np.asarray(source[f"observations/images/{PROCESSED_CAMERA[camera]}"][frame])
                .transpose(2, 0, 1)
                .astype(np.float32)
                / 255.0
                for camera in CAMERA_ORDER
            ]
        )
        expected_qpos = (
            np.asarray(source["observations/qpos"][frame], dtype=np.float32) - ours["qpos_mean"]
        ) / ours["qpos_std"]
        action = np.asarray(source["action"][max(0, frame - 1) :], dtype=np.float32)
        expected_action = np.zeros((max_action_len, action.shape[1]), dtype=np.float32)
        expected_action[: len(action)] = action
        expected_action = (expected_action - ours["action_mean"]) / ours["action_std"]

    checks = {
        "stats_match": max(stats_delta.values()) <= 1e-7,
        "max_action_len_match": max_action_len == upstream_max_action_len,
        "rgb_exact": np.array_equal(sample["image"].numpy(), expected_image),
        "qpos_close": np.allclose(sample["qpos"].numpy(), expected_qpos.astype(np.float32), atol=1e-6),
        "action_close": np.allclose(sample["action"].numpy(), expected_action.astype(np.float32), atol=1e-6),
        "rgb_shape_official": list(sample["image"].shape) == [3, 3, 480, 640],
        "depth_shape_aligned": list(sample["depth_m"].shape) == [3, 1, 480, 640],
        "xyz_shape_aligned": list(sample["xyz_map_m"].shape) == [3, 3, 480, 640],
        "depth_finite": bool(np.isfinite(sample["depth_m"].numpy()).all()),
    }
    report = {
        "passed": all(checks.values()),
        "task": args.task,
        "frame": frame,
        "camera_order": list(CAMERA_ORDER),
        "checks": checks,
        "stats_max_abs_delta": stats_delta,
        "max_action_len": max_action_len,
    }
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

