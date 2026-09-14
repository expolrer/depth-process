#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One full-resolution real-data ACT RGB-D training step")
    parser.add_argument("--project-root", type=Path, default=Path("/ssd/hhw/depth-model"))
    parser.add_argument(
        "--official-act-root",
        type=Path,
        default=Path("/ssd/hhw/depth-model/repos/RoboTwin/policy/ACT"),
    )
    parser.add_argument("--variant", default="ACT2_DUAL_SHARED")
    parser.add_argument("--task", default="stack_blocks_two")
    parser.add_argument("--task-config", default="depth_master_clean")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = Path(__file__).resolve().parent
    sys.path.insert(0, str(project))
    sys.path.insert(0, str(args.official_act_root))

    from official_act_rgbd.data import OfficialAlignedRGBDDataset, exact_official_stats, move_batch
    from official_act_rgbd.policy import DEFAULT_MODEL_CONFIG, build_policy_and_optimizer

    processed = args.project_root / "datasets/act_processed" / f"sim-{args.task}" / f"{args.task_config}-50"
    master = args.project_root / "datasets/master" / args.task / args.task_config / "data"
    stats, max_action_len = exact_official_stats(processed, list(range(50)))
    dataset = OfficialAlignedRGBDDataset(
        processed,
        master,
        [0],
        stats,
        max_action_len,
        sample_mode="frame_grid",
        frames_per_episode=1,
    )
    batch = move_batch(next(iter(DataLoader(dataset, batch_size=1))), torch.device(args.device))
    torch.manual_seed(23)
    policy, optimizer = build_policy_and_optimizer(
        DEFAULT_MODEL_CONFIG,
        args.variant,
        device=args.device,
        lingbot_repo=args.project_root / "repos/lingbot-depth",
        lingbot_checkpoint=args.project_root / "models/lingbot-depth-v0.5/model.pt",
        lingbot_vendor=args.project_root / "repos/depth-processing-vendor",
    )
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    result = policy(
        batch["qpos"],
        batch["image"],
        batch["depth_m"],
        batch["validity"],
        batch["xyz_map_m"],
        batch["action"],
        batch["is_pad"],
    )
    result["loss"].backward()
    depth_grad = 0.0
    rgb_grad = 0.0
    for name, parameter in policy.named_parameters():
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().abs().sum())
        if "geometry_encoder" in name or "fusion" in name:
            depth_grad += value
        if "rgb_joiner" in name or (args.variant == "ACT0_RGB" and "backbones" in name):
            rgb_grad += value
    optimizer.step()
    report = {
        "passed": bool(torch.isfinite(result["loss"]).item() and depth_grad > 0.0),
        "variant": args.variant,
        "input_image_shape": list(batch["image"].shape),
        "input_depth_shape": list(batch["depth_m"].shape),
        "action_shape": list(batch["action"][:, :50].shape),
        "loss": float(result["loss"].detach()),
        "depth_frontend_grad_l1": depth_grad,
        "rgb_backbone_grad_l1": rgb_grad,
        "cuda_peak_memory_mib": (
            torch.cuda.max_memory_allocated() / 1024**2 if torch.device(args.device).type == "cuda" else None
        ),
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
