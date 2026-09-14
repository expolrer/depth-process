#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete official ACT core with representative depth inputs")
    parser.add_argument(
        "--official-act-root",
        type=Path,
        default=Path("/ssd/hhw/depth-model/repos/RoboTwin/policy/ACT"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["ACT0_RGB", "ACT2_DEPTH_CNN", "ACT4_XYZMAP", "ACT5_POINT_TOKENS", "ACT7_DEPTH_TRANSFORMER"],
    )
    parser.add_argument("--backward-variant", default="ACT2_DEPTH_CNN")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = Path(__file__).resolve().parent
    sys.path.insert(0, str(project))
    sys.path.insert(0, str(args.official_act_root))

    from official_act_rgbd.policy import DEFAULT_MODEL_CONFIG, OfficialACTRGBDPolicy
    from official_act_rgbd.variants import get_variant

    device = torch.device(args.device)
    rows = []
    for variant in args.variants:
        get_variant(variant)
        torch.manual_seed(11)
        policy = OfficialACTRGBDPolicy(DEFAULT_MODEL_CONFIG, variant).to(device)
        policy.train()
        image = torch.rand(1, 3, 3, args.height, args.width, device=device)
        depth = torch.rand(1, 3, 1, args.height, args.width, device=device) * 1.5 + 0.1
        validity = torch.ones_like(depth)
        xyz = torch.cat((depth * 0.1, depth * -0.1, depth), dim=2)
        qpos = torch.randn(1, 14, device=device)
        actions = torch.randn(1, 50, 14, device=device)
        is_pad = torch.zeros(1, 50, dtype=torch.bool, device=device)
        result = policy(qpos, image, depth, validity, xyz, actions, is_pad)
        if variant == args.backward_variant:
            result["loss"].backward()
        policy.eval()
        with torch.inference_mode():
            prediction = policy(qpos, image, depth, validity, xyz)
        passed = (
            prediction.shape == (1, 50, 14)
            and torch.isfinite(prediction).all().item()
            and torch.isfinite(result["loss"]).item()
        )
        rows.append(
            {
                "variant": variant,
                "passed": bool(passed),
                "prediction_shape": list(prediction.shape),
                "loss": float(result["loss"].detach()),
                "backward": variant == args.backward_variant,
            }
        )
        del policy, image, depth, validity, xyz, qpos, actions, is_pad, result, prediction
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {"passed": all(row["passed"] for row in rows), "policies": rows}
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
