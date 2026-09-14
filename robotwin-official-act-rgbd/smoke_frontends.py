#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check every depth frontend satisfies the official Joiner contract")
    parser.add_argument(
        "--official-act-root",
        type=Path,
        default=Path("/ssd/hhw/depth-model/repos/RoboTwin/policy/ACT"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--include-lingbot", action="store_true")
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--lingbot-repo", type=Path)
    parser.add_argument("--lingbot-checkpoint", type=Path)
    parser.add_argument("--lingbot-vendor", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = Path(__file__).resolve().parent
    sys.path.insert(0, str(project))
    sys.path.insert(0, str(args.official_act_root))

    from official_act_rgbd.policy import DEFAULT_MODEL_CONFIG, build_official_model
    from official_act_rgbd.variants import SUPPORTED_VARIANTS, get_variant

    device = torch.device(args.device)
    variants = args.variants or [
        value for value in SUPPORTED_VARIANTS if args.include_lingbot or value != "ACT6_LINGBOT_DEPTH"
    ]
    rows = []
    for variant in variants:
        get_variant(variant)
        torch.manual_seed(7)
        model = build_official_model(
            DEFAULT_MODEL_CONFIG,
            variant,
            lingbot_repo=args.lingbot_repo,
            lingbot_checkpoint=args.lingbot_checkpoint,
            lingbot_vendor=args.lingbot_vendor,
        ).to(device).eval()
        frontend = model.backbones[0]
        channels = get_variant(variant).packed_channels
        packed = torch.rand(1, channels, args.height, args.width, device=device)
        if channels in {5, 7}:
            packed[:, -1].fill_(1.0)
        with torch.inference_mode():
            features, positions = frontend(packed)
        feature = features[-1]
        position = positions[-1]
        passed = (
            feature.ndim == 4
            and feature.shape[0] == 1
            and feature.shape[1] == model.input_proj.in_channels
            and feature.shape == position.shape
        )
        rows.append(
            {
                "variant": variant,
                "passed": passed,
                "input_shape": list(packed.shape),
                "feature_shape": list(feature.shape),
                "position_shape": list(position.shape),
            }
        )
        del model, frontend, packed, features, positions
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {"passed": all(row["passed"] for row in rows), "frontends": rows}
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
