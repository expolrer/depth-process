#!/usr/bin/env python3
"""Minimal local-checkpoint smoke test for LingBot-Depth."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution-level", type=int, default=4)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo))
    from mdm.model.v2 import MDMModel

    rgb_bgr = cv2.imread(str(args.rgb), cv2.IMREAD_COLOR)
    depth_mm = cv2.imread(str(args.depth), cv2.IMREAD_UNCHANGED)
    if rgb_bgr is None or depth_mm is None:
        raise RuntimeError("Failed to read smoke-test inputs")
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    depth_m = depth_mm.astype(np.float32) / 1000.0
    height, width = depth_m.shape
    intrinsics = np.loadtxt(args.intrinsics, dtype=np.float32)
    intrinsics[0, :] /= width
    intrinsics[1, :] /= height

    device = torch.device("cuda:0")
    model = MDMModel.from_pretrained(args.checkpoint).to(device).eval()
    image_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255).unsqueeze(0).to(device)
    depth_tensor = torch.from_numpy(depth_m).unsqueeze(0).to(device)
    intrinsics_tensor = torch.from_numpy(intrinsics).unsqueeze(0).to(device)

    torch.cuda.synchronize()
    started = time.perf_counter()
    result = model.infer(
        image_tensor,
        depth_in=depth_tensor,
        intrinsics=intrinsics_tensor,
        resolution_level=args.resolution_level,
        apply_mask=False,
        use_fp16=True,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    output = result["depth"][0].float().cpu().numpy()

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "depth_refined_m.npy", output.astype(np.float32))
    encoded_mm = np.clip(np.nan_to_num(output) * 1000.0, 0, 65535).astype(np.uint16)
    cv2.imwrite(str(args.output / "depth_refined_mm.png"), encoded_mm)
    print(
        f"shape={output.shape} min={np.nanmin(output):.4f}m "
        f"max={np.nanmax(output):.4f}m elapsed={elapsed:.3f}s"
    )


if __name__ == "__main__":
    main()
