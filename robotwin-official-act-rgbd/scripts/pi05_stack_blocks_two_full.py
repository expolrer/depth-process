#!/usr/bin/env python3
"""Build the official RoboTwin pi0.5 full-finetuning config without editing upstream."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import sys


CONFIG_NAME = "pi05_robotwin_stack_blocks_two_full_4gpu"
EXPERIMENT_NAME = "pi05_stack_blocks_two_rgb_full_seed0"
REPO_ID = "robotwin/stack_blocks_two_rgb_50"
TRAIN_STEPS = 20_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("check", "norm-stats", "train"))
    parser.add_argument("--project-root", default="/ssd/hhw/depth-model")
    parser.add_argument("--pi05-root", default="/ssd/hhw/depth-model/repos/RoboTwin/policy/pi05")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_config(project_root: Path, resume: bool):
    from openpi.training import config
    from openpi.training import weight_loaders

    base = config.get_config("pi05_aloha_full_base")
    cfg = dataclasses.replace(
        base,
        name=CONFIG_NAME,
        exp_name=EXPERIMENT_NAME,
        data=dataclasses.replace(base.data, repo_id=REPO_ID),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/ssd/hhw/openpi-hzh/checkpoint/pi05_base/params"
        ),
        assets_base_dir=str(project_root / "experiments/OfficialPi05/assets"),
        checkpoint_base_dir=str(project_root / "experiments/OfficialPi05/checkpoints"),
        seed=0,
        batch_size=64,
        num_workers=24,
        num_train_steps=TRAIN_STEPS,
        save_interval=1_000,
        keep_period=5_000,
        fsdp_devices=4,
        wandb_enabled=False,
        resume=resume,
        overwrite=not resume,
    )
    return config, cfg


def assert_full_finetune(cfg) -> None:
    if not cfg.model.pi05:
        raise RuntimeError("The selected model is not pi0.5")
    if cfg.fsdp_devices != 4 or cfg.batch_size % 4:
        raise RuntimeError("Four-device FSDP configuration is invalid")
    if type(cfg.freeze_filter).__name__ != "Nothing":
        raise RuntimeError("freeze_filter is not empty; this would not be full fine-tuning")


def write_manifest(project_root: Path, cfg) -> None:
    manifest = {
        "schema_version": 1,
        "upstream_config": "pi05_aloha_full_base",
        "config_name": cfg.name,
        "experiment_name": cfg.exp_name,
        "dataset_repo_id": cfg.data.repo_id,
        "modalities": ["cam_high_rgb", "cam_left_wrist_rgb", "cam_right_wrist_rgb", "joint", "prompt", "action"],
        "depth_used": False,
        "full_finetune": True,
        "freeze_filter": type(cfg.freeze_filter).__name__,
        "num_train_steps": cfg.num_train_steps,
        "global_batch_size": cfg.batch_size,
        "fsdp_devices": cfg.fsdp_devices,
        "seed": cfg.seed,
        "base_params": cfg.weight_loader.params_path,
        "checkpoint_dir": str(cfg.checkpoint_dir),
        "assets_dir": str(cfg.assets_dirs),
    }
    path = project_root / "experiments/OfficialPi05/run_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    pi05_root = Path(args.pi05_root).resolve()
    config_module, cfg = build_config(project_root, args.resume)
    assert_full_finetune(cfg)
    write_manifest(project_root, cfg)

    if args.mode == "check":
        print(f"config={cfg.name}")
        print(f"repo_id={cfg.data.repo_id}")
        print(f"steps={cfg.num_train_steps} batch={cfg.batch_size} fsdp={cfg.fsdp_devices}")
        print(f"freeze_filter={type(cfg.freeze_filter).__name__}")
        return

    config_module._CONFIGS_DICT[cfg.name] = cfg
    if args.mode == "norm-stats":
        norm = load_module("robotwin_pi05_compute_norm_stats", pi05_root / "scripts/compute_norm_stats.py")
        norm.main(cfg.name)
        return

    train = load_module("robotwin_pi05_train", pi05_root / "scripts/train.py")
    train.main(cfg)


if __name__ == "__main__":
    main()
