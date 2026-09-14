#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train depth frontends on the locked official ACT core")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-config", default="depth_master_clean")
    parser.add_argument(
        "--project-root", type=Path, default=Path("/ssd/hhw/depth-model")
    )
    parser.add_argument(
        "--official-act-root",
        type=Path,
        default=Path("/ssd/hhw/depth-model/repos/RoboTwin/policy/ACT"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth-variant", default="gt")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--validation-frames", type=int, default=16)
    parser.add_argument("--validate-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--lingbot-repo",
        type=Path,
        default=Path("/ssd/hhw/depth-model/repos/lingbot-depth"),
    )
    parser.add_argument(
        "--lingbot-checkpoint",
        type=Path,
        default=Path("/ssd/hhw/depth-model/models/lingbot-depth-v0.5/model.pt"),
    )
    parser.add_argument(
        "--lingbot-vendor",
        type=Path,
        default=Path("/ssd/hhw/depth-model/repos/depth-processing-vendor"),
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def jsonable_args(args: argparse.Namespace) -> dict[str, object]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def policy_call(policy, batch, *, training: bool):
    return policy(
        batch["qpos"],
        batch["image"],
        batch["depth_m"],
        batch["validity"],
        batch["xyz_map_m"],
        batch["action"] if training else None,
        batch["is_pad"] if training else None,
    )


@torch.inference_mode()
def validate(policy, loader, device: torch.device) -> dict[str, float]:
    from official_act_rgbd.data import move_batch

    policy.eval()
    totals = {"posterior_l1": 0.0, "posterior_kl": 0.0, "prior_l1": 0.0}
    samples = 0
    for host_batch in loader:
        batch = move_batch(host_batch, device)
        posterior = policy_call(policy, batch, training=True)
        prior_prediction = policy_call(policy, batch, training=False)
        target = batch["action"][:, : policy.model.num_queries]
        valid = (~batch["is_pad"][:, : policy.model.num_queries]).unsqueeze(-1)
        valid_elements = valid.sum().clamp_min(1) * target.shape[-1]
        prior_l1 = ((prior_prediction - target).abs() * valid).sum() / valid_elements
        batch_size = int(target.shape[0])
        totals["posterior_l1"] += float(posterior["l1"]) * batch_size
        totals["posterior_kl"] += float(posterior["kl"]) * batch_size
        totals["prior_l1"] += float(prior_l1) * batch_size
        samples += batch_size
    if samples == 0:
        return {key: math.nan for key in totals}
    return {key: value / samples for key, value in totals.items()}


def save_training_state(path, policy, optimizer, epoch, best, args, split) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_validation_prior_l1": best,
            "args": jsonable_args(args),
            "split": split,
        },
        temporary,
    )
    temporary.replace(path)


def save_policy(path: Path, policy) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(policy.state_dict(), temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(args.official_act_root))

    from official_act_rgbd.data import OfficialAlignedRGBDDataset, exact_official_stats, move_batch
    from official_act_rgbd.policy import DEFAULT_MODEL_CONFIG, build_policy_and_optimizer
    from official_act_rgbd.variants import get_variant

    variant_spec = get_variant(args.variant)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the requested device")
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    processed_dir = (
        args.project_root
        / "datasets"
        / "act_processed"
        / f"sim-{args.task}"
        / f"{args.task_config}-{args.episodes}"
    )
    master_dir = (
        args.project_root
        / "datasets"
        / "master"
        / args.task
        / args.task_config
        / "data"
    )
    derived_dir = (
        args.project_root
        / "datasets"
        / "derived"
        / "base"
        / args.task
        / args.task_config
        / "data"
    )
    if not processed_dir.is_dir():
        raise FileNotFoundError(processed_dir)
    if not master_dir.is_dir():
        raise FileNotFoundError(master_dir)

    args.output.mkdir(parents=True, exist_ok=True)
    all_episodes = list(range(args.episodes))
    shuffled = np.random.default_rng(args.seed).permutation(all_episodes).tolist()
    validation_count = max(1, args.episodes // 5)
    split = {
        "train": shuffled[:-validation_count],
        "validation": shuffled[-validation_count:],
    }
    stats, max_action_len = exact_official_stats(processed_dir, all_episodes)
    with (args.output / "dataset_stats.pkl").open("wb") as stream:
        pickle.dump(stats, stream)

    model_config = dict(DEFAULT_MODEL_CONFIG)
    model_config["camera_names"] = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    config_record = jsonable_args(args)
    config_record.update(
        {
            "processed_dir": str(processed_dir),
            "master_dir": str(master_dir),
            "derived_dir": str(derived_dir),
            "split": split,
            "max_action_len": max_action_len,
            "official_model_config": model_config,
            "checkpoint_selection": "full_validation_prior_action_l1",
        }
    )
    (args.output / "config.json").write_text(
        json.dumps(config_record, indent=2) + "\n", encoding="utf-8"
    )

    common_dataset = {
        "processed_dir": processed_dir,
        "master_dir": master_dir,
        "stats": stats,
        "max_action_len": max_action_len,
        "depth_variant": args.depth_variant,
        "derived_dir": derived_dir if args.depth_variant != "gt" else None,
        "include_depth": variant_spec.requires_depth,
        "include_xyz": variant_spec.requires_xyz,
    }
    train_dataset = OfficialAlignedRGBDDataset(
        episode_ids=split["train"], sample_mode="episode_random", **common_dataset
    )
    validation_dataset = OfficialAlignedRGBDDataset(
        episode_ids=split["validation"],
        sample_mode="frame_grid",
        frames_per_episode=args.validation_frames,
        **common_dataset,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_options,
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)

    policy, optimizer = build_policy_and_optimizer(
        model_config,
        args.variant,
        device=device,
        lingbot_repo=args.lingbot_repo,
        lingbot_checkpoint=args.lingbot_checkpoint,
        lingbot_vendor=args.lingbot_vendor,
    )
    start_epoch = 0
    best = math.inf
    training_state_path = args.output / "training_last.pt"
    if args.resume and training_state_path.exists():
        state = torch.load(training_state_path, map_location="cpu")
        policy.load_state_dict(state["policy"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"])
        best = float(state.get("best_validation_prior_l1", math.inf))

    metrics_path = args.output / "metrics.jsonl"
    started = time.time()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        policy.train()
        aggregate = {"loss": 0.0, "l1": 0.0, "kl": 0.0}
        batches = 0
        for host_batch in train_loader:
            batch = move_batch(host_batch, device)
            optimizer.zero_grad(set_to_none=True)
            result = policy_call(policy, batch, training=True)
            if not torch.isfinite(result["loss"]):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            result["loss"].backward()
            optimizer.step()
            for key in aggregate:
                aggregate[key] += float(result[key].detach())
            batches += 1

        record = {
            "epoch": epoch,
            "updates": epoch * len(train_loader),
            "train_loss": aggregate["loss"] / max(batches, 1),
            "train_l1": aggregate["l1"] / max(batches, 1),
            "train_kl": aggregate["kl"] / max(batches, 1),
            "elapsed_seconds": time.time() - started,
        }
        gate = policy.gate_summary()
        if gate is not None:
            record["depth_gate_mean"] = gate

        should_validate = epoch == 1 or epoch % args.validate_every == 0 or epoch == args.epochs
        if should_validate:
            validation = validate(policy, validation_loader, device)
            record.update({f"validation_{key}": value for key, value in validation.items()})
            if validation["prior_l1"] < best:
                best = validation["prior_l1"]
                save_policy(args.output / "policy_best.ckpt", policy)

        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_policy(args.output / f"policy_epoch_{epoch:04d}.ckpt", policy)
            save_policy(args.output / "policy_last.ckpt", policy)
            save_training_state(training_state_path, policy, optimizer, epoch, best, args, split)


if __name__ == "__main__":
    main()
