from __future__ import annotations

import json
import pickle
import sys
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import torch

from .data import CAMERA_ORDER, depth_to_xyz_map, resize_intrinsic
from .policy import DEFAULT_MODEL_CONFIG, OfficialACTRGBDPolicy


DEFAULT_INTRINSIC = np.array(
    ((358.6421814, 0.0, 160.0), (0.0, 358.6421814, 120.0), (0.0, 0.0, 1.0)),
    dtype=np.float32,
)


def _joint_vector(observation: dict) -> np.ndarray:
    joint = observation["joint_action"]
    if "vector" in joint:
        return np.asarray(joint["vector"], dtype=np.float32)
    return np.asarray(
        joint["left_arm"]
        + [joint["left_gripper"]]
        + joint["right_arm"]
        + [joint["right_gripper"]],
        dtype=np.float32,
    )


def encode_observation(observation: dict, counterfactual: str = "none") -> dict[str, np.ndarray]:
    rgb_views = []
    depth_views = []
    validity_views = []
    xyz_views = []
    for camera in CAMERA_ORDER:
        camera_observation = observation["observation"][camera]
        rgb_raw = np.asarray(camera_observation["rgb"])
        depth_raw = np.asarray(camera_observation["depth"], dtype=np.float32)
        rgb = cv2.resize(rgb_raw, (640, 480), interpolation=cv2.INTER_LINEAR)
        depth_mm = cv2.resize(depth_raw, (640, 480), interpolation=cv2.INTER_NEAREST)
        validity = np.isfinite(depth_mm) & (depth_mm > 0)
        depth_m = np.where(validity, depth_mm * 0.001, 0.0).astype(np.float32)
        intrinsic_raw = np.asarray(camera_observation.get("intrinsic_cv", DEFAULT_INTRINSIC), dtype=np.float32)
        intrinsic = resize_intrinsic(intrinsic_raw, depth_raw.shape, depth_m.shape)
        rgb_views.append(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)
        depth_views.append(depth_m[None])
        validity_views.append(validity.astype(np.float32)[None])
        xyz_views.append(depth_to_xyz_map(depth_m, validity, intrinsic))

    if counterfactual == "zero_depth":
        depth_views = [np.zeros_like(value) for value in depth_views]
        validity_views = [np.zeros_like(value) for value in validity_views]
        xyz_views = [np.zeros_like(value) for value in xyz_views]
    elif counterfactual == "shuffle_views":
        order = (1, 2, 0)
        depth_views = [depth_views[index] for index in order]
        validity_views = [validity_views[index] for index in order]
        xyz_views = [xyz_views[index] for index in order]
    elif counterfactual != "none":
        raise ValueError(f"Unsupported online counterfactual: {counterfactual}")

    return {
        "image": np.stack(rgb_views),
        "depth_m": np.stack(depth_views),
        "validity": np.stack(validity_views),
        "xyz_map_m": np.stack(xyz_views),
        "qpos": _joint_vector(observation),
    }


class OfficialACTRGBDRunner:
    def __init__(self, args: dict) -> None:
        official_act_root = str(
            args.get("official_act_root", "/ssd/hhw/depth-model/repos/RoboTwin/policy/ACT")
        )
        if official_act_root not in sys.path:
            sys.path.insert(0, official_act_root)
        self.device = torch.device(args.get("device", "cuda:0"))
        self.variant = str(args.get("variant", "ACT0_RGB"))
        self.counterfactual = str(args.get("depth_counterfactual", "none"))
        self.temporal_agg = bool(args.get("temporal_agg", False))
        self.num_queries = int(args.get("chunk_size", 50))
        self.query_frequency = 1 if self.temporal_agg else self.num_queries
        self.state_dim = int(args.get("action_dim", 14))
        self.max_timesteps = int(args.get("max_timesteps", 3000))
        self.t = 0

        checkpoint_value = args.get("checkpoint")
        checkpoint_dir = Path(args.get("ckpt_dir", ""))
        checkpoint = Path(checkpoint_value) if checkpoint_value else checkpoint_dir / "policy_best.ckpt"
        config_path = checkpoint.parent / "config.json"
        model_config = dict(DEFAULT_MODEL_CONFIG)
        if config_path.exists():
            stored = json.loads(config_path.read_text(encoding="utf-8"))
            model_config.update(stored.get("official_model_config", {}))
        model_config.update({key: value for key, value in args.items() if key in model_config})

        self.policy = OfficialACTRGBDPolicy(
            Namespace(**model_config),
            self.variant,
            lingbot_repo=args.get("lingbot_repo"),
            lingbot_checkpoint=args.get("lingbot_checkpoint"),
            lingbot_vendor=args.get("lingbot_vendor"),
        ).to(self.device)
        state = torch.load(checkpoint, map_location="cpu")
        self.policy.load_state_dict(state)
        self.policy.eval()

        stats_path = checkpoint.parent / "dataset_stats.pkl"
        with stats_path.open("rb") as stream:
            stats = pickle.load(stream)
        self.qpos_mean = np.asarray(stats["qpos_mean"], dtype=np.float32)
        self.qpos_std = np.asarray(stats["qpos_std"], dtype=np.float32)
        self.action_mean = np.asarray(stats["action_mean"], dtype=np.float32)
        self.action_std = np.asarray(stats["action_std"], dtype=np.float32)
        if self.temporal_agg:
            self._reset_temporal_buffer()

    def _reset_temporal_buffer(self) -> None:
        self.all_time_actions = torch.zeros(
            self.max_timesteps,
            self.max_timesteps + self.num_queries,
            self.state_dim,
            device=self.device,
        )

    @torch.inference_mode()
    def get_action(self, observation: dict) -> np.ndarray:
        encoded = encode_observation(observation, self.counterfactual)
        qpos = torch.from_numpy((encoded["qpos"] - self.qpos_mean) / self.qpos_std).to(
            self.device, dtype=torch.float32
        ).unsqueeze(0)
        image = torch.from_numpy(encoded["image"]).to(self.device, dtype=torch.float32).unsqueeze(0)
        depth_m = torch.from_numpy(encoded["depth_m"]).to(self.device, dtype=torch.float32).unsqueeze(0)
        validity = torch.from_numpy(encoded["validity"]).to(self.device, dtype=torch.float32).unsqueeze(0)
        xyz_map_m = torch.from_numpy(encoded["xyz_map_m"]).to(self.device, dtype=torch.float32).unsqueeze(0)

        if self.t % self.query_frequency == 0:
            self.all_actions = self.policy(qpos, image, depth_m, validity, xyz_map_m)

        if self.temporal_agg:
            self.all_time_actions[[self.t], self.t : self.t + self.num_queries] = self.all_actions
            candidates = self.all_time_actions[:, self.t]
            populated = torch.all(candidates != 0, dim=1)
            candidates = candidates[populated]
            weights = np.exp(-0.01 * np.arange(len(candidates)))
            weights = torch.from_numpy(weights / weights.sum()).to(self.device).unsqueeze(1)
            raw_action = (candidates * weights).sum(dim=0, keepdim=True)
        else:
            raw_action = self.all_actions[:, self.t % self.query_frequency]

        action = raw_action.float().cpu().numpy() * self.action_std + self.action_mean
        self.t += 1
        return action

    def reset(self) -> None:
        self.t = 0
        if self.temporal_agg:
            self._reset_temporal_buffer()


def get_model(usr_args):
    return OfficialACTRGBDRunner(usr_args)


def eval(TASK_ENV, model, observation):
    actions = model.get_action(observation)
    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
    return observation


def reset_model(model):
    model.reset()
