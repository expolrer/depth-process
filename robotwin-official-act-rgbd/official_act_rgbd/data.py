from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


CAMERA_ORDER = ("head_camera", "left_camera", "right_camera")
PROCESSED_CAMERA = {
    "head_camera": "cam_high",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}


@dataclass(frozen=True)
class FrameRef:
    episode: int
    frame: int


def resize_intrinsic(
    intrinsic: np.ndarray,
    source_hw: tuple[int, int],
    target_hw: tuple[int, int],
) -> np.ndarray:
    source_h, source_w = source_hw
    target_h, target_w = target_hw
    result = np.asarray(intrinsic, dtype=np.float32).copy()
    result[0] *= target_w / source_w
    result[1] *= target_h / source_h
    return result


def depth_to_xyz_map(depth_m: np.ndarray, validity: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    height, width = depth_m.shape
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    z = np.where(validity, depth_m, 0.0)
    x = (xx - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (yy - intrinsic[1, 2]) * z / intrinsic[1, 1]
    return np.stack((x, y, z), axis=0).astype(np.float32)


def exact_official_stats(processed_dir: Path, episode_ids: Sequence[int]) -> tuple[dict[str, np.ndarray], int]:
    """Reproduce RoboTwin ACT padding and all-episode normalization semantics."""
    qpos_values: list[torch.Tensor] = []
    action_values: list[torch.Tensor] = []
    for episode in episode_ids:
        with h5py.File(processed_dir / f"episode_{episode}.hdf5", "r") as source:
            qpos_values.append(torch.from_numpy(np.asarray(source["observations/qpos"])))
            action_values.append(torch.from_numpy(np.asarray(source["action"])))

    max_qpos_len = max(item.shape[0] for item in qpos_values)
    max_action_len = max(item.shape[0] for item in action_values)

    def pad_last(items: Iterable[torch.Tensor], length: int) -> torch.Tensor:
        padded = []
        for item in items:
            if item.shape[0] < length:
                item = torch.cat((item, item[-1:].repeat(length - item.shape[0], 1)), dim=0)
            padded.append(item)
        return torch.stack(padded)

    all_qpos = pad_last(qpos_values, max_qpos_len)
    all_action = pad_last(action_values, max_action_len)
    stats = {
        "action_mean": all_action.mean(dim=(0, 1), keepdim=True).numpy().squeeze(),
        "action_std": all_action.std(dim=(0, 1), keepdim=True).clamp_min(1e-2).numpy().squeeze(),
        "qpos_mean": all_qpos.mean(dim=(0, 1), keepdim=True).numpy().squeeze(),
        "qpos_std": all_qpos.std(dim=(0, 1), keepdim=True).clamp_min(1e-2).numpy().squeeze(),
    }
    return stats, max_action_len


class OfficialAlignedRGBDDataset(Dataset):
    """Official ACT RGB/action samples plus frame-aligned depth from the master dataset."""

    def __init__(
        self,
        processed_dir: Path,
        master_dir: Path,
        episode_ids: Sequence[int],
        stats: dict[str, np.ndarray],
        max_action_len: int,
        *,
        depth_variant: str = "gt",
        derived_dir: Path | None = None,
        sample_mode: str = "episode_random",
        frames_per_episode: int = 16,
        camera_order: Sequence[str] = CAMERA_ORDER,
        include_depth: bool = True,
        include_xyz: bool = True,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.master_dir = Path(master_dir)
        self.episode_ids = tuple(int(value) for value in episode_ids)
        self.stats = stats
        self.max_action_len = int(max_action_len)
        self.depth_variant = depth_variant
        self.derived_dir = None if derived_dir is None else Path(derived_dir)
        self.sample_mode = sample_mode
        self.camera_order = tuple(camera_order)
        self.include_depth = bool(include_depth)
        self.include_xyz = bool(include_xyz)
        if sample_mode not in {"episode_random", "frame_grid"}:
            raise ValueError("sample_mode must be episode_random or frame_grid")
        if set(self.camera_order) != set(CAMERA_ORDER):
            raise ValueError(f"camera_order must contain exactly {CAMERA_ORDER}")
        if depth_variant != "gt" and self.derived_dir is None:
            raise ValueError("derived_dir is required when depth_variant is not gt")
        if self.include_xyz and not self.include_depth:
            raise ValueError("include_xyz requires include_depth")

        self.frame_refs: list[FrameRef] = []
        if sample_mode == "frame_grid":
            for episode in self.episode_ids:
                with h5py.File(self._processed_path(episode), "r") as source:
                    episode_len = int(source["action"].shape[0])
                count = min(frames_per_episode, episode_len)
                indices = np.linspace(0, episode_len - 1, count, dtype=np.int64)
                self.frame_refs.extend(FrameRef(episode, int(frame)) for frame in np.unique(indices))

    def _processed_path(self, episode: int) -> Path:
        return self.processed_dir / f"episode_{episode}.hdf5"

    def _master_path(self, episode: int) -> Path:
        return self.master_dir / f"episode{episode}.hdf5"

    def _derived_path(self, episode: int) -> Path:
        assert self.derived_dir is not None
        return self.derived_dir / f"episode{episode}.depth_variants.hdf5"

    def __len__(self) -> int:
        if self.sample_mode == "episode_random":
            return len(self.episode_ids)
        return len(self.frame_refs)

    def _select(self, index: int) -> FrameRef:
        if self.sample_mode == "frame_grid":
            return self.frame_refs[index]
        episode = self.episode_ids[index]
        with h5py.File(self._processed_path(episode), "r") as source:
            episode_len = int(source["action"].shape[0])
        return FrameRef(episode, int(np.random.choice(episode_len)))

    def _read_depth(
        self,
        master: h5py.File,
        derived: h5py.File | None,
        camera: str,
        frame: int,
    ) -> np.ndarray:
        if self.depth_variant == "gt":
            return np.asarray(master[f"observation/{camera}/depth"][frame], dtype=np.float32)
        assert derived is not None
        return np.asarray(derived[f"depth/{camera}/{self.depth_variant}"][frame], dtype=np.float32)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ref = self._select(index)
        master_handle = None
        derived_handle = None
        with h5py.File(self._processed_path(ref.episode), "r") as processed:
            try:
                if self.include_depth:
                    master_handle = h5py.File(self._master_path(ref.episode), "r")
                if self.include_depth and self.depth_variant != "gt":
                    derived_handle = h5py.File(self._derived_path(ref.episode), "r")

                qpos = np.asarray(processed["observations/qpos"][ref.frame], dtype=np.float32)
                action_start = max(0, ref.frame - 1)
                action = np.asarray(processed["action"][action_start:], dtype=np.float32)
                action_len = min(len(action), self.max_action_len)
                padded_action = np.zeros((self.max_action_len, action.shape[1]), dtype=np.float32)
                padded_action[:action_len] = action[:action_len]
                is_pad = np.ones(self.max_action_len, dtype=bool)
                is_pad[:action_len] = False

                rgb_views = []
                depth_views = []
                validity_views = []
                xyz_views = []
                for camera in self.camera_order:
                    processed_name = PROCESSED_CAMERA[camera]
                    rgb = np.asarray(processed[f"observations/images/{processed_name}"][ref.frame], dtype=np.uint8)
                    target_hw = rgb.shape[:2]
                    rgb_views.append(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)
                    if self.include_depth:
                        assert master_handle is not None
                        depth_mm_raw = self._read_depth(master_handle, derived_handle, camera, ref.frame)
                        depth_mm = cv2.resize(
                            depth_mm_raw,
                            (target_hw[1], target_hw[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        validity = np.isfinite(depth_mm) & (depth_mm > 0)
                        depth_m = np.where(validity, depth_mm * 0.001, 0.0).astype(np.float32)
                        depth_views.append(depth_m[None])
                        validity_views.append(validity.astype(np.float32)[None])
                        if self.include_xyz:
                            intrinsic_raw = np.asarray(
                                master_handle[f"observation/{camera}/intrinsic_cv"][ref.frame], dtype=np.float32
                            )
                            intrinsic = resize_intrinsic(intrinsic_raw, depth_mm_raw.shape, target_hw)
                            xyz_views.append(depth_to_xyz_map(depth_m, validity, intrinsic))
                        else:
                            xyz_views.append(np.zeros((3, 1, 1), dtype=np.float32))
                    else:
                        depth_views.append(np.zeros((1, 1, 1), dtype=np.float32))
                        validity_views.append(np.zeros((1, 1, 1), dtype=np.float32))
                        xyz_views.append(np.zeros((3, 1, 1), dtype=np.float32))
            finally:
                if derived_handle is not None:
                    derived_handle.close()
                if master_handle is not None:
                    master_handle.close()

        normalized_action = (padded_action - self.stats["action_mean"]) / self.stats["action_std"]
        normalized_qpos = (qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        return {
            "image": torch.from_numpy(np.stack(rgb_views)),
            "depth_m": torch.from_numpy(np.stack(depth_views)),
            "validity": torch.from_numpy(np.stack(validity_views)),
            "xyz_map_m": torch.from_numpy(np.stack(xyz_views)),
            "qpos": torch.from_numpy(normalized_qpos.astype(np.float32)),
            "action": torch.from_numpy(normalized_action.astype(np.float32)),
            "is_pad": torch.from_numpy(is_pad),
            "episode": torch.tensor(ref.episode, dtype=torch.int32),
            "frame": torch.tensor(ref.frame, dtype=torch.int32),
        }


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}
