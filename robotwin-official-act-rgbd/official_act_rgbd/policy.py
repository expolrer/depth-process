from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import Normalize

from .frontends import frontend_gate_summary, install_frontend
from .variants import get_variant


DEFAULT_MODEL_CONFIG = {
    "action_dim": 14,
    "chunk_size": 50,
    "hidden_dim": 512,
    "dim_feedforward": 3200,
    "enc_layers": 4,
    "dec_layers": 7,
    "nheads": 8,
    "dropout": 0.1,
    "pre_norm": False,
    "position_embedding": "sine",
    "masks": False,
    "dilation": False,
    "backbone": "resnet18",
    "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
    "lr": 1e-5,
    "lr_backbone": 1e-5,
    "weight_decay": 1e-4,
    "kl_weight": 10.0,
}


def as_namespace(config: Mapping[str, Any] | Namespace) -> Namespace:
    if isinstance(config, Namespace):
        return config
    merged = dict(DEFAULT_MODEL_CONFIG)
    merged.update(config)
    return Namespace(**merged)


def official_kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Match RoboTwin ACT exactly: sum latent dimensions, then mean the batch."""
    if mu.ndim == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.ndim == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return klds.sum(dim=1).mean(dim=0, keepdim=True)


def build_official_model(
    config: Mapping[str, Any] | Namespace,
    variant: str,
    *,
    lingbot_repo: str | Path | None = None,
    lingbot_checkpoint: str | Path | None = None,
    lingbot_vendor: str | Path | None = None,
) -> nn.Module:
    """Build through the official constructor, then replace only the visual frontend."""
    args = as_namespace(config)
    from detr.models import build_ACT_model

    model = build_ACT_model(args)
    return install_frontend(
        model,
        variant,
        lingbot_repo=lingbot_repo,
        lingbot_checkpoint=lingbot_checkpoint,
        lingbot_vendor=lingbot_vendor,
    )


class OfficialACTRGBDPolicy(nn.Module):
    def __init__(
        self,
        config: Mapping[str, Any] | Namespace,
        variant: str,
        *,
        lingbot_repo: str | Path | None = None,
        lingbot_checkpoint: str | Path | None = None,
        lingbot_vendor: str | Path | None = None,
        max_depth_m: float = 2.0,
    ) -> None:
        super().__init__()
        self.args = as_namespace(config)
        self.variant = variant
        self.spec = get_variant(variant)
        self.kl_weight = float(self.args.kl_weight)
        self.max_depth_m = float(max_depth_m)
        self.normalize_rgb = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.model = build_official_model(
            self.args,
            variant,
            lingbot_repo=lingbot_repo,
            lingbot_checkpoint=lingbot_checkpoint,
            lingbot_vendor=lingbot_vendor,
        )

    def pack_inputs(
        self,
        image: torch.Tensor,
        depth_m: torch.Tensor | None = None,
        validity: torch.Tensor | None = None,
        xyz_map_m: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized_rgb = self.normalize_rgb(image)
        if not self.spec.requires_depth:
            return normalized_rgb
        if depth_m is None:
            raise ValueError(f"{self.variant} requires depth_m")
        if validity is None:
            validity = torch.isfinite(depth_m) & (depth_m > 0)
        validity = validity.to(dtype=depth_m.dtype)
        depth_m = torch.nan_to_num(depth_m).clamp(0.0, self.max_depth_m)

        if self.spec.frontend == "early":
            return torch.cat((normalized_rgb, depth_m / self.max_depth_m), dim=2)
        if self.spec.requires_xyz:
            if xyz_map_m is None:
                raise ValueError(f"{self.variant} requires xyz_map_m")
            xyz = torch.nan_to_num(xyz_map_m)
            xyz_scaled = torch.cat(
                (
                    (xyz[:, :, 0:1] / 1.0).clamp(-2.0, 2.0),
                    (xyz[:, :, 1:2] / 1.0).clamp(-2.0, 2.0),
                    (xyz[:, :, 2:3] / self.max_depth_m).clamp(0.0, 1.0),
                ),
                dim=2,
            )
            return torch.cat((normalized_rgb, xyz_scaled, validity), dim=2)
        if self.spec.frontend == "lingbot":
            return torch.cat((normalized_rgb, depth_m, validity), dim=2)
        return torch.cat((normalized_rgb, depth_m / self.max_depth_m, validity), dim=2)

    def forward(
        self,
        qpos: torch.Tensor,
        image: torch.Tensor,
        depth_m: torch.Tensor | None = None,
        validity: torch.Tensor | None = None,
        xyz_map_m: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
        is_pad: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        packed = self.pack_inputs(image, depth_m, validity, xyz_map_m)
        env_state = None
        if actions is None:
            prediction, _, _ = self.model(qpos, packed, env_state)
            return prediction

        actions = actions[:, : self.model.num_queries]
        if is_pad is None:
            raise ValueError("is_pad is required during training")
        is_pad = is_pad[:, : self.model.num_queries]
        prediction, _, (mu, logvar) = self.model(qpos, packed, env_state, actions, is_pad)
        all_l1 = F.l1_loss(actions, prediction, reduction="none")
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
        kl = official_kl_divergence(mu, logvar)[0]
        loss = l1 + kl * self.kl_weight
        return {"loss": loss, "l1": l1, "kl": kl, "prediction": prediction}

    def gate_summary(self) -> float | None:
        return frontend_gate_summary(self.model)


def build_policy_and_optimizer(
    config: Mapping[str, Any] | Namespace,
    variant: str,
    *,
    device: torch.device | str,
    lingbot_repo: str | Path | None = None,
    lingbot_checkpoint: str | Path | None = None,
    lingbot_vendor: str | Path | None = None,
) -> tuple[OfficialACTRGBDPolicy, torch.optim.Optimizer]:
    args = as_namespace(config)
    policy = OfficialACTRGBDPolicy(
        args,
        variant,
        lingbot_repo=lingbot_repo,
        lingbot_checkpoint=lingbot_checkpoint,
        lingbot_vendor=lingbot_vendor,
    ).to(device)
    backbone_parameters = []
    other_parameters = []
    for name, parameter in policy.model.named_parameters():
        if not parameter.requires_grad:
            continue
        destination = backbone_parameters if "backbone" in name else other_parameters
        destination.append(parameter)
    optimizer = torch.optim.AdamW(
        (
            {"params": other_parameters, "lr": float(args.lr)},
            {"params": backbone_parameters, "lr": float(args.lr_backbone)},
        ),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    return policy, optimizer
