from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .variants import get_variant


def _last_feature(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        return list(output.values())[-1]
    if isinstance(output, (tuple, list)):
        return output[-1]
    if torch.is_tensor(output):
        return output
    raise TypeError(f"Unsupported backbone output type: {type(output)!r}")


def _replace_first_conv(joiner: nn.Module, in_channels: int, depth_weight: str) -> None:
    body = joiner[0].body
    old = body["conv1"]
    replacement = nn.Conv2d(
        in_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=False,
    ).to(device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        replacement.weight.zero_()
        replacement.weight[:, :3].copy_(old.weight)
        if in_channels > 3 and depth_weight == "rgb_mean":
            replacement.weight[:, 3:4].copy_(old.weight.mean(dim=1, keepdim=True))
    body["conv1"] = replacement


def _replace_geometry_conv(joiner: nn.Module, in_channels: int) -> None:
    body = joiner[0].body
    old = body["conv1"]
    replacement = nn.Conv2d(
        in_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=False,
    ).to(device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        replacement.weight.zero_()
        replacement.weight[:, 0:1].copy_(old.weight.mean(dim=1, keepdim=True))
    body["conv1"] = replacement


class StridedGeometryCNN(nn.Module):
    """Five stride-2 stages map 480x640 inputs to the ACT 15x20 feature grid."""

    def __init__(self, in_channels: int, output_dim: int) -> None:
        super().__init__()
        channels = (in_channels, 64, 128, 256, 384, output_dim)
        layers: list[nn.Module] = []
        for source, target in zip(channels, channels[1:]):
            groups = min(32, target)
            layers.extend(
                (
                    nn.Conv2d(source, target, 3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(groups, target),
                    nn.GELU(),
                )
            )
        self.net = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class DepthResNetEncoder(nn.Module):
    def __init__(self, official_rgb_joiner: nn.Module, in_channels: int = 1) -> None:
        super().__init__()
        cloned = copy.deepcopy(official_rgb_joiner)
        _replace_geometry_conv(cloned, in_channels)
        self.body = cloned[0]

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return _last_feature(self.body(values))


class DepthTransformerEncoder(nn.Module):
    def __init__(self, in_channels: int, dim: int, layers: int = 2, heads: int = 8) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=32, stride=32)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, layers, nn.LayerNorm(dim))

    def forward(
        self,
        values: torch.Tensor,
        target_hw: tuple[int, int],
        position: torch.Tensor,
    ) -> torch.Tensor:
        feature = self.patch_embed(values)
        if feature.shape[-2:] != target_hw:
            feature = F.interpolate(feature, target_hw, mode="bilinear", align_corners=False)
        tokens = feature.flatten(2).transpose(1, 2)
        pos_tokens = position.flatten(2).transpose(1, 2).to(tokens.dtype)
        tokens = self.encoder(tokens + pos_tokens)
        return tokens.transpose(1, 2).reshape(values.shape[0], -1, *target_hw)


class PointTokenEncoder(nn.Module):
    def __init__(self, in_channels: int, dim: int, grid_hw: tuple[int, int] = (8, 10)) -> None:
        super().__init__()
        self.grid_hw = grid_hw
        self.point_mlp = nn.Sequential(
            nn.Conv2d(in_channels, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, 256, 1),
            nn.GELU(),
            nn.Conv2d(256, dim, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        sampled = F.adaptive_avg_pool2d(values, self.grid_hw)
        return self.point_mlp(sampled).flatten(2).transpose(1, 2)


class LingBotFeatureEncoder(nn.Module):
    """Frozen LingBot model kept out of checkpoints; only the 1024->512 adapter is trained."""

    def __init__(
        self,
        repo: str | Path,
        checkpoint: str | Path,
        output_dim: int,
        vendor: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.repo = str(repo)
        self.checkpoint = str(checkpoint)
        self.vendor = None if vendor is None else str(vendor)
        self.adapter = nn.Conv2d(1024, output_dim, 1)
        self.register_buffer("rgb_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))
        object.__setattr__(self, "_frozen_model", None)

    def _load(self, device: torch.device) -> nn.Module:
        model = object.__getattribute__(self, "_frozen_model")
        if model is not None:
            return model
        if self.repo not in sys.path:
            sys.path.insert(0, self.repo)
        if self.vendor is not None and self.vendor not in sys.path:
            sys.path.insert(0, self.vendor)
        module = importlib.import_module("mdm.model.v2")
        model = module.MDMModel.from_pretrained(self.checkpoint).to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        object.__setattr__(self, "_frozen_model", model)
        return model

    def forward(self, normalized_rgb: torch.Tensor, depth_m: torch.Tensor) -> torch.Tensor:
        model = self._load(normalized_rgb.device)
        rgb = normalized_rgb * self.rgb_std.to(normalized_rgb.dtype) + self.rgb_mean.to(normalized_rgb.dtype)
        with torch.no_grad():
            features, _ = model.infer_feat(
                rgb.float(),
                depth_in=depth_m[:, 0].float(),
                resolution_level=0,
                apply_mask=False,
                use_fp16=True,
            )
        # LingBot returns inference tensors; clone before the trainable adapter.
        features = features.float().to(self.adapter.weight.dtype).clone()
        return self.adapter(features)


class GatedCrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8) -> None:
        super().__init__()
        self.rgb_norm = nn.LayerNorm(dim)
        self.geometry_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout=0.1, batch_first=True)
        self.output = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.gate_logit = nn.Parameter(torch.full((dim,), -2.0))
        self.last_gate: torch.Tensor | None = None

    def forward(
        self,
        rgb_feature: torch.Tensor,
        geometry_tokens: torch.Tensor,
        quality: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels, height, width = rgb_feature.shape
        rgb_tokens = rgb_feature.flatten(2).transpose(1, 2)
        normalized_geometry = self.geometry_norm(geometry_tokens)
        cross = self.cross_attention(
            self.rgb_norm(rgb_tokens), normalized_geometry, normalized_geometry, need_weights=False
        )[0]
        if quality.shape[1] != rgb_tokens.shape[1]:
            quality = quality.mean(dim=1, keepdim=True).expand(-1, rgb_tokens.shape[1], -1)
        gate = torch.sigmoid(self.gate_logit).view(1, 1, -1) * quality.clamp(0.0, 1.0)
        fused = rgb_tokens + gate * cross
        fused = fused + gate * self.output(fused)
        self.last_gate = gate.detach()
        return fused.transpose(1, 2).reshape(batch, channels, height, width)


class FusionJoiner(nn.Module):
    """Joiner-compatible frontend consumed by the unmodified official DETRVAE."""

    def __init__(
        self,
        rgb_joiner: nn.Module,
        mode: str,
        lingbot_repo: str | Path | None = None,
        lingbot_checkpoint: str | Path | None = None,
        lingbot_vendor: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.rgb_joiner = rgb_joiner
        self.mode = mode
        self.num_channels = int(rgb_joiner.num_channels)
        self.fusion = GatedCrossAttention(self.num_channels)
        if mode == "depth_resnet":
            self.geometry_encoder: nn.Module | None = DepthResNetEncoder(rgb_joiner, 1)
        elif mode == "xyz_cnn":
            self.geometry_encoder = StridedGeometryCNN(3, self.num_channels)
        elif mode == "point_tokens":
            self.geometry_encoder = PointTokenEncoder(3, self.num_channels)
        elif mode == "depth_transformer":
            self.geometry_encoder = DepthTransformerEncoder(1, self.num_channels)
        elif mode == "lingbot":
            if lingbot_repo is None or lingbot_checkpoint is None:
                raise ValueError("ACT6_LINGBOT_DEPTH requires lingbot_repo and lingbot_checkpoint")
            self.geometry_encoder = LingBotFeatureEncoder(
                lingbot_repo,
                lingbot_checkpoint,
                self.num_channels,
                vendor=lingbot_vendor,
            )
        else:
            raise ValueError(f"Unsupported fusion mode: {mode}")

    def forward(self, packed: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        rgb = packed[:, :3]
        rgb_features, positions = self.rgb_joiner(rgb)
        rgb_feature = rgb_features[-1]
        position = positions[-1]
        target_hw = rgb_feature.shape[-2:]

        if self.mode == "depth_resnet":
            geometry_input = packed[:, 3:4]
            geometry_feature = self.geometry_encoder(geometry_input)
            geometry_feature = F.interpolate(geometry_feature, target_hw, mode="bilinear", align_corners=False)
            geometry_tokens = geometry_feature.flatten(2).transpose(1, 2)
        elif self.mode == "xyz_cnn":
            geometry_input = packed[:, 3:6]
            geometry_feature = self.geometry_encoder(geometry_input)
            geometry_feature = F.interpolate(geometry_feature, target_hw, mode="bilinear", align_corners=False)
            geometry_tokens = geometry_feature.flatten(2).transpose(1, 2)
        elif self.mode == "point_tokens":
            geometry_tokens = self.geometry_encoder(packed[:, 3:6])
        elif self.mode == "depth_transformer":
            geometry_input = packed[:, 3:4]
            geometry_feature = self.geometry_encoder(geometry_input, target_hw, position)
            geometry_tokens = geometry_feature.flatten(2).transpose(1, 2)
        else:
            geometry_input = packed[:, 3:4]
            geometry_feature = self.geometry_encoder(rgb, geometry_input)
            geometry_feature = F.interpolate(geometry_feature, target_hw, mode="bilinear", align_corners=False)
            geometry_tokens = geometry_feature.flatten(2).transpose(1, 2)

        quality = torch.ones(
            geometry_tokens.shape[0], geometry_tokens.shape[1], 1,
            device=geometry_tokens.device, dtype=geometry_tokens.dtype,
        )

        fused = self.fusion(rgb_feature, geometry_tokens, quality)
        return [fused], [position]

    def gate_summary(self) -> float | None:
        gate = self.fusion.last_gate
        return None if gate is None else float(gate.mean().cpu())


class PerViewFusionJoiner(nn.Module):
    """Route each camera to its own RGB ResNet18 and Depth ResNet18 pair."""

    def __init__(self, official_rgb_joiner: nn.Module, camera_count: int = 3) -> None:
        super().__init__()
        self.num_channels = int(official_rgb_joiner.num_channels)
        self.camera_count = camera_count
        self.views = nn.ModuleList(
            FusionJoiner(copy.deepcopy(official_rgb_joiner), "depth_resnet")
            for _ in range(camera_count)
        )

    def forward(self, packed: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if packed.shape[1] != 4 + self.camera_count:
            raise ValueError(
                f"Per-view frontend expected {4 + self.camera_count} channels, got {packed.shape[1]}"
            )
        camera_id = int(packed[:, 4:].mean(dim=(0, 2, 3)).argmax().item())
        return self.views[camera_id](packed[:, :4])

    def gate_summary(self) -> float | None:
        gates = [view.gate_summary() for view in self.views]
        values = [gate for gate in gates if gate is not None]
        return None if not values else sum(values) / len(values)


def install_frontend(
    model: nn.Module,
    variant_name: str,
    *,
    lingbot_repo: str | Path | None = None,
    lingbot_checkpoint: str | Path | None = None,
    lingbot_vendor: str | Path | None = None,
) -> nn.Module:
    spec = get_variant(variant_name)
    if spec.frontend == "official":
        return model

    official_joiner = model.backbones[0]
    if spec.frontend == "early":
        _replace_first_conv(official_joiner, spec.packed_channels, depth_weight="zero")
        return model

    if spec.frontend == "per_view_depth_resnet":
        model.backbones[0] = PerViewFusionJoiner(official_joiner)
        return model

    model.backbones[0] = FusionJoiner(
        official_joiner,
        spec.frontend,
        lingbot_repo=lingbot_repo,
        lingbot_checkpoint=lingbot_checkpoint,
        lingbot_vendor=lingbot_vendor,
    )
    return model


def frontend_gate_summary(model: nn.Module) -> float | None:
    if not getattr(model, "backbones", None):
        return None
    frontend = model.backbones[0]
    if hasattr(frontend, "gate_summary"):
        return frontend.gate_summary()
    return None
