from __future__ import annotations

from typing import Any


def build_multimodal_boundary_model(
    input_dims: dict[str, int],
    hidden_dim: int = 128,
    layers: int = 3,
    heads: int = 4,
) -> Any:
    """Build the optional trainable boundary refiner without importing torch at package import."""
    import torch
    from torch import nn

    class MultiModalBoundaryModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.names = tuple(sorted(input_dims))
            self.encoders = nn.ModuleDict(
                {
                    name: nn.Sequential(
                        nn.Linear(input_dims[name], hidden_dim),
                        nn.GELU(),
                        nn.LayerNorm(hidden_dim),
                    )
                    for name in self.names
                }
            )
            self.modality_embeddings = nn.Parameter(torch.zeros(len(self.names), hidden_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            self.heads = nn.ModuleDict(
                {level: nn.Linear(hidden_dim, 1) for level in ("task", "subtask", "event")}
            )

        def forward(
            self,
            features: dict[str, Any],
            modality_valid: dict[str, Any] | None = None,
            padding_mask: Any | None = None,
        ) -> dict[str, Any]:
            encoded = []
            weights = []
            for index, name in enumerate(self.names):
                if name not in features:
                    continue
                item = self.encoders[name](features[name]) + self.modality_embeddings[index]
                if modality_valid and name in modality_valid:
                    weight = modality_valid[name].to(item.dtype).unsqueeze(-1)
                else:
                    weight = torch.ones_like(item[..., :1])
                encoded.append(item * weight)
                weights.append(weight)
            if not encoded:
                raise ValueError("at least one configured modality must be present")
            fused = torch.stack(encoded).sum(0) / torch.stack(weights).sum(0).clamp_min(1.0)
            temporal = self.temporal(fused, src_key_padding_mask=padding_mask)
            return {name: head(temporal).squeeze(-1) for name, head in self.heads.items()}

    return MultiModalBoundaryModel()


def hierarchical_boundary_loss(
    logits: dict[str, Any],
    targets: dict[str, Any],
    valid: Any | None = None,
    positive_weights: dict[str, float] | None = None,
    hierarchy_weight: float = 0.1,
) -> Any:
    import torch
    from torch.nn import functional

    losses = []
    positive_weights = positive_weights or {}
    for level, prediction in logits.items():
        target = targets[level].to(prediction.dtype)
        pos_weight = torch.as_tensor(
            positive_weights.get(level, 1.0), device=prediction.device, dtype=prediction.dtype
        )
        loss = functional.binary_cross_entropy_with_logits(
            prediction, target, reduction="none", pos_weight=pos_weight
        )
        if valid is not None:
            mask = valid.to(loss.dtype)
            loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)
        else:
            loss = loss.mean()
        losses.append(loss)
    base_loss = torch.stack(losses).mean()
    if not {"task", "subtask", "event"}.issubset(logits):
        return base_loss
    task = torch.sigmoid(logits["task"])
    subtask = torch.sigmoid(logits["subtask"])
    event = torch.sigmoid(logits["event"])
    hierarchy = functional.relu(task - subtask) + functional.relu(subtask - event)
    if valid is not None:
        mask = valid.to(hierarchy.dtype)
        hierarchy = (hierarchy * mask).sum() / mask.sum().clamp_min(1.0)
    else:
        hierarchy = hierarchy.mean()
    return base_loss + float(hierarchy_weight) * hierarchy
