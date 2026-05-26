from __future__ import annotations

import torch
import torch.nn as nn


class SigmoidHeadModel(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.backbone(x))


def build_mobilenet_v3_small_head(
    output_dim: int,
    *,
    pretrained: bool = False,
    sigmoid_output: bool = False,
) -> nn.Module:
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, output_dim)
    if sigmoid_output:
        return SigmoidHeadModel(model)
    return model
