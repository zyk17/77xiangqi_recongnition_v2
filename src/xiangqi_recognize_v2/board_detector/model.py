from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .labels import EVIDENCE_CHANNELS, MAP_DOWNSAMPLE


class BoardDetector(nn.Module):
    """整图 -> piece_map/cross_map 空间证据图 (B, 2, H/4, W/4)。"""

    def __init__(
        self,
        features: nn.Module,
        decoder: nn.Module,
        *,
        map_downsample: int = MAP_DOWNSAMPLE,
    ) -> None:
        super().__init__()
        self.features = features
        self.decoder = decoder
        self.map_downsample = map_downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        maps = self.decoder(features)
        target_h = max(1, x.shape[2] // self.map_downsample)
        target_w = max(1, x.shape[3] // self.map_downsample)
        maps = F.interpolate(maps, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return torch.sigmoid(maps)


def build_board_detector(*, pretrained: bool = False, map_downsample: int = MAP_DOWNSAMPLE) -> BoardDetector:
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    base = mobilenet_v3_small(weights=weights)
    in_features = 576
    decoder = nn.Sequential(
        nn.Conv2d(in_features, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(128, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(64, 32, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(32, EVIDENCE_CHANNELS, kernel_size=1),
    )
    return BoardDetector(base.features, decoder, map_downsample=map_downsample)
