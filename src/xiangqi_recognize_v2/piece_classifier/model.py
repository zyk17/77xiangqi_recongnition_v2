from __future__ import annotations

import torch.nn as nn

from ..shared.mobilenet import build_mobilenet_v3_small_head


def build_piece_classifier(
    num_classes: int,
    *,
    pretrained: bool = False,
) -> nn.Module:
    return build_mobilenet_v3_small_head(num_classes, pretrained=pretrained, sigmoid_output=False)
