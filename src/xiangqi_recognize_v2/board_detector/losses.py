from __future__ import annotations

import torch
import torch.nn.functional as F

from .labels import CROSS_MAP_INDEX, PIECE_MAP_INDEX


def _heatmap_bce(pred: torch.Tensor, target: torch.Tensor, *, pos_weight: float) -> torch.Tensor:
    weight = 1.0 + float(pos_weight) * target
    return (F.binary_cross_entropy(pred, target, reduction="none") * weight).mean()


def board_detector_loss(
    pred_maps: torch.Tensor,
    target_maps: torch.Tensor,
    *,
    piece_weight: float = 1.0,
    cross_weight: float = 1.0,
    pos_weight: float = 8.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    piece_loss = _heatmap_bce(
        pred_maps[:, PIECE_MAP_INDEX],
        target_maps[:, PIECE_MAP_INDEX],
        pos_weight=pos_weight,
    )
    cross_loss = _heatmap_bce(
        pred_maps[:, CROSS_MAP_INDEX],
        target_maps[:, CROSS_MAP_INDEX],
        pos_weight=pos_weight,
    )
    total = piece_weight * piece_loss + cross_weight * cross_loss
    return total, {
        "piece": float(piece_loss.item()),
        "cross": float(cross_loss.item()),
    }
