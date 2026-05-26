from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

GRID_X_COUNT = 9
GRID_Y_COUNT = 10
GRID_CELL_COUNT = GRID_X_COUNT * GRID_Y_COUNT
GEOM_DIM = GRID_X_COUNT + GRID_Y_COUNT
MAP_DOWNSAMPLE = 4
EVIDENCE_CHANNELS = 2
PIECE_MAP_INDEX = 0
CROSS_MAP_INDEX = 1


def _point_value(point: Mapping[str, Any], key: str) -> float:
    value = point.get(key)
    if value is None:
        raise ValueError(f"grid point 缺少 {key}")
    return float(value)


def _estimate_spacing(
    grid_points: Sequence[Sequence[Mapping[str, Any]]],
) -> tuple[float, float]:
    xs = [_point_value(grid_points[0][col], "x") for col in range(GRID_X_COUNT)]
    ys = [_point_value(grid_points[row][0], "y") for row in range(GRID_Y_COUNT)]
    dx_values = [xs[col + 1] - xs[col] for col in range(GRID_X_COUNT - 1)]
    dy_values = [ys[row + 1] - ys[row] for row in range(GRID_Y_COUNT - 1)]
    dx = float(sum(dx_values) / max(len(dx_values), 1))
    dy = float(sum(dy_values) / max(len(dy_values), 1))
    return max(dx, 1.0), max(dy, 1.0)


def evidence_map_size(*, target_width: int, target_height: int, downsample: int = MAP_DOWNSAMPLE) -> tuple[int, int]:
    if target_width <= 0 or target_height <= 0:
        raise ValueError("目标尺寸非法")
    if downsample <= 0:
        raise ValueError("downsample 必须为正")
    return max(1, target_width // downsample), max(1, target_height // downsample)


def _stamp_gaussian(
    channel: torch.Tensor,
    *,
    center_x: float,
    center_y: float,
    sigma: float,
) -> None:
    map_h, map_w = channel.shape
    radius = max(1, int(math.ceil(sigma * 3.0)))
    x0 = max(0, int(math.floor(center_x - radius)))
    x1 = min(map_w, int(math.ceil(center_x + radius)) + 1)
    y0 = max(0, int(math.floor(center_y - radius)))
    y1 = min(map_h, int(math.ceil(center_y + radius)) + 1)
    if x0 >= x1 or y0 >= y1:
        return

    yy, xx = torch.meshgrid(
        torch.arange(y0, y1, dtype=torch.float32),
        torch.arange(x0, x1, dtype=torch.float32),
        indexing="ij",
    )
    gaussian = torch.exp(-(((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * sigma**2)))
    channel[y0:y1, x0:x1] = torch.maximum(channel[y0:y1, x0:x1], gaussian)


def grid_to_evidence_maps(
    payload: Mapping[str, Any],
    *,
    image_width: int,
    image_height: int,
    target_width: int,
    target_height: int,
    downsample: int = MAP_DOWNSAMPLE,
) -> torch.Tensor:
    """从 grid_points 生成 piece_map/cross_map，形状 (2, H/4, W/4)。"""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("图像尺寸非法")
    if target_width <= 0 or target_height <= 0:
        raise ValueError("目标尺寸非法")

    grid_points = payload.get("grid_points")
    if not isinstance(grid_points, list) or len(grid_points) != GRID_Y_COUNT:
        raise ValueError("grid_points 行数应为 10")

    map_w, map_h = evidence_map_size(
        target_width=target_width,
        target_height=target_height,
        downsample=downsample,
    )
    scale_x = target_width / float(image_width)
    scale_y = target_height / float(image_height)
    map_scale_x = scale_x / float(downsample)
    map_scale_y = scale_y / float(downsample)

    dx_raw, dy_raw = _estimate_spacing(grid_points)
    sigma_x = max(0.75, dx_raw * map_scale_x * 0.35)
    sigma_y = max(0.75, dy_raw * map_scale_y * 0.35)
    sigma = min(sigma_x, sigma_y)

    maps = torch.zeros((EVIDENCE_CHANNELS, map_h, map_w), dtype=torch.float32)
    for row in grid_points:
        if not isinstance(row, list) or len(row) != GRID_X_COUNT:
            raise ValueError("grid_points 每行列数应为 9")
        for point in row:
            if not isinstance(point, Mapping):
                raise ValueError("grid point 非法")
            point_type = str(point.get("type", "")).lower()
            if point_type not in {"piece", "cross"}:
                raise ValueError(f"未知格点类型: {point_type!r}")
            channel_index = PIECE_MAP_INDEX if point_type == "piece" else CROSS_MAP_INDEX
            center_x = _point_value(point, "x") * map_scale_x
            center_y = _point_value(point, "y") * map_scale_y
            _stamp_gaussian(
                maps[channel_index],
                center_x=center_x,
                center_y=center_y,
                sigma=sigma,
            )
    return maps.clamp_(0.0, 1.0)


def load_evidence_maps_file(
    label_path: Path,
    *,
    image_width: int,
    image_height: int,
    target_width: int,
    target_height: int,
    downsample: int = MAP_DOWNSAMPLE,
) -> torch.Tensor:
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"标注应为 object: {label_path}")
    return grid_to_evidence_maps(
        payload,
        image_width=image_width,
        image_height=image_height,
        target_width=target_width,
        target_height=target_height,
        downsample=downsample,
    )


def render_evidence_overlay(
    image: Image.Image,
    evidence_maps: torch.Tensor | np.ndarray,
    *,
    alpha: float = 0.55,
) -> Image.Image:
    """piece=红，cross=青，叠加到 RGB 图像上供人眼验收。"""
    maps = (
        evidence_maps.detach().cpu().numpy()
        if isinstance(evidence_maps, torch.Tensor)
        else np.asarray(evidence_maps, dtype=np.float32)
    )
    if maps.ndim != 3 or maps.shape[0] != EVIDENCE_CHANNELS:
        raise ValueError(f"evidence_maps 形状应为 ({EVIDENCE_CHANNELS}, H, W)")

    base = image.convert("RGB")
    overlay_arr = np.zeros((maps.shape[1], maps.shape[2], 3), dtype=np.float32)
    overlay_arr[..., 0] = maps[PIECE_MAP_INDEX]
    overlay_arr[..., 1] = maps[CROSS_MAP_INDEX]
    overlay_arr[..., 2] = maps[CROSS_MAP_INDEX]
    heat = Image.fromarray(np.clip(overlay_arr * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    heat = heat.resize(base.size, Image.Resampling.BILINEAR)
    return Image.blend(base, heat, alpha=alpha)


def parse_grid_label(
    payload: Mapping[str, Any],
    *,
    image_width: int,
    image_height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从合成标签解析归一化 x[9]/y[10] 与 piece 目标 (90)。保留供 decode 测试与 grid recovery。"""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("图像尺寸非法")

    grid_points = payload.get("grid_points")
    if not isinstance(grid_points, list) or len(grid_points) != GRID_Y_COUNT:
        raise ValueError("grid_points 行数应为 10")

    x_norm: list[float] = []
    y_norm: list[float] = []
    type_targets: list[float] = []

    for row_index, row in enumerate(grid_points):
        if not isinstance(row, list) or len(row) != GRID_X_COUNT:
            raise ValueError(f"grid_points 第 {row_index} 行列数应为 9")
        if row_index == 0:
            for col_index, point in enumerate(row):
                if not isinstance(point, Mapping):
                    raise ValueError(f"grid point 非法: row=0 col={col_index}")
                x_norm.append(_point_value(point, "x") / image_width)
        y_norm.append(_point_value(row[0], "y") / image_height)
        for col_index, point in enumerate(row):
            if not isinstance(point, Mapping):
                raise ValueError(f"grid point 非法: row={row_index} col={col_index}")
            point_type = str(point.get("type", "")).lower()
            if point_type not in {"piece", "cross"}:
                raise ValueError(f"未知格点类型: {point_type!r}")
            type_targets.append(1.0 if point_type == "piece" else 0.0)

    geom = torch.tensor([*x_norm, *y_norm], dtype=torch.float32).clamp_(0.0, 1.0)
    types = torch.tensor(type_targets, dtype=torch.float32).clamp_(0.0, 1.0)
    return geom, types


def load_grid_label_file(label_path: Path, *, image_width: int, image_height: int) -> tuple[torch.Tensor, torch.Tensor]:
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"标注应为 object: {label_path}")
    return parse_grid_label(payload, image_width=image_width, image_height=image_height)


def geom_to_pixel_lines(geom: torch.Tensor, *, image_width: int, image_height: int) -> tuple[torch.Tensor, torch.Tensor]:
    if geom.shape[-1] != GEOM_DIM:
        raise ValueError(f"geom 维度应为 {GEOM_DIM}")
    x = geom[..., :GRID_X_COUNT] * float(image_width)
    y = geom[..., GRID_X_COUNT:] * float(image_height)
    return x, y
