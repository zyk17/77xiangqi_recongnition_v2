from __future__ import annotations

import torch

from .labels import GRID_X_COUNT, GRID_Y_COUNT, MAP_DOWNSAMPLE, geom_to_pixel_lines


def _cell_span_along_axis(lines: torch.Tensor, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    """交点 index 对应格点在轴向上的 [low, high) 边界（相邻中线）。"""
    if index < 0 or index >= lines.shape[0]:
        raise ValueError("index 超出范围")
    if index == 0:
        step = lines[1] - lines[0]
        low = lines[0] - 0.5 * step
    else:
        low = 0.5 * (lines[index - 1] + lines[index])
    if index == lines.shape[0] - 1:
        step = lines[index] - lines[index - 1]
        high = lines[index] + 0.5 * step
    else:
        high = 0.5 * (lines[index] + lines[index + 1])
    return low, high


def cell_bboxes_from_geom(
    geom: torch.Tensor,
    *,
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    """geom (19,) -> (90, 4)。每格使用局部 x/y 间距，而非全局平均 dx/dy。"""
    if geom.ndim != 1:
        raise ValueError("cell_bboxes_from_geom 仅支持单向量 geom")
    x_lines, y_lines = geom_to_pixel_lines(geom, image_width=image_width, image_height=image_height)

    boxes: list[torch.Tensor] = []
    for row in range(GRID_Y_COUNT):
        y1, y2 = _cell_span_along_axis(y_lines, row)
        for col in range(GRID_X_COUNT):
            x1, x2 = _cell_span_along_axis(x_lines, col)
            boxes.append(torch.stack([x1, y1, x2, y2], dim=0))
    return torch.stack(boxes, dim=0)


def roi_from_cell_boxes(
    boxes: torch.Tensor,
    *,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """90 格 bbox 的外接矩形 x1,y1,x2,y2。"""
    x1 = int(round(float(boxes[:, 0].min())))
    y1 = int(round(float(boxes[:, 1].min())))
    x2 = int(round(float(boxes[:, 2].max())))
    y2 = int(round(float(boxes[:, 3].max())))
    x1 = max(0, min(image_width - 1, x1))
    y1 = max(0, min(image_height - 1, y1))
    x2 = max(x1 + 1, min(image_width, x2))
    y2 = max(y1 + 1, min(image_height, y2))
    return x1, y1, x2, y2


def _uniform_axis_peaks(count: int, length: int) -> list[int]:
    if length <= 0:
        raise ValueError("axis length 非法")
    if count <= 1:
        return [max(0, length // 2)]
    step = (length - 1) / float(count - 1)
    return [int(round(index * step)) for index in range(count)]


def _find_axis_peaks(
    projection: torch.Tensor,
    count: int,
    *,
    min_distance: int,
    allow_fallback: bool = True,
) -> tuple[list[int], bool]:
    del min_distance  # 分 bin 恢复不依赖 NMS 间距
    length = int(projection.numel())
    if length == 0:
        if allow_fallback:
            return _uniform_axis_peaks(count, 1), False
        raise ValueError("projection 为空")

    peak_floor = max(float(projection.max()) * 0.02, 1e-6)
    bin_size = length / float(count)
    peaks: list[int] = []
    strong_bins = 0
    for index in range(count):
        start = int(round(index * bin_size))
        end = int(round((index + 1) * bin_size))
        end = max(start + 1, min(length, end))
        segment = projection[start:end]
        if segment.numel() == 0:
            peaks.append(int(round((index + 0.5) * bin_size)))
            continue
        local_index = int(torch.argmax(segment))
        peak_index = start + local_index
        peaks.append(peak_index)
        if float(segment[local_index]) >= peak_floor:
            strong_bins += 1

    ok = strong_bins >= max(1, count - 1)
    if ok:
        return peaks, True
    if allow_fallback:
        return _uniform_axis_peaks(count, length), False
    raise ValueError(f"未能找到 {count} 个轴峰值")


def _peaks_to_norm(
    peaks: list[int],
    *,
    count: int,
    downsample: int,
    input_size: int,
    image_size: int,
) -> list[float]:
    scale = float(image_size) / float(input_size)
    norm: list[float] = []
    for peak in peaks[:count]:
        pixel = (peak + 0.5) * downsample * scale
        norm.append(min(1.0, max(0.0, pixel / float(image_size))))
    while len(norm) < count:
        norm.append(norm[-1] if norm else 0.5)
    return norm


def recover_geom_from_evidence_maps(
    evidence_maps: torch.Tensor,
    *,
    input_width: int,
    input_height: int,
    image_width: int,
    image_height: int,
    downsample: int = MAP_DOWNSAMPLE,
    threshold: float = 0.25,
    allow_fallback: bool = True,
) -> tuple[torch.Tensor, bool]:
    """从 piece/cross 证据图恢复归一化 x[9]/y[10]。返回 (geom, recovery_ok)。"""
    if evidence_maps.ndim != 3 or evidence_maps.shape[0] != 2:
        raise ValueError("evidence_maps 形状应为 (2, H, W)")

    combined = torch.maximum(evidence_maps[0], evidence_maps[1])
    map_h, map_w = combined.shape
    min_x_distance = max(1, map_w // (GRID_X_COUNT + 2))
    min_y_distance = max(1, map_h // (GRID_Y_COUNT + 2))

    thresholds = [threshold]
    if threshold > 0.0:
        thresholds.extend([threshold * 0.5, threshold * 0.25, 0.0])
    x_peaks: list[int] | None = None
    y_peaks: list[int] | None = None
    x_ok = False
    y_ok = False
    for active_threshold in thresholds:
        active = combined if active_threshold <= 0.0 else combined * (combined >= active_threshold).float()
        x_projection = active.sum(dim=0)
        y_projection = active.sum(dim=1)
        candidate_x, candidate_x_ok = _find_axis_peaks(
            x_projection,
            GRID_X_COUNT,
            min_distance=min_x_distance,
            allow_fallback=allow_fallback,
        )
        candidate_y, candidate_y_ok = _find_axis_peaks(
            y_projection,
            GRID_Y_COUNT,
            min_distance=min_y_distance,
            allow_fallback=allow_fallback,
        )
        if candidate_x_ok and candidate_y_ok:
            x_peaks, y_peaks = candidate_x, candidate_y
            x_ok, y_ok = True, True
            break
        if x_peaks is None or (candidate_x_ok and not x_ok):
            x_peaks, x_ok = candidate_x, candidate_x_ok
        if y_peaks is None or (candidate_y_ok and not y_ok):
            y_peaks, y_ok = candidate_y, candidate_y_ok

    assert x_peaks is not None and y_peaks is not None
    if not x_ok or not y_ok:
        if allow_fallback:
            if not x_ok:
                x_peaks, x_ok = _uniform_axis_peaks(GRID_X_COUNT, map_w), False
            if not y_ok:
                y_peaks, y_ok = _uniform_axis_peaks(GRID_Y_COUNT, map_h), False
        else:
            raise ValueError("证据图 recovery 失败")

    x_norm = _peaks_to_norm(
        x_peaks,
        count=GRID_X_COUNT,
        downsample=downsample,
        input_size=input_width,
        image_size=image_width,
    )
    y_norm = _peaks_to_norm(
        y_peaks,
        count=GRID_Y_COUNT,
        downsample=downsample,
        input_size=input_height,
        image_size=image_height,
    )
    return torch.tensor([*x_norm, *y_norm], dtype=torch.float32), bool(x_ok and y_ok)


def cell_bboxes_from_evidence_maps(
    evidence_maps: torch.Tensor,
    *,
    input_width: int,
    input_height: int,
    image_width: int,
    image_height: int,
    downsample: int = MAP_DOWNSAMPLE,
    threshold: float = 0.25,
    allow_fallback: bool = True,
) -> tuple[torch.Tensor, bool]:
    geom, recovery_ok = recover_geom_from_evidence_maps(
        evidence_maps,
        input_width=input_width,
        input_height=input_height,
        image_width=image_width,
        image_height=image_height,
        downsample=downsample,
        threshold=threshold,
        allow_fallback=allow_fallback,
    )
    boxes = cell_bboxes_from_geom(geom, image_width=image_width, image_height=image_height)
    return boxes, recovery_ok


RECOVERY_FALLBACK_PENALTY = 1.0


def batch_decoded_grid_geom_mae(
    pred_maps: torch.Tensor,
    target_maps: torch.Tensor,
    *,
    image_size: int,
    map_downsample: int,
    threshold: float,
    fallback_penalty: float = RECOVERY_FALLBACK_PENALTY,
) -> tuple[float, float, float]:
    """对比 pred/target 证据图 recovery 后的 geom MAE。

    返回:
        selection_mae: 失败 recovery 的样本记 fallback_penalty，用于选 checkpoint
        recovery_ok_rate: pred recovery 成功率
        ok_only_mae: 仅统计 pred recovery 成功样本的 MAE
    """
    if pred_maps.ndim != 4 or target_maps.ndim != 4:
        raise ValueError("pred/target maps 形状应为 (B, C, H, W)")
    batch_size = int(pred_maps.shape[0])
    if batch_size == 0:
        return 0.0, 1.0, 0.0

    selection_sum = 0.0
    ok_mae_sum = 0.0
    ok_count = 0
    for index in range(batch_size):
        target_geom, _target_ok = recover_geom_from_evidence_maps(
            target_maps[index].detach().cpu(),
            input_width=image_size,
            input_height=image_size,
            image_width=image_size,
            image_height=image_size,
            downsample=map_downsample,
            threshold=threshold,
            allow_fallback=True,
        )
        pred_geom, pred_ok = recover_geom_from_evidence_maps(
            pred_maps[index].detach().cpu(),
            input_width=image_size,
            input_height=image_size,
            image_width=image_size,
            image_height=image_size,
            downsample=map_downsample,
            threshold=threshold,
            allow_fallback=True,
        )
        if pred_ok:
            sample_mae = float((pred_geom - target_geom).abs().mean().item())
            selection_sum += sample_mae
            ok_mae_sum += sample_mae
            ok_count += 1
        else:
            selection_sum += float(fallback_penalty)
    selection_mae = selection_sum / batch_size
    ok_only_mae = ok_mae_sum / ok_count if ok_count else float(fallback_penalty)
    recovery_ok_rate = ok_count / batch_size
    return selection_mae, recovery_ok_rate, ok_only_mae
