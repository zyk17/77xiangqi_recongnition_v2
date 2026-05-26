from __future__ import annotations

import torch
from PIL import Image, ImageDraw

from xiangqi_recognize_v2.board_detector.decode import (
    RECOVERY_FALLBACK_PENALTY,
    batch_decoded_grid_geom_mae,
    cell_bboxes_from_evidence_maps,
    cell_bboxes_from_geom,
    recover_geom_from_evidence_maps,
)
from xiangqi_recognize_v2.board_detector.labels import (
    EVIDENCE_CHANNELS,
    GEOM_DIM,
    GRID_CELL_COUNT,
    grid_to_evidence_maps,
    parse_grid_label,
    render_evidence_overlay,
)


def _sample_grid_points() -> list[list[dict[str, object]]]:
    grid_points: list[list[dict[str, object]]] = []
    for row in range(10):
        row_points: list[dict[str, object]] = []
        for col in range(9):
            row_points.append(
                {
                    "x": 100.0 + col * 50.0,
                    "y": 80.0 + row * 48.0,
                    "type": "piece" if (row + col) % 2 == 0 else "cross",
                }
            )
        grid_points.append(row_points)
    return grid_points


def test_parse_grid_label_and_decode_cells() -> None:
    grid_points = _sample_grid_points()
    geom, types = parse_grid_label({"grid_points": grid_points}, image_width=640, image_height=640)
    assert geom.shape == (GEOM_DIM,)
    assert types.shape == (GRID_CELL_COUNT,)
    boxes = cell_bboxes_from_geom(geom, image_width=640, image_height=640)
    assert boxes.shape == (90, 4)
    assert float(boxes[0, 0]) < float(boxes[0, 2])
    width_col0 = float(boxes[0, 2] - boxes[0, 0])
    width_col8 = float(boxes[8, 2] - boxes[8, 0])
    assert abs(width_col0 - width_col8) < 1.0

    x_norm = [0.1, 0.2, 0.3, 0.4, 0.5, 0.62, 0.78, 0.9, 0.98]
    y_norm = [0.1 + row * 0.08 for row in range(10)]
    skew_geom = torch.tensor([*x_norm, *y_norm], dtype=torch.float32)
    skew_boxes = cell_bboxes_from_geom(skew_geom, image_width=1000, image_height=1000)
    col_widths = [float(skew_boxes[col, 2] - skew_boxes[col, 0]) for col in range(9)]
    assert max(col_widths) - min(col_widths) > 10.0


def test_grid_to_evidence_maps_and_overlay() -> None:
    grid_points = _sample_grid_points()
    payload = {"grid_points": grid_points}
    image_width, image_height = 640, 640
    target_size = 224
    maps = grid_to_evidence_maps(
        payload,
        image_width=image_width,
        image_height=image_height,
        target_width=target_size,
        target_height=target_size,
    )
    assert maps.shape == (EVIDENCE_CHANNELS, target_size // 4, target_size // 4)
    assert float(maps.max()) > 0.9
    assert float(maps[0].sum()) > 0.0
    assert float(maps[1].sum()) > 0.0

    image = Image.new("RGB", (image_width, image_height), color=(40, 60, 80))
    draw = ImageDraw.Draw(image)
    for row in grid_points:
        for point in row:
            x = int(point["x"])
            y = int(point["y"])
            color = (255, 64, 64) if point["type"] == "piece" else (64, 220, 220)
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
    overlay = render_evidence_overlay(image, maps)
    assert overlay.size == image.size

    recovered_geom, recovery_ok = recover_geom_from_evidence_maps(
        maps,
        input_width=target_size,
        input_height=target_size,
        image_width=image_width,
        image_height=image_height,
    )
    assert recovered_geom.shape == (GEOM_DIM,)
    assert recovery_ok
    recovered_boxes, _ = cell_bboxes_from_evidence_maps(
        maps,
        input_width=target_size,
        input_height=target_size,
        image_width=image_width,
        image_height=image_height,
    )
    assert recovered_boxes.shape == (90, 4)


def test_recover_geom_falls_back_on_weak_maps() -> None:
    maps = torch.zeros((EVIDENCE_CHANNELS, 56, 56), dtype=torch.float32)
    maps[0, 20, 10] = 0.4
    geom, recovery_ok = recover_geom_from_evidence_maps(
        maps,
        input_width=224,
        input_height=224,
        image_width=640,
        image_height=640,
        threshold=0.35,
        allow_fallback=True,
    )
    assert geom.shape == (GEOM_DIM,)
    assert not recovery_ok


def test_batch_decoded_grid_geom_mae_penalizes_failed_recovery() -> None:
    grid_points = _sample_grid_points()
    target_maps = grid_to_evidence_maps(
        {"grid_points": grid_points},
        image_width=640,
        image_height=640,
        target_width=224,
        target_height=224,
    )
    weak_pred = torch.zeros_like(target_maps)
    weak_pred[0, 20, 10] = 0.4
    batch_target = target_maps.unsqueeze(0)
    batch_pred = weak_pred.unsqueeze(0)

    selection_mae, recovery_ok_rate, ok_only_mae = batch_decoded_grid_geom_mae(
        batch_pred,
        batch_target,
        image_size=224,
        map_downsample=4,
        threshold=0.35,
    )
    assert recovery_ok_rate == 0.0
    assert selection_mae == RECOVERY_FALLBACK_PENALTY
    assert ok_only_mae == RECOVERY_FALLBACK_PENALTY

    good_pred = target_maps.clone()
    selection_mae_good, recovery_ok_rate_good, ok_only_mae_good = batch_decoded_grid_geom_mae(
        good_pred.unsqueeze(0),
        batch_target,
        image_size=224,
        map_downsample=4,
        threshold=0.25,
    )
    assert recovery_ok_rate_good == 1.0
    assert selection_mae_good < 0.2
    assert ok_only_mae_good < 0.2
