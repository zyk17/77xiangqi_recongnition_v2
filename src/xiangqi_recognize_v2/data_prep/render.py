from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from ..shared.augmentations import OptionalImageAugment
from ..types import Roi


def edge_mean_color(image: Image.Image) -> tuple[int, int, int]:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    edge = np.concatenate(
        [
            arr[0, :, :].reshape(-1, 3),
            arr[-1, :, :].reshape(-1, 3),
            arr[:, 0, :].reshape(-1, 3),
            arr[:, -1, :].reshape(-1, 3),
        ]
    )
    mean = edge.mean(axis=0)
    return int(mean[0]), int(mean[1]), int(mean[2])


def prepare_board_and_sampling_roi(
    board_rgb: Image.Image,
    *,
    grid_roi_xyxy: Sequence[int] | None = None,
    pad_fraction: float = 0.0,
    inset_fraction: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
) -> tuple[Image.Image, Roi]:
    rgb = board_rgb.convert("RGB")
    w, h = rgb.size
    pad = int(round(min(w, h) * pad_fraction))
    canvas = ImageOps.expand(rgb, border=pad, fill=edge_mean_color(rgb)) if pad > 0 else rgb
    if grid_roi_xyxy is not None:
        x1, y1, x2, y2 = (int(value) for value in grid_roi_xyxy)
        x1 += pad
        y1 += pad
        x2 += pad
        y2 += pad
    else:
        wp, hp = canvas.size
        left, top, right, bottom = (float(x) for x in inset_fraction)
        outer_x1 = int(round(wp * left))
        outer_y1 = int(round(hp * top))
        outer_x2 = int(round(wp * (1.0 - right)))
        outer_y2 = int(round(hp * (1.0 - bottom)))
        # Old configs describe the outer playable board crop.
        # Convert it to the grid-point ROI by shrinking half a cell per side.
        outer_w = outer_x2 - outer_x1
        outer_h = outer_y2 - outer_y1
        dx = outer_w / 9.0
        dy = outer_h / 10.0
        x1 = int(round(outer_x1 + 0.5 * dx))
        y1 = int(round(outer_y1 + 0.5 * dy))
        x2 = int(round(outer_x2 - 0.5 * dx))
        y2 = int(round(outer_y2 - 0.5 * dy))
    return canvas, Roi(x=x1, y=y1, w=x2 - x1, h=y2 - y1)


def grid_roi_spacing(roi: Roi) -> tuple[float, float]:
    return roi.w / 8.0, roi.h / 9.0


def grid_point_center(roi: Roi, row: int, col: int) -> tuple[float, float]:
    dx, dy = grid_roi_spacing(roi)
    return roi.x + col * dx, roi.y + row * dy


def grid_cell_bbox(roi: Roi, row: int, col: int) -> tuple[int, int, int, int]:
    cx, cy = grid_point_center(roi, row, col)
    dx, dy = grid_roi_spacing(roi)
    x1 = int(round(cx - 0.5 * dx))
    y1 = int(round(cy - 0.5 * dy))
    x2 = int(round(cx + 0.5 * dx))
    y2 = int(round(cy + 0.5 * dy))
    return x1, y1, x2, y2


def expand_grid_roi_to_outer_bbox(roi: Roi) -> tuple[int, int, int, int]:
    dx, dy = grid_roi_spacing(roi)
    x1 = int(round(roi.x - 0.5 * dx))
    y1 = int(round(roi.y - 0.5 * dy))
    x2 = int(round(roi.x + roi.w + 0.5 * dx))
    y2 = int(round(roi.y + roi.h + 0.5 * dy))
    return x1, y1, x2, y2


def sample_board_cells(image: Image.Image, roi: Roi) -> list[tuple[int, int, tuple[int, int, int, int], Image.Image]]:
    cells: list[tuple[int, int, tuple[int, int, int, int], Image.Image]] = []
    for row in range(10):
        for col in range(9):
            x1, y1, x2, y2 = grid_cell_bbox(roi, row, col)
            cells.append((row, col, (x1, y1, x2, y2), image.crop((x1, y1, x2, y2))))
    return cells


def apply_piece_halo(
    canvas: Image.Image,
    center_x: float,
    center_y: float,
    piece_min_edge: float,
    rng: random.Random,
    cfg: Mapping[str, Any],
) -> Image.Image:
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    outer_alpha = rng.randint(
        int(cfg.get("alpha_outer_min", 22)),
        int(cfg.get("alpha_outer_max", 58)),
    )
    radius = max(4.0, piece_min_edge * float(cfg.get("radius_frac", 0.5)))
    layers = max(2, int(cfg.get("ring_layers", 4)))
    color = (255, 72, 72) if rng.random() < float(cfg.get("red_tint_prob", 0.45)) else (255, 255, 255)
    cx = int(round(center_x))
    cy = int(round(center_y))
    for index in range(layers):
        alpha = max(6, outer_alpha - index * max(1, outer_alpha // layers))
        rr = max(2, int(round(radius * (1.0 - 0.16 * index))))
        draw.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=(*color, alpha))
    return Image.alpha_composite(canvas.convert("RGBA"), overlay)


def compose_board_with_pieces(
    board_rgb: Image.Image,
    roi: Roi,
    labels: list[list[str]],
    pieces_rgba: dict[str, Image.Image],
    *,
    piece_scale_in_cell: float,
    paste_jitter_frac: float,
    rng: random.Random,
    piece_halo_cfg: Mapping[str, Any] | None = None,
) -> Image.Image:
    canvas = board_rgb.convert("RGBA")
    dx, dy = grid_roi_spacing(roi)
    max_pw = dx * piece_scale_in_cell
    max_ph = dy * piece_scale_in_cell
    halo_enabled = bool(piece_halo_cfg and piece_halo_cfg.get("enabled", False))
    halo_prob = float(piece_halo_cfg.get("prob", 0.0)) if piece_halo_cfg else 0.0
    for row in range(10):
        for col in range(9):
            label = labels[row][col]
            if label == "empty":
                continue
            piece = pieces_rgba[label]
            scale = min(max_pw / piece.size[0], max_ph / piece.size[1])
            size = (
                max(1, int(round(piece.size[0] * scale))),
                max(1, int(round(piece.size[1] * scale))),
            )
            resized = piece.resize(size, Image.Resampling.LANCZOS)
            cx_base, cy_base = grid_point_center(roi, row, col)
            cx = cx_base + rng.uniform(-paste_jitter_frac, paste_jitter_frac) * dx
            cy = cy_base + rng.uniform(-paste_jitter_frac, paste_jitter_frac) * dy
            x = int(round(cx - size[0] / 2))
            y = int(round(cy - size[1] / 2))
            if halo_enabled and rng.random() < halo_prob:
                canvas = apply_piece_halo(
                    canvas,
                    center_x=x + 0.5 * size[0],
                    center_y=y + 0.5 * size[1],
                    piece_min_edge=min(size),
                    rng=rng,
                    cfg=piece_halo_cfg,
                )
            canvas.paste(resized, (x, y), resized)
    return canvas.convert("RGB")


def apply_realistic_nuisance(
    image: Image.Image,
    rng: random.Random,
    cfg: Mapping[str, Any] | None,
) -> Image.Image:
    if not cfg or not bool(cfg.get("enabled", False)):
        return image
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    if rng.random() < float(cfg.get("stain_prob", 0.085)):
        for _ in range(rng.randint(1, int(cfg.get("stain_max_per_image", 4)))):
            radius = rng.randint(int(cfg.get("stain_radius_px_min", 2)), int(cfg.get("stain_radius_px_max", 8)))
            alpha = int(255 * float(cfg.get("stain_alpha_max", 0.26)))
            x = rng.randint(radius, max(radius, image.size[0] - radius))
            y = rng.randint(radius, max(radius, image.size[1] - radius))
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(40, 32, 20, alpha))
    if rng.random() < float(cfg.get("edge_occlusion_prob", 0.065)):
        strip = rng.randint(int(cfg.get("edge_strip_px_min", 2)), int(cfg.get("edge_strip_px_max", 10)))
        cover_frac = rng.uniform(
            float(cfg.get("edge_cover_frac_min", 0.16)),
            float(cfg.get("edge_cover_frac_max", 0.36)),
        )
        side = rng.choice(("top", "bottom", "left", "right"))
        if side in ("top", "bottom"):
            cover = int(round(image.size[0] * cover_frac))
            start = rng.randint(0, max(0, image.size[0] - cover))
            box = (start, 0, start + cover, strip) if side == "top" else (start, image.size[1] - strip, start + cover, image.size[1])
        else:
            cover = int(round(image.size[1] * cover_frac))
            start = rng.randint(0, max(0, image.size[1] - cover))
            box = (0, start, strip, start + cover) if side == "left" else (image.size[0] - strip, start, image.size[0], start + cover)
        draw.rectangle(box, fill=(15, 15, 15, 70))
    return Image.alpha_composite(base, overlay).convert("RGB")


def apply_empty_marker_nuisance(
    image: Image.Image,
    labels: list[list[str]],
    roi: Roi,
    rng: random.Random,
    cfg: Mapping[str, Any] | None,
) -> Image.Image:
    if not cfg or not bool(cfg.get("enabled", False)):
        return image
    out = image.convert("RGBA")
    draw = ImageDraw.Draw(out, "RGBA")
    max_markers = int(cfg.get("max_markers_per_board", 28))
    modes = list(cfg.get("color_modes", ["white"]))
    marked = 0
    color_map = {"white": (255, 255, 255), "gold": (235, 191, 87), "orange": (240, 150, 60), "ui_red": (255, 92, 92)}
    for row in range(10):
        for col in range(9):
            if labels[row][col] != "empty":
                continue
            if marked >= max_markers or rng.random() >= float(cfg.get("prob_empty_cell", 0.16)):
                continue
            cx_base, cy_base = grid_point_center(roi, row, col)
            cx = int(round(cx_base + rng.uniform(-1.4, 1.4)))
            cy = int(round(cy_base + rng.uniform(-1.4, 1.4)))
            radius = rng.randint(int(cfg.get("radius_px_min", 5)), int(cfg.get("radius_px_max", 18)))
            alpha = rng.randint(int(cfg.get("alpha_min", 90)), int(cfg.get("alpha_max", 210)))
            color = color_map[rng.choice(modes)]
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=(*color, alpha), width=2)
            marked += 1
    return out.convert("RGB")


def _sample_margin_px(
    rng: random.Random,
    base: int,
    cfg: Mapping[str, Any],
    name: str,
    *,
    default_min: float,
    default_max: float,
) -> int:
    low = float(cfg.get(f"{name}_min", default_min))
    high = float(cfg.get(f"{name}_max", default_max))
    if high < low:
        low, high = high, low
    return int(round(base * rng.uniform(low, high)))


def _sample_background_palette(rng: random.Random) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    palettes = [
        ((236, 225, 206), (214, 194, 167)),
        ((228, 232, 239), (192, 201, 217)),
        ((216, 229, 224), (176, 197, 191)),
        ((238, 235, 228), (205, 198, 188)),
        ((223, 218, 232), (188, 182, 208)),
        ((241, 229, 214), (212, 189, 162)),
        ((228, 236, 221), (192, 206, 185)),
    ]
    return rng.choice(palettes)


def _make_gradient_background(size: tuple[int, int], top_color: tuple[int, int, int], bottom_color: tuple[int, int, int]) -> Image.Image:
    width, height = size
    top = np.array(top_color, dtype=np.float32)
    bottom = np.array(bottom_color, dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    rows = top * (1.0 - ramp) + bottom * ramp
    image = np.repeat(rows[:, None, :], width, axis=1).astype(np.uint8)
    return Image.fromarray(image, mode="RGB")


def _decorate_background(
    image: Image.Image,
    *,
    rng: random.Random,
    cfg: Mapping[str, Any],
    clean_scene: bool,
) -> Image.Image:
    if clean_scene:
        return image
    out = image.convert("RGBA")
    draw = ImageDraw.Draw(out, "RGBA")
    width, height = out.size
    if rng.random() < float(cfg.get("background_photo_blobs_prob", 0.0)):
        for _ in range(rng.randint(int(cfg.get("background_photo_blobs_min", 2)), int(cfg.get("background_photo_blobs_max", 5)))):
            radius = rng.randint(max(28, width // 8), max(40, width // 3))
            cx = _randint_safe(rng, -radius // 3, max(0, width + radius // 3))
            cy = _randint_safe(rng, -radius // 3, max(0, height + radius // 3))
            color = rng.choice(
                [
                    (255, 210, 170, 42),
                    (120, 170, 230, 36),
                    (180, 120, 210, 34),
                    (90, 180, 140, 30),
                    (240, 190, 120, 38),
                ]
            )
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)
    if rng.random() < float(cfg.get("background_blocks_prob", 0.55)):
        for _ in range(rng.randint(int(cfg.get("background_blocks_min", 2)), int(cfg.get("background_blocks_max", 6)))):
            bw = rng.randint(max(24, width // 10), max(32, width // 3))
            bh = rng.randint(max(12, height // 18), max(18, height // 7))
            x = _randint_safe(rng, 0, max(0, width - bw))
            y = _randint_safe(rng, 0, max(0, height - bh))
            color = rng.choice(
                [
                    (255, 255, 255, 38),
                    (20, 20, 20, 28),
                    (94, 122, 180, 34),
                    (180, 98, 82, 30),
                ]
            )
            draw.rounded_rectangle((x, y, x + bw, y + bh), radius=max(4, min(bw, bh) // 8), fill=color)
    if rng.random() < float(cfg.get("background_lines_prob", 0.45)):
        for _ in range(rng.randint(int(cfg.get("background_lines_min", 3)), int(cfg.get("background_lines_max", 8)))):
            horizontal = rng.random() < 0.6
            alpha = rng.randint(18, 42)
            if horizontal:
                y = _randint_safe(rng, 0, max(0, height - 1))
                thickness = rng.randint(1, 3)
                draw.rectangle((0, y, width, min(height, y + thickness)), fill=(255, 255, 255, alpha))
            else:
                x = _randint_safe(rng, 0, max(0, width - 1))
                thickness = rng.randint(1, 3)
                draw.rectangle((x, 0, min(width, x + thickness), height), fill=(255, 255, 255, alpha))
    return out.convert("RGB")


def _add_background_texture(image: Image.Image, rng: random.Random, cfg: Mapping[str, Any]) -> Image.Image:
    if rng.random() >= float(cfg.get("background_texture_prob", 0.0)):
        return image
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    strength = float(cfg.get("background_texture_strength", 10.0))
    rng_np = np.random.default_rng(rng.randint(0, 2**31 - 1))
    noise = rng_np.uniform(-strength, strength, size=arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8), mode="RGB")


def _load_ui_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    for raw in candidates:
        path = Path(raw)
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _draw_scene_text_nuisance(
    scene: Image.Image,
    board_bbox: tuple[int, int, int, int],
    *,
    rng: random.Random,
    cfg: Mapping[str, Any],
    clean_scene: bool,
) -> Image.Image:
    if clean_scene or rng.random() >= float(cfg.get("text_nuisance_prob", 0.0)):
        return scene
    tokens = list(
        cfg.get(
            "text_tokens",
            ["车", "马", "炮", "士", "象", "新", "游", "戏", "悔", "棋", "沙", "盘", "上", "一", "步", "下", "棋", "盘", "分", "析", "库"],
        )
    )
    if not tokens:
        return scene
    out = scene.convert("RGBA")
    draw = ImageDraw.Draw(out, "RGBA")
    width, height = out.size
    x1, y1, x2, y2 = board_bbox
    text_count = rng.randint(int(cfg.get("text_count_min", 1)), int(cfg.get("text_count_max", 3)))
    for _ in range(text_count):
        token = rng.choice(tokens)
        font_size = rng.randint(int(cfg.get("text_size_min", 12)), int(cfg.get("text_size_max", 28)))
        font = _load_ui_font(font_size)
        text_box = draw.textbbox((0, 0), token, font=font)
        text_w = max(1, text_box[2] - text_box[0])
        text_h = max(1, text_box[3] - text_box[1])
        region = rng.choice(("top", "bottom", "left", "right"))
        if region == "top":
            tx = _randint_safe(rng, 4, max(4, width - text_w - 4))
            ty = _randint_safe(rng, 2, max(2, y1 - text_h - 4))
        elif region == "bottom":
            tx = _randint_safe(rng, 4, max(4, width - text_w - 4))
            ty = _randint_safe(rng, y2 + 4, max(4, height - text_h - 4))
        elif region == "left":
            tx = _randint_safe(rng, 2, max(2, x1 - text_w - 4))
            ty = _randint_safe(rng, 4, max(4, height - text_h - 4))
        elif region == "right":
            tx = _randint_safe(rng, x2 + 4, max(4, width - text_w - 4))
            ty = _randint_safe(rng, 4, max(4, height - text_h - 4))
        else:
            continue
        text_box_xyxy = (tx, ty, tx + text_w, ty + text_h)
        if _box_intersects_board(
            text_box_xyxy,
            board_bbox,
            scene_size=(width, height),
            padding=_board_safe_padding(cfg),
        ):
            continue
        fill = rng.choice(
            [
                (35, 35, 35, 155),
                (255, 255, 255, 170),
                (180, 64, 64, 165),
                (70, 110, 185, 150),
            ]
        )
        draw.text((tx + 1, ty + 1), token, font=font, fill=(0, 0, 0, max(60, fill[3] - 70)))
        draw.text((tx, ty), token, font=font, fill=fill)
    return out.convert("RGB")


def _draw_portrait_nuisance(
    scene: Image.Image,
    board_bbox: tuple[int, int, int, int],
    rng: random.Random,
    cfg: Mapping[str, Any],
) -> Image.Image:
    if rng.random() >= float(cfg.get("portrait_prob", 0.0)):
        return scene
    out = scene.convert("RGBA")
    draw = ImageDraw.Draw(out, "RGBA")
    width, height = out.size
    x1, y1, x2, y2 = board_bbox
    portrait_count = rng.randint(int(cfg.get("portrait_count_min", 1)), int(cfg.get("portrait_count_max", 2)))
    safe_padding = _board_safe_padding(cfg)
    for _ in range(portrait_count):
        radius = max(18, int(round(rng.uniform(0.06, 0.14) * min(width, height))))
        frame = radius + rng.randint(4, 10)
        placed = False
        for _attempt in range(16):
            region = rng.choice(("left", "right", "top"))
            if region == "left" and x1 > frame + safe_padding + 8:
                cx = _randint_safe(rng, frame + 6, max(frame + 6, x1 - frame - safe_padding - 8))
                cy = _randint_safe(rng, max(frame + 4, 4), max(frame + 4, height - frame - 4))
            elif region == "right" and x2 < width - frame - safe_padding - 8:
                cx = _randint_safe(rng, x2 + frame + safe_padding + 8, max(x2 + frame + 8, width - frame - 6))
                cy = _randint_safe(rng, max(frame + 4, 4), max(frame + 4, height - frame - 4))
            elif y1 > frame + safe_padding + 8:
                cx = _randint_safe(rng, max(frame + 4, 4), max(frame + 4, width - frame - 4))
                cy = _randint_safe(rng, frame + 4, max(frame + 4, y1 - frame - safe_padding - 6))
            else:
                continue
            portrait_box = (cx - frame, cy - frame, cx + frame, cy + frame)
            if _box_intersects_board(
                portrait_box,
                board_bbox,
                scene_size=(width, height),
                padding=safe_padding,
            ):
                continue
            placed = True
            break
        if not placed:
            continue
        skin = rng.choice([(235, 198, 168), (210, 170, 140), (180, 130, 110), (150, 110, 95)])
        hair = rng.choice([(40, 30, 25), (70, 45, 30), (20, 20, 20), (120, 80, 50)])
        draw.ellipse((cx - frame, cy - frame, cx + frame, cy + frame), fill=(255, 255, 255, 210), outline=(220, 180, 90, 220), width=3)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(*skin, 255))
        draw.pieslice((cx - radius, cy - radius - radius // 2, cx + radius, cy + radius // 2), 180, 360, fill=(*hair, 255))
        draw.ellipse((cx - radius // 3, cy - radius // 8, cx - radius // 8, cy + radius // 8), fill=(40, 30, 25, 180))
        draw.ellipse((cx + radius // 8, cy - radius // 8, cx + radius // 3, cy + radius // 8), fill=(40, 30, 25, 180))
    return out.convert("RGB")


def _draw_icon_cluster_nuisance(
    scene: Image.Image,
    board_bbox: tuple[int, int, int, int],
    rng: random.Random,
    cfg: Mapping[str, Any],
) -> Image.Image:
    if rng.random() >= float(cfg.get("icon_cluster_prob", 0.0)):
        return scene
    out = scene.convert("RGBA")
    draw = ImageDraw.Draw(out, "RGBA")
    width, height = out.size
    x1, _y1, x2, y2 = board_bbox
    cluster_count = rng.randint(int(cfg.get("icon_cluster_count_min", 1)), int(cfg.get("icon_cluster_count_max", 3)))
    palette = [(255, 92, 92), (92, 160, 255), (255, 196, 72), (120, 200, 120), (180, 120, 220)]
    safe_padding = _board_safe_padding(cfg)
    for _ in range(cluster_count):
        cols = rng.randint(int(cfg.get("icon_cluster_cols_min", 3)), int(cfg.get("icon_cluster_cols_max", 6)))
        rows = rng.randint(int(cfg.get("icon_cluster_rows_min", 2)), int(cfg.get("icon_cluster_rows_max", 4)))
        icon_size = max(8, int(round(rng.uniform(0.018, 0.032) * min(width, height))))
        gap = max(3, icon_size // 3)
        cluster_w = cols * icon_size + (cols - 1) * gap
        cluster_h = rows * icon_size + (rows - 1) * gap
        placed = False
        for _attempt in range(16):
            anchor = rng.choice(("top_left", "top_right", "bottom_left", "bottom_right", "side_left", "side_right"))
            if anchor == "top_left":
                ox, oy = 6, 6
            elif anchor == "top_right":
                ox, oy = width - cluster_w - 6, 6
            elif anchor == "bottom_left":
                ox, oy = 6, height - cluster_h - 6
            elif anchor == "bottom_right":
                ox, oy = width - cluster_w - 6, height - cluster_h - 6
            elif anchor == "side_left":
                if x1 <= cluster_w + safe_padding + 8:
                    continue
                ox = _randint_safe(rng, 4, max(4, x1 - cluster_w - safe_padding - 4))
                oy = _randint_safe(rng, 4, max(4, height - cluster_h - 4))
            else:
                if x2 >= width - cluster_w - safe_padding - 8:
                    continue
                ox = _randint_safe(rng, x2 + safe_padding + 4, max(x2 + 4, width - cluster_w - 4))
                oy = _randint_safe(rng, 4, max(4, height - cluster_h - 4))
            cluster_box = (ox - 4, oy - 4, ox + cluster_w + 4, oy + cluster_h + 4)
            if _box_intersects_board(
                cluster_box,
                board_bbox,
                scene_size=(width, height),
                padding=safe_padding,
            ):
                continue
            placed = True
            break
        if not placed:
            continue
        draw.rounded_rectangle(
            (ox - 4, oy - 4, ox + cluster_w + 4, oy + cluster_h + 4),
            radius=6,
            fill=(255, 255, 255, 120),
            outline=(180, 180, 180, 150),
            width=1,
        )
        for row in range(rows):
            for col in range(cols):
                ix = ox + col * (icon_size + gap)
                iy = oy + row * (icon_size + gap)
                color = rng.choice(palette)
                draw.rounded_rectangle(
                    (ix, iy, ix + icon_size, iy + icon_size),
                    radius=max(2, icon_size // 4),
                    fill=(*color, rng.randint(150, 230)),
                )
    return out.convert("RGB")


def _draw_semi_transparent_overlay(
    scene: Image.Image,
    board_bbox: tuple[int, int, int, int],
    rng: random.Random,
    cfg: Mapping[str, Any],
) -> Image.Image:
    if rng.random() >= float(cfg.get("semi_transparent_overlay_prob", 0.0)):
        return scene
    out = scene.convert("RGBA")
    width, height = out.size
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    mode = rng.choice(("vignette", "top_wash", "bottom_wash", "side_wash"))
    alpha = rng.randint(int(cfg.get("overlay_alpha_min", 28)), int(cfg.get("overlay_alpha_max", 72)))
    x1, y1, x2, y2 = board_bbox
    if mode == "vignette":
        margin_x = int(round(width * rng.uniform(0.08, 0.18)))
        margin_y = int(round(height * rng.uniform(0.08, 0.18)))
        draw.rectangle((0, 0, width, min(margin_y, max(0, y1))), fill=(0, 0, 0, alpha))
        draw.rectangle((0, max(y2, height - margin_y), width, height), fill=(0, 0, 0, alpha))
        draw.rectangle((0, 0, min(margin_x, max(0, x1)), height), fill=(0, 0, 0, alpha))
        draw.rectangle((max(x2, width - margin_x), 0, width, height), fill=(0, 0, 0, alpha))
    elif mode == "top_wash":
        wash_h = int(round(height * rng.uniform(0.12, 0.28)))
        color = rng.choice([(20, 20, 30), (30, 50, 90), (60, 30, 30)])
        draw.rectangle((0, 0, width, min(wash_h, max(0, y1))), fill=(*color, alpha))
    elif mode == "bottom_wash":
        wash_h = int(round(height * rng.uniform(0.14, 0.32)))
        color = rng.choice([(20, 20, 30), (30, 50, 90), (60, 30, 30)])
        top = max(y2, height - wash_h)
        draw.rectangle((0, top, width, height), fill=(*color, alpha))
    else:
        side = rng.choice(("left", "right"))
        wash_w = int(round(width * rng.uniform(0.1, 0.22)))
        color = rng.choice([(20, 20, 30), (30, 50, 90), (60, 30, 30)])
        if side == "left":
            draw.rectangle((0, 0, min(wash_w, max(0, x1)), height), fill=(*color, alpha))
        else:
            draw.rectangle((max(x2, width - wash_w), 0, width, height), fill=(*color, alpha))
    overlay = _punch_board_hole_in_overlay(
        overlay,
        board_bbox,
        scene_size=(width, height),
        padding=_board_safe_padding(cfg),
    )
    return Image.alpha_composite(out, overlay).convert("RGB")


def _fit_scene_background(size: tuple[int, int], rng: random.Random, cfg: Mapping[str, Any], *, clean_scene: bool) -> Image.Image:
    top_color, bottom_color = _sample_background_palette(rng)
    background = _make_gradient_background(size, top_color, bottom_color)
    background = _decorate_background(background, rng=rng, cfg=cfg, clean_scene=clean_scene)
    background = _add_background_texture(background, rng, cfg)
    blur_prob = float(cfg.get("background_blur_prob", 0.35))
    blur_radius = 0.0
    if not clean_scene and rng.random() < blur_prob:
        blur_radius = rng.uniform(
            float(cfg.get("background_blur_radius_min", 1.0)),
            float(cfg.get("background_blur_radius_max", 6.0)),
        )
    if blur_radius > 0.0:
        background = background.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    shade_alpha = 0 if clean_scene else int(
        rng.uniform(
            float(cfg.get("background_shade_alpha_min", 8)),
            float(cfg.get("background_shade_alpha_max", 42)),
        )
    )
    shade = Image.new("RGBA", size, (0, 0, 0, shade_alpha))
    return Image.alpha_composite(background.convert("RGBA"), shade).convert("RGB")


def _resize_with_scale(image: Image.Image, scale: float) -> Image.Image:
    return image.resize(
        (
            max(32, int(round(image.size[0] * scale))),
            max(32, int(round(image.size[1] * scale))),
        ),
        Image.Resampling.LANCZOS,
    )


def _paste_shadowed_image(
    scene: Image.Image,
    overlay_image: Image.Image,
    *,
    x: int,
    y: int,
    rng: random.Random,
    shadow_alpha_range: tuple[int, int] = (48, 96),
) -> Image.Image:
    out = scene.convert("RGBA")
    shadow = Image.new("RGBA", out.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow, "RGBA")
    spread = max(4, int(round(0.018 * min(overlay_image.size))))
    dx = rng.randint(-2, 4)
    dy = rng.randint(3, 8)
    shadow_draw.rounded_rectangle(
        (x + dx - spread, y + dy - spread, x + overlay_image.size[0] + dx + spread, y + overlay_image.size[1] + dy + spread),
        radius=max(6, spread * 2),
        fill=(0, 0, 0, rng.randint(*shadow_alpha_range)),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=max(3, spread)))
    out = Image.alpha_composite(out, shadow)
    out.paste(overlay_image.convert("RGBA"), (x, y))
    return out.convert("RGB")


def _boxes_overlap(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], padding: int = 0) -> bool:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return not (ax2 + padding <= bx1 or bx2 + padding <= ax1 or ay2 + padding <= by1 or by2 + padding <= ay1)


def _protected_board_box(
    board_bbox: tuple[int, int, int, int],
    *,
    scene_size: tuple[int, int],
    padding: int = 4,
) -> tuple[int, int, int, int]:
    width, height = scene_size
    x1, y1, x2, y2 = board_bbox
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(width, x2 + padding),
        min(height, y2 + padding),
    )


def _box_intersects_board(
    box: tuple[int, int, int, int],
    board_bbox: tuple[int, int, int, int],
    *,
    scene_size: tuple[int, int],
    padding: int = 4,
) -> bool:
    protected = _protected_board_box(board_bbox, scene_size=scene_size, padding=padding)
    return _boxes_overlap(box, protected, padding=0)


def _board_safe_padding(cfg: Mapping[str, Any]) -> int:
    return max(0, int(cfg.get("board_safe_padding", 4)))


def _punch_board_hole_in_overlay(
    overlay: Image.Image,
    board_bbox: tuple[int, int, int, int],
    *,
    scene_size: tuple[int, int],
    padding: int,
) -> Image.Image:
    x1, y1, x2, y2 = _protected_board_box(board_bbox, scene_size=scene_size, padding=padding)
    if x2 <= x1 or y2 <= y1:
        return overlay
    arr = np.array(overlay.convert("RGBA"), dtype=np.uint8)
    arr[y1:y2, x1:x2, 3] = 0
    return Image.fromarray(arr, mode="RGBA")


def _randint_safe(rng: random.Random, low: int, high: int) -> int:
    if high < low:
        low, high = high, low
    return rng.randint(low, high)


def _expand_scene_to_aspect(
    scene: Image.Image,
    board_box: tuple[int, int, int, int],
    *,
    rng: random.Random,
    cfg: Mapping[str, Any],
    clean_scene: bool,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    wide_prob = float(cfg.get("wide_screen_prob", 0.5))
    is_wide = rng.random() < wide_prob
    if is_wide:
        aspect = rng.uniform(float(cfg.get("wide_aspect_min", 1.45)), float(cfg.get("wide_aspect_max", 2.2)))
    else:
        aspect = rng.uniform(float(cfg.get("narrow_aspect_min", 0.5)), float(cfg.get("narrow_aspect_max", 0.82)))

    width, height = scene.size
    target_width = width
    target_height = height
    current_aspect = width / max(height, 1)
    if current_aspect < aspect:
        target_width = max(width, int(round(height * aspect)))
    else:
        target_height = max(height, int(round(width / aspect)))

    if target_width == width and target_height == height:
        return scene, board_box

    pad_left = _randint_safe(rng, 0, max(0, target_width - width))
    pad_top = _randint_safe(rng, 0, max(0, target_height - height))

    background = _fit_scene_background((target_width, target_height), rng, cfg, clean_scene=clean_scene)
    background.paste(scene, (pad_left, pad_top))
    x1, y1, x2, y2 = board_box
    shifted_box = (x1 + pad_left, y1 + pad_top, x2 + pad_left, y2 + pad_top)
    return background, shifted_box


def _add_secondary_boards(
    scene: Image.Image,
    main_board_image: Image.Image,
    main_board_box: tuple[int, int, int, int],
    *,
    rng: random.Random,
    cfg: Mapping[str, Any],
) -> Image.Image:
    if rng.random() >= float(cfg.get("secondary_board_prob", 0.0)):
        return scene

    out = scene
    width, height = scene.size
    secondary_count = rng.randint(int(cfg.get("secondary_board_count_min", 1)), int(cfg.get("secondary_board_count_max", 2)))
    base_min_edge = min(main_board_image.size)
    for _ in range(secondary_count):
        scale = rng.uniform(float(cfg.get("secondary_scale_min", 0.22)), float(cfg.get("secondary_scale_max", 0.62)))
        rotated = _resize_with_scale(main_board_image, scale).rotate(
            rng.uniform(float(cfg.get("secondary_rotation_min", -8.0)), float(cfg.get("secondary_rotation_max", 8.0))),
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=edge_mean_color(main_board_image),
        )
        if min(rotated.size) >= base_min_edge:
            forced_scale = max(0.16, (base_min_edge - 8) / max(min(rotated.size), 1))
            rotated = _resize_with_scale(rotated, forced_scale)

        candidates = [
            (
                _randint_safe(rng, 0, max(0, width - rotated.size[0])),
                _randint_safe(rng, 0, max(0, main_board_box[1] - rotated.size[1] - 8)),
            ),
            (
                _randint_safe(rng, 0, max(0, width - rotated.size[0])),
                _randint_safe(rng, main_board_box[3] + 8, max(0, height - rotated.size[1])),
            ),
            (
                _randint_safe(rng, 0, max(0, main_board_box[0] - rotated.size[0] - 8)),
                _randint_safe(rng, 0, max(0, height - rotated.size[1])),
            ),
            (
                _randint_safe(rng, main_board_box[2] + 8, max(0, width - rotated.size[0])),
                _randint_safe(rng, 0, max(0, height - rotated.size[1])),
            ),
        ]
        placed = False
        for x, y in candidates:
            candidate_box = (x, y, x + rotated.size[0], y + rotated.size[1])
            if not _boxes_overlap(candidate_box, main_board_box, padding=12):
                out = _paste_shadowed_image(out, rotated, x=x, y=y, rng=rng, shadow_alpha_range=(28, 72))
                placed = True
                break
        if not placed:
            x = _randint_safe(rng, 0, max(0, width - rotated.size[0]))
            y = _randint_safe(rng, 0, max(0, height - rotated.size[1]))
            candidate_box = (x, y, x + rotated.size[0], y + rotated.size[1])
            if not _boxes_overlap(candidate_box, main_board_box, padding=4):
                out = _paste_shadowed_image(out, rotated, x=x, y=y, rng=rng, shadow_alpha_range=(28, 72))
    return out


def _draw_scene_ui_nuisance(
    scene: Image.Image,
    board_bbox: tuple[int, int, int, int],
    rng: random.Random,
    cfg: Mapping[str, Any],
) -> Image.Image:
    if not bool(cfg.get("enabled", False)):
        return scene
    out = scene.convert("RGBA")
    draw = ImageDraw.Draw(out, "RGBA")
    width, height = out.size
    x1, y1, x2, y2 = board_bbox

    if rng.random() < float(cfg.get("top_bar_prob", 0.45)) and y1 > 24:
        bar_h = max(12, int(round(rng.uniform(0.035, 0.08) * height)))
        margin = max(6, int(round(0.018 * width)))
        top = rng.randint(4, max(4, y1 - bar_h - 4))
        draw.rounded_rectangle((margin, top, width - margin, top + bar_h), radius=max(6, bar_h // 3), fill=(245, 245, 245, 228))
        icon_count = rng.randint(int(cfg.get("toolbar_icon_count_min", 4)), int(cfg.get("toolbar_icon_count_max", 10)))
        icon_size = max(8, int(round(bar_h * 0.42)))
        for index in range(icon_count):
            icon_x = margin + 8 + index * (icon_size + 8)
            if icon_x + icon_size >= width - margin - 4:
                break
            icon_y = top + max(3, (bar_h - icon_size) // 2)
            draw.rounded_rectangle(
                (icon_x, icon_y, icon_x + icon_size, icon_y + icon_size),
                radius=max(2, icon_size // 4),
                fill=(255, 255, 255, 180),
                outline=(160, 160, 160, 220),
                width=1,
            )
        if rng.random() < float(cfg.get("top_status_strip_prob", 0.55)):
            strip_w = rng.randint(max(48, width // 5), max(64, width // 3))
            strip_x = _randint_safe(rng, margin + 8, max(margin + 8, width - margin - strip_w - 8))
            strip_y = top + max(2, (bar_h - 6) // 2)
            draw.rounded_rectangle((strip_x, strip_y, strip_x + strip_w, strip_y + 6), radius=3, fill=(220, 70, 70, 200))
            draw.rounded_rectangle(
                (strip_x + strip_w // 3, strip_y, strip_x + strip_w, strip_y + 6),
                radius=3,
                fill=(255, 196, 72, 190),
            )
    if rng.random() < float(cfg.get("bottom_panel_prob", 0.55)) and y2 < height - 24:
        panel_h = max(16, int(round(rng.uniform(0.08, 0.2) * height)))
        margin = max(6, int(round(0.03 * width)))
        top_min = max(y2 + 4, 4)
        top_max = height - panel_h - 4
        if top_min <= top_max:
            top = rng.randint(top_min, top_max)
            draw.rounded_rectangle((margin, top, width - margin, top + panel_h), radius=max(8, panel_h // 5), fill=(248, 248, 248, 210))
            row_count = rng.randint(2, 4)
            for row in range(row_count):
                row_y = top + 6 + row * max(8, panel_h // max(row_count + 1, 2))
                row_x2 = rng.randint(max(margin + 24, width // 3), width - margin - 12)
                draw.rounded_rectangle((margin + 10, row_y, row_x2, row_y + 4), radius=2, fill=(205, 70, 70, 210))
            if rng.random() < float(cfg.get("bottom_action_buttons_prob", 0.65)):
                btn_count = rng.randint(int(cfg.get("bottom_action_buttons_min", 3)), int(cfg.get("bottom_action_buttons_max", 6)))
                btn_w = max(28, int(round((width - 2 * margin - 12) / max(btn_count, 1) - 8)))
                btn_h = max(14, int(round(panel_h * 0.28)))
                for index in range(btn_count):
                    bx = margin + 8 + index * (btn_w + 8)
                    by = top + panel_h - btn_h - 8
                    if bx + btn_w >= width - margin:
                        break
                    draw.rounded_rectangle(
                        (bx, by, bx + btn_w, by + btn_h),
                        radius=6,
                        fill=rng.choice([(255, 255, 255, 220), (214, 73, 73, 210), (65, 120, 255, 200)]),
                        outline=(160, 160, 160, 200),
                        width=1,
                    )
    if rng.random() < float(cfg.get("side_panel_prob", 0.7)):
        panel_w = max(26, int(round(rng.uniform(0.08, 0.18) * width)))
        panel_h = max(48, int(round(rng.uniform(0.16, 0.34) * height)))
        side = rng.choice(("left", "right"))
        if side == "left" and x1 > panel_w + 8:
            x = _randint_safe(rng, 4, max(4, x1 - panel_w - 6))
        else:
            x = _randint_safe(rng, x2 + 6, max(4, width - panel_w - 4))
        y = _randint_safe(rng, 4, max(4, height - panel_h - 4))
        draw.rounded_rectangle(
            (x, y, x + panel_w, y + panel_h),
            radius=max(8, panel_w // 5),
            fill=(252, 252, 252, 220),
            outline=(180, 180, 180, 210),
            width=1,
        )
        button_count = rng.randint(3, 7)
        button_h = max(10, int(round((panel_h - 12) / max(button_count, 1))))
        for index in range(button_count):
            by = y + 6 + index * button_h
            draw.rounded_rectangle(
                (x + 6, by, x + panel_w - 6, min(y + panel_h - 6, by + button_h - 4)),
                radius=4,
                fill=(255, 255, 255, 205),
            )
    if rng.random() < float(cfg.get("floating_badge_prob", 0.5)):
        radius = max(10, int(round(rng.uniform(0.025, 0.065) * min(width, height))))
        safe_padding = _board_safe_padding(cfg)
        for _attempt in range(12):
            cx = rng.randint(radius + 4, width - radius - 4)
            cy_candidates = [
                max(radius + 4, y1 - radius - safe_padding - 8),
                min(height - radius - 4, y2 + radius + safe_padding + 8),
            ]
            cy = rng.choice(cy_candidates)
            badge_box = (cx - radius, cy - radius, cx + radius, cy + radius)
            if _box_intersects_board(
                badge_box,
                board_bbox,
                scene_size=(width, height),
                padding=safe_padding,
            ):
                continue
            draw.ellipse(
                (cx - radius, cy - radius, cx + radius, cy + radius),
                fill=(255, 255, 255, 160),
                outline=(250, 210, 120, 180),
                width=2,
            )
            break
    if rng.random() < float(cfg.get("board_occluder_prob", 0.48)):
        occluder_count = rng.randint(int(cfg.get("board_occluder_count_min", 1)), int(cfg.get("board_occluder_count_max", 3)))
        for _ in range(occluder_count):
            occ_w = rng.randint(max(18, int((x2 - x1) * 0.08)), max(26, int((x2 - x1) * 0.22)))
            occ_h = rng.randint(max(12, int((y2 - y1) * 0.04)), max(18, int((y2 - y1) * 0.1)))
            occ_x = rng.randint(max(0, x1 - occ_w // 3), min(width - occ_w, x2 - occ_w // 2))
            occ_y = rng.randint(max(0, y1 - occ_h // 3), min(height - occ_h, y2 - occ_h // 2))
            fill = rng.choice(
                [
                    (255, 255, 255, 210),
                    (30, 30, 30, 180),
                    (214, 73, 73, 205),
                    (65, 120, 255, 190),
                ]
            )
            draw.rounded_rectangle((occ_x, occ_y, occ_x + occ_w, occ_y + occ_h), radius=max(4, min(occ_w, occ_h) // 4), fill=fill)
    return out.convert("RGB")


def place_board_in_scene(
    board_rgb: Image.Image,
    roi: Roi,
    *,
    rng: random.Random,
    cfg: Mapping[str, Any] | None,
) -> tuple[Image.Image, Roi]:
    if not cfg or not bool(cfg.get("enabled", False)):
        return board_rgb, roi
    clean_scene = rng.random() < float(cfg.get("clean_scene_prob", 0.0))

    board_width, board_height = board_rgb.size
    left = _sample_margin_px(rng, board_width, cfg, "left_margin_frac", default_min=0.0, default_max=0.18)
    right = _sample_margin_px(rng, board_width, cfg, "right_margin_frac", default_min=0.0, default_max=0.18)
    top = _sample_margin_px(rng, board_height, cfg, "top_margin_frac", default_min=0.04, default_max=0.55)
    bottom = _sample_margin_px(rng, board_height, cfg, "bottom_margin_frac", default_min=0.04, default_max=0.6)

    scene_size = (board_width + left + right, board_height + top + bottom)
    scene = _fit_scene_background(scene_size, rng, cfg, clean_scene=clean_scene)

    if not clean_scene and rng.random() < float(cfg.get("board_shadow_prob", 0.7)):
        shadow = Image.new("RGBA", scene_size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow, "RGBA")
        spread = max(6, int(round(rng.uniform(0.012, 0.03) * min(board_width, board_height))))
        shadow_box = (left - spread, top - spread, left + board_width + spread, top + board_height + spread)
        shadow_draw.rounded_rectangle(shadow_box, radius=max(8, spread * 2), fill=(0, 0, 0, int(rng.uniform(35, 70))))
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=max(4, spread)))
        scene = Image.alpha_composite(scene.convert("RGBA"), shadow).convert("RGB")

    if not clean_scene:
        scene = _add_secondary_boards(
            scene,
            board_rgb,
            (left, top, left + board_width, top + board_height),
            rng=rng,
            cfg=cfg,
        )
        scene = _paste_shadowed_image(scene, board_rgb, x=left, y=top, rng=rng, shadow_alpha_range=(44, 92))
    else:
        scene.paste(board_rgb, (left, top))
    board_box = (left, top, left + board_width, top + board_height)
    scene, board_box = _expand_scene_to_aspect(scene, board_box, rng=rng, cfg=cfg, clean_scene=clean_scene)
    board_left, board_top, _, _ = board_box
    placed_roi = Roi(x=roi.x + board_left, y=roi.y + board_top, w=roi.w, h=roi.h)
    if not clean_scene:
        scene = _draw_scene_ui_nuisance(
            scene,
            placed_roi.as_xyxy(),
            rng,
            cfg,
        )
        scene = _draw_portrait_nuisance(scene, placed_roi.as_xyxy(), rng, cfg)
        scene = _draw_icon_cluster_nuisance(scene, placed_roi.as_xyxy(), rng, cfg)
        scene = _draw_scene_text_nuisance(
            scene,
            placed_roi.as_xyxy(),
            rng=rng,
            cfg=cfg,
            clean_scene=clean_scene,
        )
        scene = _draw_semi_transparent_overlay(scene, placed_roi.as_xyxy(), rng, cfg)

    scale_min = float(cfg.get("scene_scale_min", 0.8))
    scale_max = float(cfg.get("scene_scale_max", 1.35))
    if scale_max < scale_min:
        scale_min, scale_max = scale_max, scale_min
    scale = rng.uniform(scale_min, scale_max)
    if abs(scale - 1.0) < 1e-6:
        return scene, placed_roi

    new_width = max(64, int(round(scene.size[0] * scale)))
    new_height = max(64, int(round(scene.size[1] * scale)))
    resized = scene.resize((new_width, new_height), Image.Resampling.LANCZOS)
    scale_x = new_width / scene.size[0]
    scale_y = new_height / scene.size[1]
    scaled_roi = Roi(
        x=int(round(placed_roi.x * scale_x)),
        y=int(round(placed_roi.y * scale_y)),
        w=int(round(placed_roi.w * scale_x)),
        h=int(round(placed_roi.h * scale_y)),
    )
    return resized, scaled_roi


def render_synthetic_board(
    *,
    board_rgb: Image.Image,
    pieces_rgba: dict[str, Image.Image],
    labels: list[list[str]],
    grid_roi_xyxy: Sequence[int] | None,
    pad_fraction: float,
    inset_fraction: Sequence[float],
    piece_scale_in_cell: float,
    rng: random.Random,
    synthetic_augmentation: Mapping[str, Any] | None,
    realistic_nuisance_cfg: Mapping[str, Any] | None,
    piece_halo_cfg: Mapping[str, Any] | None,
    empty_marker_cfg: Mapping[str, Any] | None,
    scene_placement_cfg: Mapping[str, Any] | None,
) -> tuple[Image.Image, Roi]:
    padded, roi = prepare_board_and_sampling_roi(
        board_rgb,
        grid_roi_xyxy=grid_roi_xyxy,
        pad_fraction=pad_fraction,
        inset_fraction=inset_fraction,
    )
    image = compose_board_with_pieces(
        padded,
        roi,
        labels,
        pieces_rgba,
        piece_scale_in_cell=piece_scale_in_cell,
        paste_jitter_frac=float((synthetic_augmentation or {}).get("paste_jitter_frac", 0.035)),
        rng=rng,
        piece_halo_cfg=piece_halo_cfg,
    )
    image = OptionalImageAugment(synthetic_augmentation)(image)
    image = apply_realistic_nuisance(image, rng, realistic_nuisance_cfg)
    image = apply_empty_marker_nuisance(image, labels, roi, rng, empty_marker_cfg)
    image, roi = place_board_in_scene(image, roi, rng=rng, cfg=scene_placement_cfg)
    return image, roi
